"""回答反思冻结评测集、指标和在线门禁测试。"""

from argparse import Namespace
from pathlib import Path

import pytest

from kg_crag.evaluation.grounded import (
    compute_grounded_metrics,
    evaluate_online_grounded_probe,
    evaluate_recorded_grounded_questions,
    load_grounded_questions,
    validate_recorded_web_fixture,
)
from kg_crag.evaluation.grounded_cli import _run, validate_online_gate
from kg_crag.models import AnswerStopReason, GroundedEvaluationItem, GroundedEvaluationReport
from kg_crag.providers import MockLLMProvider, MockSearchProvider
from kg_crag.settings import Settings

PROJECT_ROOT = Path(__file__).parents[2]


def test_frozen_questions_and_recorded_web_fixture_cover_contract() -> None:
    questions = load_grounded_questions(
        PROJECT_ROOT / "data/evaluation/grounded_answer_dev_questions.json"
    )
    assert len(questions.questions) == 12
    assert (
        validate_recorded_web_fixture(
            PROJECT_ROOT / "data/evaluation/fixtures/grounded-web-recorded.json"
        )
        == 2
    )


def test_metrics_keep_failed_items_in_denominators() -> None:
    passed = GroundedEvaluationItem(
        question_id="q1",
        succeeded=True,
        expected_claim_ids=["c1"],
        supported_claim_ids=["c1"],
        expected_citation_ids=["x1"],
        predicted_citation_ids=["x1"],
        valid_citation_ids=["x1"],
        required_facets=1,
        covered_required_facets=1,
        web_eligible=True,
        web_triggered=True,
        web_used=True,
        model_calls=1,
        search_calls=1,
        stop_reason=AnswerStopReason.ACCEPTED,
    )
    failed = GroundedEvaluationItem(
        question_id="q2",
        succeeded=False,
        expected_claim_ids=["c2"],
        expected_citation_ids=["x2"],
        unsupported_claims=1,
        required_facets=1,
        stop_reason=AnswerStopReason.CONSERVATIVE,
    )
    metrics = compute_grounded_metrics([passed, failed])
    assert metrics.total == 2
    assert metrics.failures == 1
    assert metrics.task_score == 0.5
    assert metrics.completeness == 0.5
    assert metrics.faithfulness == 0.5
    assert metrics.citation_precision == 1.0
    assert metrics.citation_recall == 0.5
    assert metrics.unsupported_claim_rate == 0.5


def test_recorded_report_is_identity_cached(tmp_path: Path) -> None:
    questions = load_grounded_questions(
        PROJECT_ROOT / "data/evaluation/grounded_answer_dev_questions.json"
    )
    first = evaluate_recorded_grounded_questions(
        questions, results_root=tmp_path / "results", workspace_root=tmp_path
    )
    second = evaluate_recorded_grounded_questions(
        questions, results_root=tmp_path / "results", workspace_root=tmp_path
    )
    assert second == first
    assert second.metrics.total == 12
    assert GroundedEvaluationReport.model_validate_json(second.model_dump_json()) == second


def test_online_gate_requires_confirmation_limit_and_credentials() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="openai-compatible",
        llm_api_key="test-llm",
        web_search_provider="tavily",
        web_search_api_key="test-search",
    )
    with pytest.raises(ValueError, match="both"):
        validate_online_gate(Namespace(online=True, confirm=False, limit=5), settings)
    with pytest.raises(ValueError, match="between 5 and 10"):
        validate_online_gate(Namespace(online=True, confirm=True, limit=11), settings)
    assert validate_online_gate(Namespace(online=True, confirm=True, limit=5), settings) == 5


@pytest.mark.asyncio
async def test_successful_online_identity_reuses_cached_result(tmp_path: Path) -> None:
    questions = load_grounded_questions(
        PROJECT_ROOT / "data/evaluation/grounded_answer_dev_questions.json"
    )
    llm = MockLLMProvider("bounded answer")
    search = MockSearchProvider()
    arguments = {
        "llm": llm,
        "search": search,
        "limit": 5,
        "provider_identity": "mock-live-v1",
        "results_root": tmp_path / "online",
        "workspace_root": tmp_path,
    }
    first = await evaluate_online_grounded_probe(questions, **arguments)  # type: ignore[arg-type]
    second = await evaluate_online_grounded_probe(questions, **arguments)  # type: ignore[arg-type]
    assert first == second
    assert len(llm.calls) == 5
    assert len(search.calls) == 1


@pytest.mark.asyncio
async def test_cli_dry_run_is_offline(capsys: pytest.CaptureFixture[str]) -> None:
    assert await _run(["--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "budget_preview" in output
    assert "recorded_fixture_queries" in output
