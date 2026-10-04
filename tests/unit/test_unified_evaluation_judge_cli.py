"""Judge 缓存、人工对照、外部门禁和 CLI 脱敏测试。"""

import asyncio
from pathlib import Path

import pytest

from kg_crag.evaluation.config import UnifiedEvaluationConfig
from kg_crag.evaluation.external import validate_external_services
from kg_crag.evaluation.judge import (
    evaluate_with_judge,
    human_judge_agreement,
    validate_judge_gate,
)
from kg_crag.evaluation.unified_cli import _safe_summary, main
from kg_crag.models import (
    EvaluationAnswerPoint,
    EvaluationFacetTarget,
    EvaluationItemResult,
    EvaluationResourceUsage,
    EvaluationSplit,
    EvaluationStageStatus,
    HumanReview,
    JudgeVerdict,
    KnowledgeSufficiency,
    StrategyObservation,
    StressCategory,
    UnifiedEvaluationQuestion,
    UnifiedQuestionType,
    UnifiedStrategy,
)
from kg_crag.workflow.artifacts import ContentAddressedCache


def _close_idle_pytest_loop() -> None:
    """关闭 pytest-asyncio 留给同步测试的空闲 loop，避免 asyncio.run 遗失其 socket。"""

    try:
        loop = asyncio.get_event_loop_policy().get_event_loop()
    except RuntimeError:
        return
    if not loop.is_running():
        loop.close()
        asyncio.set_event_loop(None)


def _question() -> UnifiedEvaluationQuestion:
    return UnifiedEvaluationQuestion(
        question_id="ueq-judge-001",
        split=EvaluationSplit.DEV,
        question="Judge fixture question",
        normalized_question="judge fixture question",
        question_types=[UnifiedQuestionType.FACTOID],
        stress_category=StressCategory.CONTROL,
        knowledge_sufficiency=KnowledgeSufficiency.SUFFICIENT,
        answer_points=[
            EvaluationAnswerPoint(
                point_id="point-0000000000000001",
                description="Expected",
                evidence_ids=["e-1"],
            )
        ],
        facets=[
            EvaluationFacetTarget(
                facet_id="facet-0000000000000001",
                description="Expected",
                target_evidence_ids=["e-1"],
            )
        ],
        relevant_evidence={"e-1": 3},
        minimum_sufficient_evidence_sets=[["e-1"]],
        leakage_group_id="leak-judge-001",
        source_group_ids=["source-judge-001"],
        source_fixture="tests/fixtures/judge.json#1",
    )


def _item() -> EvaluationItemResult:
    observation = StrategyObservation(
        question_id="ueq-judge-001",
        strategy=UnifiedStrategy.FACET_CORRECTIVE,
        status=EvaluationStageStatus.SUCCEEDED,
    )
    return EvaluationItemResult(
        question_id=observation.question_id,
        strategy=observation.strategy,
        observation=observation,
    )


class _Judge:
    revision = "fixture-judge-v1"

    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    async def judge(
        self,
        question: UnifiedEvaluationQuestion,
        item: EvaluationItemResult,
        *,
        prompt_hash: str,
    ) -> JudgeVerdict:
        self.calls += 1
        if self.fail:
            raise RuntimeError("fixture unavailable")
        return JudgeVerdict(
            question_id=question.question_id,
            strategy=item.strategy,
            judge_version=self.revision,
            prompt_hash=prompt_hash,
            input_hash="d" * 64,
            correct=True,
            complete=True,
            faithful=True,
            citation_correct=True,
            usage=EvaluationResourceUsage(judge_calls=1, input_tokens=10, output_tokens=2),
        )


class _Probe:
    def __init__(self, healthy: bool) -> None:
        self.value = healthy

    async def healthy(self) -> bool:
        return self.value


def test_judge_gate_rejects_offline_budget_and_scope_before_calls() -> None:
    item = _item()
    with pytest.raises(ValueError, match="online"):
        validate_judge_gate(
            UnifiedEvaluationConfig(),
            [item],
            selected_strategy=UnifiedStrategy.FACET_CORRECTIVE,
            confirmed_budget=True,
        )
    config = UnifiedEvaluationConfig(online=True, with_judge=True)
    with pytest.raises(ValueError, match="budget"):
        validate_judge_gate(
            config,
            [item],
            selected_strategy=UnifiedStrategy.FACET_CORRECTIVE,
            confirmed_budget=False,
        )
    with pytest.raises(ValueError, match="hard limit"):
        validate_judge_gate(
            config.model_copy(update={"judge_max_answers": 50}),
            [item] * 51,
            selected_strategy=UnifiedStrategy.FACET_CORRECTIVE,
            confirmed_budget=True,
        )


async def test_judge_cache_and_failure_usage_are_preserved(tmp_path: Path) -> None:
    cache = ContentAddressedCache(tmp_path / "cache", workspace_root=tmp_path)
    provider = _Judge()
    prompt_hash = "a" * 64
    first = await evaluate_with_judge(
        provider,
        {_question().question_id: _question()},
        [_item()],
        prompt_hash=prompt_hash,
        cache=cache,
    )
    second = await evaluate_with_judge(
        provider,
        {_question().question_id: _question()},
        [_item()],
        prompt_hash=prompt_hash,
        cache=cache,
    )
    assert provider.calls == 1 and first == second
    failed = await evaluate_with_judge(
        _Judge(fail=True),
        {_question().question_id: _question()},
        [_item()],
        prompt_hash="b" * 64,
        cache=cache,
    )
    assert failed[0].error is not None
    assert failed[0].usage.judge_calls == 1


def test_human_agreement_is_separate_and_optional() -> None:
    verdict = JudgeVerdict(
        question_id="ueq-judge-001",
        strategy=UnifiedStrategy.FACET_CORRECTIVE,
        judge_version="fixture-judge-v1",
        prompt_hash="a" * 64,
        input_hash="b" * 64,
        correct=True,
        complete=True,
        faithful=True,
        citation_correct=True,
        usage=EvaluationResourceUsage(judge_calls=1),
    )
    from kg_crag.models import JudgeRecord

    record = JudgeRecord(
        question_id=verdict.question_id,
        strategy=verdict.strategy,
        status=EvaluationStageStatus.SUCCEEDED,
        verdict=verdict,
        usage=verdict.usage,
    )
    assert human_judge_agreement([record], []) is None
    review = HumanReview(
        question_id=verdict.question_id,
        strategy=verdict.strategy,
        review_version="review-v1",
        correct=True,
        complete=True,
        faithful=True,
        citation_correct=True,
    )
    assert human_judge_agreement([record], [review]) == 1.0


async def test_external_service_health_failure_is_isolated_by_gate() -> None:
    config = UnifiedEvaluationConfig(online=True)
    with pytest.raises(RuntimeError, match="qdrant"):
        await validate_external_services(
            config,
            {"qdrant": _Probe(False), "neo4j": _Probe(True)},
            confirmed_budget=True,
        )


def test_cli_dry_run_and_safe_summary_do_not_echo_content(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _close_idle_pytest_loop()
    assert main(["--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "questions=40" in output and "dry-run" in output
    _safe_summary(Path("result.json"), run_id="safe-run", items=3, failures=1)
    output = capsys.readouterr().out
    assert "safe-run" in output and "prompt" not in output.casefold()
    assert "evidence" not in output.casefold()


def test_cli_rejects_unconfirmed_test_and_real_run_without_adapter() -> None:
    assert main(["--split", "test"]) == 2
    assert main([]) == 2
