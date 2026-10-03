"""压力集校验、矩阵指标、错误隔离和检查点幂等测试。"""

from pathlib import Path

import pytest

from kg_crag.correction.config import CorrectiveWorkflowConfig
from kg_crag.evaluation.corrective import (
    aggregate_metrics,
    evaluate_offline_matrix,
    load_corrective_questions,
)
from kg_crag.models import (
    CorrectiveEvaluationItem,
    CorrectiveStrategy,
    ErrorCode,
    ErrorDetail,
    StopReason,
)


def test_frozen_question_set_has_required_pressure_categories() -> None:
    questions = load_corrective_questions(Path("data/evaluation/corrective_dev_questions.json"))
    assert len(questions.questions) == 20
    assert {item.category.value for item in questions.questions} == {
        "terminology_mismatch",
        "entity_alias",
        "comparison_side",
        "multi_hop_gap",
        "metric_missing",
        "conflict",
        "internal_missing",
    }


def test_metrics_keep_failures_in_denominator() -> None:
    items = [
        CorrectiveEvaluationItem(
            question_id="q1",
            strategy=CorrectiveStrategy.FACET_CORRECTIVE,
            first_sufficient=True,
            diagnosis_correct=True,
            required_facets=2,
            covered_required_facets=2,
            stop_reason=StopReason.SUFFICIENT,
        ),
        CorrectiveEvaluationItem(
            question_id="q2",
            strategy=CorrectiveStrategy.FACET_CORRECTIVE,
            required_facets=2,
            stop_reason=StopReason.EXECUTION_FAILED,
            error=ErrorDetail(code=ErrorCode.INTERNAL, message="failed"),
        ),
    ]
    metrics = aggregate_metrics(items, CorrectiveStrategy.FACET_CORRECTIVE)
    assert metrics.total == 2 and metrics.failures == 1
    assert metrics.first_sufficiency_rate == 0.5
    assert metrics.diagnosis_accuracy == 0.5
    assert metrics.required_coverage_rate == 0.5


@pytest.mark.asyncio
async def test_offline_matrix_is_checkpointed_and_replay_is_identical(tmp_path: Path) -> None:
    questions = load_corrective_questions(Path("data/evaluation/corrective_dev_questions.json"))
    config = CorrectiveWorkflowConfig()
    first = await evaluate_offline_matrix(
        questions,
        config,
        limit=3,
        results_root=tmp_path / "results",
        workspace_root=tmp_path,
    )
    second = await evaluate_offline_matrix(
        questions,
        config,
        limit=3,
        results_root=tmp_path / "results",
        workspace_root=tmp_path,
    )
    assert len(first.items) == 9
    assert first.items == second.items
    assert all(item.error is None for item in first.items)
    assert first.metrics[CorrectiveStrategy.FACET_CORRECTIVE].llm_calls == 0
    assert first.metrics[CorrectiveStrategy.FACET_CORRECTIVE].recovery_rate == 1.0
