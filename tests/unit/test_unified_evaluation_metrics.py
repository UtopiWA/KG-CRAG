"""统一检索、facet、回答、失败分母与成本指标测试。"""

import pytest

from kg_crag.evaluation.metrics import (
    aggregate_slice,
    compute_answer_metrics,
    compute_facet_metrics,
    compute_rank_metrics,
    evaluate_item,
)
from kg_crag.models import (
    ErrorCode,
    ErrorDetail,
    EvaluationAnswerPoint,
    EvaluationFacetTarget,
    EvaluationResourceUsage,
    EvaluationSplit,
    EvaluationStageStatus,
    KnowledgeSufficiency,
    StrategyObservation,
    StressCategory,
    UnifiedEvaluationQuestion,
    UnifiedQuestionType,
    UnifiedStrategy,
)


def _question() -> UnifiedEvaluationQuestion:
    return UnifiedEvaluationQuestion(
        question_id="ueq-metrics-001",
        split=EvaluationSplit.DEV,
        question="Compare method A and method B.",
        normalized_question="compare method a and method b.",
        question_types=[UnifiedQuestionType.COMPARISON],
        stress_category=StressCategory.COMPARISON_SIDE,
        knowledge_sufficiency=KnowledgeSufficiency.SUFFICIENT,
        answer_points=[
            EvaluationAnswerPoint(
                point_id="point-0000000000000001",
                description="Method A",
                evidence_ids=["e-a"],
            ),
            EvaluationAnswerPoint(
                point_id="point-0000000000000002",
                description="Method B",
                evidence_ids=["e-b"],
            ),
        ],
        facets=[
            EvaluationFacetTarget(
                facet_id="facet-0000000000000001",
                description="Method A",
                target_evidence_ids=["e-a"],
            ),
            EvaluationFacetTarget(
                facet_id="facet-0000000000000002",
                description="Method B",
                target_evidence_ids=["e-b"],
            ),
            EvaluationFacetTarget(
                facet_id="facet-0000000000000003",
                description="Optional context",
                required=False,
                target_evidence_ids=["e-c"],
            ),
        ],
        relevant_evidence={"e-a": 3, "e-b": 2, "e-c": 1},
        minimum_sufficient_evidence_sets=[["e-a", "e-b"]],
        leakage_group_id="leak-metrics-001",
        source_group_ids=["source-metrics-001"],
        source_fixture="tests/fixtures/metrics.json#1",
    )


def _observation(**updates: object) -> StrategyObservation:
    payload: dict[str, object] = {
        "question_id": "ueq-metrics-001",
        "strategy": UnifiedStrategy.FACET_CORRECTIVE,
        "status": EvaluationStageStatus.SUCCEEDED,
        "ranked_evidence_ids": ["wrong", "e-a", "e-a", "e-b"],
        "initial_covered_facet_ids": ["facet-0000000000000001"],
        "final_covered_facet_ids": [
            "facet-0000000000000001",
            "facet-0000000000000002",
        ],
        "predicted_sufficient": True,
        "action_ids": ["action-1"],
        "loop_count": 1,
        "matched_answer_point_ids": [
            "point-0000000000000001",
            "point-0000000000000002",
        ],
        "cited_evidence_ids": ["e-a", "wrong"],
        "usage": EvaluationResourceUsage(
            tool_calls=2,
            input_tokens=10,
            output_tokens=5,
            latency_ms=20,
            estimated_cost=0.1,
        ),
    }
    payload.update(updates)
    return StrategyObservation.model_validate(payload)


def test_rank_metrics_use_deduplicated_order_and_graded_ndcg() -> None:
    metrics = compute_rank_metrics(_question(), _observation(), k=3)
    assert metrics["recall_at_3"].value == pytest.approx(2 / 3)
    assert metrics["mrr"].value == pytest.approx(0.5)
    assert 0 < (metrics["ndcg_at_3"].value or 0) < 1
    assert metrics["evidence_coverage"].value == pytest.approx(2 / 3)


def test_zero_relevance_is_not_applicable_instead_of_zero() -> None:
    question = _question().model_copy(
        update={
            "knowledge_sufficiency": KnowledgeSufficiency.INSUFFICIENT,
            "answer_points": [],
            "relevant_evidence": {},
            "minimum_sufficient_evidence_sets": [],
            "facets": [
                EvaluationFacetTarget(
                    facet_id="facet-0000000000000001",
                    description="Missing facet",
                )
            ],
        }
    )
    metrics = compute_rank_metrics(question, _observation(ranked_evidence_ids=[]), k=10)
    assert metrics["recall_at_10"].value is None
    assert metrics["recall_at_10"].not_applicable_reason


def test_facet_recovery_and_meaningless_correction_are_distinct() -> None:
    recovered = compute_facet_metrics(_question(), _observation())
    assert recovered["recovered"].value == 1.0
    assert recovered["facet_coverage_gain"].value == 1.0
    meaningless = compute_facet_metrics(
        _question(),
        _observation(
            final_covered_facet_ids=["facet-0000000000000001"],
            predicted_sufficient=False,
        ),
    )
    assert meaningless["meaningless_correction"].value == 1.0
    with pytest.raises(ValueError, match="unknown facet"):
        compute_facet_metrics(
            _question(), _observation(final_covered_facet_ids=["facet-ffffffffffffffff"])
        )


def test_answer_metrics_reject_unknown_points_and_score_supported_citations() -> None:
    metrics = compute_answer_metrics(_question(), _observation())
    assert metrics["answer_completeness"].value == 1.0
    assert metrics["citation_correctness"].value == 0.5
    with pytest.raises(ValueError, match="unknown answer point"):
        compute_answer_metrics(
            _question(), _observation(matched_answer_point_ids=["point-ffffffffffffffff"])
        )


def test_aggregate_keeps_failures_and_reports_cost_gain() -> None:
    success = evaluate_item(_question(), _observation(), k=10)
    failure_observation = StrategyObservation(
        question_id="ueq-metrics-001",
        strategy=UnifiedStrategy.FACET_CORRECTIVE,
        status=EvaluationStageStatus.FAILED,
        usage=EvaluationResourceUsage(tool_calls=1, estimated_cost=0.1),
        error=ErrorDetail(code=ErrorCode.TIMEOUT, message="fixture timeout"),
    )
    failure = evaluate_item(_question(), failure_observation, k=10)
    metrics = aggregate_slice("overall", [success, failure]).metrics
    assert metrics["success_rate"].value == 0.5
    assert metrics["failure_rate"].denominator == 2
    assert metrics["cost_normalized_gain"].value is not None
    zero_cost_item = evaluate_item(_question(), _observation(usage=EvaluationResourceUsage()), k=10)
    zero_cost = aggregate_slice("zero-cost", [zero_cost_item])
    assert zero_cost.metrics["cost_normalized_gain"].value is None
