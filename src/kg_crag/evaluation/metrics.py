"""统一评测的纯函数指标与可下钻切片聚合。"""

from __future__ import annotations

import math
from collections.abc import Iterable

from kg_crag.models import (
    EvaluationItemResult,
    EvaluationMetricValue,
    EvaluationSliceMetrics,
    EvaluationStageStatus,
    KnowledgeSufficiency,
    StrategyObservation,
    UnifiedEvaluationQuestion,
    UnifiedStrategy,
)


def _deduplicate(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _numeric(
    value: float,
    *,
    numerator: float,
    denominator: float,
    question_ids: list[str],
) -> EvaluationMetricValue:
    return EvaluationMetricValue(
        value=value,
        numerator=numerator,
        denominator=denominator,
        question_ids=question_ids,
    )


def _not_applicable(reason: str, question_ids: list[str]) -> EvaluationMetricValue:
    return EvaluationMetricValue(
        denominator=0,
        question_ids=question_ids,
        not_applicable_reason=reason,
    )


def compute_rank_metrics(
    question: UnifiedEvaluationQuestion,
    observation: StrategyObservation,
    *,
    k: int,
) -> dict[str, EvaluationMetricValue]:
    """计算去重后的 Recall/MRR/nDCG，并让失败题保留为零分。"""

    if k <= 0:
        raise ValueError("rank metric k must be positive")
    ranked = _deduplicate(observation.ranked_evidence_ids)
    relevant = question.relevant_evidence
    ids = [question.question_id]
    if not relevant:
        return {
            f"recall_at_{k}": _not_applicable("question has no internal relevant evidence", ids),
            "mrr": _not_applicable("question has no internal relevant evidence", ids),
            f"ndcg_at_{k}": _not_applicable("question has no graded relevance labels", ids),
            "evidence_coverage": _not_applicable("question has no internal relevant evidence", ids),
        }
    if observation.status is not EvaluationStageStatus.SUCCEEDED:
        ranked = []
    hits = set(ranked[:k]) & set(relevant)
    recall = len(hits) / len(relevant)
    first_rank = next((index for index, item in enumerate(ranked, start=1) if item in relevant), 0)
    mrr = 1.0 / first_rank if first_rank else 0.0
    dcg = sum(
        (2 ** relevant.get(evidence_id, 0) - 1) / math.log2(index + 2)
        for index, evidence_id in enumerate(ranked[:k])
    )
    ideal = sorted(relevant.values(), reverse=True)[:k]
    ideal_dcg = sum((2**grade - 1) / math.log2(index + 2) for index, grade in enumerate(ideal))
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0
    all_hits = set(ranked) & set(relevant)
    return {
        f"recall_at_{k}": _numeric(
            recall, numerator=len(hits), denominator=len(relevant), question_ids=ids
        ),
        "mrr": _numeric(mrr, numerator=mrr, denominator=1, question_ids=ids),
        f"ndcg_at_{k}": _numeric(ndcg, numerator=dcg, denominator=ideal_dcg, question_ids=ids),
        "evidence_coverage": _numeric(
            len(all_hits) / len(relevant),
            numerator=len(all_hits),
            denominator=len(relevant),
            question_ids=ids,
        ),
    }


def compute_facet_metrics(
    question: UnifiedEvaluationQuestion,
    observation: StrategyObservation,
) -> dict[str, EvaluationMetricValue]:
    """核算必需/可选 facet、充分性和纠错恢复，拒绝未知 facet。"""

    known = {item.facet_id for item in question.facets}
    initial = set(observation.initial_covered_facet_ids)
    final = set(observation.final_covered_facet_ids)
    if (initial | final) - known:
        raise ValueError("strategy observation references unknown facet")
    ranked = set(observation.ranked_evidence_ids)
    facets_by_id = {item.facet_id: item for item in question.facets}
    unsupported = {
        facet_id for facet_id in final if not facets_by_id[facet_id].is_covered_by(ranked)
    }
    if unsupported:
        raise ValueError(
            f"strategy observation claims facets without required Evidence: {sorted(unsupported)}"
        )
    required = {item.facet_id for item in question.facets if item.required}
    optional = known - required
    ids = [question.question_id]
    required_covered = len(final & required)
    optional_covered = len(final & optional)
    initial_required = len(initial & required)
    truth_sufficient = question.knowledge_sufficiency is KnowledgeSufficiency.SUFFICIENT
    recovery_candidate = bool(required - initial)
    recovered = recovery_candidate and not (required - final)
    gain = required_covered - initial_required
    metrics = {
        "required_facet_coverage": _numeric(
            required_covered / len(required) if required else 1.0,
            numerator=required_covered,
            denominator=len(required) or 1,
            question_ids=ids,
        ),
        "optional_facet_coverage": (
            _numeric(
                optional_covered / len(optional),
                numerator=optional_covered,
                denominator=len(optional),
                question_ids=ids,
            )
            if optional
            else _not_applicable("question has no optional facets", ids)
        ),
        "facet_coverage_gain": _numeric(
            float(gain), numerator=gain, denominator=len(required) or 1, question_ids=ids
        ),
        "sufficiency_correct": _numeric(
            float(observation.predicted_sufficient == truth_sufficient),
            numerator=float(observation.predicted_sufficient == truth_sufficient),
            denominator=1,
            question_ids=ids,
        ),
        "recovered": (
            _numeric(float(recovered), numerator=float(recovered), denominator=1, question_ids=ids)
            if recovery_candidate
            else _not_applicable("initial evidence already covers required facets", ids)
        ),
        "meaningless_correction": (
            _numeric(
                float(gain <= 0),
                numerator=float(gain <= 0),
                denominator=1,
                question_ids=ids,
            )
            if observation.action_ids
            else _not_applicable("no corrective action was attempted", ids)
        ),
    }
    return metrics


def compute_answer_metrics(
    question: UnifiedEvaluationQuestion,
    observation: StrategyObservation,
) -> dict[str, EvaluationMetricValue]:
    """以冻结答案要点与 Evidence ID 计算确定性回答/引用指标。"""

    ids = [question.question_id]
    expected_points = {item.point_id for item in question.answer_points}
    matched = set(observation.matched_answer_point_ids)
    if matched - expected_points:
        raise ValueError("strategy observation references unknown answer point")
    known_evidence = set(question.relevant_evidence)
    cited = set(observation.cited_evidence_ids)
    if not expected_points:
        completeness = _not_applicable("question has no answer key points", ids)
    else:
        completeness = _numeric(
            len(matched) / len(expected_points),
            numerator=len(matched),
            denominator=len(expected_points),
            question_ids=ids,
        )
    if not cited:
        citation = _not_applicable("observation has no citations", ids)
    else:
        supported = len(cited & known_evidence)
        citation = _numeric(
            supported / len(cited),
            numerator=supported,
            denominator=len(cited),
            question_ids=ids,
        )
    return {
        "answer_correctness": (
            _numeric(
                float(matched == expected_points),
                numerator=float(matched == expected_points),
                denominator=1,
                question_ids=ids,
            )
            if expected_points
            else _not_applicable("question has no answer key points", ids)
        ),
        "answer_completeness": completeness,
        "citation_correctness": citation,
        "faithfulness": citation,
    }


def evaluate_item(
    question: UnifiedEvaluationQuestion,
    observation: StrategyObservation,
    *,
    k: int,
    trace_sequences: list[int] | None = None,
) -> EvaluationItemResult:
    metrics = compute_rank_metrics(question, observation, k=k)
    metrics.update(compute_facet_metrics(question, observation))
    metrics.update(compute_answer_metrics(question, observation))
    metrics["loop_count"] = _numeric(
        float(observation.loop_count),
        numerator=observation.loop_count,
        denominator=1,
        question_ids=[question.question_id],
    )
    metrics["web_used"] = _numeric(
        float(observation.web_used),
        numerator=float(observation.web_used),
        denominator=1,
        question_ids=[question.question_id],
    )
    return EvaluationItemResult(
        question_id=question.question_id,
        strategy=observation.strategy,
        observation=observation,
        metrics=metrics,
        trace_sequences=trace_sequences or [],
    )


def _mean_metric(items: list[EvaluationItemResult], name: str) -> EvaluationMetricValue:
    included = [
        item for item in items if (metric := item.metrics.get(name)) and metric.value is not None
    ]
    ids = [item.question_id for item in items]
    if not included:
        return _not_applicable(f"metric {name} has no applicable questions", ids)
    value = sum(item.metrics[name].value or 0.0 for item in included) / len(included)
    return _numeric(
        value,
        numerator=sum(item.metrics[name].value or 0.0 for item in included),
        denominator=len(included),
        question_ids=[item.question_id for item in included],
    )


def aggregate_slice(
    name: str,
    items: list[EvaluationItemResult],
) -> EvaluationSliceMetrics:
    """固定全样本分母聚合失败、质量与资源指标。"""

    if not items:
        raise ValueError("cannot aggregate an empty evaluation slice")
    ids = [item.question_id for item in items]
    succeeded = sum(item.observation.status is EvaluationStageStatus.SUCCEEDED for item in items)
    usage = [item.observation.usage for item in items]
    metrics = {
        "success_rate": _numeric(
            succeeded / len(items), numerator=succeeded, denominator=len(items), question_ids=ids
        ),
        "failure_rate": _numeric(
            (len(items) - succeeded) / len(items),
            numerator=len(items) - succeeded,
            denominator=len(items),
            question_ids=ids,
        ),
    }
    metric_names = {
        name
        for item in items
        for name in item.metrics
        if name.startswith(("recall_at_", "ndcg_at_"))
    }
    metric_names.update(
        {
            "mrr",
            "evidence_coverage",
            "required_facet_coverage",
            "optional_facet_coverage",
            "facet_coverage_gain",
            "sufficiency_correct",
            "recovered",
            "meaningless_correction",
            "answer_correctness",
            "answer_completeness",
            "citation_correctness",
            "faithfulness",
            "loop_count",
            "web_used",
        }
    )
    for metric_name in sorted(metric_names):
        metrics[metric_name] = _mean_metric(items, metric_name)
    total_cost = sum(item.estimated_cost or 0.0 for item in usage)
    total_gain = sum((item.metrics["facet_coverage_gain"].value or 0.0) for item in items)
    metrics.update(
        {
            "tool_calls": _numeric(
                float(sum(item.tool_calls for item in usage)),
                numerator=sum(item.tool_calls for item in usage),
                denominator=len(items),
                question_ids=ids,
            ),
            "model_calls": _numeric(
                float(sum(item.model_calls for item in usage)),
                numerator=sum(item.model_calls for item in usage),
                denominator=len(items),
                question_ids=ids,
            ),
            "input_tokens": _numeric(
                float(sum(item.input_tokens for item in usage)),
                numerator=sum(item.input_tokens for item in usage),
                denominator=len(items),
                question_ids=ids,
            ),
            "output_tokens": _numeric(
                float(sum(item.output_tokens for item in usage)),
                numerator=sum(item.output_tokens for item in usage),
                denominator=len(items),
                question_ids=ids,
            ),
            "latency_ms": _numeric(
                float(sum(item.latency_ms for item in usage)),
                numerator=sum(item.latency_ms for item in usage),
                denominator=len(items),
                question_ids=ids,
            ),
            "estimated_cost": _numeric(
                total_cost,
                numerator=total_cost,
                denominator=len(items),
                question_ids=ids,
            ),
            "cost_normalized_gain": (
                _numeric(
                    total_gain / total_cost,
                    numerator=total_gain,
                    denominator=total_cost,
                    question_ids=ids,
                )
                if total_cost > 0
                else _not_applicable("estimated cost is zero", ids)
            ),
        }
    )
    return EvaluationSliceMetrics(slice_name=name, question_ids=ids, metrics=metrics)


def aggregate_all_slices(
    questions: list[UnifiedEvaluationQuestion],
    items: list[EvaluationItemResult],
) -> dict[str, list[EvaluationSliceMetrics]]:
    """为每个策略生成总体、问题类型与压力类型切片。"""

    question_map = {item.question_id: item for item in questions}
    output: dict[str, list[EvaluationSliceMetrics]] = {}
    for strategy in UnifiedStrategy:
        selected = [item for item in items if item.strategy is strategy]
        if not selected:
            continue
        slices = [aggregate_slice("overall", selected)]
        for question_type in sorted(
            {kind for item in questions for kind in item.question_types},
            key=lambda item: item.value,
        ):
            subset = [
                item
                for item in selected
                if question_type in question_map[item.question_id].question_types
            ]
            if subset:
                slices.append(aggregate_slice(f"question_type:{question_type.value}", subset))
        for stress in sorted(
            {item.stress_category for item in questions}, key=lambda item: item.value
        ):
            subset = [
                item
                for item in selected
                if question_map[item.question_id].stress_category is stress
            ]
            if subset:
                slices.append(aggregate_slice(f"stress:{stress.value}", subset))
        output[strategy.value] = slices
    return output
