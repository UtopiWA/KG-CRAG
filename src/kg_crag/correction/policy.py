"""基于缺口、成本和历史状态的确定性纠错策略。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from kg_crag.correction.actions import action_catalog, build_action_request
from kg_crag.correction.budget import BudgetManager, can_reserve
from kg_crag.correction.config import CorrectiveWorkflowConfig
from kg_crag.correction.identity import stable_digest
from kg_crag.models import (
    ActionEstimate,
    ActionRequest,
    BudgetLedger,
    BudgetUsage,
    CorrectionAction,
    CorrectionDecision,
    EvidenceRequirement,
    FacetKind,
    GapType,
    StopReason,
    SufficiencyAssessment,
)


class DecisionAdvisor(Protocol):
    revision: str

    async def rank(self, action_ids: list[str]) -> tuple[list[str], BudgetUsage]: ...


async def apply_advisor_order(
    candidates: list[tuple[ActionRequest, ActionEstimate, float]],
    advisor: DecisionAdvisor,
    budget: BudgetManager,
) -> list[tuple[ActionRequest, ActionEstimate, float]]:
    """可选建议只能重排既有白名单候选，失败时不重试并保留规则顺序。"""

    reservation = BudgetUsage(
        llm_calls=1, input_tokens=1000, output_tokens=500, context_chars=4000, latency_ms=5000
    )
    token = budget.reserve(reservation)
    if token is None:
        return candidates
    expected_ids = [item[0].action_id for item in candidates]
    try:
        ranked_ids, usage = await advisor.rank(expected_ids)
        if sorted(ranked_ids) != sorted(expected_ids) or len(ranked_ids) != len(set(ranked_ids)):
            raise ValueError("advisor must only reorder the supplied action IDs")
        budget.settle(token, usage)
    except Exception:
        budget.settle(token, failed=True)
        return candidates
    positions = {action_id: index for index, action_id in enumerate(ranked_ids)}
    return sorted(candidates, key=lambda item: positions[item[0].action_id])


def infer_gap(facet: EvidenceRequirement, assessment: SufficiencyAssessment) -> GapType:
    if any(item.facet_id == facet.facet_id for item in assessment.conflicts):
        return GapType.CONFLICT
    if facet.kind is FacetKind.MULTI_HOP:
        return GapType.MULTI_HOP_GAP
    if facet.kind is FacetKind.COMPARISON:
        return GapType.COMPARISON_SIDE
    if facet.kind is FacetKind.METRIC:
        return GapType.METRIC_MISSING
    if facet.target_entity:
        return GapType.ENTITY_ALIAS
    return GapType.TERMINOLOGY_MISMATCH


def estimate_for_action(
    action: CorrectionAction, required_gain: float, config: CorrectiveWorkflowConfig
) -> ActionEstimate:
    base = action_catalog(config.actions)[action].worst_case
    return base.model_copy(update={"expected_required_facet_gain": required_gain})


def calculate_utility(
    estimate: ActionEstimate, *, repeated: bool, config: CorrectiveWorkflowConfig
) -> float:
    weights = config.costs
    cost = (
        estimate.retrieval_rounds * weights.retrieval
        + estimate.llm_calls * weights.model
        + (estimate.input_tokens + estimate.output_tokens) * weights.token
        + estimate.latency_ms * weights.latency
        + (weights.repeat if repeated else 0.0)
    )
    return round(estimate.expected_required_facet_gain - cost, 8)


def candidate_actions(
    question: str,
    facets: Sequence[EvidenceRequirement],
    assessment: SufficiencyAssessment,
    ledger: BudgetLedger,
    history_action_ids: set[str],
    config: CorrectiveWorkflowConfig,
) -> list[tuple[ActionRequest, ActionEstimate, float]]:
    facet_by_id = {item.facet_id: item for item in facets}
    missing = [facet_by_id[item] for item in assessment.missing_required_facet_ids]
    catalog = action_catalog(config.actions)
    candidates: list[tuple[ActionRequest, ActionEstimate, float]] = []
    for facet in missing:
        gap = infer_gap(facet, assessment)
        for action, definition in catalog.items():
            if gap not in definition.gap_types:
                continue
            request = build_action_request(action, question, [facet], config.actions)
            estimate = estimate_for_action(action, 1.0, config)
            repeated = request.action_id in history_action_ids
            if repeated or not can_reserve(ledger, estimate):
                continue
            utility = calculate_utility(estimate, repeated=False, config=config)
            candidates.append((request, estimate, utility))
    return sorted(
        candidates,
        key=lambda item: (
            -item[2],
            -item[1].expected_required_facet_gain,
            item[1].retrieval_rounds + item[1].llm_calls,
            item[0].action.value,
            item[0].action_id,
        ),
    )


def choose_action(
    question: str,
    facets: Sequence[EvidenceRequirement],
    assessment: SufficiencyAssessment,
    ledger: BudgetLedger,
    history_action_ids: set[str],
    config: CorrectiveWorkflowConfig,
    *,
    state_improved: bool = True,
) -> CorrectionDecision:
    if not state_improved and history_action_ids:
        candidates: list[tuple[ActionRequest, ActionEstimate, float]] = []
    else:
        candidates = candidate_actions(
            question, facets, assessment, ledger, history_action_ids, config
        )
    positive = [item for item in candidates if item[2] >= config.costs.minimum_utility]
    payload = {
        "question": question,
        "assessment": assessment.assessment_id,
        "history": sorted(history_action_ids),
        "candidates": [item[0].action_id for item in candidates],
    }
    decision_id = "decision-" + stable_digest(payload)[:16]
    if not positive:
        budget_blocked = bool(assessment.missing_required_facet_ids) and not any(
            can_reserve(ledger, item.worst_case) for item in action_catalog(config.actions).values()
        )
        return CorrectionDecision(
            decision_id=decision_id,
            utility=0.0,
            reason_code="budget_exhausted" if budget_blocked else "no_positive_gain",
            candidate_action_ids=[item[0].action_id for item in candidates],
            stop_reason=(
                StopReason.BUDGET_EXHAUSTED if budget_blocked else StopReason.NO_POSITIVE_GAIN
            ),
        )
    selected, estimate, utility = positive[0]
    return CorrectionDecision(
        decision_id=decision_id,
        selected=selected,
        estimate=estimate,
        utility=utility,
        reason_code=f"selected_{selected.action.value}_for_missing_facet",
        candidate_action_ids=[item[0].action_id for item in candidates],
        rejected_reasons={item[0].action_id: "lower_stable_utility" for item in positive[1:]},
    )
