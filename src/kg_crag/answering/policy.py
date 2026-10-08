"""回答阶段的固定优先级决策表。"""

from __future__ import annotations

from kg_crag.correction.identity import stable_digest
from kg_crag.models import (
    AnswerBudgetUsage,
    AnswerCheckCode,
    AnswerEvaluation,
    GroundedAnswerCandidate,
    ReflectionAction,
    ReflectionDecision,
    StopReason,
)


def decide_reflection(
    *,
    internal_stop: StopReason,
    evaluation: AnswerEvaluation | None,
    candidate: GroundedAnswerCandidate | None,
    missing_facet_ids: list[str],
    web_allowed: bool,
    web_ready: bool,
    internal_can_retrieve: bool,
    remediation_used: bool,
) -> ReflectionDecision:
    action: ReflectionAction
    reason: str
    estimate = AnswerBudgetUsage()
    if remediation_used:
        if evaluation is not None and evaluation.acceptable:
            action, reason = ReflectionAction.ACCEPT, "remediation_checks_passed"
        else:
            action, reason = ReflectionAction.CONSERVATIVE_STOP, "remediation_already_used"
    elif internal_stop in {
        StopReason.EXECUTION_FAILED,
        StopReason.INVALID_INPUT,
        StopReason.BUDGET_EXHAUSTED,
    }:
        action, reason = ReflectionAction.CONSERVATIVE_STOP, internal_stop.value
    elif internal_stop is StopReason.INTERNAL_KNOWLEDGE_MISSING:
        if web_allowed and web_ready and missing_facet_ids:
            action, reason = ReflectionAction.WEB_SEARCH, "eligible_internal_knowledge_missing"
            estimate = AnswerBudgetUsage(
                web_calls=1,
                reflection_rounds=1,
                web_results=5,
                external_context_chars=8000,
            )
        else:
            action, reason = ReflectionAction.CONSERVATIVE_STOP, "web_unavailable"
    elif evaluation is not None and evaluation.acceptable:
        action, reason = ReflectionAction.ACCEPT, "all_checks_passed"
    elif (
        candidate is not None
        and evaluation is not None
        and any(item.code is AnswerCheckCode.ABSTENTION for item in evaluation.findings)
    ):
        # 已有引用与 facet 绑定可安全复用时，先要求模型提取证据中的答案，
        # 避免把一次格式正确但语义拒答的输出误当成检索缺失。
        action, reason = ReflectionAction.REGENERATE, "abstention_must_be_rewritten"
        estimate = AnswerBudgetUsage(reflection_rounds=1)
    elif missing_facet_ids and internal_can_retrieve:
        action, reason = ReflectionAction.RERETRIEVE, "required_facet_missing"
        estimate = AnswerBudgetUsage(reflection_rounds=1)
    elif (
        candidate is not None
        and evaluation is not None
        and any(
            item.code
            in {
                AnswerCheckCode.ABSTENTION,
                AnswerCheckCode.UNSUPPORTED,
                AnswerCheckCode.WRONG_ATTRIBUTION,
            }
            for item in evaluation.findings
        )
    ):
        action, reason = ReflectionAction.REGENERATE, "answer_can_be_narrowed"
        estimate = AnswerBudgetUsage(reflection_rounds=1)
    else:
        action, reason = ReflectionAction.CONSERVATIVE_STOP, "no_safe_remediation"
    payload = {
        "stop": internal_stop.value,
        "action": action.value,
        "reason": reason,
        "missing": sorted(missing_facet_ids),
        "candidate": [item.claim_id for item in candidate.claims] if candidate else [],
    }
    return ReflectionDecision(
        decision_id="answer-decision-" + stable_digest(payload)[:16],
        action=action,
        reason_code=reason,
        target_facet_ids=sorted(missing_facet_ids),
        allowed_claim_ids=[item.claim_id for item in candidate.claims] if candidate else [],
        estimated_usage=estimate,
    )
