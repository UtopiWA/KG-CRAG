"""不调用模型的确定性保守回答。"""

from __future__ import annotations

from kg_crag.models import CombinedBudgetSummary, GroundedAnswerCandidate


def build_conservative_candidate(
    *,
    supported: GroundedAnswerCandidate | None,
    missing_facet_ids: list[str],
    reason: str,
    internal_evidence_count: int = 0,
    external_evidence_count: int = 0,
    budget_summary: CombinedBudgetSummary | None = None,
) -> GroundedAnswerCandidate:
    claims = list(supported.claims) if supported else []
    citations = list(supported.citations) if supported else []
    supported_text = (
        supported.answer if supported and supported.claims else "暂无可安全陈述的结论。"
    )
    missing = "、".join(sorted(missing_facet_ids)) or "无可确认的完整证据范围"
    budget_text = ""
    if budget_summary is not None:
        answer_usage = budget_summary.answer.used
        budget_text = (
            f" 阶段用量: model_calls={answer_usage.answer_calls + answer_usage.critic_calls}, "
            f"web_calls={answer_usage.web_calls}, "
            f"tokens={answer_usage.input_tokens + answer_usage.output_tokens}。"
        )
    answer = (
        f"{supported_text}\n证据边界: {missing}。停止原因: {reason}。"
        f"内部证据={internal_evidence_count}，外部证据={external_evidence_count}。{budget_text}"
    )
    return GroundedAnswerCandidate(
        answer=answer,
        claims=claims,
        citations=citations,
        confidence=min(supported.confidence, 0.5) if supported else 0.0,
    )
