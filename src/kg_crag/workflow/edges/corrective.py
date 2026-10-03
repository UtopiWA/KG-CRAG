"""只读取枚举、布尔、计数和预算字段的固定条件边。"""

from __future__ import annotations

from typing import Literal

from kg_crag.models import ActionStatus, StopReason
from kg_crag.workflow.state import AgentState, correction_state_from_agent


def route_after_assess(state: AgentState) -> Literal["decide", "finalize"]:
    validated = correction_state_from_agent(state)
    return "finalize" if validated.sufficiency and validated.sufficiency.sufficient else "decide"


def route_after_decide(state: AgentState) -> Literal["execute", "finalize"]:
    validated = correction_state_from_agent(state)
    return "execute" if validated.decision and validated.decision.selected else "finalize"


def route_after_execute(state: AgentState) -> Literal["reassess", "finalize"]:
    validated = correction_state_from_agent(state)
    if not validated.action_history:
        return "finalize"
    if validated.action_history[-1].status is ActionStatus.FAILED:
        return "finalize"
    return "reassess"


def stop_reason_from_state(state: AgentState) -> StopReason:
    """终止分类不读取自由文本理由。"""

    validated = correction_state_from_agent(state)
    if validated.stop is not None:
        return validated.stop.reason
    if validated.sufficiency and validated.sufficiency.sufficient:
        return StopReason.SUFFICIENT
    if validated.action_history and validated.action_history[-1].status is ActionStatus.FAILED:
        return StopReason.EXECUTION_FAILED
    if validated.decision and validated.decision.stop_reason:
        return validated.decision.stop_reason
    if validated.retrieval_round >= validated.budget.limit.retrieval_rounds:
        if validated.action_history and validated.action_history[-1].status is ActionStatus.EMPTY:
            return StopReason.INTERNAL_KNOWLEDGE_MISSING
        return StopReason.NO_POSITIVE_GAIN
    return StopReason.INTERNAL_KNOWLEDGE_MISSING
