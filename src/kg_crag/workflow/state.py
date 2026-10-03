"""在 LangGraph 节点之间传递的显式状态。"""

from typing import TypedDict

from kg_crag.models import (
    ActionRequest,
    BudgetLedger,
    CorrectionDecision,
    CorrectionState,
    Evidence,
    EvidenceRequirement,
    RetrievalEvaluation,
    RouteDecision,
    RunIdentity,
    StopResult,
    SufficiencyAssessment,
)
from kg_crag.models import CoverageMatrix as CoverageMatrix
from kg_crag.models import TraceEvent as TraceEvent


class AgentState(TypedDict, total=False):
    """保留旧字段，并显式增加纠错工作流的所有结构化字段。"""

    query: str
    question_id: str
    identity: RunIdentity
    query_types: list[str]
    route: RouteDecision | None
    subqueries: list[str]
    retrieval_round: int
    reflection_round: int
    candidates: list[Evidence]
    selected_evidence: list[Evidence]
    facets: list[EvidenceRequirement]
    coverage_matrix: CoverageMatrix | None
    sufficiency: SufficiencyAssessment | None
    candidate_actions: list[ActionRequest]
    decision: CorrectionDecision | None
    action_history: list[dict[str, object]]
    budget: BudgetLedger
    errors: list[dict[str, object]]
    state_fingerprint: str | None
    stop: StopResult | None
    retrieval_evaluation: RetrievalEvaluation | None
    draft_answer: str | None
    answer_evaluation: dict[str, str | int | float | bool | None] | None
    citations: list[dict[str, str | int | None]]
    trace: list[TraceEvent]
    final_answer: str | None


def agent_state_from_correction(state: CorrectionState) -> AgentState:
    """通过严格模型转储创建新字典，节点不得原地修改调用方输入。"""

    from kg_crag.models import CorrectionState

    validated = CorrectionState.model_validate(state.model_dump(mode="json"))
    payload = validated.model_dump(mode="python")
    return AgentState(
        query=validated.question,
        question_id=validated.question_id,
        identity=validated.identity,
        facets=validated.facets,
        candidates=validated.candidates,
        selected_evidence=validated.selected_evidence,
        coverage_matrix=validated.coverage_matrix,
        sufficiency=validated.sufficiency,
        candidate_actions=validated.candidate_actions,
        decision=validated.decision,
        action_history=payload["action_history"],
        retrieval_round=validated.retrieval_round,
        budget=validated.budget,
        errors=payload["errors"],
        trace=validated.trace,
        state_fingerprint=validated.state_fingerprint,
        stop=validated.stop,
        query_types=[],
        route=None,
        subqueries=[],
        reflection_round=0,
        retrieval_evaluation=None,
        draft_answer=None,
        answer_evaluation=None,
        citations=[],
        final_answer=None,
    )


def correction_state_from_agent(state: AgentState) -> CorrectionState:
    """只读取新结构化字段，并在每个节点边界重新执行 Pydantic 校验。"""

    from kg_crag.models import CorrectionState

    return CorrectionState.model_validate(
        {
            "identity": state["identity"],
            "question_id": state["question_id"],
            "question": state["query"],
            "facets": state.get("facets", []),
            "candidates": state.get("candidates", []),
            "selected_evidence": state.get("selected_evidence", []),
            "coverage_matrix": state.get("coverage_matrix"),
            "sufficiency": state.get("sufficiency"),
            "candidate_actions": state.get("candidate_actions", []),
            "decision": state.get("decision"),
            "action_history": state.get("action_history", []),
            "retrieval_round": state.get("retrieval_round", 0),
            "budget": state.get("budget", BudgetLedger()),
            "errors": state.get("errors", []),
            "trace": state.get("trace", []),
            "state_fingerprint": state.get("state_fingerprint"),
            "stop": state.get("stop"),
        }
    )
