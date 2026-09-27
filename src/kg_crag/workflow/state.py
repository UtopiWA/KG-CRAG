"""在 LangGraph 节点之间传递的显式状态。"""

from typing import TypedDict

from kg_crag.models import Evidence, RetrievalEvaluation, RouteDecision
from kg_crag.models import TraceEvent as TraceEvent


class AgentState(TypedDict):
    query: str
    query_types: list[str]
    route: RouteDecision | None
    subqueries: list[str]
    retrieval_round: int
    reflection_round: int
    candidates: list[Evidence]
    selected_evidence: list[Evidence]
    retrieval_evaluation: RetrievalEvaluation | None
    draft_answer: str | None
    answer_evaluation: dict[str, str | int | float | bool | None] | None
    citations: list[dict[str, str | int | None]]
    trace: list[TraceEvent]
    final_answer: str | None
