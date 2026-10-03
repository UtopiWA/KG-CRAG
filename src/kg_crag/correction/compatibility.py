"""旧路由/检索评估模型到新结构化契约的只读迁移映射。"""

from __future__ import annotations

from kg_crag.correction.requirements import _facet
from kg_crag.models import (
    EvidenceRequirement,
    FacetKind,
    RetrievalEvaluation,
    RetrievalLabel,
    RouteDecision,
)


def requirements_from_route(
    route: RouteDecision, *, question_id: str, question: str
) -> list[EvidenceRequirement]:
    """只读取枚举与子查询，不读取自由文本 reason。"""

    descriptions = route.subqueries or [question]
    kind = FacetKind.MULTI_HOP if "multi_hop" in route.query_types else FacetKind.CONTENT
    return [_facet(question_id, kind, item, item.casefold().split()) for item in descriptions]


def legacy_sufficient(evaluation: RetrievalEvaluation) -> bool:
    """仅提供展示兼容；新工作流条件边不得调用本函数。"""

    return (
        evaluation.label is RetrievalLabel.RELEVANT
        and evaluation.scores.coverage >= 1.0
        and evaluation.scores.consistency >= 0.5
    )
