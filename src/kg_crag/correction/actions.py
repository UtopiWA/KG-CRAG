"""版本化白名单动作目录与确定性复合查询生成。"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import Field

from kg_crag.correction.config import ActionBoundsConfig
from kg_crag.correction.identity import stable_digest
from kg_crag.models import (
    ActionEstimate,
    ActionRequest,
    CorrectionAction,
    EvidenceRequirement,
    GapType,
)
from kg_crag.models.domain import StrictModel


class ActionDefinition(StrictModel):
    action: CorrectionAction
    gap_types: list[GapType] = Field(min_length=1)
    tool: str = Field(min_length=1)
    max_top_k: int = Field(ge=1, le=100)
    max_candidates: int = Field(ge=1, le=100)
    max_hops: int = Field(default=1, ge=1, le=3)
    max_subquestions: int = Field(default=0, ge=0, le=3)
    worst_case: ActionEstimate
    failure_semantics: str = Field(min_length=1, max_length=200)


def action_catalog(config: ActionBoundsConfig) -> dict[CorrectionAction, ActionDefinition]:
    """目录只包含固定内部工具，不接受配置注入任意工具名。"""

    retrieval = ActionEstimate(retrieval_rounds=1, candidates=config.max_candidates_per_action)
    compound = ActionEstimate(
        retrieval_rounds=1,
        subquestions=config.max_subquestions,
        candidates=config.max_candidates_per_action,
    )
    definitions = {
        CorrectionAction.DENSE: ActionDefinition(
            action=CorrectionAction.DENSE,
            gap_types=[GapType.COMPARISON_SIDE, GapType.METRIC_MISSING],
            tool="dense",
            max_top_k=config.max_top_k,
            max_candidates=config.max_candidates_per_action,
            worst_case=retrieval,
            failure_semantics="single dense backend failure",
        ),
        CorrectionAction.SPARSE: ActionDefinition(
            action=CorrectionAction.SPARSE,
            gap_types=[GapType.TERMINOLOGY_MISMATCH, GapType.ENTITY_ALIAS],
            tool="sparse",
            max_top_k=config.max_top_k,
            max_candidates=config.max_candidates_per_action,
            worst_case=retrieval,
            failure_semantics="single sparse backend failure",
        ),
        CorrectionAction.HYBRID: ActionDefinition(
            action=CorrectionAction.HYBRID,
            gap_types=[
                GapType.TERMINOLOGY_MISMATCH,
                GapType.ENTITY_ALIAS,
                GapType.COMPARISON_SIDE,
                GapType.METRIC_MISSING,
            ],
            tool="hybrid",
            max_top_k=config.max_top_k,
            max_candidates=config.max_candidates_per_action,
            worst_case=retrieval,
            failure_semantics="explicit hybrid degradation or failure",
        ),
        CorrectionAction.GRAPH: ActionDefinition(
            action=CorrectionAction.GRAPH,
            gap_types=[GapType.MULTI_HOP_GAP, GapType.CONFLICT],
            tool="graph",
            max_top_k=config.max_top_k,
            max_candidates=config.max_candidates_per_action,
            max_hops=config.max_graph_hops,
            worst_case=retrieval,
            failure_semantics="single graph query failure",
        ),
        CorrectionAction.REWRITE: ActionDefinition(
            action=CorrectionAction.REWRITE,
            gap_types=[GapType.TERMINOLOGY_MISMATCH, GapType.ENTITY_ALIAS],
            tool="hybrid",
            max_top_k=config.max_top_k,
            max_candidates=config.max_candidates_per_action,
            worst_case=retrieval,
            failure_semantics="deterministic rewrite then hybrid failure",
        ),
        CorrectionAction.DECOMPOSE: ActionDefinition(
            action=CorrectionAction.DECOMPOSE,
            gap_types=[GapType.MULTI_HOP_GAP, GapType.COMPARISON_SIDE],
            tool="hybrid",
            max_top_k=config.max_top_k,
            max_candidates=config.max_candidates_per_action,
            max_subquestions=config.max_subquestions,
            worst_case=compound,
            failure_semantics="bounded subquery execution failure",
        ),
        CorrectionAction.ADJUST: ActionDefinition(
            action=CorrectionAction.ADJUST,
            gap_types=[GapType.METRIC_MISSING, GapType.COMPARISON_SIDE],
            tool="hybrid",
            max_top_k=config.max_top_k,
            max_candidates=config.max_candidates_per_action,
            worst_case=retrieval,
            failure_semantics="bounded parameter adjustment failure",
        ),
    }
    return {action: definitions[action] for action in config.enabled}


def rewrite_query(question: str, facet: EvidenceRequirement) -> str:
    terms = list(dict.fromkeys(facet.condition.terms))
    suffix = " ".join(terms[:8])
    return " ".join(f"{question.strip()} {suffix}".split())[:2000]


def decompose_question(
    question: str, facets: Sequence[EvidenceRequirement], *, max_subquestions: int
) -> list[str]:
    values = [rewrite_query(question, item) for item in sorted(facets, key=lambda x: x.facet_id)]
    return list(dict.fromkeys(values))[: min(3, max_subquestions)]


def build_action_request(
    action: CorrectionAction,
    question: str,
    target_facets: Sequence[EvidenceRequirement],
    config: ActionBoundsConfig,
) -> ActionRequest:
    if action not in action_catalog(config):
        raise ValueError(f"correction action is disabled or unknown: {action}")
    facets = sorted(target_facets, key=lambda item: item.facet_id)
    if not facets:
        raise ValueError("correction action requires at least one target facet")
    query = rewrite_query(question, facets[0]) if action is CorrectionAction.REWRITE else question
    subqueries = (
        decompose_question(question, facets, max_subquestions=config.max_subquestions)
        if action is CorrectionAction.DECOMPOSE
        else []
    )
    max_hops = config.max_graph_hops if action is CorrectionAction.GRAPH else 1
    payload = {
        "action": action,
        "question": query,
        "facets": [item.facet_id for item in facets],
        "subqueries": subqueries,
        "top_k": config.default_top_k,
        "max_hops": max_hops,
    }
    return ActionRequest(
        action_id="action-" + stable_digest(payload)[:16],
        action=action,
        target_facet_ids=[item.facet_id for item in facets],
        query=re.sub(r"\s+", " ", query).strip(),
        top_k=config.default_top_k,
        max_candidates=config.max_candidates_per_action,
        max_hops=max_hops,
        subqueries=subqueries,
    )
