"""纠错动作薄适配器、统一执行器与 Evidence 入状态门禁。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from kg_crag.correction.config import ActionBoundsConfig
from kg_crag.correction.coverage import PermissiveInternalVerifier, SourceVerifier
from kg_crag.errors import KGCRAGError
from kg_crag.models import (
    ActionEstimate,
    ActionRequest,
    ActionResult,
    ActionStatus,
    BudgetUsage,
    CorrectionAction,
    ErrorCode,
    ErrorDetail,
    Evidence,
    EvidenceSourceType,
    GraphQueryRequest,
    GraphQueryTemplate,
)
from kg_crag.retrieval.base import Retriever
from kg_crag.retrieval.graph import GraphRetriever
from kg_crag.retrieval.hybrid import HybridRetrievalService


class CorrectionTool(Protocol):
    async def execute(self, request: ActionRequest, estimate: ActionEstimate) -> ActionResult: ...


def _failed(request: ActionRequest, estimate: ActionEstimate, message: str) -> ActionResult:
    return ActionResult(
        request=request,
        status=ActionStatus.FAILED,
        estimated_usage=estimate,
        actual_usage=BudgetUsage(
            **{name: getattr(estimate, name) for name in BudgetUsage.model_fields}
        ),
        error=ErrorDetail(code=ErrorCode.EXTERNAL_SERVICE, message=message, retryable=False),
    )


class RetrieverTool:
    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever

    async def execute(self, request: ActionRequest, estimate: ActionEstimate) -> ActionResult:
        try:
            evidence = await self._retriever.retrieve(
                request.query, top_k=request.top_k, filters=request.filters
            )
        except Exception:
            return _failed(request, estimate, "retriever action failed")
        evidence = evidence[: request.max_candidates]
        return ActionResult(
            request=request,
            status=ActionStatus.SUCCEEDED if evidence else ActionStatus.EMPTY,
            evidence=evidence,
            estimated_usage=estimate,
            actual_usage=BudgetUsage(retrieval_rounds=1, candidates=len(evidence)),
        )


class HybridTool:
    def __init__(self, service: HybridRetrievalService) -> None:
        self._service = service

    async def execute(self, request: ActionRequest, estimate: ActionEstimate) -> ActionResult:
        try:
            result = await self._service.retrieve(request.query, filters=request.filters)
        except Exception:
            return _failed(request, estimate, "hybrid action failed")
        evidence = result.evidence[: request.max_candidates]
        return ActionResult(
            request=request,
            status=ActionStatus.SUCCEEDED if evidence else ActionStatus.EMPTY,
            evidence=evidence,
            estimated_usage=estimate,
            actual_usage=BudgetUsage(retrieval_rounds=1, candidates=len(evidence)),
            diagnostics={
                "degraded": result.degraded,
                "degraded_stages": ",".join(result.degraded_stages),
            },
        )


class GraphTool:
    def __init__(self, retriever: GraphRetriever, *, graph_version: str) -> None:
        self._retriever = retriever
        self._graph_version = graph_version

    async def execute(self, request: ActionRequest, estimate: ActionEstimate) -> ActionResult:
        template = (
            GraphQueryTemplate.BOUNDED_PATH
            if request.max_hops > 1
            else GraphQueryTemplate.RELATION_LOOKUP
        )
        graph_request = GraphQueryRequest(
            graph_version=self._graph_version,
            template=template,
            entity_names=[request.query[:500]],
            top_k=request.top_k,
            max_candidates=request.max_candidates,
            max_hops=request.max_hops,
        )
        try:
            result = await self._retriever.retrieve(graph_request)
        except Exception:
            return _failed(request, estimate, "graph action failed")
        evidence = result.evidence[: request.max_candidates]
        return ActionResult(
            request=request,
            status=ActionStatus.SUCCEEDED if evidence else ActionStatus.EMPTY,
            evidence=evidence,
            estimated_usage=estimate,
            actual_usage=BudgetUsage(retrieval_rounds=1, candidates=len(evidence)),
            diagnostics={
                "database_calls": result.database_calls,
                "rejected_paths": result.rejected_paths,
            },
        )


class CompoundTool:
    """重写/分解/调参只委派给创建时固定的内部 Retriever。"""

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever

    async def execute(self, request: ActionRequest, estimate: ActionEstimate) -> ActionResult:
        queries = request.subqueries or [request.query]
        collected: list[Evidence] = []
        try:
            for query in queries:
                collected.extend(
                    await self._retriever.retrieve(
                        query, top_k=request.top_k, filters=request.filters
                    )
                )
        except Exception:
            return _failed(request, estimate, "compound action failed")
        evidence = deduplicate_evidence(collected, limit=request.max_candidates)
        return ActionResult(
            request=request,
            status=ActionStatus.SUCCEEDED if evidence else ActionStatus.EMPTY,
            evidence=evidence,
            estimated_usage=estimate,
            actual_usage=BudgetUsage(
                retrieval_rounds=1,
                subquestions=len(request.subqueries),
                candidates=len(evidence),
            ),
        )


def deduplicate_evidence(
    evidence: Sequence[Evidence],
    *,
    limit: int,
    verifier: SourceVerifier | None = None,
    max_content_chars: int = 10_000,
) -> list[Evidence]:
    selected_verifier = verifier or PermissiveInternalVerifier()
    selected: dict[str, Evidence] = {}
    for item in sorted(evidence, key=lambda value: value.evidence_id):
        valid, _reason = selected_verifier.verify(item)
        if not valid or item.external or item.source_type is EvidenceSourceType.WEB:
            continue
        if len(item.content) > max_content_chars:
            continue
        selected.setdefault(item.evidence_id, item.model_copy(deep=True))
        if len(selected) >= limit:
            break
    return list(selected.values())


class ActionExecutor:
    def __init__(
        self,
        tools: Mapping[CorrectionAction, CorrectionTool],
        *,
        max_candidates: int,
        bounds: ActionBoundsConfig | None = None,
        source_verifier: SourceVerifier | None = None,
        max_evidence_chars: int = 10_000,
    ) -> None:
        self._tools = dict(tools)
        self._max_candidates = max_candidates
        self._bounds = bounds or ActionBoundsConfig(max_candidates_per_action=max_candidates)
        self._source_verifier = source_verifier or PermissiveInternalVerifier()
        self._max_evidence_chars = max_evidence_chars

    async def execute(self, request: ActionRequest, estimate: ActionEstimate) -> ActionResult:
        # model_copy 可绕过 Pydantic 校验，因此在任何后端调用前重新建立严格对象。
        request = ActionRequest.model_validate(request.model_dump(mode="json"))
        if request.action not in self._bounds.enabled:
            raise ValueError("correction action is disabled")
        if (
            request.top_k > self._bounds.max_top_k
            or request.max_candidates > self._bounds.max_candidates_per_action
            or request.max_hops > self._bounds.max_graph_hops
            or len(request.subqueries) > self._bounds.max_subquestions
        ):
            raise ValueError("action exceeds configured executor boundary")
        tool = self._tools.get(request.action)
        if tool is None:
            raise KGCRAGError(
                ErrorDetail(
                    code=ErrorCode.CONFIGURATION,
                    message="correction action has no bound tool",
                    retryable=False,
                    context={"action": request.action.value},
                )
            )
        if request.max_candidates > self._max_candidates:
            raise ValueError("action exceeds executor candidate boundary")
        result = await tool.execute(request, estimate)
        validated = ActionResult.model_validate(result.model_dump(mode="json"))
        if validated.status is ActionStatus.SUCCEEDED:
            clean = deduplicate_evidence(
                validated.evidence,
                limit=self._max_candidates,
                verifier=self._source_verifier,
                max_content_chars=self._max_evidence_chars,
            )
            status = ActionStatus.SUCCEEDED if clean else ActionStatus.EMPTY
            return ActionResult.model_validate(
                validated.model_copy(update={"evidence": clean, "status": status}).model_dump(
                    mode="json"
                )
            )
        return validated
