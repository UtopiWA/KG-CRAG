"""固定调用 Dense/Sparse、显式降级并可选重排的 Hybrid 服务。"""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import time
import uuid
from pathlib import Path

from kg_crag.errors import KGCRAGError
from kg_crag.ingestion.storage import stable_json_bytes
from kg_crag.models import (
    ErrorCode,
    ErrorDetail,
    Evidence,
    HybridRetrievalResult,
    RetrievalCostSummary,
    RetrievalStageStatus,
    RetrievalStageSummary,
)
from kg_crag.retrieval.base import Reranker, Retriever
from kg_crag.retrieval.config import HybridRetrievalConfig, dense_config_hash
from kg_crag.retrieval.fusion import deduplicate_evidence_by_paper, fuse_evidence, fusion_version


class HybridRetrievalService:
    """每路最多调用一次，不隐藏重试、回答生成、Graph 或 Web 调用。"""

    def __init__(
        self,
        dense: Retriever,
        sparse: Retriever,
        config: HybridRetrievalConfig,
        *,
        collection_version: str,
        sparse_index_version: str,
        corpus_snapshot_hash: str,
        reranker: Reranker | None = None,
        workspace_root: Path | None = None,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.reranker = reranker
        self.config = config
        self.collection_version = collection_version
        self.sparse_index_version = sparse_index_version
        self.corpus_snapshot_hash = corpus_snapshot_hash
        self.workspace_root = workspace_root.resolve() if workspace_root else None

    async def retrieve(
        self,
        query: str,
        *,
        filters: dict[str, str | int | bool] | None = None,
        persist: bool = False,
    ) -> HybridRetrievalResult:
        normalized = query.strip()
        if not normalized:
            raise _hybrid_error(ErrorCode.VALIDATION, "hybrid query must not be empty")
        dense_call = self._call_retriever(
            "dense", self.dense, normalized, self.config.dense.dense.top_k, filters
        )
        sparse_call = self._call_retriever(
            "sparse", self.sparse, normalized, self.config.sparse.top_k, filters
        )
        dense_result, sparse_result = await asyncio.gather(dense_call, sparse_call)
        dense_items, dense_stage = dense_result
        sparse_items, sparse_stage = sparse_result
        failed = [
            stage.stage
            for stage in (dense_stage, sparse_stage)
            if stage.status is RetrievalStageStatus.FAILED
        ]
        if len(failed) == 2 or (failed and not self.config.hybrid.allow_partial):
            raise _hybrid_error(
                ErrorCode.EXTERNAL_SERVICE,
                "hybrid retrieval failed under the configured degradation policy",
                {"dense_failed": "dense" in failed, "sparse_failed": "sparse" in failed},
            )

        fusion_started = time.perf_counter()
        fused = fuse_evidence(dense_items, sparse_items, self.config.fusion)
        fusion_stage = RetrievalStageSummary(
            stage="fusion",
            status=RetrievalStageStatus.SUCCEEDED if fused else RetrievalStageStatus.EMPTY,
            input_candidates=len(dense_items) + len(sparse_items),
            output_candidates=len(fused),
            latency_ms=(time.perf_counter() - fusion_started) * 1000,
            call_count=1,
        )
        final, rerank_stage, reranker_calls, reranker_latency, rerank_degraded = await self._rerank(
            normalized, fused
        )
        degraded_stages = [stage for stage in failed if stage in {"dense", "sparse"}]
        if rerank_degraded:
            degraded_stages.append("rerank")
        if self.config.hybrid.deduplicate_by_paper:
            final = deduplicate_evidence_by_paper(final, max_results=self.config.hybrid.max_output)
        else:
            final = [item.model_copy(deep=True) for item in final[: self.config.hybrid.max_output]]

        version = fusion_version(self.config.fusion)
        result = HybridRetrievalResult(
            run_id=uuid.uuid4().hex,
            query_id=hashlib.sha256(normalized.encode()).hexdigest()[:24],
            query=normalized,
            config_hash=dense_config_hash(self.config),
            corpus_snapshot_hash=self.corpus_snapshot_hash,
            collection_version=self.collection_version,
            sparse_index_version=self.sparse_index_version,
            fusion_version=version,
            reranker_version=(
                f"{self.config.reranker.model}@{self.config.reranker.revision}"
                if reranker_calls
                else None
            ),
            evidence=final,
            degraded=bool(degraded_stages),
            degraded_stages=degraded_stages,
            stages=[dense_stage, sparse_stage, fusion_stage, rerank_stage],
            costs=RetrievalCostSummary(
                embedding_calls=1,
                sparse_calls=1,
                reranker_calls=reranker_calls,
                local_reranker_latency_ms=reranker_latency,
            ),
        )
        if persist:
            if self.workspace_root is None:
                raise ValueError("workspace_root is required to persist Hybrid results")
            publish_hybrid_result(
                result,
                workspace_root=self.workspace_root,
                output_root=self.config.hybrid.output_root,
            )
        return result

    async def _call_retriever(
        self,
        name: str,
        retriever: Retriever,
        query: str,
        top_k: int,
        filters: dict[str, str | int | bool] | None,
    ) -> tuple[list[Evidence], RetrievalStageSummary]:
        started = time.perf_counter()
        try:
            evidence = await retriever.retrieve(query, top_k=top_k, filters=filters)
            return evidence, RetrievalStageSummary(
                stage=name,
                status=(RetrievalStageStatus.SUCCEEDED if evidence else RetrievalStageStatus.EMPTY),
                output_candidates=len(evidence),
                latency_ms=(time.perf_counter() - started) * 1000,
                call_count=1,
            )
        except Exception as error:
            return [], RetrievalStageSummary(
                stage=name,
                status=RetrievalStageStatus.FAILED,
                latency_ms=(time.perf_counter() - started) * 1000,
                call_count=1,
                error=_safe_error(error, name),
            )

    async def _rerank(
        self, query: str, fused: list[Evidence]
    ) -> tuple[list[Evidence], RetrievalStageSummary, int, float, bool]:
        if not self.config.reranker.enabled or not fused:
            return (
                fused,
                RetrievalStageSummary(
                    stage="rerank", status=RetrievalStageStatus.SKIPPED, call_count=0
                ),
                0,
                0.0,
                False,
            )
        if self.reranker is None:
            raise _hybrid_error(ErrorCode.CONFIGURATION, "enabled reranker is not configured")
        candidates = fused[: self.config.reranker.max_candidates]
        started = time.perf_counter()
        try:
            ranked = await self.reranker.rerank(query, candidates, top_k=self.config.reranker.top_k)
            latency = (time.perf_counter() - started) * 1000
            ranked = [
                item.model_copy(
                    update={"ranks": item.ranks.model_copy(update={"rerank": rank})},
                    deep=True,
                )
                for rank, item in enumerate(ranked, start=1)
            ]
            return (
                ranked,
                RetrievalStageSummary(
                    stage="rerank",
                    status=(
                        RetrievalStageStatus.SUCCEEDED if ranked else RetrievalStageStatus.EMPTY
                    ),
                    input_candidates=len(candidates),
                    output_candidates=len(ranked),
                    latency_ms=latency,
                    call_count=1,
                ),
                1,
                latency,
                False,
            )
        except Exception as error:
            latency = (time.perf_counter() - started) * 1000
            if not self.config.hybrid.allow_rerank_fallback:
                raise _hybrid_error(ErrorCode.EXTERNAL_SERVICE, "hybrid reranker failed") from error
            return (
                fused,
                RetrievalStageSummary(
                    stage="rerank",
                    status=RetrievalStageStatus.FAILED,
                    input_candidates=len(candidates),
                    latency_ms=latency,
                    call_count=1,
                    error=_safe_error(error, "rerank"),
                ),
                1,
                latency,
                True,
            )


def publish_hybrid_result(
    result: HybridRetrievalResult,
    *,
    workspace_root: Path,
    output_root: str,
) -> Path:
    """在临时目录回读验证后原子发布一次 Hybrid 检索结果。"""

    workspace = workspace_root.resolve()
    root = (workspace / output_root).resolve()
    if not root.is_relative_to(workspace):
        raise ValueError("hybrid output root escapes the workspace")
    destination = root / result.run_id
    temporary = root / f".{result.run_id}.tmp-{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    try:
        temporary.mkdir(exist_ok=False)
        data = stable_json_bytes(result)
        path = temporary / "result.json"
        path.write_bytes(data)
        HybridRetrievalResult.model_validate_json(path.read_text(encoding="utf-8"))
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination / "result.json"


def _safe_error(error: Exception, stage: str) -> ErrorDetail:
    if isinstance(error, KGCRAGError):
        return error.detail
    return ErrorDetail(
        code=ErrorCode.INTERNAL,
        message=f"{stage} stage failed",
        retryable=False,
        context={"error_type": type(error).__name__, "stage": stage},
    )


def _hybrid_error(
    code: ErrorCode,
    message: str,
    context: dict[str, str | int | float | bool | None] | None = None,
) -> KGCRAGError:
    return KGCRAGError(
        ErrorDetail(code=code, message=message, retryable=False, context=context or {})
    )
