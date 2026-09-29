"""有界、幂等且逐篇隔离的 Dense 索引流水线。"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from kg_crag.errors import KGCRAGError
from kg_crag.indexing.processed import ProcessedPaper
from kg_crag.ingestion.storage import stable_json_bytes
from kg_crag.models import (
    ErrorCode,
    ErrorDetail,
    IndexItemResult,
    IndexItemStatus,
    IndexRunManifest,
    VectorCollectionIdentity,
)
from kg_crag.providers import EmbeddingService
from kg_crag.retrieval.config import DenseRAGConfig, dense_config_hash
from kg_crag.vector_store import VectorStore, collection_identity, corpus_snapshot_hash


@dataclass(frozen=True)
class IndexPlan:
    collection: VectorCollectionIdentity
    corpus_snapshot_hash: str
    items: list[IndexItemResult]


class DenseIndexPipeline:
    def __init__(
        self,
        config: DenseRAGConfig,
        embedding: EmbeddingService,
        store: VectorStore,
        *,
        workspace_root: Path,
    ) -> None:
        self.config = config
        self.embedding = embedding
        self.store = store
        self.workspace_root = workspace_root.resolve()
        self.identity = collection_identity(config)

    async def plan(self, papers: list[ProcessedPaper], *, rebuild: bool = False) -> IndexPlan:
        chunks = [chunk for paper in papers for chunk in paper.chunks]
        items: list[IndexItemResult] = []
        for paper in papers:
            # 规划阶段只比较远端状态与本地稳定字段，不加载模型也不写入集合。
            existing = {} if rebuild else await self.store.record_state(paper.paper_id)
            target = {chunk.chunk_id: chunk for chunk in paper.chunks}
            added = sum(chunk_id not in existing for chunk_id in target)
            updated = sum(
                chunk_id in existing
                and (
                    existing[chunk_id].content_hash != chunk.content_hash
                    or existing[chunk_id].processing_version != chunk.processing_version
                )
                for chunk_id, chunk in target.items()
            )
            skipped = len(target) - added - updated
            deleted = len(set(existing) - set(target))
            status = (
                IndexItemStatus.SKIPPED
                if skipped == len(target) and deleted == 0
                else IndexItemStatus.PLANNED
            )
            items.append(
                IndexItemResult(
                    paper_id=paper.paper_id,
                    status=status,
                    chunk_count=len(target),
                    added=added,
                    updated=updated,
                    skipped=skipped,
                    deleted=deleted,
                )
            )
        return IndexPlan(
            collection=self.identity,
            corpus_snapshot_hash=corpus_snapshot_hash(chunks),
            items=items,
        )

    async def run(
        self,
        papers: list[ProcessedPaper],
        *,
        dry_run: bool,
        rebuild: bool = False,
    ) -> IndexRunManifest:
        started = datetime.now(UTC)
        plan = await self.plan(papers, rebuild=rebuild)
        if dry_run:
            return self._manifest(started, plan, plan.items, dry_run=True, path=None)

        # dry-run 已提前返回，因此集合创建、重建和后续写入只发生在实际运行路径。
        await self.store.ensure_collection(rebuild=rebuild)
        results: list[IndexItemResult] = []
        for paper, item_plan in zip(papers, plan.items, strict=True):
            item_started = time.perf_counter()
            if item_plan.status is IndexItemStatus.SKIPPED:
                results.append(
                    item_plan.model_copy(
                        update={"latency_ms": (time.perf_counter() - item_started) * 1000}
                    )
                )
                continue
            try:
                existing = await self.store.record_state(paper.paper_id)
                changed = [
                    chunk
                    for chunk in paper.chunks
                    if chunk.chunk_id not in existing
                    or existing[chunk.chunk_id].content_hash != chunk.content_hash
                    or existing[chunk.chunk_id].processing_version != chunk.processing_version
                ]
                # 先完整向量化本篇所有变更 Chunk；失败时不会留下半篇新向量。
                embedded = await self.embedding.embed(
                    [chunk.text for chunk in changed], input_type="document"
                )
                for offset in range(0, len(changed), self.config.indexing.upsert_batch_size):
                    batch_chunks = changed[offset : offset + self.config.indexing.upsert_batch_size]
                    batch_vectors = embedded.vectors[
                        offset : offset + self.config.indexing.upsert_batch_size
                    ]
                    await self.store.upsert(list(zip(batch_chunks, batch_vectors, strict=True)))
                # 新记录写入成功后才删除陈旧点，保证失败时旧索引仍可服务。
                deleted = await self.store.delete_stale(
                    paper.paper_id, {chunk.chunk_id for chunk in paper.chunks}
                )
                results.append(
                    item_plan.model_copy(
                        update={
                            "status": IndexItemStatus.SUCCEEDED,
                            "deleted": deleted,
                            "cache_hits": embedded.cache_hits,
                            "latency_ms": (time.perf_counter() - item_started) * 1000,
                        }
                    )
                )
            except Exception as error:
                results.append(
                    IndexItemResult(
                        paper_id=paper.paper_id,
                        status=IndexItemStatus.FAILED,
                        chunk_count=len(paper.chunks),
                        latency_ms=(time.perf_counter() - item_started) * 1000,
                        error=self._safe_error(error),
                    )
                )
        config_prefix = dense_config_hash(self.config)[:8]
        run_id = f"{started:%Y%m%dT%H%M%SZ}-{config_prefix}-{uuid.uuid4().hex[:8]}"
        relative = Path(self.config.indexing.manifest_root) / run_id / "manifest.json"
        manifest = self._manifest(
            started,
            plan,
            results,
            dry_run=False,
            path=relative.as_posix(),
            run_id=run_id,
        )
        self._write_manifest(relative, manifest)
        return manifest

    def _manifest(
        self,
        started: datetime,
        plan: IndexPlan,
        items: list[IndexItemResult],
        *,
        dry_run: bool,
        path: str | None,
        run_id: str | None = None,
    ) -> IndexRunManifest:
        return IndexRunManifest(
            run_id=run_id or f"dry-run-{dense_config_hash(self.config)[:12]}",
            dry_run=dry_run,
            config_hash=dense_config_hash(self.config),
            corpus_snapshot_hash=plan.corpus_snapshot_hash,
            collection=plan.collection,
            started_at=started,
            finished_at=datetime.now(UTC),
            items=items,
            manifest_path=path,
        )

    def _write_manifest(self, relative: Path, manifest: IndexRunManifest) -> None:
        destination = (self.workspace_root / relative).resolve()
        if not destination.is_relative_to(self.workspace_root):
            raise ValueError("index manifest destination escapes the workspace")
        destination.parent.mkdir(parents=True, exist_ok=False)
        temporary = destination.with_suffix(".json.tmp")
        try:
            data = stable_json_bytes(manifest)
            temporary.write_bytes(data)
            if temporary.read_bytes() != data:
                raise OSError("index manifest staging verification failed")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _safe_error(error: Exception) -> ErrorDetail:
        if isinstance(error, KGCRAGError):
            return error.detail
        return ErrorDetail(
            code=ErrorCode.INTERNAL,
            message="paper indexing failed",
            retryable=False,
            context={"error_type": type(error).__name__},
        )
