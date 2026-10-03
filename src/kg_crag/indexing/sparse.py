"""Sparse 索引的有界规划、断点恢复和逐篇失败隔离。"""

from __future__ import annotations

import json
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
    IndexItemStatus,
    SparseIndexIdentity,
    SparseIndexItemResult,
    SparseIndexRunManifest,
)
from kg_crag.retrieval.config import HybridRetrievalConfig, dense_config_hash
from kg_crag.sparse_store import SparseStore


@dataclass(frozen=True)
class SparseIndexPlan:
    index: SparseIndexIdentity
    corpus_snapshot_hash: str
    items: list[SparseIndexItemResult]


class SparseIndexPipeline:
    """只在正式运行时创建索引、检查点和运行清单。"""

    def __init__(
        self,
        config: HybridRetrievalConfig,
        store: SparseStore,
        *,
        workspace_root: Path,
    ) -> None:
        self.config = config
        self.store = store
        self.workspace_root = workspace_root.resolve()
        self.identity = store.identity

    async def plan(self, papers: list[ProcessedPaper], *, rebuild: bool = False) -> SparseIndexPlan:
        if self.store.is_initialized and not rebuild:
            await self.store.ensure_index(self.identity)
        items: list[SparseIndexItemResult] = []
        for paper in papers:
            existing = (
                {}
                if rebuild or not self.store.is_initialized
                else await self.store.record_state(paper.paper_id)
            )
            target = {chunk.chunk_id: chunk for chunk in paper.chunks}
            added = len(target.keys() - existing.keys())
            deleted = len(existing.keys() - target.keys())
            updated = sum(
                chunk_id in existing
                and (
                    existing[chunk_id].content_hash != chunk.content_hash
                    or existing[chunk_id].processing_version != chunk.processing_version
                )
                for chunk_id, chunk in target.items()
            )
            skipped = len(target) - added - updated
            status = (
                IndexItemStatus.SKIPPED
                if skipped == len(target) and deleted == 0
                else IndexItemStatus.PLANNED
            )
            items.append(
                SparseIndexItemResult(
                    paper_id=paper.paper_id,
                    status=status,
                    chunk_count=len(target),
                    added=added,
                    updated=updated,
                    skipped=skipped,
                    deleted=deleted,
                )
            )
        return SparseIndexPlan(
            index=self.identity,
            corpus_snapshot_hash=self.identity.corpus_snapshot_hash,
            items=items,
        )

    async def run(
        self,
        papers: list[ProcessedPaper],
        *,
        dry_run: bool,
        rebuild: bool = False,
    ) -> SparseIndexRunManifest:
        started = datetime.now(UTC)
        plan = await self.plan(papers, rebuild=rebuild)
        if dry_run:
            return self._manifest(started, plan, plan.items, dry_run=True, path=None)

        await self.store.ensure_index(self.identity, rebuild=rebuild)
        checkpoint_path = self._checkpoint_path()
        completed = set() if rebuild else self._load_checkpoint(checkpoint_path)
        results: list[SparseIndexItemResult] = []
        for paper, item_plan in zip(papers, plan.items, strict=True):
            item_started = time.perf_counter()
            if item_plan.status is IndexItemStatus.SKIPPED:
                results.append(
                    item_plan.model_copy(
                        update={"latency_ms": (time.perf_counter() - item_started) * 1000}
                    )
                )
                completed.add(paper.paper_id)
                self._write_checkpoint(checkpoint_path, completed)
                continue
            try:
                synced = await self.store.sync_paper(paper.paper_id, list(paper.chunks))
                results.append(
                    SparseIndexItemResult(
                        paper_id=paper.paper_id,
                        status=IndexItemStatus.SUCCEEDED,
                        chunk_count=len(paper.chunks),
                        added=synced.added,
                        updated=synced.updated,
                        skipped=synced.skipped,
                        deleted=synced.deleted,
                        latency_ms=(time.perf_counter() - item_started) * 1000,
                    )
                )
                completed.add(paper.paper_id)
                self._write_checkpoint(checkpoint_path, completed)
            except Exception as error:
                results.append(
                    SparseIndexItemResult(
                        paper_id=paper.paper_id,
                        status=IndexItemStatus.FAILED,
                        chunk_count=len(paper.chunks),
                        latency_ms=(time.perf_counter() - item_started) * 1000,
                        error=self._safe_error(error),
                    )
                )

        config_prefix = dense_config_hash(self.config)[:8]
        run_id = f"{started:%Y%m%dT%H%M%SZ}-{config_prefix}-{uuid.uuid4().hex[:8]}"
        relative = Path(self.config.sparse.manifest_root) / run_id / "manifest.json"
        manifest = self._manifest(
            started,
            plan,
            results,
            dry_run=False,
            path=relative.as_posix(),
            run_id=run_id,
        )
        self._write_model(relative, manifest)
        return manifest

    def _manifest(
        self,
        started: datetime,
        plan: SparseIndexPlan,
        items: list[SparseIndexItemResult],
        *,
        dry_run: bool,
        path: str | None,
        run_id: str | None = None,
    ) -> SparseIndexRunManifest:
        return SparseIndexRunManifest(
            run_id=run_id or f"dry-run-{dense_config_hash(self.config)[:12]}",
            dry_run=dry_run,
            config_hash=dense_config_hash(self.config),
            corpus_snapshot_hash=plan.corpus_snapshot_hash,
            index=plan.index,
            started_at=started,
            finished_at=datetime.now(UTC),
            items=items,
            manifest_path=path,
        )

    def _checkpoint_path(self) -> Path:
        relative = (
            Path(self.config.sparse.manifest_root)
            / "checkpoints"
            / f"{self.identity.index_version}.json"
        )
        return self._contained(relative)

    def _load_checkpoint(self, path: Path) -> set[str]:
        if not path.is_file():
            return set()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload.get("schema_version") != "v1"
                or payload.get("index_version") != self.identity.index_version
            ):
                raise ValueError("checkpoint identity mismatch")
            completed = payload.get("completed_paper_ids")
            if not isinstance(completed, list) or not all(
                isinstance(item, str) for item in completed
            ):
                raise ValueError("checkpoint completed_paper_ids is invalid")
            return set(completed)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise ValueError("sparse checkpoint is invalid") from error

    def _write_checkpoint(self, path: Path, completed: set[str]) -> None:
        payload = {
            "schema_version": "v1",
            "index_version": self.identity.index_version,
            "completed_paper_ids": sorted(completed),
        }
        self._write_bytes(
            path,
            (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
        )

    def _write_model(self, relative: Path, model: SparseIndexRunManifest) -> None:
        self._write_bytes(self._contained(relative), stable_json_bytes(model))

    def _contained(self, relative: Path) -> Path:
        destination = (self.workspace_root / relative).resolve()
        if not destination.is_relative_to(self.workspace_root):
            raise ValueError("sparse index output escapes the workspace")
        return destination

    @staticmethod
    def _write_bytes(destination: Path, data: bytes) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(f"{destination.suffix}.tmp")
        try:
            temporary.write_bytes(data)
            if temporary.read_bytes() != data:
                raise OSError("sparse output staging verification failed")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _safe_error(error: Exception) -> ErrorDetail:
        if isinstance(error, KGCRAGError):
            return error.detail
        return ErrorDetail(
            code=ErrorCode.INTERNAL,
            message="paper sparse indexing failed",
            retryable=False,
            context={"error_type": type(error).__name__},
        )
