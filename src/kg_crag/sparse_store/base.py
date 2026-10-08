"""Sparse 索引后端的公共契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from kg_crag.models import Chunk, Evidence, SparseIndexIdentity


@dataclass(frozen=True)
class SparseRecordState:
    content_hash: str
    processing_version: str


@dataclass(frozen=True)
class SparseSyncResult:
    paper_id: str
    added: int
    updated: int
    skipped: int
    deleted: int


@runtime_checkable
class SparseStore(Protocol):
    """以论文为事务边界同步 Chunk，并执行有界词法检索。"""

    @property
    def identity(self) -> SparseIndexIdentity: ...

    @property
    def is_initialized(self) -> bool: ...

    async def ensure_index(
        self, identity: SparseIndexIdentity, *, rebuild: bool = False
    ) -> None: ...

    async def sync_paper(self, paper_id: str, chunks: list[Chunk]) -> SparseSyncResult: ...

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        offset: int = 0,
        filters: dict[str, str | int | bool] | None = None,
    ) -> list[Evidence]: ...

    async def record_state(self, paper_id: str) -> dict[str, SparseRecordState]: ...

    async def paper_context(self, paper_id: str, *, limit: int) -> list[Evidence]: ...
