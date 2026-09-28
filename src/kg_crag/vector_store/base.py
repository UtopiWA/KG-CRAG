"""向量存储接口。"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from kg_crag.models import Chunk, Evidence


@dataclass(frozen=True)
class VectorRecordState:
    content_hash: str
    processing_version: str


@runtime_checkable
class VectorStore(Protocol):
    """持久化向量，并提供与后端无关的相似度检索。"""

    async def ensure_collection(self, *, rebuild: bool = False) -> None: ...

    async def upsert(self, records: list[tuple[Chunk, list[float]]]) -> None: ...

    async def search(
        self,
        vector: list[float],
        *,
        top_k: int,
        filters: dict[str, str | int | bool] | None = None,
    ) -> list[Evidence]: ...

    async def delete_paper(self, paper_id: str) -> None: ...

    async def record_state(self, paper_id: str) -> dict[str, VectorRecordState]: ...

    async def delete_stale(self, paper_id: str, keep_chunk_ids: set[str]) -> int: ...
