"""向量存储接口。"""

from typing import Protocol, runtime_checkable

from kg_crag.models import Chunk, Evidence


@runtime_checkable
class VectorStore(Protocol):
    """持久化向量，并提供与后端无关的相似度检索。"""

    async def upsert(self, records: list[tuple[Chunk, list[float]]]) -> None: ...

    async def search(
        self,
        vector: list[float],
        *,
        top_k: int,
        filters: dict[str, str | int | bool] | None = None,
    ) -> list[Evidence]: ...

    async def delete_paper(self, paper_id: str) -> None: ...
