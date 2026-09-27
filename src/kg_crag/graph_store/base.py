"""知识图谱存储接口。"""

from typing import Protocol, runtime_checkable

from kg_crag.models import Chunk, Evidence, Paper


@runtime_checkable
class GraphStore(Protocol):
    """持久化带溯源信息的图记录，并执行受控读取。"""

    async def upsert_paper(self, paper: Paper, chunks: list[Chunk]) -> None: ...

    async def retrieve(
        self,
        entity_names: list[str],
        *,
        max_hops: int,
        top_k: int,
    ) -> list[Evidence]: ...

    async def delete_paper(self, paper_id: str) -> None: ...
