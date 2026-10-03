"""版本化知识图谱存储接口。"""

from typing import Protocol, runtime_checkable

from kg_crag.models import (
    Chunk,
    Evidence,
    GraphBundle,
    GraphIdentity,
    GraphPathHit,
    GraphQueryRequest,
    GraphRecordState,
    GraphSyncResult,
    Paper,
)


@runtime_checkable
class GraphStore(Protocol):
    """管理图身份、论文级目标状态与受控只读查询。"""

    async def ensure_graph(self, identity: GraphIdentity, *, rebuild: bool = False) -> None: ...

    async def record_state(self, paper_id: str) -> GraphRecordState | None: ...

    async def sync_paper(self, bundle: GraphBundle) -> GraphSyncResult: ...

    async def query(self, request: GraphQueryRequest) -> list[GraphPathHit]: ...

    async def close(self) -> None: ...

    # 旧接口只用于迁移与早期离线测试，新编排不得依赖它们。
    async def upsert_paper(self, paper: Paper, chunks: list[Chunk]) -> None: ...

    async def retrieve(
        self,
        entity_names: list[str],
        *,
        max_hops: int,
        top_k: int,
    ) -> list[Evidence]: ...

    async def delete_paper(self, paper_id: str) -> None: ...
