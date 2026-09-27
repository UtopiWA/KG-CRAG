"""检索与重排序接口。"""

from typing import Protocol, runtime_checkable

from kg_crag.models import Evidence


@runtime_checkable
class Retriever(Protocol):
    """返回与后端实现无关的规范化证据。"""

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, str | int | bool] | None = None,
    ) -> list[Evidence]: ...


@runtime_checkable
class Reranker(Protocol):
    """为单个查询排序证据，并保留溯源字段。"""

    async def rerank(
        self,
        query: str,
        candidates: list[Evidence],
        *,
        top_k: int,
    ) -> list[Evidence]: ...
