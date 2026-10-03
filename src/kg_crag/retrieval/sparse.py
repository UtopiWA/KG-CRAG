"""单次、有界且不触发其他工具的 Sparse Retriever。"""

from __future__ import annotations

from kg_crag.errors import KGCRAGError
from kg_crag.models import ErrorCode, ErrorDetail, Evidence
from kg_crag.retrieval.config import SparseRetrievalConfig
from kg_crag.sparse_store import SparseStore, prepare_sparse_query


def _invalid(message: str, **context: str | int | bool) -> KGCRAGError:
    return KGCRAGError(
        ErrorDetail(
            code=ErrorCode.VALIDATION,
            message=message,
            retryable=False,
            context=context,
        )
    )


class SparseRetriever:
    """在调用 Store 前完成全部输入边界校验，且每次只搜索一次。"""

    def __init__(self, store: SparseStore, config: SparseRetrievalConfig) -> None:
        self.store = store
        self.config = config

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, str | int | bool] | None = None,
    ) -> list[Evidence]:
        prepare_sparse_query(
            query,
            max_chars=self.config.max_query_chars,
            max_tokens=self.config.max_query_tokens,
        )
        if top_k <= 0 or top_k > self.config.max_top_k:
            raise _invalid("top_k is outside the configured sparse boundary", top_k=top_k)
        selected_filters = dict(filters or {})
        unknown = set(selected_filters) - set(self.config.allowed_filters)
        if unknown:
            raise _invalid("unsupported sparse filter field", field=sorted(unknown)[0])
        results = await self.store.search(
            query,
            top_k=top_k,
            filters=selected_filters,
        )
        results.sort(
            key=lambda item: (
                -(item.scores.sparse if item.scores.sparse is not None else float("-inf")),
                item.source_id,
            )
        )
        return [
            item.model_copy(update={"ranks": item.ranks.model_copy(update={"sparse": rank})})
            for rank, item in enumerate(results[:top_k], start=1)
        ]
