"""单次、有界且可追溯的 Dense Retriever。"""

from __future__ import annotations

from kg_crag.errors import KGCRAGError
from kg_crag.models import ErrorCode, ErrorDetail, Evidence
from kg_crag.providers import EmbeddingService
from kg_crag.retrieval.config import DenseRAGConfig
from kg_crag.vector_store import VectorStore


def _invalid(message: str, **context: str | int | bool) -> KGCRAGError:
    return KGCRAGError(
        ErrorDetail(
            code=ErrorCode.VALIDATION,
            message=message,
            retryable=False,
            context=context,
        )
    )


class DenseRetriever:
    def __init__(
        self,
        embedding: EmbeddingService,
        store: VectorStore,
        config: DenseRAGConfig,
    ) -> None:
        self.embedding = embedding
        self.store = store
        self.config = config

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, str | int | bool] | None = None,
        deduplicate_by_paper: bool | None = None,
    ) -> list[Evidence]:
        normalized = query.strip()
        if not normalized:
            raise _invalid("query must not be empty")
        if top_k <= 0 or top_k > self.config.dense.max_top_k:
            raise _invalid("top_k is outside the configured boundary", top_k=top_k)
        selected_filters = dict(filters or {})
        unknown = set(selected_filters) - set(self.config.collection.allowed_filters)
        if unknown:
            raise _invalid("unsupported filter field", field=sorted(unknown)[0])
        deduplicate = (
            self.config.dense.deduplicate_by_paper
            if deduplicate_by_paper is None
            else deduplicate_by_paper
        )
        candidate_count = top_k
        if deduplicate:
            candidate_count = min(
                top_k * self.config.dense.candidate_multiplier,
                self.config.dense.max_candidates,
            )
        embedded = await self.embedding.embed([normalized], input_type="query")
        candidates = await self.store.search(
            embedded.vectors[0],
            top_k=candidate_count,
            filters=selected_filters,
        )
        candidates.sort(
            key=lambda item: (
                -(item.scores.dense if item.scores.dense is not None else float("-inf")),
                item.source_id,
            )
        )
        if not deduplicate:
            return candidates[:top_k]
        selected: list[Evidence] = []
        seen_papers: set[str] = set()
        for candidate in candidates:
            key = candidate.paper_id or f"__missing__:{candidate.source_id}"
            if key in seen_papers:
                continue
            seen_papers.add(key)
            selected.append(candidate)
            if len(selected) == top_k:
                break
        return selected
