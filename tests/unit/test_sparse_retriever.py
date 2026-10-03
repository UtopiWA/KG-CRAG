"""Sparse Retriever 的提前校验、科研术语召回与溯源测试。"""

import pytest

from kg_crag.errors import KGCRAGError
from kg_crag.models import Chunk, ErrorCode, ErrorDetail, Evidence
from kg_crag.retrieval import SparseRetrievalConfig, SparseRetriever
from kg_crag.sparse_store import InMemorySparseStore


def _chunk(chunk_id: str, paper_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        section="Abstract",
        page_start=2,
        page_end=2,
        text=text,
        token_count=max(1, len(text.split())),
        content_hash=f"content-{chunk_id}",
    )


async def test_sparse_retriever_recalls_scientific_terms_and_round_trips_chunks() -> None:
    store = InMemorySparseStore()
    chunks = [
        _chunk("c1", "p1", "GPT-4 is evaluated on HumanEval."),
        _chunk("c2", "p2", "LLaMA-2 uses the MMLU dataset."),
        _chunk("c3", "p3", "A rare entity named Zorblax appears."),
    ]
    for chunk in chunks:
        await store.sync_paper(chunk.paper_id, [chunk])
    retriever = SparseRetriever(store, SparseRetrievalConfig())

    for query, expected in [("GPT-4", "c1"), ("LLaMA-2", "c2"), ("MMLU", "c2")]:
        found = await retriever.retrieve(query, top_k=3)
        assert found[0].source_id == expected
        original = next(chunk for chunk in chunks if chunk.chunk_id == expected)
        assert found[0].paper_id == original.paper_id
        assert found[0].content == original.text
        assert found[0].metadata["content_hash"] == original.content_hash
        assert found[0].external is False


async def test_sparse_retriever_rejects_before_store_call() -> None:
    store = InMemorySparseStore()
    retriever = SparseRetriever(
        store, SparseRetrievalConfig(top_k=2, max_top_k=2, max_candidates=2)
    )

    with pytest.raises(KGCRAGError, match="empty"):
        await retriever.retrieve(" ", top_k=1)
    with pytest.raises(KGCRAGError, match="top_k"):
        await retriever.retrieve("valid", top_k=3)
    with pytest.raises(KGCRAGError, match="unsupported sparse filter"):
        await retriever.retrieve("valid", top_k=1, filters={"unknown": "x"})
    assert store.calls == []


async def test_sparse_retriever_preserves_empty_and_backend_failure_semantics() -> None:
    store = InMemorySparseStore()
    retriever = SparseRetriever(store, SparseRetrievalConfig())
    assert await retriever.retrieve("not-found", top_k=2) == []

    class FailingStore(InMemorySparseStore):
        async def search(
            self,
            query: str,
            *,
            top_k: int,
            offset: int = 0,
            filters: dict[str, str | int | bool] | None = None,
        ) -> list[Evidence]:
            raise KGCRAGError(
                ErrorDetail(
                    code=ErrorCode.EXTERNAL_SERVICE,
                    message="sparse backend unavailable",
                    retryable=True,
                )
            )

    failing = SparseRetriever(FailingStore(), SparseRetrievalConfig())
    with pytest.raises(KGCRAGError, match="backend unavailable") as captured:
        await failing.retrieve("query", top_k=1)
    assert captured.value.detail.retryable is True
