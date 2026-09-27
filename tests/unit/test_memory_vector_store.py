"""内存 Vector Store 的确定性行为测试。"""

import pytest

from kg_crag.errors import KGCRAGError
from kg_crag.models import Chunk
from kg_crag.vector_store import InMemoryVectorStore


def _chunk(chunk_id: str, paper_id: str, section: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        section=section,
        page_start=1,
        page_end=1,
        text=text,
        token_count=len(text.split()),
        content_hash=f"hash---{chunk_id}",
    )


async def test_vector_store_ranks_stably_and_upserts_idempotently() -> None:
    store = InMemoryVectorStore()
    first = _chunk("c-b", "p1", "Methods", "old text")
    replacement = _chunk("c-b", "p1", "Methods", "replacement text")
    tied = _chunk("c-a", "p2", "Results", "tied text")

    await store.upsert([(first, [1.0, 0.0]), (tied, [1.0, 0.0])])
    await store.upsert([(replacement, [1.0, 0.0])])
    results = await store.search([1.0, 0.0], top_k=5)

    assert store.record_count == 2
    assert store.dimension == 2
    assert [item.source_id for item in results] == ["c-a", "c-b"]
    assert results[1].content == "replacement text"
    assert results[0].scores.dense == pytest.approx(1.0)


async def test_vector_store_filters_and_deletes_only_target_paper() -> None:
    store = InMemoryVectorStore()
    await store.upsert(
        [
            (_chunk("c1", "p1", "Methods", "one"), [1.0, 0.0]),
            (_chunk("c2", "p1", "Results", "two"), [0.9, 0.1]),
            (_chunk("c3", "p2", "Methods", "three"), [0.8, 0.2]),
        ]
    )

    filtered = await store.search(
        [1.0, 0.0], top_k=5, filters={"paper_id": "p1", "section": "Methods"}
    )
    assert [item.source_id for item in filtered] == ["c1"]
    assert await store.search([1.0, 0.0], top_k=5, filters={"unknown": "x"}) == []

    await store.delete_paper("p1")
    remaining = await store.search([1.0, 0.0], top_k=5)
    assert [item.source_id for item in remaining] == ["c3"]


async def test_vector_store_rejects_invalid_boundaries_atomically() -> None:
    store = InMemoryVectorStore()
    with pytest.raises(KGCRAGError, match="dimension") as batch_error:
        await store.upsert(
            [
                (_chunk("c1", "p1", "Methods", "one"), [1.0, 0.0]),
                (_chunk("c2", "p2", "Methods", "two"), [1.0]),
            ]
        )
    assert batch_error.value.detail.retryable is False
    assert store.record_count == 0

    await store.upsert([(_chunk("c1", "p1", "Methods", "one"), [1.0, 0.0])])
    with pytest.raises(KGCRAGError, match="dimension"):
        await store.search([1.0], top_k=1)
    with pytest.raises(KGCRAGError, match="top_k"):
        await store.search([1.0, 0.0], top_k=0)
