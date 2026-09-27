"""内存 Graph Store 的确定性行为测试。"""

import pytest

from kg_crag.errors import KGCRAGError
from kg_crag.graph_store import InMemoryGraphStore
from kg_crag.models import Chunk, Paper


def _paper(paper_id: str, title: str) -> Paper:
    return Paper(paper_id=paper_id, title=title)


def _chunk(chunk_id: str, paper_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        section="Methods",
        page_start=1,
        text=text,
        token_count=len(text.split()),
        content_hash=f"hash---{chunk_id}",
    )


async def test_graph_store_upserts_idempotently_and_matches_all_entities() -> None:
    store = InMemoryGraphStore()
    paper_b = _paper("p-b", "Corrective RAG")
    paper_a = _paper("p-a", "Graph Agent")
    await store.upsert_paper(paper_b, [_chunk("c-b", "p-b", "Uses retrieval feedback")])
    await store.upsert_paper(paper_a, [_chunk("c-a", "p-a", "RAG retrieval loop")])
    await store.upsert_paper(paper_a, [_chunk("c-a", "p-a", "RAG retrieval revised")])

    results = await store.retrieve(["RAG", "retrieval"], max_hops=2, top_k=10)

    assert store.paper_count == 2
    assert [item.paper_id for item in results] == ["p-a", "p-b"]
    assert results[0].content == "RAG retrieval revised"
    assert results[0].metadata["max_hops"] == 2
    assert store.calls[-1].entity_names == ("RAG", "retrieval")  # type: ignore[union-attr]


async def test_graph_store_deletes_only_target_paper() -> None:
    store = InMemoryGraphStore()
    await store.upsert_paper(_paper("p1", "RAG"), [_chunk("c1", "p1", "agent")])
    await store.upsert_paper(_paper("p2", "RAG"), [_chunk("c2", "p2", "agent")])

    await store.delete_paper("p1")
    results = await store.retrieve(["agent"], max_hops=0, top_k=5)

    assert [item.paper_id for item in results] == ["p2"]


@pytest.mark.parametrize(
    ("top_k", "max_hops", "message"),
    [(0, 1, "top_k"), (1, -1, "max_hops")],
)
async def test_graph_store_rejects_invalid_boundaries(
    top_k: int, max_hops: int, message: str
) -> None:
    store = InMemoryGraphStore()
    with pytest.raises(KGCRAGError, match=message) as caught:
        await store.retrieve(["entity"], max_hops=max_hops, top_k=top_k)
    assert caught.value.detail.retryable is False


async def test_graph_store_rejects_chunk_from_another_paper() -> None:
    store = InMemoryGraphStore()
    with pytest.raises(KGCRAGError, match="paper_id"):
        await store.upsert_paper(_paper("p1", "RAG"), [_chunk("c2", "p2", "agent")])
    assert store.paper_count == 0
