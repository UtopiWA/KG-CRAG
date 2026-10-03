"""Sparse 分词器和内存 Store 的共享契约测试。"""

import pytest

from kg_crag.errors import KGCRAGError
from kg_crag.models import Chunk
from kg_crag.sparse_store import (
    InMemorySparseStore,
    SparseIndexIdentity,
    prepare_sparse_query,
    tokenize_scientific_text,
)


def _chunk(chunk_id: str, paper_id: str, text: str, *, version: str = "v1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        section="Methods",
        page_start=1,
        page_end=1,
        text=text,
        token_count=max(1, len(text.split())),
        content_hash=f"content-{chunk_id}-{version}",
        processing_version=version,
    )


def test_scientific_tokenizer_preserves_identifiers_and_blocks_fts_syntax() -> None:
    assert tokenize_scientific_text("GPT-4、LLaMA-2 与 DATA_SET Café") == (
        "gpt-4",
        "llama-2",
        "与",
        "data_set",
        "café",
    )
    prepared = prepare_sparse_query('GPT-4" OR secret* NOT (x)', max_chars=100, max_tokens=10)
    assert prepared.tokens == ("gpt-4", "or", "secret", "not", "x")
    assert prepared.fts_query == '"gpt-4" OR "or" OR "secret" OR "not" OR "x"'

    with pytest.raises(KGCRAGError, match="no searchable tokens"):
        prepare_sparse_query("   !!! ", max_chars=100, max_tokens=10)
    with pytest.raises(KGCRAGError, match="character limit"):
        prepare_sparse_query("abcd", max_chars=3, max_tokens=10)
    with pytest.raises(KGCRAGError, match="token limit"):
        prepare_sparse_query("a b c", max_chars=100, max_tokens=2)


async def test_memory_sparse_store_syncs_per_paper_and_ranks_stably() -> None:
    store = InMemorySparseStore()
    first = _chunk("c-b", "p1", "GPT-4 benchmark")
    tied = _chunk("c-a", "p2", "GPT-4 benchmark")

    result = await store.sync_paper("p1", [first])
    await store.sync_paper("p2", [tied])
    repeated = await store.sync_paper("p1", [first])
    found = await store.search("GPT-4", top_k=5)

    assert (result.added, result.updated, result.deleted) == (1, 0, 0)
    assert repeated.skipped == 1
    assert [item.source_id for item in found] == ["c-a", "c-b"]
    assert all(item.scores.sparse is not None for item in found)
    assert all(item.external is False for item in found)


async def test_memory_sparse_store_replaces_stale_records_and_filters() -> None:
    store = InMemorySparseStore()
    await store.sync_paper(
        "p1",
        [_chunk("c1", "p1", "old term"), _chunk("c2", "p1", "keep term")],
    )
    replacement = _chunk("c2", "p1", "updated term", version="v2")
    result = await store.sync_paper("p1", [replacement])

    assert (result.updated, result.deleted) == (1, 1)
    assert await store.search("old", top_k=5) == []
    filtered = await store.search("updated", top_k=5, filters={"processing_version": "v2"})
    assert [item.source_id for item in filtered] == ["c2"]
    assert (await store.record_state("p1"))["c2"].processing_version == "v2"


async def test_memory_sparse_store_rejects_invalid_operations_atomically() -> None:
    store = InMemorySparseStore(max_top_k=2)
    original = _chunk("c1", "p1", "stable content")
    await store.sync_paper("p1", [original])

    with pytest.raises(KGCRAGError, match="paper_id"):
        await store.sync_paper("p1", [_chunk("bad", "p2", "wrong paper")])
    assert store.record_count == 1
    with pytest.raises(KGCRAGError, match="top_k"):
        await store.search("stable", top_k=3)
    with pytest.raises(KGCRAGError, match="unknown sparse filter"):
        await store.search("stable", top_k=1, filters={"unknown": "x"})
    assert len(store.calls) == 0

    incompatible = SparseIndexIdentity(
        schema_version="v2",
        tokenizer="scientific-unicode-v1",
        bm25_version="bm25",
        python_version="3.11",
        sqlite_version="3",
        fts5_enabled=True,
        fts5_version="sqlite-builtin-3",
        corpus_snapshot_hash="1" * 64,
        index_version="2" * 64,
    )
    with pytest.raises(KGCRAGError, match="identity"):
        await store.ensure_index(incompatible)
