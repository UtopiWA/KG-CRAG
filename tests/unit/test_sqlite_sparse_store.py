"""SQLite FTS5 Sparse Store 的身份、事务和检索契约测试。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from kg_crag.errors import KGCRAGError
from kg_crag.models import Chunk, SparseIndexIdentity
from kg_crag.sparse_store import SQLiteSparseStore, build_sqlite_index_identity

HASH = "a" * 64


def _identity(*, snapshot: str = HASH) -> SparseIndexIdentity:
    return build_sqlite_index_identity(
        schema_version="v1",
        tokenizer="scientific-unicode-v1",
        bm25_version="sqlite-fts5-bm25-v1",
        corpus_snapshot_hash=snapshot,
    )


def _chunk(chunk_id: str, paper_id: str, text: str, *, version: str = "v1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        section="Methods",
        page_start=1,
        page_end=2,
        text=text,
        token_count=max(1, len(text.split())),
        content_hash=f"content-{chunk_id}-{version}",
        processing_version=version,
    )


async def test_sqlite_index_create_reuse_reject_and_explicit_rebuild(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite3"
    identity = _identity()
    store = SQLiteSparseStore(path, identity)
    await store.ensure_index(identity)

    assert path.is_file()
    assert store.manifest_path.is_file()
    await SQLiteSparseStore(path, identity).ensure_index(identity)

    incompatible = _identity(snapshot="b" * 64)
    with pytest.raises(KGCRAGError, match="incompatible"):
        await SQLiteSparseStore(path, incompatible).ensure_index(incompatible)

    store.manifest_path.write_text("not-json", encoding="utf-8")
    with pytest.raises(KGCRAGError, match="manifest is invalid"):
        await store.ensure_index(identity)
    await store.ensure_index(identity, rebuild=True)
    assert await store.record_state("missing") == {}


async def test_sqlite_index_rejects_missing_or_damaged_identity(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite3"
    identity = _identity()
    store = SQLiteSparseStore(path, identity)
    await store.ensure_index(identity)

    store.manifest_path.unlink()
    with pytest.raises(KGCRAGError, match="missing"):
        await store.ensure_index(identity)
    await store.ensure_index(identity, rebuild=True)

    connection = sqlite3.connect(path)
    with connection:
        connection.execute("DELETE FROM metadata WHERE key = 'identity'")
    connection.close()
    with pytest.raises(KGCRAGError, match="identity is missing"):
        await store.ensure_index(identity)


async def test_sqlite_sync_search_filter_and_pagination(tmp_path: Path) -> None:
    identity = _identity()
    store = SQLiteSparseStore(tmp_path / "index.sqlite3", identity, max_candidates=10)
    await store.ensure_index(identity)
    first = _chunk("c-b", "p1", "GPT-4 benchmark")
    stale = _chunk("c-old", "p1", "obsolete term")
    tied = _chunk("c-a", "p2", "GPT-4 benchmark")

    initial = await store.sync_paper("p1", [first, stale])
    await store.sync_paper("p2", [tied])
    repeated = await store.sync_paper("p1", [first, stale])
    found = await store.search("GPT-4", top_k=2)

    assert initial.added == 2
    assert repeated.skipped == 2
    assert [item.source_id for item in found] == ["c-a", "c-b"]
    assert found[0].ranks.sparse == 1
    assert found[0].metadata["content_hash"] == tied.content_hash
    assert found[0].external is False
    page = await store.search("GPT-4", top_k=1, offset=1)
    assert [item.source_id for item in page] == ["c-b"]
    assert page[0].ranks.sparse == 2

    replacement = _chunk("c-b", "p1", "LLaMA-2 updated", version="v2")
    changed = await store.sync_paper("p1", [replacement])
    assert (changed.updated, changed.deleted) == (1, 1)
    assert await store.search("obsolete", top_k=5) == []
    assert await store.search("GPT-4", top_k=5, filters={"paper_id": "p1"}) == []
    filtered = await store.search(
        "LLaMA-2", top_k=5, filters={"paper_id": "p1", "processing_version": "v2"}
    )
    assert [item.source_id for item in filtered] == ["c-b"]


async def test_sqlite_paper_context_reads_frozen_chunks_abstract_first(tmp_path: Path) -> None:
    identity = _identity()
    store = SQLiteSparseStore(tmp_path / "index.sqlite3", identity)
    await store.ensure_index(identity)
    methods = _chunk("methods", "p1", "Method details.").model_copy(update={"ordinal": 1})
    abstract = _chunk("abstract", "p1", "Abstract Direct answer.").model_copy(
        update={"section": "Abstract", "ordinal": 2}
    )
    await store.sync_paper("p1", [methods, abstract])

    context = await store.paper_context("p1", limit=1)

    assert [item.source_id for item in context] == ["abstract"]
    assert context[0].metadata["paper_context_expansion"] is True


async def test_sqlite_sync_rolls_back_whole_paper_on_write_failure(tmp_path: Path) -> None:
    identity = _identity()

    class FailingStore(SQLiteSparseStore):
        def _write_chunk(self, connection: sqlite3.Connection, chunk: Chunk) -> None:
            if chunk.chunk_id == "c2":
                raise sqlite3.IntegrityError("forced test failure")
            super()._write_chunk(connection, chunk)

    path = tmp_path / "index.sqlite3"
    stable_store = SQLiteSparseStore(path, identity)
    await stable_store.ensure_index(identity)
    original = _chunk("c1", "p1", "stable old content")
    await stable_store.sync_paper("p1", [original])

    failing_store = FailingStore(path, identity)
    with pytest.raises(KGCRAGError, match="transaction failed"):
        await failing_store.sync_paper(
            "p1",
            [_chunk("c1", "p1", "changed content", version="v2"), _chunk("c2", "p1", "new")],
        )

    found = await stable_store.search("stable", top_k=5)
    assert [item.source_id for item in found] == ["c1"]
    state = await stable_store.record_state("p1")
    assert state["c1"].processing_version == "v1"
    assert "c2" not in state


async def test_sqlite_search_rejects_invalid_window_and_filter_before_open(tmp_path: Path) -> None:
    identity = _identity()
    store = SQLiteSparseStore(tmp_path / "missing.sqlite3", identity, max_candidates=2)
    with pytest.raises(KGCRAGError, match="candidate limit"):
        await store.search("query", top_k=2, offset=1)
    with pytest.raises(KGCRAGError, match="unknown sparse filter"):
        await store.search("query", top_k=1, filters={"unknown": "x"})
    assert not store.db_path.exists()
