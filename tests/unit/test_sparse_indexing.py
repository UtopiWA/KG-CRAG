"""Sparse 索引规划、断点和 CLI 边界测试。"""

from pathlib import Path

import pytest

from kg_crag.indexing import sparse_cli
from kg_crag.indexing.processed import ProcessedPaper
from kg_crag.indexing.sparse import SparseIndexPipeline
from kg_crag.indexing.sparse_cli import _run, parse_args, validate_cli_bounds
from kg_crag.models import Chunk, IndexItemStatus
from kg_crag.retrieval import load_hybrid_retrieval_config
from kg_crag.sparse_store import (
    InMemorySparseStore,
    SparseSyncResult,
    SQLiteSparseStore,
    build_sqlite_index_identity,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HASH = "a" * 64


def _chunk(chunk_id: str, paper_id: str, text: str = "sparse text") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        text=text,
        token_count=2,
        content_hash=f"content-{chunk_id}",
    )


def _paper(paper_id: str, *chunks: Chunk, directory: Path) -> ProcessedPaper:
    return ProcessedPaper(paper_id, "v1", chunks, directory)


def _pipeline(tmp_path: Path, store: InMemorySparseStore) -> SparseIndexPipeline:
    config = load_hybrid_retrieval_config(PROJECT_ROOT / "configs/retrieval.yaml")
    return SparseIndexPipeline(config, store, workspace_root=tmp_path)


def _store() -> InMemorySparseStore:
    identity = build_sqlite_index_identity(
        schema_version="v1",
        tokenizer="scientific-unicode-v1",
        bm25_version="sqlite-fts5-bm25-v1",
        corpus_snapshot_hash=HASH,
    )
    return InMemorySparseStore(identity=identity)


async def test_sparse_index_dry_run_writes_nothing_and_actual_run_is_idempotent(
    tmp_path: Path,
) -> None:
    store = _store()
    pipeline = _pipeline(tmp_path, store)
    papers = [_paper("p1", _chunk("c1", "p1"), directory=tmp_path)]

    planned = await pipeline.run(papers, dry_run=True)
    assert planned.dry_run is True
    assert planned.items[0].status is IndexItemStatus.PLANNED
    assert list(tmp_path.rglob("*.json")) == []

    first = await pipeline.run(papers, dry_run=False)
    second = await pipeline.run(papers, dry_run=False)
    assert first.items[0].status is IndexItemStatus.SUCCEEDED
    assert second.items[0].status is IndexItemStatus.SKIPPED
    assert len(list(tmp_path.glob("data/processed/sparse-index-runs/*/manifest.json"))) == 2
    checkpoints = list(tmp_path.glob("data/processed/sparse-index-runs/checkpoints/*.json"))
    assert len(checkpoints) == 1


async def test_sparse_index_continues_after_one_paper_failure(tmp_path: Path) -> None:
    class FailingStore(InMemorySparseStore):
        async def sync_paper(self, paper_id: str, chunks: list[Chunk]) -> SparseSyncResult:
            if paper_id == "p-bad":
                raise RuntimeError("fixture failure")
            return await super().sync_paper(paper_id, chunks)

    base = _store()
    store = FailingStore(identity=base.identity)
    pipeline = _pipeline(tmp_path, store)
    papers = [
        _paper("p-bad", _chunk("c-bad", "p-bad"), directory=tmp_path),
        _paper("p-good", _chunk("c-good", "p-good"), directory=tmp_path),
    ]

    manifest = await pipeline.run(papers, dry_run=False)
    assert [item.status for item in manifest.items] == [
        IndexItemStatus.FAILED,
        IndexItemStatus.SUCCEEDED,
    ]
    assert set(await store.record_state("p-good")) == {"c-good"}


def test_sparse_cli_rejects_conflicting_or_unbounded_selection() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--pilot", "--all"])
    with pytest.raises(ValueError, match="explicit limit"):
        validate_cli_bounds(parse_args(["--all"]))
    with pytest.raises(ValueError, match="unique"):
        validate_cli_bounds(parse_args(["--paper-id", "p1", "--paper-id", "p1"]))


async def test_sparse_cli_dry_run_success_rebuild_confirmation_and_partial_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "retrieval.yaml"
    config_path.write_text(
        (PROJECT_ROOT / "configs/retrieval.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    papers = [
        _paper("p-bad", _chunk("c-bad", "p-bad"), directory=tmp_path),
        _paper("p-good", _chunk("c-good", "p-good"), directory=tmp_path),
    ]
    monkeypatch.setattr(sparse_cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sparse_cli, "discover_processed", lambda _root: papers)

    base = [
        "--paper-id",
        "p-good",
        "--config",
        str(config_path),
        "--processed-root",
        str(tmp_path),
    ]
    assert await _run([*base, "--dry-run"]) == 0
    assert not (tmp_path / "data").exists()
    with pytest.raises(ValueError, match="exact planned target"):
        await _run([*base, "--rebuild-index-version", "0" * 64])
    assert await _run(base) == 0
    assert await _run(base) == 0

    class FailingSQLiteStore(SQLiteSparseStore):
        async def sync_paper(self, paper_id: str, chunks: list[Chunk]) -> SparseSyncResult:
            if paper_id == "p-bad":
                raise RuntimeError("fixture failure")
            return await super().sync_paper(paper_id, chunks)

    monkeypatch.setattr(sparse_cli, "SQLiteSparseStore", FailingSQLiteStore)
    partial = [
        "--paper-id",
        "p-bad",
        "--paper-id",
        "p-good",
        "--config",
        str(config_path),
        "--processed-root",
        str(tmp_path),
    ]
    assert await _run(partial) == 2
