"""已发布 Chunk 的发现、哈希重验与有界选择。"""

import hashlib
import json
from pathlib import Path

import pytest

from kg_crag.indexing import discover_processed, select_processed
from kg_crag.models import Chunk, QualityReport


def _publish(root: Path, paper_id: str = "paper-1") -> Path:
    directory = root / "paper-abc" / "pv1"
    directory.mkdir(parents=True)
    chunk = Chunk(
        chunk_id="chunk-1",
        paper_id=paper_id,
        section="Abstract",
        page_start=1,
        page_end=1,
        text="Published evidence.",
        token_count=2,
        content_hash="content-hash",
        processing_version="pv1",
    )
    chunks = (json.dumps(chunk.model_dump(mode="json"), sort_keys=True) + "\n").encode()
    (directory / "chunks.jsonl").write_bytes(chunks)
    report = QualityReport(
        paper_id=paper_id,
        processing_version="pv1",
        page_count=1,
        block_count=1,
        section_count=1,
        chunk_count=1,
        character_count=len(chunk.text),
        min_chunk_tokens=2,
        max_chunk_tokens=2,
        average_chunk_tokens=2,
        pages_with_chunks=[1],
        page_coverage_ratio=1,
        passed=True,
        artifact_hashes={"chunks.jsonl": hashlib.sha256(chunks).hexdigest()},
    )
    (directory / "quality_report.json").write_text(
        report.model_dump_json(),
        encoding="utf-8",
    )
    return directory


def test_discover_and_select_processed_artifacts(tmp_path: Path) -> None:
    _publish(tmp_path)
    papers = discover_processed(tmp_path)
    assert [paper.paper_id for paper in papers] == ["paper-1"]
    assert (
        select_processed(
            papers,
            paper_ids=["paper-1"],
            max_papers=2,
        )
        == papers
    )
    with pytest.raises(ValueError, match="unavailable"):
        select_processed(papers, paper_ids=["missing"], max_papers=2)
    with pytest.raises(ValueError, match="exactly one"):
        select_processed(papers, paper_ids=["paper-1"], select_all=True, max_papers=2)
    with pytest.raises(ValueError, match="explicit limit"):
        select_processed(papers, select_all=True, max_papers=2)


def test_discovery_rejects_corrupted_chunk_artifact(tmp_path: Path) -> None:
    directory = _publish(tmp_path)
    (directory / "chunks.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        discover_processed(tmp_path)
