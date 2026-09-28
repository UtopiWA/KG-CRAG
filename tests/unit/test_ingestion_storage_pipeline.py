"""产物发布、复用、质量统计和批次隔离测试。"""

import json
from pathlib import Path

import pytest

from kg_crag.ingestion.base import MockDocumentParser
from kg_crag.ingestion.config import IngestionConfig
from kg_crag.ingestion.pipeline import IngestionPipeline
from kg_crag.ingestion.raw import RawPaperInput
from kg_crag.ingestion.storage import ArtifactStore, paper_key
from kg_crag.models import (
    BoundingBox,
    DocumentBlock,
    DocumentPage,
    DocumentSection,
    IngestionStatus,
    ParsedDocument,
)

SHA = "a" * 64


def _parsed(paper_id: str = "arxiv:2401.00001") -> ParsedDocument:
    text = "Deterministic scientific evidence for an offline ingestion test. " * 30
    block = DocumentBlock(
        block_id="block-1",
        page_number=1,
        order=0,
        bbox=BoundingBox(x0=10, y0=100, x1=500, y1=700),
        text=text,
    )
    return ParsedDocument(
        paper_id=paper_id,
        input_sha256=SHA,
        parser_name="mock",
        parser_version="1",
        page_count=1,
        pages=[DocumentPage(page_number=1, width=600, height=800, blocks=[block])],
        sections=[
            DocumentSection(
                section_id="section-1",
                title="Method",
                page_start=1,
                page_end=1,
                block_ids=[block.block_id],
            )
        ],
    )


def _raw(tmp_path: Path, paper_id: str = "arxiv:2401.00001") -> RawPaperInput:
    pdf_path = tmp_path / f"{paper_id.replace(':', '-')}.pdf"
    pdf_path.write_bytes(b"%PDF-raw-immutable")
    metadata_path = tmp_path / f"{paper_id.replace(':', '-')}.json"
    payload = {
        "paper_id": paper_id,
        "title": "Pipeline Fixture",
        "authors": [{"name": "Ada Lovelace"}],
        "arxiv_id": paper_id.removeprefix("arxiv:"),
        "source_url": "https://arxiv.org/abs/2401.00001",
        "pdf_path": pdf_path.name,
    }
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    return RawPaperInput(
        paper_id=paper_id,
        arxiv_id=str(payload["arxiv_id"]),
        metadata_path=metadata_path,
        pdf_path=pdf_path,
        metadata=payload,
        input_sha256=SHA,
        size_bytes=pdf_path.stat().st_size,
    )


def _pipeline(tmp_path: Path, *, store: ArtifactStore | None = None) -> IngestionPipeline:
    return IngestionPipeline(
        IngestionConfig(),
        MockDocumentParser(_parsed()),
        store or ArtifactStore(tmp_path / "interim", tmp_path / "processed"),
    )


def test_paper_key_is_windows_safe_and_serialized_output_is_reusable(tmp_path: Path) -> None:
    assert paper_key("doi:10.1/a/b:c") == paper_key("doi:10.1/a/b:c")
    assert all(char.isalnum() or char == "-" for char in paper_key("doi:10.1/a/b:c"))
    raw = _raw(tmp_path)
    before = raw.pdf_path.read_bytes()
    pipeline = _pipeline(tmp_path)
    first = pipeline.process(raw)
    second = pipeline.process(raw)
    forced = pipeline.process(raw, force=True)
    assert first.status is IngestionStatus.SUCCEEDED
    assert second.status is IngestionStatus.SKIPPED
    assert forced.status is IngestionStatus.SUCCEEDED
    assert first.artifacts == second.artifacts == forced.artifacts
    assert raw.pdf_path.read_bytes() == before
    interim, processed = pipeline.store.version_directories(
        raw.paper_id, first.processing_version or ""
    )
    assert {path.name for path in interim.iterdir()} == {
        "parsed_document.json",
        "cleaned_document.json",
    }
    assert {path.name for path in processed.iterdir()} == {
        "paper.json",
        "chunks.jsonl",
        "quality_report.json",
    }


def test_corrupt_or_missing_artifact_is_rebuilt_byte_identically(tmp_path: Path) -> None:
    raw = _raw(tmp_path)
    pipeline = _pipeline(tmp_path)
    first = pipeline.process(raw)
    _, processed = pipeline.store.version_directories(raw.paper_id, first.processing_version or "")
    expected = {path.name: path.read_bytes() for path in processed.iterdir()}
    (processed / "chunks.jsonl").write_bytes(b"corrupt")
    rebuilt = pipeline.process(raw)
    assert rebuilt.status is IngestionStatus.SUCCEEDED
    assert {path.name: path.read_bytes() for path in processed.iterdir()} == expected


def test_publish_failure_leaves_existing_version_untouched(tmp_path: Path) -> None:
    raw = _raw(tmp_path)
    initial = _pipeline(tmp_path)
    first = initial.process(raw)
    directories = initial.store.version_directories(raw.paper_id, first.processing_version or "")
    before = [
        {path.name: path.read_bytes() for path in directory.iterdir()} for directory in directories
    ]

    def fail(_: Path) -> None:
        raise OSError("injected staging failure")

    failing_store = ArtifactStore(tmp_path / "interim", tmp_path / "processed", before_publish=fail)
    with pytest.raises(OSError, match="injected"):
        _pipeline(tmp_path, store=failing_store).process(raw, force=True)
    after = [
        {path.name: path.read_bytes() for path in directory.iterdir()} for directory in directories
    ]
    assert after == before
    assert not list((tmp_path / "interim").rglob("*.tmp-*"))


def test_batch_isolates_failure_and_writes_safe_manifest_last(tmp_path: Path) -> None:
    valid = _raw(tmp_path)
    invalid = _raw(tmp_path, "arxiv:2401.00002")
    # Mock 文档故意绑定第一篇 ID，第二篇会在清洗前产生契约冲突。
    pipeline = _pipeline(tmp_path)
    manifest = pipeline.run_batch([valid, invalid], mode="selected")
    assert [item.status for item in manifest.items] == [
        IngestionStatus.SUCCEEDED,
        IngestionStatus.FAILED,
    ]
    manifest_path = tmp_path / "processed" / "runs" / manifest.run_id / "manifest.json"
    text = manifest_path.read_text(encoding="utf-8")
    assert manifest_path.is_file()
    assert str(tmp_path) not in text
    assert "%PDF" not in text
    assert "token" not in text.casefold()


def test_quality_report_contains_coverage_and_token_metrics(tmp_path: Path) -> None:
    raw = _raw(tmp_path)
    pipeline = _pipeline(tmp_path)
    result = pipeline.process(raw)
    report = pipeline.store.read_quality(raw.paper_id, result.processing_version or "")
    assert report is not None and report.passed
    assert report.page_count == 1
    assert report.block_count == report.section_count == 1
    assert report.chunk_count >= 1
    assert report.character_count > 0
    assert report.min_chunk_tokens <= report.average_chunk_tokens <= report.max_chunk_tokens
    assert report.pages_with_chunks == [1]
    assert report.page_coverage_ratio == 1.0
    assert all(len(value) == 64 for value in report.artifact_hashes.values())
