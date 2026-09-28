"""从 raw 风格输入到版本化产物的完全离线集成测试。"""

import hashlib
import json
import shutil
from pathlib import Path

import pymupdf

from kg_crag.ingestion.config import IngestionConfig
from kg_crag.ingestion.pdf import PyMuPDFParser
from kg_crag.ingestion.pipeline import IngestionPipeline
from kg_crag.ingestion.raw import discover_raw_inputs
from kg_crag.ingestion.storage import ArtifactStore
from kg_crag.models import IngestionStatus


def _write_raw(root: Path, arxiv_id: str, *, empty: bool = False) -> None:
    metadata_root = root / "metadata"
    papers_root = root / "papers"
    metadata_root.mkdir(parents=True, exist_ok=True)
    papers_root.mkdir(parents=True, exist_ok=True)
    pdf_path = papers_root / f"{arxiv_id.replace('.', '_')}.pdf"
    document = pymupdf.open()
    for page_number in range(2):
        page = document.new_page(width=600, height=800)
        if not empty:
            page.insert_text((50, 45), f"{page_number + 1} Method", fontsize=18)
            page.insert_textbox(
                pymupdf.Rect(50, 100, 280, 700),
                "Left column deterministic evidence. " * 30,
                fontsize=9,
            )
            page.insert_textbox(
                pymupdf.Rect(320, 100, 550, 700),
                "Right column deterministic evidence. " * 30,
                fontsize=9,
            )
    document.save(pdf_path)
    document.close()
    data = pdf_path.read_bytes()
    payload = {
        "paper_id": f"arxiv:{arxiv_id}",
        "title": f"Fixture {arxiv_id}",
        "authors": [{"name": "Integration Author"}],
        "arxiv_id": arxiv_id,
        "source_url": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf_path": f"data/raw/papers/{pdf_path.name}",
        "acquisition": {
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        },
    }
    (metadata_root / f"{arxiv_id.replace('.', '_')}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _pipeline(tmp_path: Path, config: IngestionConfig) -> IngestionPipeline:
    return IngestionPipeline(
        config,
        PyMuPDFParser(config.parsing),
        ArtifactStore(tmp_path / "interim", tmp_path / "processed"),
    )


def test_offline_pipeline_reuse_version_change_failure_isolation_and_rebuild(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    _write_raw(raw_root, "2401.00001")
    _write_raw(raw_root, "2401.00002", empty=True)
    inputs = discover_raw_inputs(raw_root, max_pdf_bytes=10_000_000, max_papers=10)
    valid, invalid = inputs
    first_pipeline = _pipeline(tmp_path, IngestionConfig())
    first = first_pipeline.process(valid)
    skipped = first_pipeline.process(valid)
    assert first.status is IngestionStatus.SUCCEEDED
    assert skipped.status is IngestionStatus.SKIPPED

    old_directories = first_pipeline.store.version_directories(
        valid.paper_id, first.processing_version or ""
    )
    old_bytes = [
        {path.name: path.read_bytes() for path in directory.iterdir()}
        for directory in old_directories
    ]
    changed_config = IngestionConfig.model_validate(
        {
            "chunking": {
                "target_tokens": 450,
                "min_tokens": 300,
                "max_tokens": 700,
                "overlap_tokens": 75,
            }
        }
    )
    changed = _pipeline(tmp_path, changed_config).process(valid)
    assert changed.processing_version != first.processing_version

    manifest = first_pipeline.run_batch([invalid, valid], mode="selected")
    assert [item.status for item in manifest.items] == [
        IngestionStatus.FAILED,
        IngestionStatus.SKIPPED,
    ]

    for directory in old_directories:
        shutil.rmtree(directory)
    rebuilt = first_pipeline.process(valid)
    assert rebuilt.status is IngestionStatus.SUCCEEDED
    rebuilt_bytes = [
        {path.name: path.read_bytes() for path in directory.iterdir()}
        for directory in old_directories
    ]
    assert rebuilt_bytes == old_bytes
