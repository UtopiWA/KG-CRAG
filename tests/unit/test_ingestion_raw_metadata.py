"""原始输入边界与元数据规范化测试。"""

import hashlib
import json
from pathlib import Path

import pytest

from kg_crag.errors import KGCRAGError
from kg_crag.ingestion.metadata import canonical_paper_id, normalize_doi, normalize_paper
from kg_crag.ingestion.raw import discover_raw_inputs, validate_raw_record


def _raw_pair(root: Path, *, pdf: bytes = b"%PDF-1.4\nfixture") -> Path:
    metadata_root = root / "metadata"
    papers_root = root / "papers"
    metadata_root.mkdir(parents=True)
    papers_root.mkdir()
    pdf_path = papers_root / "paper.pdf"
    pdf_path.write_bytes(pdf)
    payload = {
        "paper_id": "arxiv:2401.00001",
        "title": "  A   title  ",
        "authors": [{"name": " Ada   Lovelace "}],
        "arxiv_id": "2401.00001v2",
        "doi": "https://doi.org/10.1000/ABC.",
        "source_url": "https://arxiv.org/abs/2401.00001",
        "pdf_path": "data/raw/papers/paper.pdf",
        "acquisition": {
            "sha256": hashlib.sha256(pdf).hexdigest(),
            "size_bytes": len(pdf),
        },
    }
    metadata_path = metadata_root / "paper.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    return metadata_path


def test_valid_record_and_discovery_are_stable(tmp_path: Path) -> None:
    metadata = _raw_pair(tmp_path)
    record = validate_raw_record(metadata, tmp_path / "papers", max_pdf_bytes=1024)
    assert record.input_sha256 == hashlib.sha256(record.pdf_path.read_bytes()).hexdigest()
    assert discover_raw_inputs(tmp_path, max_pdf_bytes=1024, max_papers=2) == [record]


@pytest.mark.parametrize("mutation", ["signature", "hash", "missing"])
def test_raw_validation_rejects_invalid_pairs(tmp_path: Path, mutation: str) -> None:
    metadata = _raw_pair(tmp_path)
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    if mutation == "signature":
        (tmp_path / "papers" / "paper.pdf").write_bytes(b"not-pdf")
        payload["acquisition"]["size_bytes"] = 7
        payload["acquisition"]["sha256"] = hashlib.sha256(b"not-pdf").hexdigest()
    elif mutation == "hash":
        payload["acquisition"]["sha256"] = "0" * 64
    else:
        (tmp_path / "papers" / "paper.pdf").unlink()
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(KGCRAGError):
        validate_raw_record(metadata, tmp_path / "papers", max_pdf_bytes=1024)


def test_raw_validation_rejects_symlink_when_supported(tmp_path: Path) -> None:
    metadata = _raw_pair(tmp_path)
    target = tmp_path / "papers" / "paper.pdf"
    link = tmp_path / "papers" / "link.pdf"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("当前平台不允许创建测试符号链接")
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["pdf_path"] = "link.pdf"
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(KGCRAGError, match="regular"):
        validate_raw_record(metadata, tmp_path / "papers", max_pdf_bytes=1024)


@pytest.mark.parametrize("pdf_value", ["../../outside.pdf", "paper.pdf/subfile"])
def test_raw_validation_rejects_path_traversal(tmp_path: Path, pdf_value: str) -> None:
    metadata = _raw_pair(tmp_path)
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["pdf_path"] = pdf_value
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(KGCRAGError):
        validate_raw_record(metadata, tmp_path / "papers", max_pdf_bytes=1024)


def test_discovery_rejects_duplicate_pdf_hashes(tmp_path: Path) -> None:
    first = _raw_pair(tmp_path)
    payload = json.loads(first.read_text(encoding="utf-8"))
    payload["paper_id"] = "arxiv:2401.00002"
    payload["arxiv_id"] = "2401.00002"
    payload["pdf_path"] = "copy.pdf"
    (tmp_path / "papers" / "copy.pdf").write_bytes((tmp_path / "papers" / "paper.pdf").read_bytes())
    (tmp_path / "metadata" / "copy.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(KGCRAGError, match="duplicate PDF"):
        discover_raw_inputs(tmp_path, max_pdf_bytes=1024, max_papers=3)


def test_metadata_normalization_preserves_legacy_id_and_unicode() -> None:
    payload = {
        "paper_id": "arxiv:2401.00001",
        "title": " Cafe\u0301   Agent ",
        "authors": [" Ada  Lovelace "],
        "arxiv_id": "arXiv:2401.00001v3",
        "doi": "https://doi.org/10.1000/ABC.",
        "source_url": "https://arxiv.org/abs/2401.00001",
    }
    paper = normalize_paper(payload, processing_version="version")
    assert paper.paper_id == "arxiv:2401.00001"
    assert paper.title == "Café Agent"
    assert paper.arxiv_id == "2401.00001"
    assert paper.doi == "10.1000/abc"


def test_new_identifier_priority_is_doi_then_arxiv_then_title() -> None:
    assert normalize_doi("DOI: 10.1/ABC") == "10.1/abc"
    assert canonical_paper_id({"doi": "10.1/ABC", "arxiv_id": "2401.00001"}) == ("doi:10.1/abc")
    assert canonical_paper_id({"title": " Same   Title "}) == canonical_paper_id(
        {"title": "same title"}
    )
    with pytest.raises(ValueError, match="conflicts"):
        canonical_paper_id({"paper_id": "wrong:id", "arxiv_id": "2401.00001"})
