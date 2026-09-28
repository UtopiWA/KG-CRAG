"""摄取 CLI 的选择、安全边界和退出码测试。"""

import hashlib
import json
from pathlib import Path

import pymupdf
import pytest

from kg_crag.ingestion.cli import main, parse_args


def _raw_fixture(root: Path, *, empty: bool = False) -> str:
    metadata = root / "metadata"
    papers = root / "papers"
    metadata.mkdir(parents=True)
    papers.mkdir()
    paper_id = "arxiv:2401.00001"
    pdf_path = papers / "fixture.pdf"
    document = pymupdf.open()
    page = document.new_page(width=600, height=800)
    if not empty:
        page.insert_textbox(
            pymupdf.Rect(50, 50, 550, 750),
            "1 Introduction\n" + "Offline deterministic evidence. " * 80,
            fontsize=10,
        )
    document.save(pdf_path)
    document.close()
    data = pdf_path.read_bytes()
    payload = {
        "paper_id": paper_id,
        "title": "CLI Fixture",
        "authors": [{"name": "Fixture Author"}],
        "arxiv_id": "2401.00001",
        "source_url": "https://arxiv.org/abs/2401.00001",
        "pdf_path": "data/raw/papers/fixture.pdf",
        "acquisition": {
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        },
    }
    (metadata / "fixture.json").write_text(json.dumps(payload), encoding="utf-8")
    return paper_id


def _arguments(raw: Path, output: Path, paper_id: str) -> list[str]:
    return [
        "--paper-id",
        paper_id,
        "--raw-root",
        str(raw),
        "--interim-root",
        str(output / "interim"),
        "--processed-root",
        str(output / "processed"),
    ]


def test_selectors_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--pilot", "--all"])


def test_dry_run_creates_no_output_and_limit_boundaries(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    paper_id = _raw_fixture(raw)
    arguments = _arguments(raw, output, paper_id)
    assert main([*arguments, "--limit", "1", "--dry-run"]) == 0
    assert not output.exists()
    assert main([*arguments, "--limit", "0", "--dry-run"]) == 1
    assert main([*arguments, "--limit", "201", "--dry-run"]) == 1


def test_cli_returns_zero_for_success_and_verified_skip(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    paper_id = _raw_fixture(raw)
    arguments = _arguments(raw, output, paper_id)
    assert main(arguments) == 0
    assert main(arguments) == 0
    manifests = list((output / "processed" / "runs").glob("*/manifest.json"))
    assert len(manifests) == 2


def test_cli_returns_two_when_batch_item_fails(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    paper_id = _raw_fixture(raw, empty=True)
    assert main(_arguments(raw, output, paper_id)) == 2
