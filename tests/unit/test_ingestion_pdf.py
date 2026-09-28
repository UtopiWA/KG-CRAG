"""Parser Protocol 与 PyMuPDF 适配器的离线测试。"""

import hashlib
from pathlib import Path

import pymupdf
import pytest

from kg_crag.errors import KGCRAGError
from kg_crag.ingestion.base import DocumentParser, MockDocumentParser
from kg_crag.ingestion.pdf import PyMuPDFParser, _block_kind, _overlap_exists, _RawBlock
from kg_crag.models import BlockKind, ParseRequest


def _write_pdf(
    path: Path, *, pages: int = 1, empty: bool = False, encrypted: bool = False
) -> bytes:
    document = pymupdf.open()
    for page_index in range(pages):
        page = document.new_page(width=600, height=800)
        if not empty:
            page.insert_text((50, 50), "1 Introduction", fontsize=18)
            page.insert_textbox(
                pymupdf.Rect(50, 100, 280, 700),
                f"Left column page {page_index + 1}. " * 20,
                fontsize=10,
            )
            page.insert_textbox(
                pymupdf.Rect(320, 100, 550, 700),
                "Right column includes x = y + z and evaluation details. " * 15,
                fontsize=10,
            )
    options = {}
    if encrypted:
        options = {
            "encryption": pymupdf.PDF_ENCRYPT_AES_256,
            "owner_pw": "owner",
            "user_pw": "user",
        }
    document.save(path, **options)
    document.close()
    return path.read_bytes()


def _request(data: bytes, **overrides: object) -> ParseRequest:
    payload: dict[str, object] = {
        "paper_id": "arxiv:2401.00001",
        "input_sha256": hashlib.sha256(data).hexdigest(),
        "max_pdf_bytes": 1_000_000,
        "max_pages": 10,
        "min_document_chars": 50,
        "full_width_ratio": 0.72,
    }
    payload.update(overrides)
    return ParseRequest.model_validate(payload)


def test_parser_protocol_and_stable_two_column_output(tmp_path: Path) -> None:
    path = tmp_path / "valid.pdf"
    data = _write_pdf(path, pages=2)
    parser = PyMuPDFParser()
    assert isinstance(parser, DocumentParser)
    first = parser.parse(path, _request(data))
    second = parser.parse(path, _request(data))
    assert first == second
    assert first.page_count == 2
    assert all(
        block.page_number == page.page_number for page in first.pages for block in page.blocks
    )
    assert any(warning.code.value == "multi_column" for warning in first.warnings)
    texts = [block.text for block in first.pages[0].blocks]
    assert texts[0] == "1 Introduction"
    assert next(index for index, text in enumerate(texts) if text.startswith("Left")) < next(
        index for index, text in enumerate(texts) if text.startswith("Right")
    )
    mock = MockDocumentParser(first)
    assert isinstance(mock, DocumentParser)
    assert mock.parse(path, _request(data)) == first


def test_checked_in_pdf_fixture_is_parseable_offline() -> None:
    path = Path("tests/fixtures/ingestion/sample.pdf")
    data = path.read_bytes()
    parsed = PyMuPDFParser().parse(path, _request(data))
    assert parsed.page_count == 1
    assert "Offline parser fixture" in parsed.pages[0].blocks[-1].text


def test_space_only_spans_are_preserved(tmp_path: Path) -> None:
    path = tmp_path / "spaces.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_htmlbox(
        pymupdf.Rect(50, 50, 500, 300),
        "<p><b>General</b> <i>AI</i> Assistants preserve spaces in spans.</p>" * 8,
    )
    pdf.save(path)
    pdf.close()
    data = path.read_bytes()
    parsed = PyMuPDFParser().parse(path, _request(data))
    text = " ".join(block.text for block in parsed.pages[0].blocks)
    assert "General AI Assistants" in text


@pytest.mark.parametrize("case", ["corrupt", "empty", "pages", "bytes", "encrypted"])
def test_parser_returns_structured_errors_for_invalid_documents(tmp_path: Path, case: str) -> None:
    path = tmp_path / f"{case}.pdf"
    if case == "corrupt":
        data = b"%PDF-corrupt"
        path.write_bytes(data)
        request = _request(data)
    else:
        data = _write_pdf(
            path,
            pages=2 if case == "pages" else 1,
            empty=case == "empty",
            encrypted=case == "encrypted",
        )
        overrides: dict[str, object] = {}
        if case == "pages":
            overrides["max_pages"] = 1
        if case == "bytes":
            overrides["max_pdf_bytes"] = 10
        request = _request(data, **overrides)
    with pytest.raises(KGCRAGError) as error:
        PyMuPDFParser().parse(path, request)
    assert error.value.detail.context == {"paper_id": "arxiv:2401.00001"}


def test_coordinates_are_rounded_and_sections_are_stable(tmp_path: Path) -> None:
    path = tmp_path / "sections.pdf"
    data = _write_pdf(path)
    document = PyMuPDFParser().parse(path, _request(data))
    assert document.sections
    assert all(
        value == round(value, 3)
        for page in document.pages
        for block in page.blocks
        for value in (block.bbox.x0, block.bbox.y0, block.bbox.x1, block.bbox.y1)
    )


def test_complex_layout_emits_controlled_warnings(tmp_path: Path) -> None:
    path = tmp_path / "complex.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page(width=600, height=800)
    page.insert_text((40, 40), "2 Method", fontsize=18)
    page.insert_text((40, 100), "x = y + z = 1", fontsize=10)
    page.insert_textbox(pymupdf.Rect(40, 130, 300, 180), "overlapping text", fontsize=10)
    page.insert_textbox(pymupdf.Rect(200, 140, 500, 190), "second overlapping text", fontsize=10)
    page.insert_textbox(pymupdf.Rect(40, 200, 270, 700), "Left column. " * 60, fontsize=9)
    page.insert_textbox(pymupdf.Rect(330, 200, 560, 700), "Right column. " * 60, fontsize=9)
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 10, 10), False)
    pixmap.clear_with(240)
    page.insert_image(pymupdf.Rect(280, 600, 320, 640), stream=pixmap.tobytes("png"))
    pdf.save(path)
    pdf.close()
    data = path.read_bytes()
    parsed = PyMuPDFParser().parse(path, _request(data))
    codes = {warning.code.value for warning in parsed.warnings}
    assert {"multi_column", "formula_detected", "image_skipped"} <= codes
    first = _RawBlock(0, (0.0, 0.0, 10.0, 10.0), "one", 10, False, BlockKind.TEXT)
    second = _RawBlock(1, (5.0, 5.0, 15.0, 15.0), "two", 10, False, BlockKind.TEXT)
    assert _overlap_exists([first, second])
    assert _block_kind("a b c\nd e f\ng h i", line_count=3, span_count=9) is BlockKind.TABLE


def test_heading_fallback_keeps_all_blocks(tmp_path: Path) -> None:
    path = tmp_path / "fallback.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page(width=600, height=800)
    page.insert_textbox(
        pymupdf.Rect(50, 100, 550, 700),
        "lowercase prose without a heading. " * 40,
        fontsize=10,
    )
    pdf.save(path)
    pdf.close()
    data = path.read_bytes()
    first = PyMuPDFParser().parse(path, _request(data))
    second = PyMuPDFParser().parse(path, _request(data))
    assert [section.title for section in first.sections] == ["__document__"]
    assert first.sections == second.sections
    assert first.sections[0].block_ids == [
        block.block_id for page in first.pages for block in page.blocks
    ]
