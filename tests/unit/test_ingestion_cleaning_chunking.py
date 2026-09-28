"""正文清洗、溯源和章节切分测试。"""

import hashlib
from copy import deepcopy

from kg_crag.ingestion.chunking import chunk_document, token_count
from kg_crag.ingestion.cleaning import clean_document
from kg_crag.ingestion.config import ChunkingConfig, CleaningConfig
from kg_crag.models import (
    BoundingBox,
    DocumentBlock,
    DocumentPage,
    DocumentSection,
    ParsedDocument,
)

SHA = "a" * 64


def _document(page_count: int = 4) -> ParsedDocument:
    pages = []
    all_ids = []
    for page_number in range(1, page_count + 1):
        blocks = [
            DocumentBlock(
                block_id=f"header-{page_number}",
                page_number=page_number,
                order=0,
                bbox=BoundingBox(x0=10, y0=5, x1=300, y1=20),
                text=f"Conference 2026 - {page_number}",
            ),
            DocumentBlock(
                block_id=f"body-{page_number}",
                page_number=page_number,
                order=1,
                bbox=BoundingBox(x0=10, y0=100, x1=500, y1=700),
                text=("Café\u00ad agent inter-\nnational evidence. " * 25),
            ),
        ]
        pages.append(DocumentPage(page_number=page_number, width=600, height=800, blocks=blocks))
        all_ids.extend(block.block_id for block in blocks)
    return ParsedDocument(
        paper_id="arxiv:fixture",
        input_sha256=SHA,
        parser_name="mock",
        parser_version="1",
        page_count=page_count,
        pages=pages,
        sections=[
            DocumentSection(
                section_id="section-method",
                title="Method",
                page_start=1,
                page_end=page_count,
                block_ids=all_ids,
            )
        ],
    )


def test_cleaning_is_immutable_deterministic_and_preserves_sources() -> None:
    source = _document()
    before = deepcopy(source)
    first = clean_document(source, CleaningConfig())
    second = clean_document(source, CleaningConfig())
    assert source == before
    assert first == second
    assert first.statistics["header_footer_blocks_removed"] == 4
    assert all("Conference" not in block.text for page in first.pages for block in page.blocks)
    assert all("international" in block.text for page in first.pages for block in page.blocks)
    assert all(block.source_block_ids for page in first.pages for block in page.blocks)


def test_margin_rule_retains_low_coverage_and_repeated_body_content() -> None:
    source = _document(page_count=4)
    source.pages[-1].blocks[0].text = "unique footer"
    cleaned = clean_document(source, CleaningConfig(repeated_margin_coverage=0.9))
    assert any("Conference" in block.text for page in cleaned.pages for block in page.blocks)
    assert (
        len([block for page in cleaned.pages for block in page.blocks if "Café" in block.text]) == 4
    )


def test_adjacent_duplicate_is_removed_but_non_adjacent_copy_remains() -> None:
    source = _document()
    first_body = source.pages[0].blocks[1]
    duplicate = first_body.model_copy(update={"block_id": "duplicate", "order": 2})
    source.pages[0].blocks.append(duplicate)
    cleaned = clean_document(source, CleaningConfig())
    assert cleaned.statistics["adjacent_duplicates_removed"] == 1
    assert sum("Café" in block.text for page in cleaned.pages for block in page.blocks) == 4


def test_regex_tokenizer_and_section_windows_are_stable() -> None:
    assert token_count("Agent: café 2.0") == 6
    cleaned = clean_document(_document(), CleaningConfig())
    config = ChunkingConfig(target_tokens=90, min_tokens=60, max_tokens=100, overlap_tokens=20)
    first = chunk_document(cleaned, config, processing_version="pv1")
    second = chunk_document(cleaned, config, processing_version="pv1")
    changed = chunk_document(cleaned, config, processing_version="pv2")
    assert first == second
    assert first.chunks
    assert any(warning.code.value == "oversized_block" for warning in first.warnings)
    assert all(0 < chunk.token_count <= 100 for chunk in first.chunks)
    assert [chunk.ordinal for chunk in first.chunks] == list(range(len(first.chunks)))
    assert [chunk.page_start for chunk in first.chunks] == sorted(
        chunk.page_start for chunk in first.chunks
    )
    assert {chunk.chunk_id for chunk in first.chunks}.isdisjoint(
        chunk.chunk_id for chunk in changed.chunks
    )
    assert all(
        chunk.content_hash == hashlib.sha256(chunk.text.encode()).hexdigest()
        for chunk in first.chunks
    )
