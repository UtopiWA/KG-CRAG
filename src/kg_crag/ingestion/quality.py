"""从阶段产物计算可审计的自动摄取质量指标。"""

from __future__ import annotations

from collections import Counter

from kg_crag.ingestion.chunking import ChunkingResult
from kg_crag.models import CleanedDocument, ParsedDocument, QualityReport


def build_quality_report(
    parsed: ParsedDocument,
    cleaned: CleanedDocument,
    chunking: ChunkingResult,
    *,
    processing_version: str,
) -> QualityReport:
    blocks = [block for page in cleaned.pages for block in page.blocks]
    empty_pages = [page.page_number for page in cleaned.pages if not page.blocks]
    warning_counts = Counter(warning.code.value for warning in cleaned.warnings)
    warning_counts.update(warning.code.value for warning in chunking.warnings)
    tokens = [chunk.token_count for chunk in chunking.chunks]
    covered = {
        page
        for chunk in chunking.chunks
        if chunk.page_start is not None and chunk.page_end is not None
        for page in range(chunk.page_start, chunk.page_end + 1)
    }
    text_pages = {page.page_number for page in cleaned.pages if page.blocks}
    coverage = len(covered & text_pages) / len(text_pages) if text_pages else 0.0
    failures: list[str] = []
    if not blocks:
        failures.append("cleaned document has no text blocks")
    if not chunking.chunks:
        failures.append("document produced no chunks")
    if coverage < 0.8:
        failures.append("chunk page coverage is below 80%")
    return QualityReport(
        paper_id=cleaned.paper_id,
        processing_version=processing_version,
        page_count=parsed.page_count,
        block_count=len(blocks),
        section_count=len(cleaned.sections),
        chunk_count=len(chunking.chunks),
        character_count=sum(len(block.text) for block in blocks),
        empty_text_pages=empty_pages,
        warning_counts=dict(sorted(warning_counts.items())),
        min_chunk_tokens=min(tokens, default=0),
        max_chunk_tokens=max(tokens, default=0),
        average_chunk_tokens=(sum(tokens) / len(tokens) if tokens else 0.0),
        pages_with_chunks=sorted(covered),
        page_coverage_ratio=coverage,
        passed=not failures,
        failures=failures,
    )
