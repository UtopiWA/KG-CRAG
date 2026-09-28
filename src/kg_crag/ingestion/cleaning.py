"""确定性正文清洗，同时保留页面和原始 Block 溯源。"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

from kg_crag.ingestion.config import CleaningConfig
from kg_crag.models import (
    BlockKind,
    CleanedDocument,
    DocumentBlock,
    DocumentPage,
    DocumentSection,
    ParsedDocument,
    ParseWarning,
    ParseWarningCode,
    WarningSeverity,
)

_LINE_HYPHEN = re.compile(r"(?<=[^\W\d_])-\s*\n\s*(?=[^\W\d_])", re.UNICODE)
_SPACES = re.compile(r"[^\S\n]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")


def _normalized_signature(text: str) -> str:
    value = unicodedata.normalize("NFC", text).casefold()
    value = re.sub(r"\d+", "#", value)
    return " ".join(value.split())


def _unsafe_control_count(text: str) -> int:
    return sum(
        1 for char in text if unicodedata.category(char).startswith("C") and char not in "\n\t"
    )


def _normalize_block_text(text: str, kind: BlockKind) -> tuple[str, dict[str, int]]:
    statistics = {
        "soft_hyphens_removed": text.count("\u00ad"),
        "controls_removed": 0,
        "line_hyphens_joined": 0,
    }
    normalized = unicodedata.normalize("NFC", text).replace("\u00ad", "")
    statistics["controls_removed"] = _unsafe_control_count(normalized)
    normalized = "".join(
        char
        for char in normalized
        if not unicodedata.category(char).startswith("C") or char in "\n\t"
    )
    if kind not in {BlockKind.FORMULA, BlockKind.TABLE}:
        normalized, count = _LINE_HYPHEN.subn("", normalized)
        statistics["line_hyphens_joined"] = count
    normalized = "\n".join(_SPACES.sub(" ", line).strip() for line in normalized.splitlines())
    normalized = _MANY_NEWLINES.sub("\n\n", normalized).strip()
    return normalized, statistics


def _repeated_margin_ids(document: ParsedDocument, config: CleaningConfig) -> set[str]:
    by_signature: dict[str, set[int]] = {}
    block_ids: dict[str, set[str]] = {}
    for page in document.pages:
        for block in page.blocks:
            in_margin = (
                block.bbox.y1 <= page.height * config.margin_ratio
                or block.bbox.y0 >= page.height * (1 - config.margin_ratio)
            )
            if not in_margin:
                continue
            signature = _normalized_signature(block.text)
            if not signature:
                continue
            by_signature.setdefault(signature, set()).add(page.page_number)
            block_ids.setdefault(signature, set()).add(block.block_id)
    threshold = max(
        config.repeated_margin_min_pages,
        math.ceil(document.page_count * config.repeated_margin_coverage),
    )
    return {
        block_id
        for signature, pages in by_signature.items()
        if len(pages) >= threshold
        for block_id in block_ids[signature]
    }


def clean_document(
    document: ParsedDocument,
    config: CleaningConfig,
) -> CleanedDocument:
    """创建独立清洗模型，绝不原地修改 ParsedDocument。"""

    removed_margin_ids = _repeated_margin_ids(document, config)
    statistics: Counter[str] = Counter()
    warnings = list(document.warnings)
    pages: list[DocumentPage] = []
    retained_ids: set[str] = set()
    for source_page in document.pages:
        cleaned_blocks: list[DocumentBlock] = []
        previous_text: str | None = None
        for source in source_page.blocks:
            if source.block_id in removed_margin_ids:
                statistics["header_footer_blocks_removed"] += 1
                continue
            text, changes = _normalize_block_text(source.text, source.kind)
            statistics.update(changes)
            if not text:
                statistics["empty_blocks_removed"] += 1
                continue
            signature = _normalized_signature(text)
            if config.remove_adjacent_duplicates and signature == previous_text:
                statistics["adjacent_duplicates_removed"] += 1
                warnings.append(
                    ParseWarning(
                        code=ParseWarningCode.ADJACENT_DUPLICATE_REMOVED,
                        severity=WarningSeverity.INFO,
                        page_number=source.page_number,
                        block_id=source.block_id,
                        message="已移除与前一文本块相同的相邻重复块。",
                    )
                )
                continue
            block = source.model_copy(
                update={
                    "order": len(cleaned_blocks),
                    "text": text,
                    "source_block_ids": source.source_block_ids or [source.block_id],
                }
            )
            cleaned_blocks.append(block)
            retained_ids.add(block.block_id)
            previous_text = signature
        pages.append(source_page.model_copy(update={"blocks": cleaned_blocks}))
    if removed_margin_ids:
        warnings.append(
            ParseWarning(
                code=ParseWarningCode.HEADER_FOOTER_REMOVED,
                severity=WarningSeverity.INFO,
                message="已按页边覆盖率阈值移除重复页眉或页脚。",
                details={"removed_blocks": len(removed_margin_ids)},
            )
        )
    sections: list[DocumentSection] = []
    for section in document.sections:
        ids = [block_id for block_id in section.block_ids if block_id in retained_ids]
        if not ids:
            continue
        relevant = [block for page in pages for block in page.blocks if block.block_id in set(ids)]
        sections.append(
            section.model_copy(
                update={
                    "block_ids": ids,
                    "page_start": min(block.page_number for block in relevant),
                    "page_end": max(block.page_number for block in relevant),
                }
            )
        )
    if not sections:
        remaining = [block for page in pages for block in page.blocks]
        if not remaining:
            raise ValueError("cleaning removed every text block")
        sections = [
            DocumentSection(
                section_id="section-cleaned-document",
                title="__document__",
                page_start=min(block.page_number for block in remaining),
                page_end=max(block.page_number for block in remaining),
                block_ids=[block.block_id for block in remaining],
            )
        ]
    return CleanedDocument(
        paper_id=document.paper_id,
        input_sha256=document.input_sha256,
        cleaning_version=config.version,
        pages=pages,
        sections=sections,
        warnings=warnings,
        statistics=dict(sorted(statistics.items())),
    )
