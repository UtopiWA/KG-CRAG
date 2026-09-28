# mypy: disable-error-code="no-untyped-call"
"""基于 PyMuPDF 的确定性学术 PDF 文本与版面解析器。"""

from __future__ import annotations

import hashlib
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf

from kg_crag.errors import KGCRAGError
from kg_crag.ingestion.config import ParsingConfig
from kg_crag.models import (
    BlockKind,
    BoundingBox,
    DocumentBlock,
    DocumentPage,
    DocumentSection,
    ErrorCode,
    ErrorDetail,
    ParsedDocument,
    ParseRequest,
    ParseWarning,
    ParseWarningCode,
    WarningSeverity,
)

_NUMBERED_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*\.?|[IVX]+\.?)\s+[A-Z\u00C0-\u024F]", re.I)
_KNOWN_HEADING = re.compile(
    r"^(?:abstract|introduction|background|related work|method(?:ology)?|experiments?|"
    r"results?|discussion|conclusion|references|appendix)(?:\s|$)",
    re.I,
)


@dataclass(frozen=True)
class _RawBlock:
    index: int
    bbox: tuple[float, float, float, float]
    text: str
    font_size: float | None
    bold: bool
    kind: BlockKind


def _parse_error(message: str, paper_id: str) -> KGCRAGError:
    return KGCRAGError(
        ErrorDetail(
            code=ErrorCode.DATA,
            message=message,
            retryable=False,
            context={"paper_id": paper_id},
        )
    )


def _rounded_bbox(values: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return tuple(round(max(0.0, float(value)), 3) for value in values)  # type: ignore[return-value]


def _block_kind(text: str, *, line_count: int, span_count: int) -> BlockKind:
    formula_symbols = sum(text.count(symbol) for symbol in "=∑∫√≤≥≈→")
    if formula_symbols >= 2 and len(text) < 500:
        return BlockKind.FORMULA
    if line_count >= 3 and ("\t" in text or span_count >= line_count * 3):
        return BlockKind.TABLE
    return BlockKind.TEXT


def _extract_raw_blocks(page: pymupdf.Page) -> tuple[list[_RawBlock], int]:
    payload: dict[str, Any] = page.get_text("dict", sort=False)
    blocks: list[_RawBlock] = []
    image_count = 0
    for index, raw in enumerate(payload.get("blocks", [])):
        if raw.get("type") != 0:
            image_count += 1
            continue
        lines = raw.get("lines", [])
        line_texts: list[str] = []
        sizes: list[float] = []
        bold = False
        span_count = 0
        for line in lines:
            texts: list[str] = []
            for span in line.get("spans", []):
                text = str(span.get("text", ""))
                texts.append(text)
                if text.strip():
                    sizes.append(float(span.get("size", 0.0)))
                    bold = bold or bool(int(span.get("flags", 0)) & 16)
                    span_count += 1
            if texts:
                line_texts.append("".join(texts).strip())
        text = "\n".join(line_texts).strip()
        if not text:
            continue
        bbox = _rounded_bbox(tuple(raw.get("bbox", (0, 0, 0, 0))))
        blocks.append(
            _RawBlock(
                index=index,
                bbox=bbox,
                text=text,
                font_size=max(sizes) if sizes else None,
                bold=bold,
                kind=_block_kind(text, line_count=len(line_texts), span_count=span_count),
            )
        )
    return blocks, image_count


def _sort_blocks(
    blocks: list[_RawBlock], page_width: float, full_width_ratio: float
) -> tuple[list[_RawBlock], bool]:
    """以全宽块为锚点，在每个垂直区间内先左栏后右栏。"""

    full = [
        block for block in blocks if block.bbox[2] - block.bbox[0] >= page_width * full_width_ratio
    ]
    narrow = [block for block in blocks if block not in full]
    full.sort(key=lambda item: (item.bbox[1], item.bbox[0], item.index))
    ordered: list[_RawBlock] = []
    lower = 0.0
    has_left = False
    has_right = False
    for anchor in [*full, None]:
        upper = anchor.bbox[1] if anchor is not None else float("inf")
        segment = [item for item in narrow if lower <= (item.bbox[1] + item.bbox[3]) / 2 < upper]
        left = [item for item in segment if (item.bbox[0] + item.bbox[2]) / 2 < page_width / 2]
        right = [item for item in segment if item not in left]
        has_left = has_left or bool(left)
        has_right = has_right or bool(right)
        ordered.extend(sorted(left, key=lambda item: (item.bbox[1], item.bbox[0], item.index)))
        ordered.extend(sorted(right, key=lambda item: (item.bbox[1], item.bbox[0], item.index)))
        if anchor is not None:
            ordered.append(anchor)
            lower = anchor.bbox[3]
    # 对跨锚点的异常块采用稳定兜底，确保不丢内容。
    missing = [item for item in blocks if item not in ordered]
    ordered.extend(sorted(missing, key=lambda item: (item.bbox[1], item.bbox[0], item.index)))
    return ordered, has_left and has_right


def _stable_block_id(
    input_sha256: str,
    page_number: int,
    order: int,
    block: _RawBlock,
) -> str:
    coordinate = ",".join(f"{value:.3f}" for value in block.bbox)
    text_hash = hashlib.sha256(block.text.encode()).hexdigest()
    value = f"{input_sha256}\n{page_number}\n{order}\n{coordinate}\n{text_hash}"
    return "block-" + hashlib.sha256(value.encode()).hexdigest()


def _overlap_exists(blocks: list[_RawBlock]) -> bool:
    for index, first in enumerate(blocks):
        a = first.bbox
        for second in blocks[index + 1 :]:
            b = second.bbox
            width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            if width * height > 4.0:
                return True
    return False


def _identify_sections(
    paper_id: str,
    pages: list[DocumentPage],
    config: ParsingConfig,
) -> tuple[list[DocumentSection], list[ParseWarning]]:
    blocks = [block for page in pages for block in page.blocks]
    sizes = [block.font_size for block in blocks if block.font_size and block.font_size > 0]
    median = statistics.median(sizes) if sizes else 0.0
    headings: set[str] = set()
    for block in blocks:
        text = " ".join(block.text.split())
        numbered = bool(_NUMBERED_HEADING.match(text) or _KNOWN_HEADING.match(text))
        emphasized = bool(
            block.font_size
            and median
            and block.font_size >= median * config.heading_font_ratio
            and (block.bold or len(text) <= 80)
        )
        if len(text) <= config.heading_max_chars and "\n" not in text and (numbered or emphasized):
            headings.add(block.block_id)
            block.kind = BlockKind.TITLE
    if not headings:
        block_ids = [block.block_id for block in blocks]
        pages_used = [block.page_number for block in blocks]
        section = DocumentSection(
            section_id="section-" + hashlib.sha256(f"{paper_id}:__document__".encode()).hexdigest(),
            title="__document__",
            page_start=min(pages_used),
            page_end=max(pages_used),
            block_ids=block_ids,
        )
        warning = ParseWarning(
            code=ParseWarningCode.HEADING_FALLBACK,
            severity=WarningSeverity.INFO,
            message="未可靠识别章节标题，已使用整篇文档回退章节。",
        )
        return [section], [warning]

    groups: list[tuple[str, list[DocumentBlock]]] = []
    current_title = "__document__"
    current: list[DocumentBlock] = []
    for block in blocks:
        if block.block_id in headings and current:
            groups.append((current_title, current))
            current = []
        if block.block_id in headings:
            current_title = " ".join(block.text.split())
        current.append(block)
    if current:
        groups.append((current_title, current))
    sections: list[DocumentSection] = []
    for index, (title, items) in enumerate(groups):
        key = f"{paper_id}\n{index}\n{title}\n{items[0].block_id}"
        sections.append(
            DocumentSection(
                section_id="section-" + hashlib.sha256(key.encode()).hexdigest(),
                title=title,
                page_start=min(item.page_number for item in items),
                page_end=max(item.page_number for item in items),
                block_ids=[item.block_id for item in items],
            )
        )
    return sections, []


class PyMuPDFParser:
    """不向公共模型泄漏 PyMuPDF 对象的同步解析器。"""

    name = "pymupdf"
    version = f"{pymupdf.__version__}+adapter-v1"

    def __init__(self, config: ParsingConfig | None = None) -> None:
        self.config = config or ParsingConfig()

    def parse(self, pdf_path: Path, request: ParseRequest) -> ParsedDocument:
        if pdf_path.stat().st_size > request.max_pdf_bytes:
            raise _parse_error("PDF exceeds the configured size limit", request.paper_id)
        try:
            document = pymupdf.open(pdf_path)
        except Exception as exc:
            raise _parse_error("PDF cannot be opened", request.paper_id) from exc
        try:
            if document.needs_pass:
                raise _parse_error("encrypted PDF requires a password", request.paper_id)
            if document.page_count < 1 or document.page_count > request.max_pages:
                raise _parse_error(
                    "PDF page count is outside the configured limit", request.paper_id
                )
            pages: list[DocumentPage] = []
            warnings: list[ParseWarning] = []
            character_count = 0
            for page_index in range(document.page_count):
                source_page = document.load_page(page_index)
                raw_blocks, image_count = _extract_raw_blocks(source_page)
                ordered, multi_column = _sort_blocks(
                    raw_blocks,
                    float(source_page.rect.width),
                    request.full_width_ratio,
                )
                page_number = page_index + 1
                blocks = [
                    DocumentBlock(
                        block_id=_stable_block_id(request.input_sha256, page_number, order, block),
                        page_number=page_number,
                        order=order,
                        bbox=BoundingBox(
                            x0=block.bbox[0],
                            y0=block.bbox[1],
                            x1=block.bbox[2],
                            y1=block.bbox[3],
                        ),
                        text=block.text,
                        kind=block.kind,
                        font_size=block.font_size,
                        bold=block.bold,
                    )
                    for order, block in enumerate(ordered)
                ]
                page_chars = sum(len(block.text) for block in blocks)
                character_count += page_chars
                if page_chars == 0:
                    warnings.append(
                        ParseWarning(
                            code=ParseWarningCode.LOW_TEXT,
                            severity=WarningSeverity.WARNING,
                            page_number=page_number,
                            message="页面没有可提取文本。",
                        )
                    )
                if multi_column:
                    warnings.append(
                        ParseWarning(
                            code=ParseWarningCode.MULTI_COLUMN,
                            severity=WarningSeverity.INFO,
                            page_number=page_number,
                            message="检测到至多双栏版面，已按确定性列顺序排列。",
                        )
                    )
                if image_count:
                    warnings.append(
                        ParseWarning(
                            code=ParseWarningCode.IMAGE_SKIPPED,
                            severity=WarningSeverity.INFO,
                            page_number=page_number,
                            message="页面包含图像；已保留文本块但未执行图像识别。",
                            details={"image_count": image_count},
                        )
                    )
                if _overlap_exists(raw_blocks):
                    warnings.append(
                        ParseWarning(
                            code=ParseWarningCode.COMPLEX_LAYOUT,
                            severity=WarningSeverity.WARNING,
                            page_number=page_number,
                            message="检测到重叠文本块，阅读顺序采用稳定降级策略。",
                        )
                    )
                for block in blocks:
                    if block.kind is BlockKind.TABLE:
                        warnings.append(
                            ParseWarning(
                                code=ParseWarningCode.TABLE_DETECTED,
                                severity=WarningSeverity.INFO,
                                page_number=page_number,
                                block_id=block.block_id,
                                message="疑似表格文本已按普通块保留。",
                            )
                        )
                    elif block.kind is BlockKind.FORMULA:
                        warnings.append(
                            ParseWarning(
                                code=ParseWarningCode.FORMULA_DETECTED,
                                severity=WarningSeverity.INFO,
                                page_number=page_number,
                                block_id=block.block_id,
                                message="疑似公式文本已按普通块保留。",
                            )
                        )
                pages.append(
                    DocumentPage(
                        page_number=page_number,
                        width=round(float(source_page.rect.width), 3),
                        height=round(float(source_page.rect.height), 3),
                        blocks=blocks,
                    )
                )
            if character_count < request.min_document_chars:
                raise _parse_error("PDF does not contain enough extractable text", request.paper_id)
            sections, section_warnings = _identify_sections(request.paper_id, pages, self.config)
            warnings.extend(section_warnings)
            return ParsedDocument(
                paper_id=request.paper_id,
                input_sha256=request.input_sha256,
                parser_name=self.name,
                parser_version=self.version,
                page_count=len(pages),
                pages=pages,
                sections=sections,
                warnings=warnings,
            )
        finally:
            document.close()
