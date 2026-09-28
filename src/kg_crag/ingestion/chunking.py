"""版本化正则 Token 计数与章节内稳定窗口切分。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from kg_crag.ingestion.config import ChunkingConfig
from kg_crag.models import Chunk, CleanedDocument, ParseWarning, ParseWarningCode, WarningSeverity

_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class ChunkingResult:
    chunks: list[Chunk]
    warnings: list[ParseWarning]


def tokenize(text: str) -> list[str]:
    """`regex-v1`: Unicode 单词或单个非空白标点。"""

    return _TOKEN.findall(text)


def token_count(text: str) -> int:
    return len(tokenize(text))


def _window_ranges(total: int, config: ChunkingConfig) -> list[tuple[int, int]]:
    if total == 0:
        return []
    if total <= config.max_tokens:
        return [(0, total)]
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < total:
        end = min(start + config.target_tokens, total)
        remainder = total - end
        if 0 < remainder < config.min_tokens:
            candidate = total - config.min_tokens
            if candidate > start and candidate - start >= config.min_tokens:
                end = candidate
            elif total - start <= config.max_tokens:
                end = total
        ranges.append((start, end))
        if end == total:
            break
        next_start = end - config.overlap_tokens
        if next_start <= start:
            raise ValueError("chunk overlap did not advance the token window")
        start = next_start
    return ranges


def _chunk_id(
    paper_id: str,
    processing_version: str,
    section_id: str,
    ordinal: int,
    page_start: int,
    page_end: int,
    content_hash: str,
) -> str:
    canonical = "\n".join(
        [
            paper_id,
            processing_version,
            section_id,
            str(ordinal),
            str(page_start),
            str(page_end),
            content_hash,
        ]
    )
    return "chunk-" + hashlib.sha256(canonical.encode()).hexdigest()


def chunk_document(
    document: CleanedDocument,
    config: ChunkingConfig,
    *,
    processing_version: str,
) -> ChunkingResult:
    """把各 Section 的文本独立切分，并保持页码单调与稳定 ID。"""

    by_id = {block.block_id: block for page in document.pages for block in page.blocks}
    chunks: list[Chunk] = []
    warnings: list[ParseWarning] = []
    ordinal = 0
    for section in document.sections:
        tokens: list[str] = []
        pages: list[int] = []
        for block_id in section.block_ids:
            block = by_id.get(block_id)
            if block is None:
                raise ValueError(f"section references an unknown block: {block_id}")
            block_tokens = tokenize(block.text)
            if len(block_tokens) > config.max_tokens:
                warnings.append(
                    ParseWarning(
                        code=ParseWarningCode.OVERSIZED_BLOCK,
                        severity=WarningSeverity.WARNING,
                        page_number=block.page_number,
                        block_id=block.block_id,
                        message="单个文本块超过最大窗口，已在 Token 边界确定性切分。",
                        details={"unit_count": len(block_tokens)},
                    )
                )
            tokens.extend(block_tokens)
            pages.extend([block.page_number] * len(block_tokens))
        for start, end in _window_ranges(len(tokens), config):
            text = " ".join(tokens[start:end]).strip()
            if not text:
                continue
            page_start = min(pages[start:end])
            page_end = max(pages[start:end])
            content_hash = hashlib.sha256(text.encode()).hexdigest()
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(
                        document.paper_id,
                        processing_version,
                        section.section_id,
                        ordinal,
                        page_start,
                        page_end,
                        content_hash,
                    ),
                    paper_id=document.paper_id,
                    section=section.title,
                    page_start=page_start,
                    page_end=page_end,
                    text=text,
                    token_count=end - start,
                    content_hash=content_hash,
                    ordinal=ordinal,
                    processing_version=processing_version,
                )
            )
            ordinal += 1
    return ChunkingResult(chunks=chunks, warnings=warnings)
