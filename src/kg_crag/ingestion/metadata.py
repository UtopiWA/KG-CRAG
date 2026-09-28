"""将来源元数据确定性规范化为公共 Paper 契约。"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Mapping
from typing import Any

from kg_crag.ingestion.arxiv import normalize_arxiv_id
from kg_crag.models import Author, Paper

_SPACE = re.compile(r"\s+")
_DOI_PREFIX = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.I)


def normalize_text(value: str) -> str:
    return _SPACE.sub(" ", unicodedata.normalize("NFC", value)).strip()


def normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    normalized = _DOI_PREFIX.sub("", normalize_text(value)).casefold().rstrip(".")
    return normalized or None


def canonical_paper_id(payload: Mapping[str, Any]) -> str:
    """按固定优先级生成 ID，同时兼容已落盘的合法 arXiv v1 ID。"""

    arxiv_raw = str(payload.get("arxiv_id", "")).strip()
    arxiv_id = normalize_arxiv_id(arxiv_raw) if arxiv_raw else None
    legacy = str(payload.get("paper_id", "")).strip()
    if legacy and arxiv_id and legacy.casefold() == f"arxiv:{arxiv_id}".casefold():
        return f"arxiv:{arxiv_id}"
    doi = normalize_doi(str(payload.get("doi", "")) or None)
    candidate: str | None = None
    if doi:
        candidate = f"doi:{doi}"
    elif arxiv_id:
        candidate = f"arxiv:{arxiv_id}"
    source = payload.get("source_metadata")
    external_ids = source.get("external_ids", {}) if isinstance(source, Mapping) else {}
    if candidate is None and isinstance(external_ids, Mapping):
        for provider in sorted(external_ids):
            value = normalize_text(str(external_ids[provider]))
            if value:
                candidate = f"{str(provider).casefold()}:{value}"
                break
    if candidate is not None:
        if legacy and legacy.casefold() != candidate.casefold():
            raise ValueError("legacy paper_id conflicts with normalized source identifiers")
        return candidate
    title = normalize_text(str(payload.get("title", ""))).casefold()
    if not title:
        raise ValueError("title is required when no external identifier is available")
    candidate = f"title-sha256:{hashlib.sha256(title.encode()).hexdigest()}"
    if legacy and legacy.casefold() != candidate.casefold():
        raise ValueError("legacy paper_id conflicts with normalized source identifiers")
    return candidate


def normalize_paper(payload: Mapping[str, Any], *, processing_version: str) -> Paper:
    """规范字段并让 Pydantic 执行最终范围与 URL 校验。"""

    authors_value = payload.get("authors", [])
    if not isinstance(authors_value, list):
        raise ValueError("authors must be a list")
    authors: list[Author] = []
    for item in authors_value:
        if isinstance(item, Mapping):
            name = normalize_text(str(item.get("name", "")))
            author_id = normalize_text(str(item.get("author_id", ""))) or None
        else:
            name = normalize_text(str(item))
            author_id = None
        if name:
            authors.append(Author(author_id=author_id, name=name))
    arxiv_value = str(payload.get("arxiv_id", "")).strip()
    doi = normalize_doi(str(payload.get("doi", "")) or None)
    return Paper(
        paper_id=canonical_paper_id(payload),
        title=normalize_text(str(payload.get("title", ""))),
        abstract=normalize_text(str(payload.get("abstract", ""))),
        year=payload.get("year"),
        authors=authors,
        doi=doi,
        arxiv_id=normalize_arxiv_id(arxiv_value) if arxiv_value else None,
        semantic_scholar_id=(
            normalize_text(str(payload.get("semantic_scholar_id")))
            if payload.get("semantic_scholar_id")
            else None
        ),
        venue=normalize_text(str(payload.get("venue", ""))) or None,
        source_url=payload.get("source_url"),
        pdf_path=str(payload.get("pdf_path")) if payload.get("pdf_path") else None,
        references=[normalize_text(str(item)) for item in payload.get("references", [])],
        ingestion_version=processing_version,
    )
