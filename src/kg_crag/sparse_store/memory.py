"""默认测试使用的确定性内存 Sparse Store。"""

from __future__ import annotations

import hashlib
import math
import platform
import sqlite3
from collections import Counter
from dataclasses import dataclass

from kg_crag.errors import KGCRAGError
from kg_crag.models import (
    Chunk,
    ErrorCode,
    ErrorDetail,
    Evidence,
    EvidenceLocation,
    EvidenceRanks,
    EvidenceScores,
    EvidenceSourceType,
    SparseIndexIdentity,
)
from kg_crag.sparse_store.base import (
    SparseRecordState,
    SparseSyncResult,
)
from kg_crag.sparse_store.tokenizer import prepare_sparse_query, tokenize_scientific_text

FilterValue = str | int | bool
_ALLOWED_FILTERS = frozenset({"paper_id", "section", "processing_version"})


@dataclass(frozen=True)
class SparseSearchCall:
    tokens: tuple[str, ...]
    top_k: int
    offset: int
    filters: tuple[tuple[str, FilterValue], ...]


class InMemorySparseStore:
    """使用小型 BM25 计算的离线契约替身，不访问文件或网络。"""

    def __init__(
        self,
        *,
        identity: SparseIndexIdentity | None = None,
        max_top_k: int = 100,
        max_candidates: int = 100,
        max_query_chars: int = 2000,
        max_query_tokens: int = 64,
    ) -> None:
        self._identity = identity or _memory_identity()
        self._records: dict[str, Chunk] = {}
        self.max_top_k = max_top_k
        self.max_candidates = max_candidates
        self.max_query_chars = max_query_chars
        self.max_query_tokens = max_query_tokens
        self.calls: list[SparseSearchCall] = []

    @property
    def identity(self) -> SparseIndexIdentity:
        return self._identity

    @property
    def is_initialized(self) -> bool:
        return True

    @property
    def record_count(self) -> int:
        return len(self._records)

    async def ensure_index(self, identity: SparseIndexIdentity, *, rebuild: bool = False) -> None:
        if rebuild:
            self._records.clear()
            self._identity = identity
            return
        if identity != self._identity:
            raise _invalid_parameter("sparse index identity does not match")

    async def sync_paper(self, paper_id: str, chunks: list[Chunk]) -> SparseSyncResult:
        """先校验完整批次，再一次性替换该论文的全部记录。"""

        if not paper_id.strip():
            raise _invalid_parameter("paper_id must not be empty")
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise _invalid_parameter("chunk IDs must be unique within a paper sync")
        mismatch = next((chunk.chunk_id for chunk in chunks if chunk.paper_id != paper_id), None)
        if mismatch is not None:
            raise _invalid_parameter(
                "chunk paper_id must match the paper being synchronized",
                {"paper_id": paper_id, "chunk_id": mismatch},
            )

        previous = {
            key: value for key, value in self._records.items() if value.paper_id == paper_id
        }
        incoming = {chunk.chunk_id: chunk.model_copy(deep=True) for chunk in chunks}
        added = len(incoming.keys() - previous.keys())
        deleted = len(previous.keys() - incoming.keys())
        updated = sum(
            1
            for key in incoming.keys() & previous.keys()
            if incoming[key].content_hash != previous[key].content_hash
            or incoming[key].processing_version != previous[key].processing_version
        )
        skipped = len(incoming) - added - updated

        retained = {
            key: value for key, value in self._records.items() if value.paper_id != paper_id
        }
        retained.update(incoming)
        self._records = retained
        return SparseSyncResult(paper_id, added, updated, skipped, deleted)

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        offset: int = 0,
        filters: dict[str, FilterValue] | None = None,
    ) -> list[Evidence]:
        if top_k <= 0 or top_k > self.max_top_k:
            raise _invalid_parameter(
                "top_k is outside the sparse search limit",
                {"top_k": top_k, "max_top_k": self.max_top_k},
            )
        if offset < 0 or offset + top_k > self.max_candidates:
            raise _invalid_parameter(
                "sparse result window exceeds candidate limit",
                {"offset": offset, "window": offset + top_k, "max_count": self.max_candidates},
            )
        selected_filters = dict(filters or {})
        unknown = set(selected_filters) - _ALLOWED_FILTERS
        if unknown:
            raise _invalid_parameter("unknown sparse filter", {"filter": sorted(unknown)[0]})
        prepared = prepare_sparse_query(
            query,
            max_chars=self.max_query_chars,
            max_tokens=self.max_query_tokens,
        )
        self.calls.append(
            SparseSearchCall(
                prepared.tokens, top_k, offset, tuple(sorted(selected_filters.items()))
            )
        )
        documents = [
            chunk for chunk in self._records.values() if _matches_filters(chunk, selected_filters)
        ]
        scored = _bm25_scores(documents, prepared.tokens)
        scored.sort(key=lambda item: (-item[1], item[0].chunk_id))
        return [
            _to_evidence(self.identity.index_version, chunk, score, rank)
            for rank, (chunk, score) in enumerate(scored[offset : offset + top_k], start=offset + 1)
        ]

    async def record_state(self, paper_id: str) -> dict[str, SparseRecordState]:
        return {
            chunk_id: SparseRecordState(chunk.content_hash, chunk.processing_version)
            for chunk_id, chunk in self._records.items()
            if chunk.paper_id == paper_id
        }

    async def paper_context(self, paper_id: str, *, limit: int) -> list[Evidence]:
        if not paper_id.strip() or limit <= 0 or limit > 20:
            raise _invalid_parameter("invalid paper context request")
        chunks = sorted(
            (item for item in self._records.values() if item.paper_id == paper_id),
            key=lambda item: (
                0
                if "abstract" in (item.section or "").casefold()
                or item.text.lstrip().casefold().startswith("abstract")
                else 1,
                item.ordinal,
                item.chunk_id,
            ),
        )[:limit]
        return [_to_context_evidence(self.identity.index_version, chunk) for chunk in chunks]


def _bm25_scores(
    documents: list[Chunk], query_tokens: tuple[str, ...]
) -> list[tuple[Chunk, float]]:
    """实现足够小且确定的 BM25，用于共享契约和排序测试。"""

    if not documents:
        return []
    tokenized = [(chunk, tokenize_scientific_text(chunk.text)) for chunk in documents]
    average_length = sum(len(tokens) for _, tokens in tokenized) / len(tokenized)
    document_frequency = Counter(
        token for token in set(query_tokens) for _, tokens in tokenized if token in tokens
    )
    scored: list[tuple[Chunk, float]] = []
    for chunk, tokens in tokenized:
        counts = Counter(tokens)
        score = 0.0
        for token in set(query_tokens):
            frequency = counts[token]
            if not frequency:
                continue
            df = document_frequency[token]
            inverse_frequency = math.log(1 + (len(documents) - df + 0.5) / (df + 0.5))
            length_ratio = len(tokens) / average_length if average_length else 0.0
            denominator = frequency + 1.2 * (1 - 0.75 + 0.75 * length_ratio)
            score += inverse_frequency * frequency * 2.2 / denominator
        if score > 0.0:
            scored.append((chunk, score))
    return scored


def _memory_identity() -> SparseIndexIdentity:
    """为测试替身生成满足公共格式但不冒充 FTS5 的稳定身份。"""

    version_payload = b"memory|v1|scientific-unicode-v1|memory-bm25-v1"
    return SparseIndexIdentity(
        schema_version="v1",
        tokenizer="scientific-unicode-v1",
        bm25_version="memory-bm25-v1",
        python_version=platform.python_version(),
        sqlite_version=sqlite3.sqlite_version,
        fts5_enabled=False,
        fts5_version="not-used-by-memory-backend",
        corpus_snapshot_hash="0" * 64,
        index_version=hashlib.sha256(version_payload).hexdigest(),
    )


def _matches_filters(chunk: Chunk, filters: dict[str, FilterValue]) -> bool:
    values: dict[str, str | int | bool | None] = {
        "paper_id": chunk.paper_id,
        "section": chunk.section,
        "processing_version": chunk.processing_version,
    }
    return all(values[key] == expected for key, expected in filters.items())


def _to_evidence(index_version: str, chunk: Chunk, score: float, rank: int) -> Evidence:
    return Evidence(
        evidence_id=f"sparse:{index_version}:{chunk.chunk_id}",
        content=chunk.text,
        source_type=EvidenceSourceType.CHUNK,
        source_id=chunk.chunk_id,
        paper_id=chunk.paper_id,
        location=EvidenceLocation(section=chunk.section, page=chunk.page_start),
        scores=EvidenceScores(sparse=score),
        ranks=EvidenceRanks(sparse=rank),
        external=False,
        metadata={
            "index_version": index_version,
            "content_hash": chunk.content_hash,
            "processing_version": chunk.processing_version,
        },
    )


def _to_context_evidence(index_version: str, chunk: Chunk) -> Evidence:
    return Evidence(
        evidence_id=f"paper-context:{index_version}:{chunk.chunk_id}",
        content=chunk.text,
        source_type=EvidenceSourceType.CHUNK,
        source_id=chunk.chunk_id,
        paper_id=chunk.paper_id,
        location=EvidenceLocation(section=chunk.section, page=chunk.page_start),
        external=False,
        metadata={
            "index_version": index_version,
            "content_hash": chunk.content_hash,
            "processing_version": chunk.processing_version,
            "paper_context_expansion": True,
        },
    )


def _invalid_parameter(
    message: str,
    context: dict[str, str | int | float | bool | None] | None = None,
) -> KGCRAGError:
    return KGCRAGError(
        ErrorDetail(
            code=ErrorCode.VALIDATION,
            message=message,
            retryable=False,
            context=context or {},
        )
    )
