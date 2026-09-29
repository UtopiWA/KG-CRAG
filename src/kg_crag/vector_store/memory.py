"""仅供离线测试使用的确定性内存向量存储。"""

from __future__ import annotations

import math
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
)
from kg_crag.vector_store.base import VectorRecordState

FilterValue = str | int | bool


@dataclass(frozen=True)
class VectorUpsertCall:
    """一次写入的不可变记录快照。"""

    chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class VectorSearchCall:
    """一次搜索的不可变参数快照。"""

    vector: tuple[float, ...]
    top_k: int
    filters: tuple[tuple[str, FilterValue], ...]


@dataclass(frozen=True)
class VectorDeleteCall:
    """一次按论文删除的不可变参数快照。"""

    paper_id: str


VectorStoreCall = VectorUpsertCall | VectorSearchCall | VectorDeleteCall


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


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot_product / (left_norm * right_norm)


def _matches_filters(chunk: Chunk, filters: dict[str, FilterValue]) -> bool:
    for key, expected in filters.items():
        if key == "paper_id":
            actual: str | int | bool | None = chunk.paper_id
        elif key == "section":
            actual = chunk.section
        elif key == "processing_version":
            actual = chunk.processing_version
        else:
            return False
        if actual != expected:
            return False
    return True


class InMemoryVectorStore:
    """实现 VectorStore Protocol 的轻量测试替身。"""

    def __init__(self, *, collection_version: str = "memory-v1") -> None:
        self._records: dict[str, tuple[Chunk, tuple[float, ...]]] = {}
        self._dimension: int | None = None
        self.collection_version = collection_version
        self.calls: list[VectorStoreCall] = []

    async def ensure_collection(self, *, rebuild: bool = False) -> None:
        if rebuild:
            self._records.clear()
            self._dimension = None

    @property
    def dimension(self) -> int | None:
        return self._dimension

    @property
    def record_count(self) -> int:
        return len(self._records)

    async def upsert(self, records: list[tuple[Chunk, list[float]]]) -> None:
        """先校验整个批次，再按稳定 Chunk ID 幂等覆盖。"""

        self.calls.append(VectorUpsertCall(tuple(chunk.chunk_id for chunk, _ in records)))
        if not records:
            return

        expected_dimension = self._dimension or len(records[0][1])
        if expected_dimension <= 0:
            raise _invalid_parameter("vectors must not be empty")
        # 先验证整个批次，再修改内存状态，保持与真实后端一致的原子失败语义。
        for _, vector in records:
            if len(vector) != expected_dimension:
                raise _invalid_parameter(
                    "vector dimension does not match store dimension",
                    {
                        "expected_dimension": expected_dimension,
                        "actual_dimension": len(vector),
                    },
                )

        self._dimension = expected_dimension
        for chunk, vector in records:
            self._records[chunk.chunk_id] = (
                chunk.model_copy(deep=True),
                tuple(vector),
            )

    async def search(
        self,
        vector: list[float],
        *,
        top_k: int,
        filters: dict[str, FilterValue] | None = None,
    ) -> list[Evidence]:
        """以余弦相似度降序、Chunk ID 升序稳定返回结果。"""

        if top_k <= 0:
            raise _invalid_parameter("top_k must be positive", {"top_k": top_k})
        if not vector:
            raise _invalid_parameter("search vector must not be empty")
        if self._dimension is not None and len(vector) != self._dimension:
            raise _invalid_parameter(
                "vector dimension does not match store dimension",
                {
                    "expected_dimension": self._dimension,
                    "actual_dimension": len(vector),
                },
            )

        selected_filters = dict(filters or {})
        self.calls.append(
            VectorSearchCall(
                vector=tuple(vector),
                top_k=top_k,
                filters=tuple(sorted(selected_filters.items())),
            )
        )
        query_vector = tuple(vector)
        scored = [
            (chunk, _cosine_similarity(query_vector, stored_vector))
            for chunk, stored_vector in self._records.values()
            if _matches_filters(chunk, selected_filters)
        ]
        scored.sort(key=lambda item: (-item[1], item[0].chunk_id))
        return [
            self._to_evidence(chunk, score, rank)
            for rank, (chunk, score) in enumerate(scored[:top_k], start=1)
        ]

    async def delete_paper(self, paper_id: str) -> None:
        """只删除属于指定论文的记录。"""

        self.calls.append(VectorDeleteCall(paper_id))
        chunk_ids = [
            chunk_id for chunk_id, (chunk, _) in self._records.items() if chunk.paper_id == paper_id
        ]
        for chunk_id in chunk_ids:
            del self._records[chunk_id]

    async def record_state(self, paper_id: str) -> dict[str, VectorRecordState]:
        return {
            chunk_id: VectorRecordState(
                content_hash=chunk.content_hash,
                processing_version=chunk.processing_version,
            )
            for chunk_id, (chunk, _) in self._records.items()
            if chunk.paper_id == paper_id
        }

    async def delete_stale(self, paper_id: str, keep_chunk_ids: set[str]) -> int:
        stale = [
            chunk_id
            for chunk_id, (chunk, _) in self._records.items()
            if chunk.paper_id == paper_id and chunk_id not in keep_chunk_ids
        ]
        for chunk_id in stale:
            del self._records[chunk_id]
        return len(stale)

    def _to_evidence(self, chunk: Chunk, score: float, rank: int) -> Evidence:
        return Evidence(
            evidence_id=f"dense:{self.collection_version}:{chunk.chunk_id}",
            content=chunk.text,
            source_type=EvidenceSourceType.CHUNK,
            source_id=chunk.chunk_id,
            paper_id=chunk.paper_id,
            location=EvidenceLocation(section=chunk.section, page=chunk.page_start),
            scores=EvidenceScores(dense=score),
            ranks=EvidenceRanks(dense=rank),
            external=False,
            metadata={
                "collection_version": self.collection_version,
                "content_hash": chunk.content_hash,
                "processing_version": chunk.processing_version,
            },
        )
