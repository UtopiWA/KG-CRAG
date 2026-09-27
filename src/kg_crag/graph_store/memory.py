"""仅供离线测试使用的确定性内存图存储。"""

from __future__ import annotations

from dataclasses import dataclass

from kg_crag.errors import KGCRAGError
from kg_crag.models import (
    Chunk,
    ErrorCode,
    ErrorDetail,
    Evidence,
    EvidenceLocation,
    EvidenceScores,
    EvidenceSourceType,
    Paper,
)


@dataclass(frozen=True)
class GraphUpsertCall:
    """一次论文写入的不可变参数快照。"""

    paper_id: str
    chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class GraphRetrieveCall:
    """一次图检索的不可变参数快照。"""

    entity_names: tuple[str, ...]
    max_hops: int
    top_k: int


@dataclass(frozen=True)
class GraphDeleteCall:
    """一次按论文删除的不可变参数快照。"""

    paper_id: str


GraphStoreCall = GraphUpsertCall | GraphRetrieveCall | GraphDeleteCall


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


class InMemoryGraphStore:
    """实现 GraphStore Protocol 的词法匹配测试替身。"""

    def __init__(self) -> None:
        self._papers: dict[str, tuple[Paper, tuple[Chunk, ...]]] = {}
        self.calls: list[GraphStoreCall] = []

    @property
    def paper_count(self) -> int:
        return len(self._papers)

    async def upsert_paper(self, paper: Paper, chunks: list[Chunk]) -> None:
        """按 Paper ID 覆盖整组论文与 Chunk，避免残留旧记录。"""

        self.calls.append(GraphUpsertCall(paper.paper_id, tuple(item.chunk_id for item in chunks)))
        mismatched = [item.chunk_id for item in chunks if item.paper_id != paper.paper_id]
        if mismatched:
            raise _invalid_parameter(
                "chunk paper_id must match the paper being upserted",
                {"paper_id": paper.paper_id, "chunk_id": mismatched[0]},
            )
        self._papers[paper.paper_id] = (
            paper.model_copy(deep=True),
            tuple(item.model_copy(deep=True) for item in chunks),
        )

    async def retrieve(
        self,
        entity_names: list[str],
        *,
        max_hops: int,
        top_k: int,
    ) -> list[Evidence]:
        """匹配全部实体词，并按 Paper ID、Chunk ID 稳定排序。"""

        if top_k <= 0:
            raise _invalid_parameter("top_k must be positive", {"top_k": top_k})
        if max_hops < 0:
            raise _invalid_parameter("max_hops must be non-negative", {"max_hops": max_hops})
        self.calls.append(GraphRetrieveCall(tuple(entity_names), max_hops, top_k))

        normalized_entities = tuple(
            name.strip().casefold() for name in entity_names if name.strip()
        )
        if not normalized_entities:
            return []

        matches: list[tuple[Paper, Chunk]] = []
        for paper, chunks in self._papers.values():
            for chunk in chunks:
                searchable = f"{paper.title}\n{chunk.text}".casefold()
                if all(entity in searchable for entity in normalized_entities):
                    matches.append((paper, chunk))
        matches.sort(key=lambda item: (item[0].paper_id, item[1].chunk_id))
        return [self._to_evidence(paper, chunk, max_hops) for paper, chunk in matches[:top_k]]

    async def delete_paper(self, paper_id: str) -> None:
        """删除指定论文及其全部 Chunk。"""

        self.calls.append(GraphDeleteCall(paper_id))
        self._papers.pop(paper_id, None)

    @staticmethod
    def _to_evidence(paper: Paper, chunk: Chunk, max_hops: int) -> Evidence:
        return Evidence(
            evidence_id=f"graph:{paper.paper_id}:{chunk.chunk_id}",
            content=chunk.text,
            source_type=EvidenceSourceType.GRAPH,
            source_id=chunk.chunk_id,
            paper_id=paper.paper_id,
            location=EvidenceLocation(section=chunk.section, page=chunk.page_start),
            scores=EvidenceScores(graph=1.0),
            metadata={"backend": "memory", "max_hops": max_hops},
        )
