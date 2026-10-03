"""将受控图路径解析为统一、可引用 Evidence。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from kg_crag.errors import KGCRAGError
from kg_crag.graph_store import GraphStore
from kg_crag.models import (
    Chunk,
    ErrorCode,
    ErrorDetail,
    Evidence,
    EvidenceLocation,
    EvidenceScores,
    EvidenceSourceType,
    GraphPathHit,
    GraphQueryRequest,
)

if TYPE_CHECKING:
    from kg_crag.indexing.processed import ProcessedPaper


@dataclass(frozen=True)
class GraphRetrievalResult:
    evidence: list[Evidence]
    paths: list[GraphPathHit]
    rejected_paths: int
    database_calls: int


class ProcessedSourceResolver:
    """只信任当前已发布 processed 数据，不使用图库中的正文副本。"""

    def __init__(self, papers: list[ProcessedPaper]) -> None:
        self.papers = {item.paper_id: item for item in papers}
        self.chunks: dict[str, Chunk] = {
            chunk.chunk_id: chunk for item in papers for chunk in item.chunks
        }
        self.paper_hashes = {
            item.paper_id: item.paper_hash for item in papers if item.paper_hash is not None
        }

    def resolve_chunk(self, source_id: str, paper_id: str, source_hash: str) -> Chunk:
        chunk = self.chunks.get(source_id)
        if chunk is None or chunk.paper_id != paper_id or chunk.content_hash != source_hash:
            raise KGCRAGError(
                ErrorDetail(
                    code=ErrorCode.DATA,
                    message="graph path provenance no longer matches processed corpus",
                    retryable=False,
                    context={"paper_id": paper_id, "source_id": source_id},
                )
            )
        return chunk

    def validate_metadata(self, source_id: str, paper_id: str, source_hash: str) -> None:
        if source_id != paper_id or self.paper_hashes.get(paper_id) != source_hash:
            raise KGCRAGError(
                ErrorDetail(
                    code=ErrorCode.DATA,
                    message="graph metadata provenance no longer matches processed corpus",
                    retryable=False,
                    context={"paper_id": paper_id, "source_id": source_id},
                )
            )


class GraphRetriever:
    def __init__(self, store: GraphStore, resolver: ProcessedSourceResolver) -> None:
        self.store = store
        self.resolver = resolver

    async def retrieve(self, request: GraphQueryRequest) -> GraphRetrievalResult:
        """每次调用只访问一次图后端；来源漂移时拒绝整条路径。"""

        hits = await self.store.query(request)
        valid: list[GraphPathHit] = []
        evidence_rows: list[tuple[GraphPathHit, str, Evidence]] = []
        rejected = 0
        for hit in hits:
            try:
                rows = self._path_evidence(hit)
            except KGCRAGError:
                rejected += 1
                continue
            valid.append(hit)
            evidence_rows.extend(rows)
        evidence_rows.sort(
            key=lambda item: (
                -item[0].score,
                item[0].hop_count,
                -min(provenance.confidence for provenance in item[0].provenances),
                item[0].path_id,
                item[1],
            )
        )
        evidence: list[Evidence] = []
        seen: set[str] = set()
        for rank, (_hit, _source, item) in enumerate(evidence_rows, 1):
            if item.evidence_id in seen:
                continue
            seen.add(item.evidence_id)
            evidence.append(
                item.model_copy(update={"metadata": {**item.metadata, "graph_rank": rank}})
            )
            if len(evidence) >= request.top_k:
                break
        return GraphRetrievalResult(
            evidence=evidence,
            paths=valid,
            rejected_paths=rejected,
            database_calls=1,
        )

    def _path_evidence(self, hit: GraphPathHit) -> list[tuple[GraphPathHit, str, Evidence]]:
        chunks: dict[str, Chunk] = {}
        metadata_sources: dict[str, str] = {}
        for provenance in hit.provenances:
            if provenance.source_kind == "chunk":
                chunk = self.resolver.resolve_chunk(
                    provenance.source_id, provenance.paper_id, provenance.source_hash
                )
                chunks[chunk.chunk_id] = chunk
            else:
                self.resolver.validate_metadata(
                    provenance.source_id, provenance.paper_id, provenance.source_hash
                )
                metadata_sources[provenance.source_id] = provenance.paper_id
        rows: list[tuple[GraphPathHit, str, Evidence]] = []
        fact_ids = ",".join(hit.fact_ids)
        for source_id, chunk in sorted(chunks.items()):
            rows.append(
                (
                    hit,
                    source_id,
                    Evidence(
                        evidence_id=(f"graph:{hit.graph_version}:{hit.path_id}:{source_id}"),
                        content=chunk.text,
                        source_type=EvidenceSourceType.GRAPH,
                        source_id=source_id,
                        paper_id=chunk.paper_id,
                        location=EvidenceLocation(
                            section=chunk.section,
                            page=chunk.page_start,
                        ),
                        scores=EvidenceScores(graph=hit.score),
                        external=False,
                        metadata={
                            "graph_version": hit.graph_version,
                            "template": hit.template.value,
                            "path_id": hit.path_id,
                            "fact_ids": fact_ids,
                            "hop_count": hit.hop_count,
                            "source_kind": "chunk",
                            "content_hash": chunk.content_hash,
                        },
                    ),
                )
            )
        if not chunks:
            for source_id, paper_id in sorted(metadata_sources.items()):
                rows.append(
                    (
                        hit,
                        source_id,
                        Evidence(
                            evidence_id=(f"graph:{hit.graph_version}:{hit.path_id}:{source_id}"),
                            content=hit.summary,
                            source_type=EvidenceSourceType.GRAPH,
                            source_id=source_id,
                            paper_id=paper_id,
                            scores=EvidenceScores(graph=hit.score),
                            external=False,
                            metadata={
                                "graph_version": hit.graph_version,
                                "template": hit.template.value,
                                "path_id": hit.path_id,
                                "fact_ids": fact_ids,
                                "hop_count": hit.hop_count,
                                "source_kind": "metadata",
                            },
                        ),
                    )
                )
        return rows
