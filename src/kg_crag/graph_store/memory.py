"""默认离线测试使用的版本化确定性内存图存储。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from kg_crag.errors import KGCRAGError
from kg_crag.graph.schema import bundle_hash, normalize_name
from kg_crag.models import (
    Chunk,
    ErrorCode,
    ErrorDetail,
    Evidence,
    EvidenceLocation,
    EvidenceScores,
    EvidenceSourceType,
    GraphBundle,
    GraphIdentity,
    GraphPathHit,
    GraphQueryRequest,
    GraphQueryTemplate,
    GraphRecordState,
    GraphSyncResult,
    Paper,
)


@dataclass(frozen=True)
class GraphUpsertCall:
    paper_id: str
    chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class GraphRetrieveCall:
    entity_names: tuple[str, ...]
    max_hops: int
    top_k: int


@dataclass(frozen=True)
class GraphDeleteCall:
    paper_id: str


@dataclass(frozen=True)
class GraphQueryCall:
    request: GraphQueryRequest


GraphStoreCall = GraphUpsertCall | GraphRetrieveCall | GraphDeleteCall | GraphQueryCall


def _invalid_parameter(message: str, context: dict[str, str | int] | None = None) -> KGCRAGError:
    return KGCRAGError(
        ErrorDetail(
            code=ErrorCode.VALIDATION,
            message=message,
            retryable=False,
            context=context or {},
        )
    )


class InMemoryGraphStore:
    """图版本隔离、论文级覆盖同步和稳定路径查询的内存实现。"""

    def __init__(self) -> None:
        self._identity: GraphIdentity | None = None
        self._bundles: dict[str, GraphBundle] = {}
        self._papers: dict[str, tuple[Paper, tuple[Chunk, ...]]] = {}
        self.calls: list[GraphStoreCall] = []

    @property
    def paper_count(self) -> int:
        return len(set(self._papers) | set(self._bundles))

    async def ensure_graph(self, identity: GraphIdentity, *, rebuild: bool = False) -> None:
        if self._identity is None or rebuild:
            if rebuild:
                self._bundles.clear()
            self._identity = identity.model_copy(deep=True)
            return
        if self._identity != identity:
            raise _invalid_parameter("graph identity is incompatible")

    def _require_identity(self, graph_version: str | None = None) -> GraphIdentity:
        if self._identity is None:
            raise _invalid_parameter("graph identity has not been initialized")
        if graph_version is not None and self._identity.graph_version != graph_version:
            raise _invalid_parameter("requested graph version is unavailable")
        return self._identity

    async def record_state(self, paper_id: str) -> GraphRecordState | None:
        bundle = self._bundles.get(paper_id)
        if bundle is None:
            return None
        return GraphRecordState(
            paper_id=paper_id,
            bundle_hash=bundle_hash(bundle),
            entity_count=len(bundle.entities),
            fact_count=len(bundle.facts),
        )

    async def sync_paper(self, bundle: GraphBundle) -> GraphSyncResult:
        self._require_identity(bundle.identity.graph_version)
        if self._identity != bundle.identity:
            raise _invalid_parameter("bundle identity is incompatible with graph")
        digest = bundle_hash(bundle)
        previous = self._bundles.get(bundle.paper_id)
        if previous is not None and bundle_hash(previous) == digest:
            status = "unchanged"
        else:
            status = "updated" if previous is not None else "created"
            self._bundles[bundle.paper_id] = bundle.model_copy(deep=True)
        return GraphSyncResult(
            paper_id=bundle.paper_id,
            status=status,
            entity_count=len(bundle.entities),
            fact_count=len(bundle.facts),
        )

    async def query(self, request: GraphQueryRequest) -> list[GraphPathHit]:
        self._require_identity(request.graph_version)
        self.calls.append(GraphQueryCall(request.model_copy(deep=True)))
        entities = {
            entity.entity_id: entity
            for bundle in self._bundles.values()
            for entity in bundle.entities
        }
        facts = {fact.fact_id: fact for bundle in self._bundles.values() for fact in bundle.facts}
        start_ids = set(request.entity_ids)
        names = {normalize_name(name) for name in request.entity_names}
        start_ids.update(
            item_id
            for item_id, entity in entities.items()
            if entity.normalized_name in names or entity.name.casefold() in names
        )
        adjacency: dict[str, list[tuple[str, str]]] = {}
        for fact in facts.values():
            if fact.provenance.confidence < request.min_confidence:
                continue
            if request.relation_types and fact.relation not in request.relation_types:
                continue
            if request.paper_ids and fact.provenance.paper_id not in request.paper_ids:
                continue
            adjacency.setdefault(fact.source_entity_id, []).append(
                (fact.target_entity_id, fact.fact_id)
            )
            adjacency.setdefault(fact.target_entity_id, []).append(
                (fact.source_entity_id, fact.fact_id)
            )
        max_hops = request.max_hops if request.template is GraphQueryTemplate.BOUNDED_PATH else 1
        hits: list[GraphPathHit] = []
        for start in sorted(start_ids):
            frontier: list[tuple[str, list[str], list[str]]] = [(start, [start], [])]
            while frontier:
                current, path_entities, path_facts = frontier.pop(0)
                if path_facts:
                    terminal = entities[path_entities[-1]]
                    if not request.node_types or terminal.node_type in request.node_types:
                        provenances = [facts[item].provenance for item in path_facts]
                        key = json.dumps(
                            [
                                request.graph_version,
                                request.template.value,
                                path_entities,
                                path_facts,
                            ],
                            separators=(",", ":"),
                        )
                        hits.append(
                            GraphPathHit(
                                path_id=hashlib.sha256(key.encode()).hexdigest(),
                                graph_version=request.graph_version,
                                template=request.template,
                                entity_ids=path_entities,
                                fact_ids=path_facts,
                                provenances=provenances,
                                summary=" -> ".join(entities[item].name for item in path_entities),
                                score=min(item.confidence for item in provenances)
                                / len(path_facts),
                                hop_count=len(path_facts),
                            )
                        )
                if len(path_facts) >= max_hops:
                    continue
                for target, item_fact_id in sorted(adjacency.get(current, [])):
                    if target not in path_entities:
                        frontier.append(
                            (target, [*path_entities, target], [*path_facts, item_fact_id])
                        )
        unique = {item.path_id: item for item in hits}
        return sorted(
            unique.values(), key=lambda item: (-item.score, item.hop_count, item.path_id)
        )[: request.max_candidates]

    async def close(self) -> None:
        return None

    async def upsert_paper(self, paper: Paper, chunks: list[Chunk]) -> None:
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
        self, entity_names: list[str], *, max_hops: int, top_k: int
    ) -> list[Evidence]:
        if top_k <= 0:
            raise _invalid_parameter("top_k must be positive", {"top_k": top_k})
        if max_hops < 0:
            raise _invalid_parameter("max_hops must be non-negative", {"max_hops": max_hops})
        self.calls.append(GraphRetrieveCall(tuple(entity_names), max_hops, top_k))
        normalized = tuple(name.strip().casefold() for name in entity_names if name.strip())
        matches: list[tuple[Paper, Chunk]] = []
        for paper, chunks in self._papers.values():
            for chunk in chunks:
                searchable = f"{paper.title}\n{chunk.text}".casefold()
                if normalized and all(name in searchable for name in normalized):
                    matches.append((paper, chunk))
        matches.sort(key=lambda item: (item[0].paper_id, item[1].chunk_id))
        return [self._to_evidence(paper, chunk, max_hops) for paper, chunk in matches[:top_k]]

    async def delete_paper(self, paper_id: str) -> None:
        self.calls.append(GraphDeleteCall(paper_id))
        self._papers.pop(paper_id, None)
        self._bundles.pop(paper_id, None)

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
