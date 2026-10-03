"""图存储合同、只读模板和 Evidence 溯源测试。"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from kg_crag.graph.config import load_graph_config
from kg_crag.graph.metadata import plan_metadata_graph
from kg_crag.graph.schema import entity_id, fact_id, normalize_name
from kg_crag.graph_store import InMemoryGraphStore
from kg_crag.graph_store.memory import GraphQueryCall
from kg_crag.graph_store.queries import QUERY_TEMPLATES
from kg_crag.indexing.processed import ProcessedPaper
from kg_crag.models import (
    Chunk,
    GraphBundle,
    GraphEntity,
    GraphFact,
    GraphNodeType,
    GraphProvenance,
    GraphQueryRequest,
    GraphQueryTemplate,
    GraphRelationType,
    Paper,
)
from kg_crag.retrieval import GraphRetriever, ProcessedSourceResolver


def _processed() -> ProcessedPaper:
    paper = Paper(paper_id="p1", title="Agent Retrieval")
    chunk = Chunk(
        chunk_id="c1",
        paper_id="p1",
        section="Methods",
        page_start=2,
        text="The CRAG method uses BGE-M3 for retrieval.",
        token_count=8,
        content_hash="c" * 64,
        processing_version="pv1",
    )
    return ProcessedPaper("p1", "pv1", (chunk,), Path("."), paper, "a" * 64)


def _bundle_with_chunk_fact() -> tuple[ProcessedPaper, GraphBundle]:
    config = load_graph_config(Path("configs/retrieval.yaml"))
    processed = _processed()
    base = plan_metadata_graph([processed], config, model="mock")
    bundle = base.bundles[0]
    paper_node = next(item for item in bundle.entities if item.node_type is GraphNodeType.PAPER)
    method = GraphEntity(
        entity_id=entity_id(GraphNodeType.METHOD, "CRAG", paper_scope="p1"),
        node_type=GraphNodeType.METHOD,
        name="CRAG",
        normalized_name=normalize_name("CRAG"),
        paper_scope="p1",
    )
    model = GraphEntity(
        entity_id=entity_id(GraphNodeType.MODEL, "BGE-M3", paper_scope="p1"),
        node_type=GraphNodeType.MODEL,
        name="BGE-M3",
        normalized_name=normalize_name("BGE-M3"),
        paper_scope="p1",
    )
    metadata = GraphProvenance(
        source_kind="metadata",
        paper_id="p1",
        source_id="p1",
        source_hash="a" * 64,
    )
    chunk_source = GraphProvenance(
        source_kind="chunk",
        paper_id="p1",
        source_id="c1",
        source_hash="c" * 64,
        confidence=0.95,
        extractor_version="e1",
        prompt_version="p1",
        created_at=datetime.now(UTC),
        evidence_quote="uses BGE-M3",
    )

    def make_fact(
        relation: GraphRelationType,
        source: GraphEntity,
        target: GraphEntity,
        provenance: GraphProvenance,
    ) -> GraphFact:
        return GraphFact(
            fact_id=fact_id(
                base.identity.schema_version,
                relation,
                source.entity_id,
                target.entity_id,
                provenance,
            ),
            relation=relation,
            source_entity_id=source.entity_id,
            source_type=source.node_type,
            target_entity_id=target.entity_id,
            target_type=target.node_type,
            provenance=provenance,
            graph_schema_version=base.identity.schema_version,
            corpus_snapshot=base.identity.corpus_snapshot,
        )

    facts = [
        *bundle.facts,
        make_fact(GraphRelationType.HAS_METHOD, paper_node, method, metadata),
        make_fact(GraphRelationType.USES_MODEL, method, model, chunk_source),
    ]
    return processed, bundle.model_copy(
        update={"entities": [*bundle.entities, method, model], "facts": facts}
    )


async def test_memory_store_sync_is_idempotent_and_queries_bounded_paths() -> None:
    _processed_item, bundle = _bundle_with_chunk_fact()
    store = InMemoryGraphStore()
    await store.ensure_graph(bundle.identity)
    first = await store.sync_paper(bundle)
    second = await store.sync_paper(bundle)
    request = GraphQueryRequest(
        graph_version=bundle.identity.graph_version,
        template=GraphQueryTemplate.BOUNDED_PATH,
        entity_names=["Agent Retrieval"],
        max_hops=2,
        top_k=10,
    )
    hits = await store.query(request)
    assert first.status == "created"
    assert second.status == "unchanged"
    assert any(item.hop_count == 2 for item in hits)
    assert sum(isinstance(item, GraphQueryCall) for item in store.calls) == 1


async def test_graph_retriever_returns_chunk_and_metadata_evidence() -> None:
    processed, bundle = _bundle_with_chunk_fact()
    store = InMemoryGraphStore()
    await store.ensure_graph(bundle.identity)
    await store.sync_paper(bundle)
    request = GraphQueryRequest(
        graph_version=bundle.identity.graph_version,
        template=GraphQueryTemplate.BOUNDED_PATH,
        entity_names=["Agent Retrieval"],
        max_hops=2,
        top_k=20,
    )
    result = await GraphRetriever(store, ProcessedSourceResolver([processed])).retrieve(request)
    assert result.database_calls == 1
    assert any(item.source_id == "c1" for item in result.evidence)
    chunk_evidence = next(item for item in result.evidence if item.source_id == "c1")
    assert chunk_evidence.external is False
    assert chunk_evidence.location.section == "Methods"
    assert chunk_evidence.metadata["source_kind"] == "chunk"


async def test_graph_retriever_rejects_whole_path_on_source_drift() -> None:
    processed, bundle = _bundle_with_chunk_fact()
    drifted_chunk = processed.chunks[0].model_copy(update={"content_hash": "d" * 64})
    drifted = ProcessedPaper(
        processed.paper_id,
        processed.processing_version,
        (drifted_chunk,),
        processed.directory,
        processed.paper,
        processed.paper_hash,
    )
    store = InMemoryGraphStore()
    await store.ensure_graph(bundle.identity)
    await store.sync_paper(bundle)
    result = await GraphRetriever(store, ProcessedSourceResolver([drifted])).retrieve(
        GraphQueryRequest(
            graph_version=bundle.identity.graph_version,
            template=GraphQueryTemplate.BOUNDED_PATH,
            entity_names=["CRAG"],
            max_hops=1,
        )
    )
    assert result.rejected_paths >= 1
    assert all(item.source_id != "c1" for item in result.evidence)


def test_cypher_templates_are_read_only_bounded_and_parameterized() -> None:
    forbidden = (" CREATE ", " DELETE ", " SET ", " MERGE ", " CALL ")
    for query in QUERY_TEMPLATES.values():
        normalized = f" {query.upper()} "
        assert not any(keyword in normalized for keyword in forbidden)
        assert "$graph_version" in query
        assert "$max_candidates" in query
        assert "LIMIT $max_candidates" in query
        assert "[*" not in query


@pytest.mark.parametrize("max_hops", [0, 4])
def test_query_rejects_hop_bounds_before_store(max_hops: int) -> None:
    with pytest.raises(ValueError):
        GraphQueryRequest(
            graph_version="v1",
            template=GraphQueryTemplate.BOUNDED_PATH,
            entity_names=["CRAG"],
            max_hops=max_hops,
        )
