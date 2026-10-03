"""图公共模型、身份函数和严格配置测试。"""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from kg_crag.graph import entity_id, fact_id, load_graph_config, normalize_name
from kg_crag.models import (
    GraphFact,
    GraphIdentity,
    GraphNodeType,
    GraphProvenance,
    GraphQueryRequest,
    GraphQueryTemplate,
    GraphRelationType,
)


def _identity() -> GraphIdentity:
    return GraphIdentity(
        schema_version="v1",
        graph_version="graph-v1",
        corpus_snapshot="a" * 64,
        metadata_builder_version="m1",
        selector_version="s1",
        extractor_version="e1",
        prompt_version="p1",
        model="mock",
        model_revision="r1",
        normalization_version="n1",
        query_template_version="q1",
    )


def test_graph_identity_and_fact_ids_are_stable_and_versioned() -> None:
    assert normalize_name("  BGE\u2013M3  ") == normalize_name("bge-m3")
    first = entity_id(GraphNodeType.MODEL, "BGE\u0301", paper_scope="p1")
    assert first == entity_id(GraphNodeType.MODEL, "BGÉ", paper_scope="p1")
    assert first != entity_id(GraphNodeType.MODEL, "BGÉ", paper_scope="p2")
    provenance = GraphProvenance(
        source_kind="chunk",
        paper_id="p1",
        source_id="c1",
        source_hash="b" * 64,
        extractor_version="e1",
        prompt_version="p1",
        created_at=datetime.now(UTC),
        evidence_quote="BGE-M3",
    )
    source = entity_id(GraphNodeType.METHOD, "retrieval", paper_scope="p1")
    target = entity_id(GraphNodeType.MODEL, "BGE-M3", paper_scope="p1")
    assert fact_id("v1", GraphRelationType.USES_MODEL, source, target, provenance) != fact_id(
        "v2", GraphRelationType.USES_MODEL, source, target, provenance
    )


def test_graph_models_reject_incomplete_provenance_and_illegal_endpoints() -> None:
    with pytest.raises(ValidationError, match="requires extractor"):
        GraphProvenance(
            source_kind="chunk",
            paper_id="p1",
            source_id="c1",
            source_hash="a" * 64,
        )
    provenance = GraphProvenance(
        source_kind="metadata",
        paper_id="p1",
        source_id="p1",
        source_hash="a" * 64,
    )
    with pytest.raises(ValidationError, match="endpoints"):
        GraphFact(
            fact_id="f1",
            relation=GraphRelationType.USES_MODEL,
            source_entity_id="paper",
            source_type=GraphNodeType.PAPER,
            target_entity_id="model",
            target_type=GraphNodeType.MODEL,
            provenance=provenance,
            graph_schema_version="v1",
            corpus_snapshot="a" * 64,
        )


def test_graph_query_has_no_raw_language_and_enforces_bounds() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        GraphQueryRequest.model_validate(
            {
                "graph_version": "v1",
                "template": "entity_neighbors",
                "entity_names": ["RAG"],
                "raw_query": "MATCH (n) DELETE n",
            }
        )
    with pytest.raises(ValidationError, match="only bounded_path"):
        GraphQueryRequest(
            graph_version="v1",
            template=GraphQueryTemplate.ENTITY_NEIGHBORS,
            entity_names=["RAG"],
            max_hops=2,
        )


def test_graph_config_loads_and_rejects_unknown_or_escaping_paths(tmp_path: Path) -> None:
    config = load_graph_config(Path("configs/retrieval.yaml"))
    assert config.extraction.max_requests == 100
    assert config.extraction.max_input_tokens == 160_000
    payload = (Path("configs/retrieval.yaml")).read_text(encoding="utf-8")
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(payload.replace("data/processed/graph-runs", "../escape"), encoding="utf-8")
    with pytest.raises(ValidationError, match="workspace-relative"):
        load_graph_config(invalid)


def test_identity_fixture_is_valid() -> None:
    assert _identity().corpus_snapshot == "a" * 64
