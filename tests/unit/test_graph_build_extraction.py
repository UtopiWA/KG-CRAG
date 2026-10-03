"""基础图、代表 Chunk、抽取预算和保守规范化测试。"""

from pathlib import Path

import pytest

from kg_crag.graph.config import load_graph_config
from kg_crag.graph.extraction import (
    GraphExtractionProvider,
    MockGraphExtractionProvider,
    select_representative_chunks,
    validate_and_materialize,
)
from kg_crag.graph.metadata import plan_metadata_graph
from kg_crag.graph.normalization import (
    apply_review_decisions,
    load_review_decisions,
    normalize_entities,
    write_review_report,
)
from kg_crag.graph.pipeline import GraphExtractionPipeline, MetadataGraphPipeline
from kg_crag.graph_store import InMemoryGraphStore
from kg_crag.indexing.processed import ProcessedPaper, select_processed
from kg_crag.models import (
    Author,
    Chunk,
    EntityReviewDecision,
    GraphEntity,
    GraphExtractionCandidate,
    GraphExtractionResponse,
    GraphNodeType,
    GraphQueryRequest,
    GraphQueryTemplate,
    GraphRelationType,
    Paper,
)
from kg_crag.retrieval.graph import GraphRetriever, ProcessedSourceResolver


def _chunk(chunk_id: str, paper_id: str, section: str, ordinal: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        section=section,
        page_start=ordinal + 1,
        text=text,
        token_count=max(1, len(text.split())),
        content_hash=(chunk_id[-1] if chunk_id[-1].isalnum() else "a") * 64,
        ordinal=ordinal,
        processing_version="pv1",
    )


def _processed(paper: Paper, chunks: tuple[Chunk, ...]) -> ProcessedPaper:
    return ProcessedPaper(paper.paper_id, "pv1", chunks, Path("."), paper, "f" * 64)


class _FailingProvider(GraphExtractionProvider):
    def __init__(self) -> None:
        self.calls = 0

    async def extract(self, prompt: str) -> GraphExtractionResponse:
        self.calls += 1
        raise RuntimeError("secret provider failure")


def test_metadata_builder_resolves_only_explicit_unique_references() -> None:
    config = load_graph_config(Path("configs/retrieval.yaml"))
    target = Paper(paper_id="p2", title="Target", doi="10.1/target")
    source = Paper(
        paper_id="p1",
        title="Source",
        authors=[Author(author_id="a1", name="Ada")],
        references=["10.1/target", "missing"],
    )
    papers = [
        _processed(source, (_chunk("c1", "p1", "Abstract", 0, "A" * 250),)),
        _processed(target, (_chunk("c2", "p2", "Methods", 0, "B" * 250),)),
    ]
    bounded = config.model_copy(
        update={"selection": config.selection.model_copy(update={"max_papers": 15})}
    )
    plan = plan_metadata_graph(papers, bounded, model="mock")
    bundle = plan.bundles[0]
    assert {fact.relation for fact in bundle.facts} >= {
        GraphRelationType.AUTHORED_BY,
        GraphRelationType.CONTAINS,
        GraphRelationType.CITES,
    }
    assert bundle.unresolved_references == ["missing"]


def test_selection_is_deterministic_bounded_and_does_not_duplicate() -> None:
    config = load_graph_config(Path("configs/retrieval.yaml"))
    chunks = tuple(
        _chunk(f"c{index}", "p1", section, index, (section + " evidence ") * 30)
        for index, section in enumerate(
            ["Abstract", "Methods", "Experiments", "Results", "Conclusion", "Appendix"]
        )
    )
    selected = select_representative_chunks(chunks, config)
    assert 3 <= len(selected) <= 5
    assert selected == select_representative_chunks(tuple(reversed(chunks)), config)
    assert len({item.chunk_id for item in selected}) == len(selected)


def test_extraction_rejects_quote_outside_chunk_atomically() -> None:
    config = load_graph_config(Path("configs/retrieval.yaml"))
    paper = Paper(paper_id="p1", title="Source")
    chunk = _chunk("c1", "p1", "Methods", 0, "Our method uses BGE-M3.")
    identity = plan_metadata_graph([_processed(paper, (chunk,))], config, model="mock").identity
    response = GraphExtractionResponse(
        candidates=[
            GraphExtractionCandidate(
                source_type=GraphNodeType.METHOD,
                source_name="method",
                relation=GraphRelationType.USES_MODEL,
                target_type=GraphNodeType.MODEL,
                target_name="BGE-M3",
                evidence_quote="not in source",
                confidence=0.9,
            )
        ]
    )
    with pytest.raises(ValueError, match="not contained"):
        validate_and_materialize(response, chunk=chunk, identity=identity)


async def test_metadata_pipeline_is_idempotent_and_dry_run_has_no_writes(tmp_path: Path) -> None:
    config = load_graph_config(Path("configs/retrieval.yaml"))
    config = config.model_copy(
        update={"paths": config.paths.model_copy(update={"manifest_root": "runs"})}
    )
    paper = Paper(paper_id="p1", title="Source")
    plan = plan_metadata_graph(
        [_processed(paper, (_chunk("c1", "p1", "Abstract", 0, "A" * 250),))],
        config,
        model="mock",
    )
    store = InMemoryGraphStore()
    pipeline = MetadataGraphPipeline(config, store, workspace_root=tmp_path)
    dry = await pipeline.run(plan, dry_run=True)
    assert dry["database_writes"] == 0
    assert await pipeline.run(plan, dry_run=False)
    second = await pipeline.run(plan, dry_run=False)
    assert second["items"][0]["status"] == "unchanged"  # type: ignore[index]


async def test_extraction_dry_run_and_disabled_mode_never_call_provider(tmp_path: Path) -> None:
    config = load_graph_config(Path("configs/retrieval.yaml"))
    config = config.model_copy(
        update={
            "paths": config.paths.model_copy(
                update={"prompt_path": "prompt.txt", "cache_root": "cache", "manifest_root": "runs"}
            ),
            "selection": config.selection.model_copy(update={"min_chunk_chars": 1}),
        }
    )
    (tmp_path / "prompt.txt").write_text("extract", encoding="utf-8")
    paper = Paper(paper_id="p1", title="Source")
    processed = _processed(paper, (_chunk("c1", "p1", "Methods", 0, "uses BGE-M3"),))
    identity = plan_metadata_graph([processed], config, model="mock").identity
    provider = MockGraphExtractionProvider()
    pipeline = GraphExtractionPipeline(config, provider, workspace_root=tmp_path, model="mock")
    dry = await pipeline.run([processed], identity, dry_run=True)
    disabled = await pipeline.run([processed], identity, dry_run=False)
    assert dry["usage"]["requests"] == disabled["usage"]["requests"] == 0  # type: ignore[index]
    assert provider.calls == []


async def test_failed_provider_call_consumes_request_budget_without_retry(tmp_path: Path) -> None:
    config = load_graph_config(Path("configs/retrieval.yaml"))
    config = config.model_copy(
        update={
            "paths": config.paths.model_copy(
                update={"prompt_path": "prompt.txt", "cache_root": "cache", "manifest_root": "runs"}
            ),
            "selection": config.selection.model_copy(
                update={"min_chunk_chars": 1, "max_chunks_per_paper": 3}
            ),
            "extraction": config.extraction.model_copy(update={"enabled": True, "max_requests": 1}),
        }
    )
    (tmp_path / "prompt.txt").write_text("extract", encoding="utf-8")
    paper = Paper(paper_id="p1", title="Source")
    processed = _processed(
        paper,
        (
            _chunk("c1", "p1", "Methods", 0, "uses BGE-M3"),
            _chunk("c2", "p1", "Results", 1, "reports a result"),
        ),
    )
    identity = plan_metadata_graph([processed], config, model="mock").identity
    provider = _FailingProvider()
    manifest = await GraphExtractionPipeline(
        config, provider, workspace_root=tmp_path, model="mock"
    ).run([processed], identity, dry_run=False)

    assert provider.calls == 1
    usage = manifest["usage"]
    items = manifest["items"]
    assert isinstance(usage, dict)
    assert isinstance(items, list)
    assert usage["requests"] == 1
    statuses = [item["status"] for item in items if isinstance(item, dict)]
    assert statuses == ["failed", "budget_exhausted"]
    assert "secret" not in str(manifest)


async def test_successful_extraction_publishes_stable_bundle_and_reuses_cache(
    tmp_path: Path,
) -> None:
    config = load_graph_config(Path("configs/retrieval.yaml"))
    config = config.model_copy(
        update={
            "paths": config.paths.model_copy(
                update={
                    "prompt_path": "prompt.txt",
                    "cache_root": "cache",
                    "manifest_root": "runs",
                    "review_root": "reviews",
                }
            ),
            "selection": config.selection.model_copy(update={"min_chunk_chars": 1}),
            "extraction": config.extraction.model_copy(update={"enabled": True}),
        }
    )
    (tmp_path / "prompt.txt").write_text("extract", encoding="utf-8")
    paper = Paper(paper_id="p1", title="Source")
    processed = _processed(
        paper,
        (_chunk("c1", "p1", "Methods", 0, "Source proposes Agent Retrieval."),),
    )
    response = GraphExtractionResponse(
        candidates=[
            GraphExtractionCandidate(
                source_type=GraphNodeType.PAPER,
                source_name="Source",
                relation=GraphRelationType.HAS_METHOD,
                target_type=GraphNodeType.METHOD,
                target_name="Agent Retrieval",
                evidence_quote="Source proposes Agent Retrieval.",
                confidence=0.95,
            )
        ]
    )
    identity = plan_metadata_graph([processed], config, model="mock").identity
    provider = MockGraphExtractionProvider(response)
    store = InMemoryGraphStore()
    pipeline = GraphExtractionPipeline(
        config,
        provider,
        workspace_root=tmp_path,
        model="mock",
        store=store,
    )
    first = await pipeline.run([processed], identity, dry_run=False)
    first_state = await store.record_state("p1")
    second = await pipeline.run([processed], identity, dry_run=False)
    second_state = await store.record_state("p1")

    assert len(provider.calls) == 1
    assert first["usage"]["requests"] == 1  # type: ignore[index]
    assert second["usage"]["requests"] == 0  # type: ignore[index]
    assert second["items"][0]["status"] == "cached"  # type: ignore[index]
    assert second["bundles"][0]["status"] == "unchanged"  # type: ignore[index]
    assert first_state == second_state
    result = await GraphRetriever(store, ProcessedSourceResolver([processed])).retrieve(
        GraphQueryRequest(
            graph_version=identity.graph_version,
            template=GraphQueryTemplate.RELATION_LOOKUP,
            entity_names=["Source"],
            relation_types=[GraphRelationType.HAS_METHOD],
        )
    )
    assert result.database_calls == 1
    assert result.evidence[0].source_id == "c1"
    assert result.evidence[0].external is False


def test_normalization_keeps_ambiguity_separate_and_replays_safe_decision() -> None:
    entities = [
        GraphEntity(
            entity_id="m1", node_type=GraphNodeType.MODEL, name="BGE", normalized_name="bge"
        ),
        GraphEntity(
            entity_id="d1", node_type=GraphNodeType.DATASET, name="BGE", normalized_name="bge"
        ),
    ]
    mapping, reviews = normalize_entities(
        entities, auto_merge_min_confidence=0.9, review_threshold=0.8
    )
    assert mapping == {"m1": "m1", "d1": "d1"}
    assert reviews[0].reason == "cross_type_name"
    decision = EntityReviewDecision(
        review_id=reviews[0].review_id,
        left_entity_id="m1",
        right_entity_id="d1",
        decision="merge",
        reason="manual",
        normalization_version="n1",
        source_ids=["chunk-1"],
    )
    with pytest.raises(ValueError, match="different entity types"):
        apply_review_decisions(mapping, entities, [decision], version="n1")


def test_normalization_flags_abbreviation_low_confidence_and_stale_decision(
    tmp_path: Path,
) -> None:
    entities = [
        GraphEntity(
            entity_id="m1",
            node_type=GraphNodeType.METHOD,
            name="Large Language Model",
            normalized_name="large language model",
        ),
        GraphEntity(
            entity_id="m2", node_type=GraphNodeType.METHOD, name="LLM", normalized_name="llm"
        ),
    ]
    mapping, reviews = normalize_entities(
        entities,
        auto_merge_min_confidence=0.9,
        review_threshold=0.8,
        confidences={"m1": 0.7, "m2": 1.0},
        source_ids={"m1": ["chunk-1"], "m2": ["chunk-2"]},
    )
    assert mapping == {"m1": "m1", "m2": "m2"}
    assert reviews[0].reason == "abbreviation"
    assert reviews[0].source_ids == ["chunk-1", "chunk-2"]

    low_confidence_entities = [
        GraphEntity(
            entity_id="x1",
            node_type=GraphNodeType.METHOD,
            name="Agent Retrieval",
            normalized_name="agent retrieval",
        ),
        GraphEntity(
            entity_id="x2",
            node_type=GraphNodeType.METHOD,
            name="Agent-Retrieval",
            normalized_name="agent retrieval",
        ),
    ]
    low_mapping, low_reviews = normalize_entities(
        low_confidence_entities,
        auto_merge_min_confidence=0.9,
        review_threshold=0.8,
        confidences={"x1": 0.6, "x2": 1.0},
    )
    assert low_mapping == {"x1": "x1", "x2": "x2"}
    assert low_reviews[0].reason == "low_confidence"

    decision_path = tmp_path / "decisions.json"
    decision_path.write_text(
        '{"decisions":[{"review_id":"r1","left_entity_id":"m1",'
        '"right_entity_id":"m2","decision":"merge","reason":"manual",'
        '"normalization_version":"n1","source_ids":["missing"]}]}',
        encoding="utf-8",
    )
    decisions = load_review_decisions(decision_path)
    with pytest.raises(ValueError, match="stale source"):
        apply_review_decisions(
            mapping,
            entities,
            decisions,
            version="n1",
            available_source_ids={"chunk-1", "chunk-2"},
        )


def test_review_report_is_stable_and_contains_no_source_text(tmp_path: Path) -> None:
    entities = [
        GraphEntity(
            entity_id="m1", node_type=GraphNodeType.MODEL, name="BGE", normalized_name="bge"
        ),
        GraphEntity(
            entity_id="d1", node_type=GraphNodeType.DATASET, name="BGE", normalized_name="bge"
        ),
    ]
    _mapping, reviews = normalize_entities(
        entities, auto_merge_min_confidence=0.9, review_threshold=0.8
    )
    destination = tmp_path / "review.json"
    write_review_report(destination, reviews, normalization_version="n1")
    first = destination.read_bytes()
    write_review_report(destination, list(reversed(reviews)), normalization_version="n1")
    assert destination.read_bytes() == first
    assert b"complete chunk text" not in first


def test_all_selection_requires_limit_before_work() -> None:
    with pytest.raises(ValueError, match="explicit limit"):
        select_processed([], select_all=True, max_papers=15)
