"""从已发布 Paper/Chunk 确定性构建无需 LLM 的基础图。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from kg_crag.graph.config import GraphConfig
from kg_crag.graph.schema import corpus_snapshot, entity_id, fact_id, normalize_name
from kg_crag.indexing.processed import ProcessedPaper
from kg_crag.models import (
    GraphBundle,
    GraphEntity,
    GraphFact,
    GraphIdentity,
    GraphNodeType,
    GraphProvenance,
    GraphRelationType,
    Paper,
)


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def make_graph_identity(
    papers: list[ProcessedPaper], config: GraphConfig, *, model: str
) -> GraphIdentity:
    snapshot = corpus_snapshot(
        (
            item.paper_id,
            item.processing_version,
            item.paper_hash or _sha256_json([chunk.content_hash for chunk in item.chunks]),
        )
        for item in papers
    )
    versions = config.schema_config
    material = (
        versions.schema_version,
        snapshot,
        versions.metadata_builder_version,
        versions.selector_version,
        versions.extractor_version,
        versions.prompt_version,
        model,
        versions.model_revision,
        versions.normalization_version,
        versions.query_template_version,
    )
    version = hashlib.sha256("\0".join(material).encode()).hexdigest()
    return GraphIdentity(
        schema_version=versions.schema_version,
        graph_version=version,
        corpus_snapshot=snapshot,
        metadata_builder_version=versions.metadata_builder_version,
        selector_version=versions.selector_version,
        extractor_version=versions.extractor_version,
        prompt_version=versions.prompt_version,
        model=model,
        model_revision=versions.model_revision,
        normalization_version=versions.normalization_version,
        query_template_version=versions.query_template_version,
    )


def _paper_entity(paper: Paper) -> GraphEntity:
    return GraphEntity(
        entity_id=entity_id(GraphNodeType.PAPER, paper.title, external_id=paper.paper_id),
        node_type=GraphNodeType.PAPER,
        name=paper.title,
        normalized_name=normalize_name(paper.title),
        external_id=paper.paper_id,
        properties={"year": paper.year, "doi": paper.doi, "arxiv_id": paper.arxiv_id},
    )


def _reference_keys(paper: Paper) -> set[str]:
    return {
        normalize_name(item)
        for item in (paper.paper_id, paper.doi, paper.arxiv_id)
        if item is not None
    }


def reference_index(papers: list[ProcessedPaper]) -> dict[str, list[Paper]]:
    index: dict[str, list[Paper]] = {}
    for item in papers:
        if item.paper is None:
            continue
        for key in _reference_keys(item.paper):
            index.setdefault(key, []).append(item.paper)
    return index


def build_metadata_bundle(
    processed: ProcessedPaper,
    *,
    identity: GraphIdentity,
    references: dict[str, list[Paper]],
) -> GraphBundle:
    """构建一篇论文的完整基础图目标状态，并保留未解析引用。"""

    paper = processed.paper
    if paper is None or processed.paper_hash is None:
        raise ValueError(f"processed paper metadata is unavailable: {processed.paper_id}")
    paper_node = _paper_entity(paper)
    entities: dict[str, GraphEntity] = {paper_node.entity_id: paper_node}
    facts: list[GraphFact] = []
    provenance = GraphProvenance(
        source_kind="metadata",
        paper_id=paper.paper_id,
        source_id=paper.paper_id,
        source_hash=processed.paper_hash,
        confidence=1.0,
    )

    def add_fact(relation: GraphRelationType, source: GraphEntity, target: GraphEntity) -> None:
        entities[source.entity_id] = source
        entities[target.entity_id] = target
        facts.append(
            GraphFact(
                fact_id=fact_id(
                    identity.schema_version,
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
                graph_schema_version=identity.schema_version,
                corpus_snapshot=identity.corpus_snapshot,
            )
        )

    for author in paper.authors:
        author_node = GraphEntity(
            entity_id=entity_id(
                GraphNodeType.AUTHOR,
                author.name,
                external_id=author.author_id,
                paper_scope=paper.paper_id,
            ),
            node_type=GraphNodeType.AUTHOR,
            name=author.name,
            normalized_name=normalize_name(author.name),
            external_id=author.author_id,
            paper_scope=None if author.author_id else paper.paper_id,
        )
        add_fact(GraphRelationType.AUTHORED_BY, paper_node, author_node)
    for chunk in processed.chunks:
        chunk_node = GraphEntity(
            entity_id=entity_id(GraphNodeType.CHUNK, chunk.chunk_id, external_id=chunk.chunk_id),
            node_type=GraphNodeType.CHUNK,
            name=chunk.chunk_id,
            normalized_name=normalize_name(chunk.chunk_id),
            external_id=chunk.chunk_id,
            paper_scope=paper.paper_id,
            properties={
                "section": chunk.section,
                "page_start": chunk.page_start,
                "content_hash": chunk.content_hash,
            },
        )
        add_fact(GraphRelationType.CONTAINS, paper_node, chunk_node)

    unresolved: list[str] = []
    for reference in paper.references:
        matches = references.get(normalize_name(reference), [])
        unique = {item.paper_id: item for item in matches}
        if len(unique) != 1:
            unresolved.append(reference)
            continue
        target = _paper_entity(next(iter(unique.values())))
        add_fact(GraphRelationType.CITES, paper_node, target)
    unique_facts = {item.fact_id: item for item in facts}
    return GraphBundle(
        identity=identity,
        paper_id=paper.paper_id,
        entities=sorted(entities.values(), key=lambda item: item.entity_id),
        facts=sorted(unique_facts.values(), key=lambda item: item.fact_id),
        unresolved_references=sorted(unresolved),
    )


@dataclass(frozen=True)
class MetadataGraphPlan:
    identity: GraphIdentity
    bundles: tuple[GraphBundle, ...]


def plan_metadata_graph(
    papers: list[ProcessedPaper], config: GraphConfig, *, model: str
) -> MetadataGraphPlan:
    if len(papers) > config.selection.max_papers:
        raise ValueError("selected papers exceed graph max_papers")
    identity = make_graph_identity(papers, config, model=model)
    references = reference_index(papers)
    bundles = tuple(
        build_metadata_bundle(item, identity=identity, references=references) for item in papers
    )
    return MetadataGraphPlan(identity=identity, bundles=bundles)


def load_processed_paper(path: Path) -> Paper:
    return Paper.model_validate_json(path.read_text(encoding="utf-8"))
