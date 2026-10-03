"""基础图导入和有界正文抽取的可恢复编排。"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from kg_crag.errors import KGCRAGError
from kg_crag.graph.artifacts import atomic_write_json, workspace_path
from kg_crag.graph.config import GraphConfig
from kg_crag.graph.extraction import (
    ExtractionBudget,
    ExtractionCache,
    GraphExtractionProvider,
    extraction_cache_key,
    extraction_prompt,
    select_representative_chunks,
    validate_and_materialize,
)
from kg_crag.graph.metadata import MetadataGraphPlan, plan_metadata_graph
from kg_crag.graph.normalization import (
    apply_review_decisions,
    load_review_decisions,
    normalize_entities,
    write_review_report,
)
from kg_crag.graph.schema import fact_id
from kg_crag.graph_store import GraphStore
from kg_crag.indexing.processed import ProcessedPaper
from kg_crag.models import (
    EntityReviewItem,
    ExtractionUsage,
    GraphBundle,
    GraphEntity,
    GraphFact,
    GraphIdentity,
)


def _safe_error(error: Exception) -> dict[str, object]:
    if isinstance(error, KGCRAGError):
        return {
            "code": error.detail.code.value,
            "message": error.detail.message,
            "retryable": error.detail.retryable,
        }
    return {"code": "internal_error", "message": "graph item failed", "retryable": False}


class MetadataGraphPipeline:
    def __init__(self, config: GraphConfig, store: GraphStore, *, workspace_root: Path) -> None:
        self.config = config
        self.store = store
        self.workspace_root = workspace_root.resolve()

    async def run(
        self,
        plan: MetadataGraphPlan,
        *,
        dry_run: bool,
        rebuild: bool = False,
    ) -> dict[str, object]:
        """逐篇隔离同步，最后原子发布可恢复运行清单。"""

        started = datetime.now(UTC)
        planned = [
            {
                "paper_id": bundle.paper_id,
                "status": "planned",
                "entity_count": len(bundle.entities),
                "fact_count": len(bundle.facts),
                "unresolved_references": bundle.unresolved_references,
            }
            for bundle in plan.bundles
        ]
        if dry_run:
            return {
                "schema_version": "v1",
                "dry_run": True,
                "identity": plan.identity.model_dump(mode="json"),
                "config_hash": hashlib.sha256(self.config.model_dump_json().encode()).hexdigest(),
                "started_at": started.isoformat(),
                "finished_at": datetime.now(UTC).isoformat(),
                "items": planned,
                "database_writes": 0,
            }
        await self.store.ensure_graph(plan.identity, rebuild=rebuild)
        items: list[dict[str, object]] = []
        for bundle in plan.bundles:
            item_started = time.perf_counter()
            try:
                result = await self.store.sync_paper(bundle)
                payload = result.model_dump(mode="json")
            except Exception as error:
                payload = {
                    "paper_id": bundle.paper_id,
                    "status": "failed",
                    "entity_count": 0,
                    "fact_count": 0,
                    "error": _safe_error(error),
                }
            payload["latency_ms"] = (time.perf_counter() - item_started) * 1000
            payload["unresolved_references"] = bundle.unresolved_references
            items.append(payload)
        run_id = f"{started:%Y%m%dT%H%M%SZ}-{plan.identity.graph_version[:12]}"
        relative = f"{self.config.paths.manifest_root}/{run_id}/manifest.json"
        manifest: dict[str, object] = {
            "schema_version": "v1",
            "run_id": run_id,
            "dry_run": False,
            "identity": plan.identity.model_dump(mode="json"),
            "config_hash": hashlib.sha256(self.config.model_dump_json().encode()).hexdigest(),
            "started_at": started.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "items": items,
            "database_writes": sum(item.get("status") in {"created", "updated"} for item in items),
            "manifest_path": relative,
        }
        atomic_write_json(workspace_path(self.workspace_root, relative), manifest)
        return manifest


class GraphExtractionPipeline:
    def __init__(
        self,
        config: GraphConfig,
        provider: GraphExtractionProvider,
        *,
        workspace_root: Path,
        model: str,
        store: GraphStore | None = None,
    ) -> None:
        self.config = config
        self.provider = provider
        self.workspace_root = workspace_root.resolve()
        self.model = model
        self.store = store
        self.cache = ExtractionCache(workspace_path(self.workspace_root, config.paths.cache_root))

    def _normalized_bundles(
        self,
        papers: list[ProcessedPaper],
        identity: GraphIdentity,
        extracted_entities: dict[str, dict[str, GraphEntity]],
        extracted_facts: dict[str, dict[str, GraphFact]],
    ) -> tuple[list[GraphBundle], list[EntityReviewItem]]:
        """把基础事实与已完整校验的正文事实合成为论文级目标状态。"""

        plan = plan_metadata_graph(papers, self.config, model=self.model)
        if plan.identity != identity:
            raise ValueError("extraction identity does not match the selected corpus")
        all_entities = [
            entity
            for bundle in plan.bundles
            for entity in [*bundle.entities, *extracted_entities.get(bundle.paper_id, {}).values()]
        ]
        confidences: dict[str, float] = {}
        source_ids: dict[str, list[str]] = {}
        for paper_facts in extracted_facts.values():
            for fact in paper_facts.values():
                for entity_id in (fact.source_entity_id, fact.target_entity_id):
                    confidences[entity_id] = min(
                        confidences.get(entity_id, 1.0), fact.provenance.confidence
                    )
                    source_ids.setdefault(entity_id, []).append(fact.provenance.source_id)
        mapping, reviews = normalize_entities(
            all_entities,
            auto_merge_min_confidence=self.config.normalization.auto_merge_min_confidence,
            review_threshold=self.config.normalization.review_similarity_threshold,
            confidences=confidences,
            source_ids=source_ids,
        )
        decision_path = workspace_path(self.workspace_root, self.config.paths.review_decisions)
        if decision_path.is_file():
            mapping = apply_review_decisions(
                mapping,
                all_entities,
                load_review_decisions(decision_path),
                version=self.config.schema_config.normalization_version,
                available_source_ids={
                    fact.provenance.source_id
                    for facts in extracted_facts.values()
                    for fact in facts.values()
                }
                | {bundle.paper_id for bundle in plan.bundles},
                known_review_ids={item.review_id for item in reviews},
            )
        canonical: dict[str, GraphEntity] = {}
        for entity in all_entities:
            # 基础图实体排在前面；ID 相同时保留其可信标题和结构化属性。
            canonical.setdefault(entity.entity_id, entity)
        bundles: list[GraphBundle] = []
        for base in plan.bundles:
            original_facts = [*base.facts, *extracted_facts.get(base.paper_id, {}).values()]
            facts: dict[str, GraphFact] = {}
            required_ids = {mapping[entity.entity_id] for entity in base.entities}
            for original in original_facts:
                source_id = mapping[original.source_entity_id]
                target_id = mapping[original.target_entity_id]
                rewritten = original.model_copy(
                    update={
                        "fact_id": fact_id(
                            identity.schema_version,
                            original.relation,
                            source_id,
                            target_id,
                            original.provenance,
                        ),
                        "source_entity_id": source_id,
                        "target_entity_id": target_id,
                    }
                )
                facts[rewritten.fact_id] = rewritten
                required_ids.update((source_id, target_id))
            bundles.append(
                GraphBundle(
                    identity=identity,
                    paper_id=base.paper_id,
                    entities=sorted(
                        (canonical[entity_id] for entity_id in required_ids),
                        key=lambda entity: entity.entity_id,
                    ),
                    facts=sorted(facts.values(), key=lambda fact: fact.fact_id),
                    unresolved_references=base.unresolved_references,
                )
            )
        return bundles, reviews

    async def run(
        self,
        papers: list[ProcessedPaper],
        identity: GraphIdentity,
        *,
        dry_run: bool,
    ) -> dict[str, object]:
        started = datetime.now(UTC)
        if len(papers) > self.config.selection.max_papers:
            raise ValueError("graph extraction selection exceeds max_papers")
        template_path = workspace_path(self.workspace_root, self.config.paths.prompt_path)
        template = template_path.read_text(encoding="utf-8")
        budget = ExtractionBudget()
        items: list[dict[str, object]] = []
        extracted_entities: dict[str, dict[str, GraphEntity]] = {}
        extracted_facts: dict[str, dict[str, GraphFact]] = {}
        for paper in papers:
            selected = select_representative_chunks(paper.chunks, self.config)
            if dry_run or not self.config.extraction.enabled:
                items.extend(
                    {
                        "paper_id": paper.paper_id,
                        "chunk_id": chunk.chunk_id,
                        "status": "planned" if dry_run else "disabled",
                        "latency_ms": 0.0,
                    }
                    for chunk in selected
                )
                continue
            for chunk in selected:
                item_started = time.perf_counter()
                key = extraction_cache_key(chunk, identity, self.config, model=self.model)
                cached = self.cache.load(key)
                if cached is not None:
                    cached_response, extracted_at = cached
                    try:
                        entities, facts = validate_and_materialize(
                            cached_response,
                            chunk=chunk,
                            identity=identity,
                            created_at=extracted_at,
                        )
                    except ValueError:
                        cached = None
                    else:
                        extracted_entities.setdefault(paper.paper_id, {}).update(
                            {entity.entity_id: entity for entity in entities}
                        )
                        extracted_facts.setdefault(paper.paper_id, {}).update(
                            {fact.fact_id: fact for fact in facts}
                        )
                        items.append(
                            {
                                "paper_id": paper.paper_id,
                                "chunk_id": chunk.chunk_id,
                                "status": "cached",
                                "entity_count": len(entities),
                                "fact_count": len(facts),
                                "cache_key": key,
                                "latency_ms": (time.perf_counter() - item_started) * 1000,
                            }
                        )
                        continue
                prompt = extraction_prompt(template, chunk)
                estimated_input = budget.estimate_input(prompt, self.config)
                estimated_output = budget.estimate_input(
                    "x" * self.config.extraction.max_output_chars, self.config
                )
                if not budget.can_spend(
                    input_tokens=estimated_input,
                    output_tokens=estimated_output,
                    config=self.config,
                ):
                    items.append(
                        {
                            "paper_id": paper.paper_id,
                            "chunk_id": chunk.chunk_id,
                            "status": "budget_exhausted",
                            "latency_ms": (time.perf_counter() - item_started) * 1000,
                        }
                    )
                    continue
                # Provider 调用本身也可能失败，因此先预留并计入一次请求；失败调用同样受硬上限约束。
                reserved_usage = ExtractionUsage(
                    input_tokens=estimated_input,
                    output_tokens=estimated_output,
                    estimated=True,
                )
                budget.spend(reserved_usage)
                try:
                    response = await self.provider.extract(prompt)
                    extracted_at = datetime.now(UTC)
                    entities, facts = validate_and_materialize(
                        response,
                        chunk=chunk,
                        identity=identity,
                        created_at=extracted_at,
                    )
                    usage = response.usage or ExtractionUsage(
                        input_tokens=estimated_input,
                        output_tokens=min(
                            estimated_output,
                            max(
                                1,
                                len(response.model_dump_json())
                                // self.config.extraction.chars_per_token_estimate,
                            ),
                        ),
                        estimated=True,
                    )
                    budget.settle(reserved_usage, usage)
                    if (
                        budget.input_tokens > self.config.extraction.max_input_tokens
                        or budget.output_tokens > self.config.extraction.max_output_tokens
                    ):
                        raise ValueError("provider usage exceeds graph extraction budget")
                    self.cache.store(key, response, extracted_at=extracted_at)
                    extracted_entities.setdefault(paper.paper_id, {}).update(
                        {entity.entity_id: entity for entity in entities}
                    )
                    extracted_facts.setdefault(paper.paper_id, {}).update(
                        {fact.fact_id: fact for fact in facts}
                    )
                    items.append(
                        {
                            "paper_id": paper.paper_id,
                            "chunk_id": chunk.chunk_id,
                            "status": "succeeded",
                            "entity_count": len(entities),
                            "fact_count": len(facts),
                            "cache_key": key,
                            "usage": usage.model_dump(mode="json"),
                            "latency_ms": (time.perf_counter() - item_started) * 1000,
                        }
                    )
                except Exception as error:
                    items.append(
                        {
                            "paper_id": paper.paper_id,
                            "chunk_id": chunk.chunk_id,
                            "status": "failed",
                            "error": _safe_error(error),
                            "usage": reserved_usage.model_dump(mode="json"),
                            "latency_ms": (time.perf_counter() - item_started) * 1000,
                        }
                    )
        identity_material = json.dumps(
            [identity.graph_version, self.model, dry_run, self.config.extraction.enabled],
            separators=(",", ":"),
        )
        run_id = hashlib.sha256(identity_material.encode()).hexdigest()[:20]
        bundle_items: list[dict[str, object]] = []
        review_count = 0
        review_path_relative: str | None = None
        if not dry_run and self.config.extraction.enabled:
            bundles, reviews = self._normalized_bundles(
                papers, identity, extracted_entities, extracted_facts
            )
            review_count = len(reviews)
            review_path_relative = f"{self.config.paths.review_root}/{run_id}.json"
            review_path = workspace_path(self.workspace_root, review_path_relative)
            write_review_report(
                review_path,
                reviews,
                normalization_version=self.config.schema_config.normalization_version,
            )
            if self.store is not None:
                await self.store.ensure_graph(identity)
            for bundle in bundles:
                relative = (
                    f"{self.config.paths.manifest_root}/extraction-{run_id}/bundles/"
                    f"{hashlib.sha256(bundle.paper_id.encode()).hexdigest()}.json"
                )
                atomic_write_json(workspace_path(self.workspace_root, relative), bundle)
                state: dict[str, object] = {
                    "paper_id": bundle.paper_id,
                    "bundle_path": relative,
                    "entity_count": len(bundle.entities),
                    "fact_count": len(bundle.facts),
                    "status": "published",
                }
                if self.store is not None:
                    try:
                        state.update((await self.store.sync_paper(bundle)).model_dump(mode="json"))
                    except Exception as error:
                        state.update(status="failed", error=_safe_error(error))
                bundle_items.append(state)
        manifest = {
            "schema_version": "v1",
            "run_id": run_id,
            "dry_run": dry_run,
            "identity": identity.model_dump(mode="json"),
            "graph_version": identity.graph_version,
            "config_hash": hashlib.sha256(self.config.model_dump_json().encode()).hexdigest(),
            "model": self.model,
            "started_at": started.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "budget": {
                "max_requests": self.config.extraction.max_requests,
                "max_input_tokens": self.config.extraction.max_input_tokens,
                "max_output_tokens": self.config.extraction.max_output_tokens,
            },
            "usage": {
                "requests": budget.requests,
                "input_tokens": budget.input_tokens,
                "output_tokens": budget.output_tokens,
            },
            "review_count": review_count,
            "review_path": review_path_relative,
            "bundles": bundle_items,
            "database_writes": sum(
                item.get("status") in {"created", "updated"} for item in bundle_items
            ),
            "items": items,
        }
        if not dry_run:
            relative = f"{self.config.paths.manifest_root}/extraction-{run_id}/manifest.json"
            atomic_write_json(workspace_path(self.workspace_root, relative), manifest)
        return manifest
