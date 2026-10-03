"""代表 Chunk 选择、结构化事实抽取、缓存与预算控制。"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from kg_crag.graph.artifacts import atomic_write_json
from kg_crag.graph.config import GraphConfig
from kg_crag.graph.schema import entity_id, fact_id, normalize_name
from kg_crag.models import (
    Chunk,
    ExtractionUsage,
    GraphEntity,
    GraphExtractionResponse,
    GraphFact,
    GraphIdentity,
    GraphNodeType,
    GraphProvenance,
    GraphRelationType,
)
from kg_crag.providers import LLMProvider


@runtime_checkable
class GraphExtractionProvider(Protocol):
    async def extract(self, prompt: str) -> GraphExtractionResponse: ...


class MockGraphExtractionProvider:
    """返回预设严格载荷并记录调用，默认测试不访问网络。"""

    def __init__(self, response: GraphExtractionResponse | None = None) -> None:
        self.response = response or GraphExtractionResponse()
        self.calls: list[str] = []

    async def extract(self, prompt: str) -> GraphExtractionResponse:
        self.calls.append(prompt)
        return self.response.model_copy(deep=True)


class LLMGraphExtractionProvider:
    """复用通用 LLM 文本生成能力，但只在此边界解析图抽取载荷。"""

    def __init__(self, llm: LLMProvider, *, max_output_chars: int) -> None:
        self.llm = llm
        self.max_output_chars = max_output_chars

    async def extract(self, prompt: str) -> GraphExtractionResponse:
        response = await self.llm.generate(prompt, temperature=0.0)
        if len(response) > self.max_output_chars:
            raise ValueError("graph extraction response exceeds configured character limit")
        # 通用文本接口拿不到可信 Provider usage；忽略模型可能自行生成的 usage 字段。
        return GraphExtractionResponse.model_validate_json(response).model_copy(
            update={"usage": None}
        )


_SECTION_GROUPS = (
    ("abstract", "summary"),
    ("method", "approach", "architecture"),
    ("experiment", "evaluation", "result"),
    ("conclusion", "discussion"),
)


def select_representative_chunks(chunks: tuple[Chunk, ...], config: GraphConfig) -> list[Chunk]:
    """先覆盖关键章节，再按字符密度和原文顺序稳定补足。"""

    eligible = [
        item
        for item in chunks
        if config.selection.min_chunk_chars <= len(item.text) <= config.selection.max_chunk_chars
    ]
    selected: list[Chunk] = []
    selected_ids: set[str] = set()
    for aliases in _SECTION_GROUPS:
        matches = [
            item
            for item in eligible
            if any(alias in (item.section or "").casefold() for alias in aliases)
        ]
        if matches:
            chosen = min(matches, key=lambda item: (item.ordinal, item.chunk_id))
            if chosen.chunk_id not in selected_ids:
                selected.append(chosen)
                selected_ids.add(chosen.chunk_id)
    remaining = [item for item in eligible if item.chunk_id not in selected_ids]
    remaining.sort(
        key=lambda item: (
            -len(set(item.text.casefold().split())),
            item.ordinal,
            item.chunk_id,
        )
    )
    selected.extend(remaining[: max(0, config.selection.max_chunks_per_paper - len(selected))])
    selected = selected[: config.selection.max_chunks_per_paper]
    return sorted(selected, key=lambda item: (item.ordinal, item.chunk_id))


def extraction_prompt(template: str, chunk: Chunk) -> str:
    schema = (
        "节点: Paper,Author,Institution,Method,Model,Dataset,Task,Metric,Result,Chunk; "
        "关系必须使用项目 graph schema 白名单。"
    )
    return f"{template.rstrip()}\n\n{schema}\n\nChunk:\n{chunk.text}"


def extraction_cache_key(
    chunk: Chunk, identity: GraphIdentity, config: GraphConfig, *, model: str
) -> str:
    material = {
        "chunk_id": chunk.chunk_id,
        "content_hash": chunk.content_hash,
        "schema": identity.schema_version,
        "prompt": identity.prompt_version,
        "model": model,
        "revision": identity.model_revision,
        "extractor": identity.extractor_version,
        "max_output_chars": config.extraction.max_output_chars,
        "contract": "graph-extraction-response-v1",
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_and_materialize(
    response: GraphExtractionResponse,
    *,
    chunk: Chunk,
    identity: GraphIdentity,
    created_at: datetime | None = None,
) -> tuple[list[GraphEntity], list[GraphFact]]:
    """完整验证一个 Chunk 后一次性生成事实，禁止部分发布。"""

    normalized_text = normalize_name(chunk.text)
    entities: dict[str, GraphEntity] = {}
    facts: list[GraphFact] = []
    provenance = GraphProvenance(
        source_kind="chunk",
        paper_id=chunk.paper_id,
        source_id=chunk.chunk_id,
        source_hash=chunk.content_hash,
        confidence=1.0,
        extractor_version=identity.extractor_version,
        prompt_version=identity.prompt_version,
        created_at=created_at or datetime.now(UTC),
    )
    for candidate in response.candidates:
        if normalize_name(candidate.evidence_quote) not in normalized_text:
            raise ValueError("extraction evidence quote is not contained in the source chunk")
        if candidate.relation in {GraphRelationType.CITES, GraphRelationType.CONTAINS}:
            raise ValueError("metadata-owned relations cannot be created by text extraction")
        if candidate.target_type is GraphNodeType.PAPER:
            raise ValueError("text extraction cannot identify a target Paper safely")
        source_external_id = (
            chunk.paper_id if candidate.source_type is GraphNodeType.PAPER else None
        )
        source_id = entity_id(
            candidate.source_type,
            candidate.source_name,
            external_id=source_external_id,
            paper_scope=chunk.paper_id,
        )
        target_id = entity_id(
            candidate.target_type, candidate.target_name, paper_scope=chunk.paper_id
        )
        source = GraphEntity(
            entity_id=source_id,
            node_type=candidate.source_type,
            name=candidate.source_name,
            normalized_name=normalize_name(candidate.source_name),
            external_id=source_external_id,
            paper_scope=None if source_external_id else chunk.paper_id,
        )
        target = GraphEntity(
            entity_id=target_id,
            node_type=candidate.target_type,
            name=candidate.target_name,
            normalized_name=normalize_name(candidate.target_name),
            paper_scope=chunk.paper_id,
        )
        entities[source_id] = source
        entities[target_id] = target
        item_provenance = provenance.model_copy(
            update={
                "confidence": candidate.confidence,
                "evidence_quote": candidate.evidence_quote,
            }
        )
        facts.append(
            GraphFact(
                fact_id=fact_id(
                    identity.schema_version,
                    candidate.relation,
                    source_id,
                    target_id,
                    item_provenance,
                ),
                relation=candidate.relation,
                source_entity_id=source_id,
                source_type=candidate.source_type,
                target_entity_id=target_id,
                target_type=candidate.target_type,
                provenance=item_provenance,
                graph_schema_version=identity.schema_version,
                corpus_snapshot=identity.corpus_snapshot,
            )
        )
    return sorted(entities.values(), key=lambda item: item.entity_id), sorted(
        facts, key=lambda item: item.fact_id
    )


@dataclass
class ExtractionBudget:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def estimate_input(self, prompt: str, config: GraphConfig) -> int:
        return math.ceil(len(prompt) / config.extraction.chars_per_token_estimate)

    def can_spend(self, *, input_tokens: int, output_tokens: int, config: GraphConfig) -> bool:
        limits = config.extraction
        return (
            self.requests + 1 <= limits.max_requests
            and self.input_tokens + input_tokens <= limits.max_input_tokens
            and self.output_tokens + output_tokens <= limits.max_output_tokens
        )

    def spend(self, usage: ExtractionUsage) -> None:
        self.requests += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens

    def settle(self, reserved: ExtractionUsage, actual: ExtractionUsage) -> None:
        """以实际/响应估算用量替换调用前预留，但不重复增加请求数。"""

        self.input_tokens += actual.input_tokens - reserved.input_tokens
        self.output_tokens += actual.output_tokens - reserved.output_tokens


class ExtractionCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def load(self, key: str) -> tuple[GraphExtractionResponse, datetime] | None:
        path = self.root / key[:2] / f"{key}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            response = GraphExtractionResponse.model_validate(payload["response"])
            extracted_at = datetime.fromisoformat(payload["extracted_at"])
            if extracted_at.tzinfo is None:
                return None
            return response, extracted_at
        except (OSError, KeyError, TypeError, ValueError):
            return None

    def store(self, key: str, response: GraphExtractionResponse, *, extracted_at: datetime) -> None:
        atomic_write_json(
            self.root / key[:2] / f"{key}.json",
            {
                "response": response.model_dump(mode="json"),
                "extracted_at": extracted_at.isoformat(),
            },
        )
