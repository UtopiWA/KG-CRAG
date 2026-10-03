"""知识图谱流水线的严格、资源有界配置。"""

from __future__ import annotations

from pathlib import Path, PurePath
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import Field, field_validator, model_validator

from kg_crag.models.domain import StrictModel


def _relative_path(value: str) -> str:
    path = PurePath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("graph paths must be workspace-relative and contained")
    return path.as_posix()


class GraphSchemaConfig(StrictModel):
    schema_version: Literal["v1"] = "v1"
    metadata_builder_version: str = "metadata-v1"
    selector_version: str = "representative-v1"
    extractor_version: str = "structured-v1"
    prompt_version: str = "graph-extraction-v1"
    normalization_version: str = "conservative-v1"
    query_template_version: str = "v1"
    model_revision: str = Field(default="configured", min_length=1)


class GraphPathsConfig(StrictModel):
    manifest_root: str = "data/processed/graph-runs"
    cache_root: str = "data/processed/graph-extraction-cache"
    review_root: str = "data/processed/graph-review"
    review_decisions: str = "data/processed/graph-review/decisions.json"
    prompt_path: str = "prompts/graph-extraction-v1.txt"
    evaluation_questions: str = "data/evaluation/graph_dev_questions.json"
    evaluation_results_root: str = "data/evaluation/results/graph-retrieval"

    _validate_paths = field_validator(
        "manifest_root",
        "cache_root",
        "review_root",
        "review_decisions",
        "prompt_path",
        "evaluation_questions",
        "evaluation_results_root",
    )(_relative_path)


class GraphSelectionConfig(StrictModel):
    min_chunks_per_paper: int = Field(default=3, ge=1, le=5)
    max_chunks_per_paper: int = Field(default=5, ge=3, le=5)
    min_chunk_chars: int = Field(default=200, ge=1, le=10_000)
    max_chunk_chars: int = Field(default=8000, ge=100, le=50_000)
    max_papers: int = Field(default=15, ge=1, le=200)

    @model_validator(mode="after")
    def limits_are_consistent(self) -> GraphSelectionConfig:
        if self.min_chunks_per_paper > self.max_chunks_per_paper:
            raise ValueError("minimum selected chunks must not exceed maximum")
        if self.min_chunk_chars > self.max_chunk_chars:
            raise ValueError("minimum chunk chars must not exceed maximum")
        return self


class GraphExtractionBudgetConfig(StrictModel):
    enabled: bool = False
    max_requests: int = Field(default=100, ge=1, le=100)
    max_input_tokens: int = Field(default=160_000, ge=1, le=160_000)
    max_output_tokens: int = Field(default=40_000, ge=1, le=40_000)
    max_output_chars: int = Field(default=16_000, ge=100, le=100_000)
    batch_size: int = Field(default=1, ge=1, le=10)
    concurrency: int = Field(default=1, ge=1, le=4)
    chars_per_token_estimate: int = Field(default=3, ge=1, le=8)

    @model_validator(mode="after")
    def batch_is_bounded(self) -> GraphExtractionBudgetConfig:
        if self.concurrency > self.batch_size:
            raise ValueError("graph extraction concurrency must not exceed batch_size")
        return self


class GraphNormalizationConfig(StrictModel):
    auto_merge_min_confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    review_similarity_threshold: float = Field(default=0.8, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def thresholds_are_ordered(self) -> GraphNormalizationConfig:
        if self.review_similarity_threshold > self.auto_merge_min_confidence:
            raise ValueError("review threshold must not exceed automatic merge threshold")
        return self


class GraphQueryConfig(StrictModel):
    top_k: int = Field(default=20, ge=1, le=100)
    max_top_k: int = Field(default=100, ge=1, le=100)
    max_candidates: int = Field(default=100, ge=1, le=500)
    max_hops: int = Field(default=3, ge=1, le=3)
    timeout_seconds: float = Field(default=10.0, gt=0.0, le=30.0)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def limits_are_consistent(self) -> GraphQueryConfig:
        if self.top_k > self.max_top_k or self.max_top_k > self.max_candidates:
            raise ValueError("graph query limits must be monotonically increasing")
        return self


class GraphConfig(StrictModel):
    schema_config: GraphSchemaConfig
    paths: GraphPathsConfig
    selection: GraphSelectionConfig
    extraction: GraphExtractionBudgetConfig
    normalization: GraphNormalizationConfig
    query: GraphQueryConfig


def load_graph_config(path: Path) -> GraphConfig:
    """只加载 graph 段，避免改变现有 Dense/Hybrid 配置语义。"""

    with path.open("r", encoding="utf-8") as handle:
        payload: Any = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict) or not isinstance(payload.get("graph"), dict):
        raise ValueError("graph configuration must be a mapping")
    return GraphConfig.model_validate(payload["graph"])
