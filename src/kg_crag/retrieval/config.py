"""Dense RAG 的严格配置加载与稳定哈希。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePath
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import Field, field_validator, model_validator

from kg_crag.models.domain import StrictModel


class DenseEmbeddingConfig(StrictModel):
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    dimensions: int = Field(ge=1, le=65_536)
    normalization: Literal["nfc-strip-v1"] = "nfc-strip-v1"
    batch_size: int = Field(ge=1, le=1024)
    max_batch_size: int = Field(ge=1, le=1024)
    normalize: bool = True
    query_prefix: str = ""
    document_prefix: str = ""
    cache_root: str = "data/processed/embedding-cache"

    @field_validator("cache_root")
    @classmethod
    def cache_root_is_contained(cls, value: str) -> str:
        return _relative_path(value)

    @model_validator(mode="after")
    def batch_size_is_bounded(self) -> DenseEmbeddingConfig:
        if self.batch_size > self.max_batch_size:
            raise ValueError("embedding batch_size must not exceed max_batch_size")
        return self


def _default_allowed_filters() -> list[Literal["paper_id", "section", "processing_version"]]:
    return ["paper_id", "section", "processing_version"]


class VectorCollectionConfig(StrictModel):
    prefix: str = Field(pattern=r"^[a-z][a-z0-9_]{1,47}$")
    schema_version: str = Field(min_length=1)
    distance: Literal["cosine", "dot", "euclid"] = "cosine"
    allowed_filters: list[Literal["paper_id", "section", "processing_version"]] = Field(
        default_factory=_default_allowed_filters
    )

    @field_validator("allowed_filters")
    @classmethod
    def filters_are_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("allowed_filters must be unique")
        return value


class IndexingConfig(StrictModel):
    max_papers: int = Field(default=200, ge=1, le=10_000)
    upsert_batch_size: int = Field(default=64, ge=1, le=1024)
    manifest_root: str = "data/processed/index-runs"

    @field_validator("manifest_root")
    @classmethod
    def manifest_root_is_contained(cls, value: str) -> str:
        return _relative_path(value)


class DenseRetrievalConfig(StrictModel):
    top_k: int = Field(default=20, ge=1, le=1000)
    max_top_k: int = Field(default=100, ge=1, le=1000)
    deduplicate_by_paper: bool = False
    candidate_multiplier: int = Field(default=3, ge=1, le=20)
    max_candidates: int = Field(default=100, ge=1, le=5000)
    min_score: float = Field(default=0.0, ge=-1.0, le=1.0)

    @model_validator(mode="after")
    def retrieval_limits_are_consistent(self) -> DenseRetrievalConfig:
        if self.top_k > self.max_top_k:
            raise ValueError("dense top_k must not exceed max_top_k")
        if self.max_candidates < self.max_top_k:
            raise ValueError("max_candidates must be at least max_top_k")
        return self


class DenseGenerationConfig(StrictModel):
    prompt_path: str = "prompts/dense-rag-v1.txt"
    max_evidence: int = Field(default=8, ge=1, le=100)
    max_chars_per_evidence: int = Field(default=4000, ge=100, le=100_000)
    max_context_chars: int = Field(default=20_000, ge=100, le=1_000_000)
    max_response_chars: int = Field(default=20_000, ge=100, le=1_000_000)
    min_evidence: int = Field(default=1, ge=1, le=100)
    min_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    output_root: str = "data/processed/query-runs"

    @field_validator("prompt_path", "output_root")
    @classmethod
    def paths_are_contained(cls, value: str) -> str:
        return _relative_path(value)

    @model_validator(mode="after")
    def generation_limits_are_consistent(self) -> DenseGenerationConfig:
        if self.min_evidence > self.max_evidence:
            raise ValueError("min_evidence must not exceed max_evidence")
        if self.max_chars_per_evidence > self.max_context_chars:
            raise ValueError("max_chars_per_evidence must not exceed max_context_chars")
        return self


class DenseRAGConfig(StrictModel):
    embedding: DenseEmbeddingConfig
    collection: VectorCollectionConfig
    indexing: IndexingConfig
    dense: DenseRetrievalConfig
    generation: DenseGenerationConfig


class DenseEvaluationConfig(StrictModel):
    questions_path: str = "data/evaluation/dense_pilot_questions.json"
    results_root: str = "data/evaluation/results/dense-rag"
    k_values: list[int] = Field(default_factory=lambda: [5, 10, 20], min_length=1)
    max_questions: int = Field(default=200, ge=20, le=10_000)

    @field_validator("questions_path", "results_root")
    @classmethod
    def paths_are_contained(cls, value: str) -> str:
        return _relative_path(value)

    @field_validator("k_values")
    @classmethod
    def k_values_are_valid(cls, value: list[int]) -> list[int]:
        if any(item <= 0 or item > 1000 for item in value) or value != sorted(set(value)):
            raise ValueError("k_values must be unique positive values in ascending order")
        return value


def _default_hybrid_strategies() -> list[
    Literal["dense", "sparse", "rrf", "weighted", "fusion_rerank"]
]:
    return ["dense", "sparse", "rrf", "weighted", "fusion_rerank"]


class HybridEvaluationConfig(StrictModel):
    """零 LLM 开发集矩阵及唯一回答验证的资源边界。"""

    questions_path: str = "data/evaluation/hybrid_dev_questions.json"
    results_root: str = "data/evaluation/results/hybrid-retrieval"
    strategies: list[Literal["dense", "sparse", "rrf", "weighted", "fusion_rerank"]] = Field(
        default_factory=_default_hybrid_strategies, min_length=1
    )
    k_values: list[int] = Field(default_factory=lambda: [5, 8, 10, 20], min_length=1)
    max_questions: int = Field(default=40, ge=20, le=40)
    smoke_questions: int = Field(default=5, ge=1, le=5)
    answer_validation_max_configs: Literal[1] = 1
    answer_validation_calls_per_question: Literal[1] = 1

    @field_validator("questions_path", "results_root")
    @classmethod
    def paths_are_contained(cls, value: str) -> str:
        return _relative_path(value)

    @field_validator("strategies")
    @classmethod
    def strategies_are_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("hybrid evaluation strategies must be unique")
        return value

    @field_validator("k_values")
    @classmethod
    def k_values_are_valid(cls, value: list[int]) -> list[int]:
        if any(item <= 0 or item > 100 for item in value) or value != sorted(set(value)):
            raise ValueError("k_values must be unique positive values in ascending order")
        return value


class SparseRetrievalConfig(StrictModel):
    """版本化 Sparse 索引和有界查询配置。"""

    schema_version: Literal["v1"] = "v1"
    tokenizer: Literal["scientific-unicode-v1"] = "scientific-unicode-v1"
    bm25_version: Literal["sqlite-fts5-bm25-v1"] = "sqlite-fts5-bm25-v1"
    index_root: str = "data/processed/sparse-index"
    manifest_root: str = "data/processed/sparse-index-runs"
    top_k: int = Field(default=20, ge=1, le=100)
    max_top_k: int = Field(default=100, ge=1, le=100)
    max_candidates: int = Field(default=100, ge=1, le=500)
    max_query_chars: int = Field(default=2000, ge=1, le=20_000)
    max_query_tokens: int = Field(default=64, ge=1, le=256)
    allowed_filters: list[Literal["paper_id", "section", "processing_version"]] = Field(
        default_factory=_default_allowed_filters
    )

    @field_validator("index_root", "manifest_root")
    @classmethod
    def paths_are_contained(cls, value: str) -> str:
        return _relative_path(value)

    @field_validator("allowed_filters")
    @classmethod
    def filters_are_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("allowed_filters must be unique")
        return value

    @model_validator(mode="after")
    def limits_are_consistent(self) -> SparseRetrievalConfig:
        if self.top_k > self.max_top_k:
            raise ValueError("sparse top_k must not exceed max_top_k")
        if self.max_candidates < self.max_top_k:
            raise ValueError("sparse max_candidates must be at least max_top_k")
        return self


class FusionConfig(StrictModel):
    """可重放的 RRF/加权融合边界。"""

    method: Literal["rrf", "weighted"] = "rrf"
    normalization: Literal["min-max-v1"] = "min-max-v1"
    rrf_k: int = Field(default=60, ge=1, le=1000)
    dense_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    sparse_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    final_top_k: int = Field(default=10, ge=1, le=100)

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> FusionConfig:
        if abs(self.dense_weight + self.sparse_weight - 1.0) > 1e-6:
            raise ValueError("fusion weights must sum to 1.0")
        return self


class RerankerConfig(StrictModel):
    """本地 Cross-Encoder 的惰性加载与资源上限。"""

    enabled: bool = True
    model: str = "BAAI/bge-reranker-base"
    revision: str = "2cfc18c9415c912f9d8155881c133215df768a70"
    device: str = Field(default="cpu", pattern=r"^(cpu|cuda(?::[0-9]+)?)$")
    cache_root: str = "data/processed/model-cache"
    batch_size: int = Field(default=8, ge=1, le=32)
    max_batch_size: int = Field(default=16, ge=1, le=32)
    max_candidates: int = Field(default=20, ge=1, le=20)
    top_k: int = Field(default=8, ge=1, le=8)

    @model_validator(mode="after")
    def limits_are_consistent(self) -> RerankerConfig:
        if self.batch_size > self.max_batch_size:
            raise ValueError("reranker batch_size must not exceed max_batch_size")
        if self.top_k > self.max_candidates:
            raise ValueError("reranker top_k must not exceed max_candidates")
        return self


class HybridRuntimeConfig(StrictModel):
    """Hybrid 编排的显式降级和输出边界。"""

    allow_partial: bool = True
    allow_rerank_fallback: bool = True
    deduplicate_by_paper: bool = False
    max_output: int = Field(default=8, ge=1, le=100)
    output_root: str = "data/processed/hybrid-query-runs"

    @field_validator("output_root")
    @classmethod
    def output_root_is_contained(cls, value: str) -> str:
        return _relative_path(value)


class HybridRetrievalConfig(StrictModel):
    """Dense 基线之上的向后兼容 Hybrid 配置。"""

    dense: DenseRAGConfig
    sparse: SparseRetrievalConfig
    fusion: FusionConfig
    reranker: RerankerConfig
    hybrid: HybridRuntimeConfig

    @model_validator(mode="after")
    def output_limits_are_consistent(self) -> HybridRetrievalConfig:
        if self.hybrid.max_output > self.fusion.final_top_k:
            raise ValueError("hybrid max_output must not exceed fusion final_top_k")
        if self.reranker.enabled and self.hybrid.max_output > self.reranker.top_k:
            raise ValueError("hybrid max_output must not exceed reranker top_k")
        return self


def _relative_path(value: str) -> str:
    path = PurePath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Dense RAG paths must be workspace-relative and contained")
    return path.as_posix()


def _load_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a mapping")
    return payload


def load_dense_rag_config(path: Path) -> DenseRAGConfig:
    """只加载 Dense RAG 段，旧检索段留给后续迭代。"""

    payload = _load_mapping(path)
    required = {"dense_embedding", "vector_collection", "indexing", "dense", "generation"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"missing Dense RAG configuration sections: {sorted(missing)}")
    return DenseRAGConfig.model_validate(
        {
            "embedding": payload["dense_embedding"],
            "collection": payload["vector_collection"],
            "indexing": payload["indexing"],
            "dense": payload["dense"],
            "generation": payload["generation"],
        }
    )


def load_dense_evaluation_config(path: Path) -> DenseEvaluationConfig:
    payload = _load_mapping(path)
    section = payload.get("dense_rag")
    if not isinstance(section, dict):
        raise ValueError("dense_rag evaluation configuration must be a mapping")
    return DenseEvaluationConfig.model_validate(section)


def load_hybrid_evaluation_config(path: Path) -> HybridEvaluationConfig:
    payload = _load_mapping(path)
    section = payload.get("hybrid_retrieval")
    if not isinstance(section, dict):
        raise ValueError("hybrid_retrieval evaluation configuration must be a mapping")
    return HybridEvaluationConfig.model_validate(section)


def load_hybrid_retrieval_config(path: Path) -> HybridRetrievalConfig:
    """加载 Hybrid 专用段，同时复用经过验证的 Dense 配置。"""

    payload = _load_mapping(path)
    dense = load_dense_rag_config(path)
    required = {"sparse", "fusion", "reranker", "hybrid"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"missing Hybrid retrieval configuration sections: {sorted(missing)}")
    return HybridRetrievalConfig.model_validate(
        {
            "dense": dense,
            "sparse": payload["sparse"],
            "fusion": payload["fusion"],
            "reranker": payload["reranker"],
            "hybrid": payload["hybrid"],
        }
    )


ConfigModel = (
    DenseRAGConfig | DenseEvaluationConfig | HybridRetrievalConfig | HybridEvaluationConfig
)


def canonical_config_bytes(config: ConfigModel) -> bytes:
    payload = config.model_dump(mode="json")
    # 缓存目录是机器部署细节，不应让同一实验配置产生不同身份。
    if isinstance(config, HybridRetrievalConfig):
        payload["reranker"].pop("cache_root", None)
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def dense_config_hash(config: ConfigModel) -> str:
    return hashlib.sha256(canonical_config_bytes(config)).hexdigest()
