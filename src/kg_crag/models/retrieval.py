"""Sparse 索引、Hybrid 检索与开发集评测的公共契约。"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from kg_crag.models.domain import Evidence, StrictModel
from kg_crag.models.foundation import ErrorDetail
from kg_crag.models.rag import AnswerClaim, Citation, EvaluationStatus, IndexItemStatus


class SparseIndexIdentity(StrictModel):
    """绑定实现环境、分词规则、语料快照和索引内容的稳定身份。"""

    schema_version: str = Field(min_length=1)
    tokenizer: str = Field(min_length=1)
    bm25_version: str = Field(min_length=1)
    python_version: str = Field(min_length=1)
    sqlite_version: str = Field(min_length=1)
    fts5_enabled: bool
    fts5_version: str = Field(min_length=1)
    corpus_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    index_version: str = Field(pattern=r"^[0-9a-f]{64}$")


class SparseIndexItemResult(StrictModel):
    paper_id: str = Field(min_length=1)
    status: IndexItemStatus
    chunk_count: int = Field(default=0, ge=0)
    added: int = Field(default=0, ge=0)
    updated: int = Field(default=0, ge=0)
    skipped: int = Field(default=0, ge=0)
    deleted: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    error: ErrorDetail | None = None

    @model_validator(mode="after")
    def status_matches_error(self) -> SparseIndexItemResult:
        if self.status is IndexItemStatus.FAILED and self.error is None:
            raise ValueError("failed sparse index item requires an error")
        if self.status is not IndexItemStatus.FAILED and self.error is not None:
            raise ValueError("only failed sparse index item may contain an error")
        return self


class SparseIndexRunManifest(StrictModel):
    run_id: str = Field(min_length=1)
    dry_run: bool
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    index: SparseIndexIdentity
    started_at: datetime
    finished_at: datetime
    items: list[SparseIndexItemResult]
    manifest_path: str | None = None

    @model_validator(mode="after")
    def run_is_consistent(self) -> SparseIndexRunManifest:
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        if self.index.corpus_snapshot_hash != self.corpus_snapshot_hash:
            raise ValueError("index and run corpus snapshot hashes must match")
        return self


class RetrievalStageStatus(StrEnum):
    SUCCEEDED = "succeeded"
    EMPTY = "empty"
    FAILED = "failed"
    SKIPPED = "skipped"


class RetrievalStageSummary(StrictModel):
    stage: Literal["dense", "sparse", "fusion", "rerank"]
    status: RetrievalStageStatus
    input_candidates: int = Field(default=0, ge=0)
    output_candidates: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    call_count: int = Field(default=0, ge=0, le=1)
    error: ErrorDetail | None = None

    @model_validator(mode="after")
    def status_is_consistent(self) -> RetrievalStageSummary:
        if self.status is RetrievalStageStatus.FAILED and self.error is None:
            raise ValueError("failed retrieval stage requires an error")
        if self.status is not RetrievalStageStatus.FAILED and self.error is not None:
            raise ValueError("only failed retrieval stage may contain an error")
        if self.status in {RetrievalStageStatus.EMPTY, RetrievalStageStatus.SKIPPED}:
            if self.output_candidates != 0:
                raise ValueError("empty or skipped stage must not contain output candidates")
        if self.status is RetrievalStageStatus.SKIPPED and self.call_count != 0:
            raise ValueError("skipped stage must not record a call")
        return self


class RetrievalCostSummary(StrictModel):
    embedding_calls: int = Field(default=0, ge=0)
    sparse_calls: int = Field(default=0, ge=0)
    reranker_calls: int = Field(default=0, ge=0)
    local_reranker_latency_ms: float = Field(default=0.0, ge=0.0)
    llm_calls: Literal[0] = 0


class HybridRetrievalResult(StrictModel):
    run_id: str = Field(min_length=1)
    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    collection_version: str = Field(min_length=1)
    sparse_index_version: str = Field(min_length=1)
    fusion_version: str = Field(min_length=1)
    reranker_version: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    degraded: bool = False
    degraded_stages: list[Literal["dense", "sparse", "rerank"]] = Field(default_factory=list)
    stages: list[RetrievalStageSummary] = Field(default_factory=list)
    costs: RetrievalCostSummary = Field(default_factory=RetrievalCostSummary)

    @model_validator(mode="after")
    def degradation_is_consistent(self) -> HybridRetrievalResult:
        if self.degraded != bool(self.degraded_stages):
            raise ValueError("degraded must match degraded_stages")
        if len(self.degraded_stages) != len(set(self.degraded_stages)):
            raise ValueError("degraded_stages must be unique")
        return self


class HybridQueryResult(StrictModel):
    """一次 Hybrid 查询及其可选的证据约束回答。"""

    run_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    retrieval: HybridRetrievalResult
    retrieval_path: list[Literal["dense", "sparse"]] = Field(min_length=1, max_length=2)
    with_answer: bool = False
    answer: str | None = None
    claims: list[AnswerClaim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    insufficient_evidence: bool = False
    prompt_version: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    llm_calls: Literal[0, 1] = 0

    @model_validator(mode="after")
    def answer_state_is_consistent(self) -> HybridQueryResult:
        if self.run_id != self.retrieval.run_id or self.question != self.retrieval.query:
            raise ValueError("query identity must match the nested retrieval result")
        if len(self.retrieval_path) != len(set(self.retrieval_path)):
            raise ValueError("retrieval_path must be unique")
        if not self.with_answer:
            if any(
                (
                    self.answer is not None,
                    bool(self.claims),
                    bool(self.citations),
                    self.confidence is not None,
                    self.insufficient_evidence,
                    self.prompt_version is not None,
                    self.llm_calls != 0,
                )
            ):
                raise ValueError("retrieval-only result must not contain answer state")
        elif self.answer is None or self.confidence is None or self.prompt_version is None:
            raise ValueError("answer-enabled result requires answer metadata")
        elif self.insufficient_evidence:
            if self.claims or self.citations or self.llm_calls != 0:
                raise ValueError("insufficient answer must not contain generated claims")
        elif not self.claims or self.llm_calls != 1:
            raise ValueError("generated answer requires claims and exactly one LLM call")
        return self


class HybridEvaluationMetrics(StrictModel):
    recall_at_k: dict[str, float] = Field(default_factory=dict)
    mrr: float = Field(ge=0.0, le=1.0)
    ndcg_at_k: dict[str, float] = Field(default_factory=dict)
    evidence_coverage: float = Field(ge=0.0, le=1.0)
    failure_count: int = Field(ge=0)
    mean_latency_ms: float = Field(ge=0.0)
    mean_stage_latency_ms: dict[str, float] = Field(default_factory=dict)


HybridStrategy = Literal["dense", "sparse", "rrf", "weighted", "fusion_rerank"]


class HybridEvaluationItem(StrictModel):
    question_id: str = Field(min_length=1)
    strategy: HybridStrategy
    status: EvaluationStatus
    target_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    latency_ms: float = Field(ge=0.0)
    stage_latency_ms: dict[str, float] = Field(default_factory=dict)
    stage_call_counts: dict[str, int] = Field(default_factory=dict)
    error: ErrorDetail | None = None

    @model_validator(mode="after")
    def status_matches_error(self) -> HybridEvaluationItem:
        if self.status is EvaluationStatus.FAILED and self.error is None:
            raise ValueError("failed evaluation item requires an error")
        if self.status is not EvaluationStatus.FAILED and self.error is not None:
            raise ValueError("successful evaluation item must not contain an error")
        return self


class HybridEvaluationReport(StrictModel):
    run_id: str = Field(min_length=1)
    evaluation_version: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    collection_version: str = Field(min_length=1)
    sparse_index_version: str = Field(min_length=1)
    fusion_versions: dict[HybridStrategy, str] = Field(default_factory=dict)
    reranker_version: str | None = None
    started_at: datetime
    finished_at: datetime
    answer_validation_enabled: bool = False
    metrics: dict[HybridStrategy, HybridEvaluationMetrics]
    items: list[HybridEvaluationItem]

    @model_validator(mode="after")
    def times_are_ordered(self) -> HybridEvaluationReport:
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        return self
