"""Dense RAG 索引、回答与评测的严格公共契约。"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from kg_crag.models.domain import Evidence, StrictModel
from kg_crag.models.foundation import ErrorDetail, TraceEvent


class VectorCollectionIdentity(StrictModel):
    schema_version: str = Field(min_length=1)
    embedding_provider: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    embedding_revision: str = Field(min_length=1)
    dimensions: int = Field(ge=1)
    distance: Literal["cosine", "dot", "euclid"]
    payload_fields: list[str] = Field(min_length=1)
    collection_version: str = Field(pattern=r"^[0-9a-f]{64}$")
    collection_name: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")


class IndexItemStatus(StrEnum):
    SUCCEEDED = "succeeded"
    SKIPPED = "skipped"
    FAILED = "failed"
    PLANNED = "planned"


class IndexItemResult(StrictModel):
    paper_id: str = Field(min_length=1)
    status: IndexItemStatus
    chunk_count: int = Field(default=0, ge=0)
    added: int = Field(default=0, ge=0)
    updated: int = Field(default=0, ge=0)
    skipped: int = Field(default=0, ge=0)
    deleted: int = Field(default=0, ge=0)
    cache_hits: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    error: ErrorDetail | None = None


class IndexRunManifest(StrictModel):
    run_id: str = Field(min_length=1)
    dry_run: bool
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    collection: VectorCollectionIdentity
    started_at: datetime
    finished_at: datetime
    items: list[IndexItemResult]
    manifest_path: str | None = None

    @model_validator(mode="after")
    def times_are_ordered(self) -> IndexRunManifest:
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        return self


class ClaimType(StrEnum):
    FACT = "fact"
    SYNTHESIS = "synthesis"
    INFERENCE = "inference"
    UNCERTAIN = "uncertain"
    CONFLICT = "conflict"


class AnswerClaim(StrictModel):
    text: str = Field(min_length=1, max_length=10_000)
    claim_type: ClaimType
    citation_ids: list[str] = Field(default_factory=list)

    @field_validator("citation_ids")
    @classmethod
    def citations_are_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("citation_ids must be unique")
        return value

    @model_validator(mode="after")
    def supported_claims_have_citations(self) -> AnswerClaim:
        if self.claim_type is not ClaimType.UNCERTAIN and not self.citation_ids:
            raise ValueError("non-uncertain claims require citations")
        return self


class Citation(StrictModel):
    citation_id: str = Field(pattern=r"^E[1-9][0-9]*$")
    evidence_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    section: str | None = None
    page: int | None = Field(default=None, ge=1)


def _dense_retrieval_path() -> list[Literal["dense"]]:
    return ["dense"]


class DenseRAGResult(StrictModel):
    run_id: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    claims: list[AnswerClaim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    insufficient_evidence: bool
    retrieval_path: list[Literal["dense"]] = Field(default_factory=_dense_retrieval_path)
    used_external: Literal[False] = False
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_version: str = Field(pattern=r"^[0-9a-f]{64}$")
    collection_version: str = Field(pattern=r"^[0-9a-f]{64}$")
    trace: list[TraceEvent] = Field(default_factory=list)


class PilotQuestion(StrictModel):
    question_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    question: str = Field(min_length=5)
    target_chunk_ids: list[str] = Field(min_length=1)
    rationale: str = Field(min_length=1, max_length=1000)

    @field_validator("target_chunk_ids", mode="before")
    @classmethod
    def normalize_target_ids(cls, value: object) -> object:
        if isinstance(value, list):
            return list(dict.fromkeys(value))
        return value


class PilotQuestionSet(StrictModel):
    schema_version: Literal["v1"] = "v1"
    corpus_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    questions: list[PilotQuestion] = Field(min_length=20, max_length=200)

    @model_validator(mode="after")
    def question_ids_are_unique(self) -> PilotQuestionSet:
        ids = [item.question_id for item in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("question IDs must be unique")
        return self


class EvaluationStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class DenseEvaluationItem(StrictModel):
    question_id: str = Field(min_length=1)
    status: EvaluationStatus
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    cited_chunk_ids: list[str] = Field(default_factory=list)
    latency_ms: float = Field(ge=0)
    result: DenseRAGResult | None = None
    error: ErrorDetail | None = None


class DenseEvaluationMetrics(StrictModel):
    recall_at_k: dict[str, float] = Field(default_factory=dict)
    mrr: float = Field(ge=0.0, le=1.0)
    ndcg_at_k: dict[str, float] = Field(default_factory=dict)
    citation_precision: float = Field(ge=0.0, le=1.0)
    citation_recall: float = Field(ge=0.0, le=1.0)
    failure_count: int = Field(ge=0)


class DenseEvaluationReport(StrictModel):
    baseline_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str = Field(min_length=1)
    corpus_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_version: str = Field(pattern=r"^[0-9a-f]{64}$")
    collection_version: str = Field(pattern=r"^[0-9a-f]{64}$")
    started_at: datetime
    finished_at: datetime
    metrics: DenseEvaluationMetrics
    items: list[DenseEvaluationItem]
