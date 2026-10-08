"""查询应用、演示界面与运行诊断使用的严格公共契约。"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, HttpUrl, field_validator, model_validator

from kg_crag.models.domain import StrictModel
from kg_crag.models.foundation import _reject_sensitive_keys


class ApplicationMode(StrEnum):
    """调用方可选择的有限运行模式。"""

    LIVE = "live"
    REPLAY = "replay"


class PublicSourceType(StrEnum):
    INTERNAL = "internal"
    DENSE = "dense"
    SPARSE = "sparse"
    GRAPH = "graph"
    WEB = "web"


class FacetStatus(StrEnum):
    COVERED = "covered"
    MISSING = "missing"
    CONFLICTING = "conflicting"


class TokenUsageSource(StrEnum):
    NONE = "none"
    ACTUAL = "actual"
    ESTIMATED = "estimated"
    MIXED = "mixed"


class ComponentStatus(StrEnum):
    READY = "ready"
    WARMING = "warming"
    DEGRADED = "degraded"
    DISABLED = "disabled"


class ApplicationErrorCode(StrEnum):
    VALIDATION = "validation_error"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    BUDGET_STOPPED = "budget_stopped"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    INTERNAL = "internal_error"


class QueryRequest(StrictModel):
    question: str = Field(min_length=1, max_length=2000)
    mode: ApplicationMode = ApplicationMode.REPLAY
    replay_case_id: str | None = Field(default=None, pattern=r"^[a-z0-9-]{3,64}$")
    allow_web: bool = False
    include_trace: bool = False

    @model_validator(mode="after")
    def replay_case_matches_mode(self) -> QueryRequest:
        if self.mode is ApplicationMode.LIVE and self.replay_case_id is not None:
            raise ValueError("replay_case_id is only valid in replay mode")
        if self.mode is ApplicationMode.REPLAY and self.allow_web:
            raise ValueError("allow_web is only valid in live mode")
        return self


class CitationSummary(StrictModel):
    citation_id: str = Field(min_length=1, max_length=64)
    evidence_id: str = Field(min_length=1, max_length=160)
    title: str = Field(min_length=1, max_length=300)
    source_channels: list[PublicSourceType] = Field(min_length=1, max_length=5)
    paper_id: str | None = Field(default=None, max_length=160)
    section: str | None = Field(default=None, max_length=200)
    page: int | None = Field(default=None, ge=1)
    url: HttpUrl | None = None
    external: bool = False

    @model_validator(mode="after")
    def source_and_provenance_are_consistent(self) -> CitationSummary:
        web = PublicSourceType.WEB in self.source_channels
        if self.external != web:
            raise ValueError("external citation flag must agree with web source channel")
        if self.external and self.url is None:
            raise ValueError("external citations require a URL")
        return self


class FacetSummary(StrictModel):
    facet_id: str = Field(min_length=1, max_length=160)
    label: str = Field(min_length=1, max_length=300)
    status: FacetStatus
    evidence_ids: list[str] = Field(default_factory=list, max_length=20)
    reason: str | None = Field(default=None, max_length=300)
    conflict_ids: list[str] = Field(default_factory=list, max_length=20)


class ActionSummary(StrictModel):
    sequence: int = Field(ge=0, le=100)
    action: str = Field(min_length=1, max_length=100)
    reason: str = Field(min_length=1, max_length=300)
    target_facet_ids: list[str] = Field(default_factory=list, max_length=20)
    status: str = Field(min_length=1, max_length=100)


class ApplicationBudgetSummary(StrictModel):
    tool_calls: int = Field(default=0, ge=0, le=100)
    model_calls: int = Field(default=0, ge=0, le=20)
    web_calls: int = Field(default=0, ge=0, le=5)
    input_tokens: int = Field(default=0, ge=0, le=100_000)
    output_tokens: int = Field(default=0, ge=0, le=100_000)
    token_usage_source: TokenUsageSource = TokenUsageSource.NONE
    elapsed_ms: int = Field(default=0, ge=0, le=600_000)
    stopped: bool = False
    stop_reason: str | None = Field(default=None, max_length=200)


class RuntimeIdentitySummary(StrictModel):
    """公开运行身份只包含可复现版本，不包含路径或连接信息。"""

    corpus_snapshot: str = Field(pattern=r"^[a-f0-9]{64}$")
    dense_version: str = Field(pattern=r"^[a-f0-9]{64}$")
    sparse_version: str = Field(pattern=r"^[a-f0-9]{64}$")
    graph_version: str | None = Field(default=None, max_length=128)


class QueryResponse(StrictModel):
    schema_version: Literal["v1"] = "v1"
    mode: ApplicationMode
    request_id: str = Field(min_length=1, max_length=64)
    trace_id: str = Field(min_length=1, max_length=128)
    answer: str = Field(min_length=1, max_length=12_000)
    citations: list[CitationSummary] = Field(default_factory=list, max_length=20)
    facets: list[FacetSummary] = Field(default_factory=list, max_length=40)
    actions: list[ActionSummary] = Field(default_factory=list, max_length=20)
    retrieval_path: list[str] = Field(default_factory=list, max_length=10)
    budget: ApplicationBudgetSummary = Field(default_factory=ApplicationBudgetSummary)
    stop_reason: str = Field(min_length=1, max_length=200)
    runtime_identity: RuntimeIdentitySummary | None = None
    replay_fixture_version: str | None = Field(default=None, max_length=64)
    replay_notice: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def replay_metadata_is_consistent(self) -> QueryResponse:
        has_replay_metadata = (
            self.replay_fixture_version is not None and self.replay_notice is not None
        )
        if (self.mode is ApplicationMode.REPLAY) != has_replay_metadata:
            raise ValueError("replay responses require both fixture version and notice")
        if self.mode is ApplicationMode.REPLAY and self.runtime_identity is not None:
            raise ValueError("replay responses cannot claim a live runtime identity")
        return self


class DocumentSummary(StrictModel):
    schema_version: Literal["v1"] = "v1"
    paper_id: str = Field(min_length=1, max_length=160)
    title: str = Field(min_length=1, max_length=500)
    abstract: str = Field(default="", max_length=2000)
    year: int | None = Field(default=None, ge=1900, le=2100)
    authors: list[str] = Field(default_factory=list, max_length=100)
    processing_version: str = Field(min_length=1, max_length=128)
    chunk_ids: list[str] = Field(default_factory=list, max_length=100)
    quality_passed: bool


class IngestionRunRequest(StrictModel):
    paper_ids: list[str] = Field(min_length=1, max_length=5)
    dry_run: bool = True
    confirm: bool = False
    force: bool = False

    @model_validator(mode="after")
    def execution_requires_confirmation(self) -> IngestionRunRequest:
        if not self.dry_run and not self.confirm:
            raise ValueError("actual ingestion requires confirm=true")
        if len(self.paper_ids) != len(set(self.paper_ids)):
            raise ValueError("paper_ids must be unique")
        return self


class IngestionItemSummary(StrictModel):
    paper_id: str = Field(min_length=1, max_length=160)
    status: str = Field(min_length=1, max_length=40)
    processing_version: str | None = Field(default=None, max_length=128)
    artifact_names: list[str] = Field(default_factory=list, max_length=10)
    warning_count: int = Field(default=0, ge=0, le=10_000)
    reason: str | None = Field(default=None, max_length=300)


class IngestionRunResponse(StrictModel):
    schema_version: Literal["v1"] = "v1"
    request_id: str = Field(min_length=1, max_length=64)
    run_id: str = Field(min_length=1, max_length=160)
    dry_run: bool
    complete: bool
    items: list[IngestionItemSummary] = Field(min_length=1, max_length=5)


class TraceEventSummary(StrictModel):
    sequence: int = Field(ge=0)
    phase: str = Field(min_length=1, max_length=100)
    event: str = Field(min_length=1, max_length=100)
    status: str | None = Field(default=None, max_length=100)
    occurred_at: datetime
    details: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict,
        max_length=16,
    )

    @field_validator("details")
    @classmethod
    def details_are_bounded_and_safe(
        cls,
        value: dict[str, str | int | float | bool | None],
    ) -> dict[str, str | int | float | bool | None]:
        _reject_sensitive_keys(value)
        if any(isinstance(item, str) and len(item) > 300 for item in value.values()):
            raise ValueError("trace detail strings must be bounded")
        return value


class TraceSummary(StrictModel):
    schema_version: Literal["v1"] = "v1"
    trace_id: str = Field(min_length=1, max_length=128)
    mode: ApplicationMode
    events: list[TraceEventSummary] = Field(default_factory=list, max_length=200)
    truncated: bool = False
    replay_fixture_version: str | None = Field(default=None, max_length=64)


class ComponentReadiness(StrictModel):
    component: str = Field(min_length=1, max_length=100)
    status: ComponentStatus
    required: bool
    message: str = Field(min_length=1, max_length=300)


class ReadinessResponse(StrictModel):
    schema_version: Literal["v1"] = "v1"
    status: ComponentStatus
    version: str = Field(min_length=1)
    available_modes: list[ApplicationMode] = Field(min_length=1, max_length=2)
    components: list[ComponentReadiness] = Field(min_length=1, max_length=20)


class ApplicationErrorDetail(StrictModel):
    code: ApplicationErrorCode
    message: str = Field(min_length=1, max_length=500)
    retryable: bool = False
    fields: dict[str, str] = Field(default_factory=dict, max_length=16)

    @field_validator("fields")
    @classmethod
    def fields_are_bounded_and_safe(cls, value: dict[str, str]) -> dict[str, str]:
        _reject_sensitive_keys(dict(value))
        if any(len(item) > 200 for item in value.values()):
            raise ValueError("error field values must be bounded")
        return value


class ApplicationErrorResponse(StrictModel):
    schema_version: Literal["v1"] = "v1"
    request_id: str = Field(min_length=1, max_length=64)
    error: ApplicationErrorDetail
