"""跨模块共享的基础运行契约。"""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, field_validator, model_validator

from kg_crag.models.domain import StrictModel

ScalarValue: TypeAlias = str | int | float | bool | None

_SENSITIVE_KEY_MARKERS = (
    "password",
    "token",
    "apikey",
    "authorization",
    "credential",
    "secret",
    "prompt",
    "reasoning",
    "rawresponse",
    "excerpt",
    "content",
    "body",
)
_SAFE_TOKEN_COUNTERS = {"inputtokens", "outputtokens", "tokencount", "tokensused", "totaltokens"}


def _reject_sensitive_keys(values: dict[str, ScalarValue]) -> dict[str, ScalarValue]:
    """拒绝可能承载凭据的字段名，避免上下文进入错误或 Trace。"""

    for key in values:
        # 去掉大小写和分隔符差异，防止 api_key、API-Key 等变体绕过检查。
        normalized = re.sub(r"[^a-z0-9]", "", key.casefold())
        if normalized in _SAFE_TOKEN_COUNTERS:
            continue
        if any(marker in normalized for marker in _SENSITIVE_KEY_MARKERS):
            raise ValueError("sensitive context keys are not allowed")
    return values


class ErrorCode(StrEnum):
    """供调用方稳定分支的错误代码。"""

    VALIDATION = "validation_error"
    CONFIGURATION = "configuration_error"
    EXTERNAL_SERVICE = "external_service_error"
    TIMEOUT = "timeout_error"
    NOT_FOUND = "not_found"
    DATA = "data_error"
    INTERNAL = "internal_error"


class ErrorDetail(StrictModel):
    """可安全序列化的统一错误详情。"""

    code: ErrorCode
    message: str = Field(min_length=1, max_length=1000)
    retryable: bool = False
    context: dict[str, ScalarValue] = Field(default_factory=dict, max_length=32)

    _validate_context = field_validator("context")(_reject_sensitive_keys)


class TraceEvent(StrictModel):
    """工作流 TraceEvent v1 的最小稳定结构。"""

    schema_version: Literal["v1"] = "v1"
    trace_id: str = Field(min_length=1)
    sequence: int = Field(ge=0)
    node: str = Field(min_length=1)
    event: str = Field(min_length=1)
    occurred_at: datetime
    details: dict[str, ScalarValue] = Field(default_factory=dict, max_length=64)

    _validate_details = field_validator("details")(_reject_sensitive_keys)

    @field_validator("occurred_at")
    @classmethod
    def occurred_at_is_timezone_aware(cls, value: datetime) -> datetime:
        """要求调用方提供可重放、无本地时区歧义的时间。"""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must include a timezone")
        return value


class EvaluationTracePhase(StrEnum):
    """统一评测允许记录的有限阶段。"""

    VALIDATE = "validate"
    RETRIEVE = "retrieve"
    ASSESS = "assess"
    ANSWER = "answer"
    JUDGE = "judge"
    AGGREGATE = "aggregate"
    PUBLISH = "publish"


class EvaluationTraceStatus(StrEnum):
    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    BUDGET_STOPPED = "budget_stopped"


class TraceResourceUsage(StrictModel):
    """单个事件记录的资源增量，而非完整账本。"""

    tool_calls: int = Field(default=0, ge=0, le=1000)
    model_calls: int = Field(default=0, ge=0, le=100)
    web_calls: int = Field(default=0, ge=0, le=20)
    input_tokens: int = Field(default=0, ge=0, le=1_000_000)
    output_tokens: int = Field(default=0, ge=0, le=1_000_000)
    latency_ms: int = Field(default=0, ge=0, le=3_600_000)


class EvaluationTraceEvent(StrictModel):
    """关联运行与题目的脱敏 TraceEvent v2。"""

    schema_version: Literal["v2"] = "v2"
    trace_id: str = Field(min_length=1, max_length=128)
    run_id: str = Field(pattern=r"^unified-run-[a-f0-9]{32}$")
    question_id: str | None = Field(default=None, pattern=r"^ueq-[a-z0-9-]{3,120}$")
    sequence: int = Field(ge=0)
    phase: EvaluationTracePhase
    event: str = Field(min_length=1, max_length=100)
    status: EvaluationTraceStatus
    occurred_at: datetime
    details: dict[str, ScalarValue] = Field(default_factory=dict, max_length=64)
    usage: TraceResourceUsage = Field(default_factory=TraceResourceUsage)

    _validate_details = field_validator("details")(_reject_sensitive_keys)

    @field_validator("occurred_at")
    @classmethod
    def occurred_at_is_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must include a timezone")
        return value

    @model_validator(mode="after")
    def question_scope_matches_phase(self) -> EvaluationTraceEvent:
        if (
            self.phase
            in {
                EvaluationTracePhase.RETRIEVE,
                EvaluationTracePhase.ASSESS,
                EvaluationTracePhase.ANSWER,
                EvaluationTracePhase.JUDGE,
            }
            and self.question_id is None
        ):
            raise ValueError("question-scoped evaluation trace phase requires question_id")
        for value in self.details.values():
            if isinstance(value, str) and len(value) > 300:
                raise ValueError("trace string detail exceeds the bounded length")
        return self


TraceRecord: TypeAlias = Annotated[
    TraceEvent | EvaluationTraceEvent,
    Field(discriminator="schema_version"),
]


class NormalizedTraceEvent(StrictModel):
    """跨版本只读视图；未知字段保持为空，不推测历史状态。"""

    source_schema_version: Literal["v1", "v2"]
    trace_id: str
    run_id: str | None = None
    question_id: str | None = None
    sequence: int = Field(ge=0)
    phase: str
    event: str
    status: str | None = None
    occurred_at: datetime
    details: dict[str, ScalarValue] = Field(default_factory=dict, max_length=64)
    usage: TraceResourceUsage | None = None


class HealthResponse(StrictModel):
    """不探测外部依赖的进程健康响应。"""

    status: Literal["ok"] = "ok"
    version: str = Field(min_length=1)
    environment: str = Field(min_length=1)
