"""统一评测 Trace v2 的有界写入与跨版本只读适配。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import TypeAdapter

from kg_crag.models import (
    EvaluationTraceEvent,
    EvaluationTracePhase,
    EvaluationTraceStatus,
    NormalizedTraceEvent,
    TraceEvent,
    TraceRecord,
    TraceResourceUsage,
)
from kg_crag.workflow.trace import safe_trace_details

_TRACE_ADAPTER: TypeAdapter[TraceRecord] = TypeAdapter(TraceRecord)


def parse_trace_record(payload: str | bytes) -> TraceEvent | EvaluationTraceEvent:
    """按 schema_version 判别解析，未知版本由 Pydantic 明确拒绝。"""

    return _TRACE_ADAPTER.validate_json(payload)


def normalize_trace_event(event: TraceEvent | EvaluationTraceEvent) -> NormalizedTraceEvent:
    """生成不推测缺失字段的跨版本只读视图。"""

    if isinstance(event, EvaluationTraceEvent):
        return NormalizedTraceEvent(
            source_schema_version="v2",
            trace_id=event.trace_id,
            run_id=event.run_id,
            question_id=event.question_id,
            sequence=event.sequence,
            phase=event.phase.value,
            event=event.event,
            status=event.status.value,
            occurred_at=event.occurred_at,
            details=dict(event.details),
            usage=event.usage,
        )
    return NormalizedTraceEvent(
        source_schema_version="v1",
        trace_id=event.trace_id,
        sequence=event.sequence,
        phase=event.node,
        event=event.event,
        occurred_at=event.occurred_at,
        details=dict(event.details),
    )


def append_evaluation_trace(
    existing: list[EvaluationTraceEvent],
    *,
    trace_id: str,
    run_id: str,
    phase: EvaluationTracePhase,
    event: str,
    status: EvaluationTraceStatus,
    question_id: str | None = None,
    details: dict[str, Any] | None = None,
    usage: TraceResourceUsage | None = None,
    max_events: int = 20_000,
    occurred_at: datetime | None = None,
) -> list[EvaluationTraceEvent]:
    """追加序号连续的 v2 事件，并复用现有脱敏规则。"""

    if len(existing) >= max_events:
        raise ValueError("evaluation trace event limit reached")
    if existing:
        previous = existing[-1]
        if previous.trace_id != trace_id or previous.run_id != run_id:
            raise ValueError("evaluation trace identity changed within one stream")
        if previous.sequence != len(existing) - 1:
            raise ValueError("evaluation trace sequence is not contiguous")
    item = EvaluationTraceEvent(
        trace_id=trace_id,
        run_id=run_id,
        question_id=question_id,
        sequence=len(existing),
        phase=phase,
        event=event,
        status=status,
        occurred_at=occurred_at or datetime.now(UTC),
        details=safe_trace_details(details or {}),
        usage=usage or TraceResourceUsage(),
    )
    return [*existing, item]
