"""有界、脱敏且序号稳定的纠错 Trace 记录器。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from kg_crag.models import TraceEvent
from kg_crag.models.foundation import ScalarValue

_FORBIDDEN = (
    "prompt",
    "reasoning",
    "password",
    "secret",
    "api_key",
    "authorization",
    "credential",
    "excerpt",
    "content",
    "body",
    "raw_response",
)


def safe_trace_details(details: dict[str, Any]) -> dict[str, ScalarValue]:
    """Trace 只保留有限标量，禁止正文、Prompt、凭据和私有推理。"""

    safe: dict[str, ScalarValue] = {}
    for key, value in details.items():
        normalized = key.casefold().replace("-", "_")
        if any(marker in normalized for marker in _FORBIDDEN):
            raise ValueError(f"trace detail key is forbidden: {key}")
        if not isinstance(value, str | int | float | bool) and value is not None:
            raise ValueError("trace details must be bounded scalar values")
        if isinstance(value, str) and len(value) > 300:
            raise ValueError("trace string detail exceeds the bounded length")
        safe[key] = value
    return safe


def append_trace(
    existing: list[TraceEvent],
    *,
    trace_id: str,
    node: str,
    event: str,
    details: dict[str, Any] | None = None,
    max_events: int = 100,
    occurred_at: datetime | None = None,
) -> list[TraceEvent]:
    if len(existing) >= max_events:
        raise ValueError("trace event limit reached")
    item = TraceEvent(
        trace_id=trace_id,
        sequence=len(existing),
        node=node,
        event=event,
        occurred_at=occurred_at or datetime.now(UTC),
        details=safe_trace_details(details or {}),
    )
    return [*existing, item]
