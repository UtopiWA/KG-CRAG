"""基础运行契约的边界测试。"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from kg_crag.errors import KGCRAGError
from kg_crag.models import ErrorCode, ErrorDetail, HealthResponse, TraceEvent
from kg_crag.workflow.state import AgentState
from kg_crag.workflow.state import TraceEvent as WorkflowTraceEvent


def test_error_detail_rejects_unknown_and_sensitive_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        ErrorDetail.model_validate(
            {"code": ErrorCode.VALIDATION, "message": "invalid", "unknown": True}
        )

    for key in ("password", "Access-Token", "api_key", "Authorization", "client_secret"):
        with pytest.raises(ValidationError, match="sensitive context keys"):
            ErrorDetail(
                code=ErrorCode.VALIDATION,
                message="invalid",
                context={key: "must-not-be-recorded"},
            )


def test_error_detail_preserves_retryable_semantics() -> None:
    detail = ErrorDetail(
        code=ErrorCode.EXTERNAL_SERVICE,
        message="temporary upstream failure",
        retryable=True,
        context={"service": "example"},
    )
    assert detail.retryable is True
    assert detail.code is ErrorCode.EXTERNAL_SERVICE


def test_trace_event_validates_time_sequence_and_sensitive_details() -> None:
    event = TraceEvent(
        trace_id="trace-1",
        sequence=0,
        node="router",
        event="completed",
        occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
        details={"strategy": "hybrid"},
    )
    assert event.schema_version == "v1"

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        TraceEvent.model_validate(event.model_dump() | {"sequence": -1})
    with pytest.raises(ValidationError, match="include a timezone"):
        TraceEvent(
            trace_id="trace-1",
            sequence=0,
            node="router",
            event="completed",
            occurred_at=datetime(2026, 1, 1),
        )
    with pytest.raises(ValidationError, match="sensitive context keys"):
        TraceEvent(
            trace_id="trace-1",
            sequence=0,
            node="router",
            event="completed",
            occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
            details={"api-key": "must-not-be-recorded"},
        )


def test_health_response_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        HealthResponse.model_validate(
            {"status": "ok", "version": "0.1.0", "environment": "test", "token": "secret"}
        )


def test_kg_crag_error_keeps_context_out_of_string_representation() -> None:
    detail = ErrorDetail(
        code=ErrorCode.EXTERNAL_SERVICE,
        message="temporary upstream failure",
        retryable=True,
        context={"request_id": "private-request-id"},
    )
    error = KGCRAGError(detail)

    assert error.detail is detail
    assert "private-request-id" not in str(error)
    assert str(error) == "external_service_error: temporary upstream failure"


def test_error_codes_distinguish_retryable_external_and_invalid_input_errors() -> None:
    external = KGCRAGError(
        ErrorDetail(
            code=ErrorCode.EXTERNAL_SERVICE,
            message="service unavailable",
            retryable=True,
        )
    )
    invalid = KGCRAGError(
        ErrorDetail(
            code=ErrorCode.VALIDATION,
            message="top_k must be positive",
            retryable=False,
        )
    )

    assert (external.detail.code, external.detail.retryable) == (
        ErrorCode.EXTERNAL_SERVICE,
        True,
    )
    assert (invalid.detail.code, invalid.detail.retryable) == (ErrorCode.VALIDATION, False)


def test_agent_state_uses_public_trace_event_without_changing_existing_fields() -> None:
    event = WorkflowTraceEvent(
        trace_id="trace-1",
        sequence=0,
        node="router",
        event="completed",
        occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    state = AgentState(
        query="What is corrective RAG?",
        query_types=["semantic"],
        route=None,
        subqueries=[],
        retrieval_round=0,
        reflection_round=0,
        candidates=[],
        selected_evidence=[],
        retrieval_evaluation=None,
        draft_answer=None,
        answer_evaluation=None,
        citations=[],
        trace=[event],
        final_answer=None,
    )

    assert WorkflowTraceEvent is TraceEvent
    assert state["trace"] == [event]
    assert set(state) == {
        "query",
        "query_types",
        "route",
        "subqueries",
        "retrieval_round",
        "reflection_round",
        "candidates",
        "selected_evidence",
        "retrieval_evaluation",
        "draft_answer",
        "answer_evaluation",
        "citations",
        "trace",
        "final_answer",
    }
