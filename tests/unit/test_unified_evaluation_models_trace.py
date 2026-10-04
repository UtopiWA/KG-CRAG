"""统一评测模型与 Trace v2 兼容性测试。"""

from datetime import UTC, datetime

import pytest
from pydantic import TypeAdapter, ValidationError

from kg_crag.models import (
    EvaluationAnswerPoint,
    EvaluationFacetTarget,
    EvaluationSplit,
    EvaluationTraceEvent,
    EvaluationTracePhase,
    EvaluationTraceStatus,
    KnowledgeSufficiency,
    StressCategory,
    TraceEvent,
    TraceRecord,
    UnifiedEvaluationQuestion,
    UnifiedQuestionType,
)
from kg_crag.observability import (
    append_evaluation_trace,
    normalize_trace_event,
    parse_trace_record,
)


def _question(**updates: object) -> UnifiedEvaluationQuestion:
    payload: dict[str, object] = {
        "question_id": "ueq-fixture-001",
        "split": EvaluationSplit.DEV,
        "question": "Which evidence supports the fixture?",
        "normalized_question": "which evidence supports the fixture?",
        "question_types": [UnifiedQuestionType.FACTOID],
        "stress_category": StressCategory.CONTROL,
        "knowledge_sufficiency": KnowledgeSufficiency.SUFFICIENT,
        "answer_points": [
            EvaluationAnswerPoint(
                point_id="point-0000000000000001",
                description="The target evidence supports the answer.",
                evidence_ids=["evidence-1"],
            )
        ],
        "facets": [
            EvaluationFacetTarget(
                facet_id="facet-0000000000000001",
                description="Core answer facet.",
                target_evidence_ids=["evidence-1"],
            )
        ],
        "relevant_evidence": {"evidence-1": 3},
        "minimum_sufficient_evidence_sets": [["evidence-1"]],
        "leakage_group_id": "leak-fixture-001",
        "source_group_ids": ["source-fixture-001"],
        "source_fixture": "tests/fixtures/unified.json#fixture-001",
    }
    payload.update(updates)
    return UnifiedEvaluationQuestion.model_validate(payload)


def test_question_rejects_unknown_fields_and_dangling_evidence() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        _question(unknown=True)
    with pytest.raises(ValidationError, match="unknown relevant evidence"):
        _question(
            answer_points=[
                EvaluationAnswerPoint(
                    point_id="point-0000000000000001",
                    description="Invalid target.",
                    evidence_ids=["missing"],
                )
            ]
        )


def test_question_rejects_incomplete_minimum_set_and_invalid_missing_truth() -> None:
    second = EvaluationFacetTarget(
        facet_id="facet-0000000000000002",
        description="Second required facet.",
        target_evidence_ids=["evidence-2"],
    )
    with pytest.raises(ValidationError, match="does not cover every required facet"):
        _question(
            facets=[*_question().facets, second],
            relevant_evidence={"evidence-1": 3, "evidence-2": 2},
        )
    with pytest.raises(ValidationError, match="knowledge-missing"):
        _question(knowledge_sufficiency=KnowledgeSufficiency.INSUFFICIENT)


def test_trace_v2_requires_question_scope_timezone_and_bounded_details() -> None:
    base = {
        "trace_id": "trace-fixture",
        "run_id": "unified-run-" + "a" * 32,
        "sequence": 0,
        "phase": EvaluationTracePhase.RETRIEVE,
        "event": "started",
        "status": EvaluationTraceStatus.STARTED,
        "occurred_at": datetime.now(UTC),
    }
    with pytest.raises(ValidationError, match="requires question_id"):
        EvaluationTraceEvent(**base)
    with pytest.raises(ValidationError, match="timezone"):
        EvaluationTraceEvent.model_validate(
            {**base, "question_id": "ueq-fixture-001", "occurred_at": datetime.now()}
        )
    with pytest.raises(ValidationError, match="bounded length"):
        EvaluationTraceEvent(
            **base,
            question_id="ueq-fixture-001",
            details={"label": "x" * 301},
        )


def test_trace_append_sequence_and_sensitive_keys_are_enforced() -> None:
    events = append_evaluation_trace(
        [],
        trace_id="trace-fixture",
        run_id="unified-run-" + "b" * 32,
        question_id="ueq-fixture-001",
        phase=EvaluationTracePhase.ASSESS,
        event="finished",
        status=EvaluationTraceStatus.SUCCEEDED,
    )
    events = append_evaluation_trace(
        events,
        trace_id="trace-fixture",
        run_id="unified-run-" + "b" * 32,
        phase=EvaluationTracePhase.AGGREGATE,
        event="aggregated",
        status=EvaluationTraceStatus.SUCCEEDED,
    )
    assert [item.sequence for item in events] == [0, 1]
    with pytest.raises(ValueError, match="forbidden"):
        append_evaluation_trace(
            events,
            trace_id="trace-fixture",
            run_id="unified-run-" + "b" * 32,
            phase=EvaluationTracePhase.PUBLISH,
            event="publish",
            status=EvaluationTraceStatus.STARTED,
            details={"raw_response": "secret"},
        )
    with pytest.raises(ValueError, match="limit"):
        append_evaluation_trace(
            events,
            trace_id="trace-fixture",
            run_id="unified-run-" + "b" * 32,
            phase=EvaluationTracePhase.PUBLISH,
            event="publish",
            status=EvaluationTraceStatus.STARTED,
            max_events=2,
        )


def test_v1_and_v2_parse_to_read_only_normalized_views() -> None:
    old = TraceEvent(
        trace_id="old-trace",
        sequence=0,
        node="retrieve",
        event="done",
        occurred_at=datetime.now(UTC),
        details={"count": 2},
    )
    old_json = old.model_dump_json()
    parsed = parse_trace_record(old_json)
    normalized = normalize_trace_event(parsed)
    assert normalized.source_schema_version == "v1"
    assert normalized.run_id is None and normalized.question_id is None
    assert old.model_dump_json() == old_json

    new = append_evaluation_trace(
        [],
        trace_id="new-trace",
        run_id="unified-run-" + "c" * 32,
        phase=EvaluationTracePhase.VALIDATE,
        event="validated",
        status=EvaluationTraceStatus.SUCCEEDED,
    )[0]
    assert normalize_trace_event(parse_trace_record(new.model_dump_json())).run_id == new.run_id
    with pytest.raises(ValidationError):
        TypeAdapter(TraceRecord).validate_python({"schema_version": "v9"})
