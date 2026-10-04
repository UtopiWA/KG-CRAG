"""回答状态、身份、原子检查点和 Trace 脱敏测试。"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from kg_crag.answering import build_answer_identity
from kg_crag.correction.identity import build_run_identity
from kg_crag.models import (
    AnswerRunIdentity,
    AnswerWorkflowStage,
    AnswerWorkflowState,
    BudgetLedger,
    CombinedBudgetSummary,
    CorrectionRunResult,
    CorrectionState,
    GroundedAnswerResult,
    StopReason,
    StopResult,
)
from kg_crag.workflow.answer_artifacts import AnswerCheckpointStore
from kg_crag.workflow.state import agent_state_from_correction, correction_state_from_agent
from kg_crag.workflow.trace import append_trace


def _correction() -> CorrectionRunResult:
    identity = build_run_identity(
        "question",
        [],
        [],
        config_hash="a" * 64,
        prompt_version="prompt-v1",
        model_revision="model-v1",
        corpus_snapshot="corpus-v1",
        dense_version="dense-v1",
        sparse_version="sparse-v1",
        graph_version="graph-v1",
        rules_version="rules-v1",
        coverage_version="coverage-v1",
        policy_version="policy-v1",
    )
    budget = BudgetLedger()
    stop = StopResult(
        reason=StopReason.SUFFICIENT,
        message="sufficient",
        budget=budget,
    )
    state = CorrectionState(
        identity=identity,
        question_id="q1",
        question="question",
        budget=budget,
        stop=stop,
    )
    now = datetime.now(UTC)
    return CorrectionRunResult(
        identity=identity,
        state=state,
        stop=stop,
        started_at=now,
        finished_at=now,
    )


def _answer_identity(
    correction: CorrectionRunResult, *, provider: str = "disabled"
) -> AnswerRunIdentity:
    return build_answer_identity(
        correction,
        config_hash="b" * 64,
        answer_prompt_version="c" * 64,
        critic_prompt_version="d" * 64,
        checker_version="checker-v1",
        policy_version="policy-v1",
        model_revision="model-v1",
        search_provider=provider,
        search_provider_version="provider-v1",
    )


def test_agent_state_keeps_corrective_roundtrip_and_strict_answer_defaults() -> None:
    correction = _correction()
    agent = agent_state_from_correction(correction.state)
    assert agent["draft_answer"] is None
    assert agent["answer_evaluation"] is None
    assert agent["external_evidence"] == []
    assert correction_state_from_agent(agent) == correction.state


def test_answer_identity_binds_provider_and_input_versions() -> None:
    correction = _correction()
    first = _answer_identity(correction)
    assert first == _answer_identity(correction)
    assert first.run_id != _answer_identity(correction, provider="mock").run_id


def test_answer_checkpoint_is_atomic_contiguous_and_result_is_recoverable(
    tmp_path: Path,
) -> None:
    correction = _correction()
    identity = _answer_identity(correction)
    store = AnswerCheckpointStore(tmp_path / "runs", workspace_root=tmp_path)
    state = AnswerWorkflowState(identity=identity, correction=correction)
    store.save(state, 0)
    checked = state.model_copy(update={"stage": AnswerWorkflowStage.CHECK})
    store.save(checked, 1)
    directory = tmp_path / "runs" / identity.run_id
    (directory / "002-decide.json").write_text("{", encoding="utf-8")
    latest = store.load_latest(identity)
    assert latest is not None and latest.sequence == 1
    now = datetime.now(UTC)
    result = GroundedAnswerResult(
        identity=identity,
        trace_id=identity.run_id,
        question="question",
        answer="grounded answer",
        internal_stop_reason=StopReason.SUFFICIENT,
        stop_reason="accepted",
        used_external=False,
        confidence=0.8,
        budget=CombinedBudgetSummary(
            internal=correction.state.budget,
            answer=state.budget,
        ),
        started_at=now,
        finished_at=now,
    )
    store.save_result(result)
    assert store.load_result(identity) == result
    manifest = store.save_manifest(identity)
    assert manifest.is_file()


def test_atomic_failure_cleans_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    correction = _correction()
    identity = _answer_identity(correction)
    store = AnswerCheckpointStore(tmp_path / "runs", workspace_root=tmp_path)
    state = AnswerWorkflowState(identity=identity, correction=correction)

    def fail_replace(self: Path, target: Path) -> Path:
        del target
        raise OSError("injected replace failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        store.save(state, 0)
    directory = tmp_path / "runs" / identity.run_id
    assert not list(directory.glob(".*.tmp"))


def test_trace_rejects_web_body_and_enforces_event_limit() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        append_trace(
            [],
            trace_id="t",
            node="web",
            event="result",
            details={"excerpt": "page body"},
        )
    first = append_trace(
        [],
        trace_id="t",
        node="web",
        event="selected",
        details={"domain": "arxiv.org", "result_count": 1, "input_tokens": 10},
        max_events=1,
    )
    with pytest.raises(ValueError, match="limit"):
        append_trace(first, trace_id="t", node="web", event="done", max_events=1)
