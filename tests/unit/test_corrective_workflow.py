"""固定 LangGraph 拓扑、节点纯度、停止路径、Trace 与检查点测试。"""

from pathlib import Path

import pytest

from kg_crag.correction.config import CorrectiveWorkflowConfig
from kg_crag.correction.executor import ActionExecutor, RetrieverTool
from kg_crag.correction.identity import build_run_identity
from kg_crag.models import (
    BudgetLedger,
    CorrectionAction,
    CorrectionState,
    Evidence,
    EvidenceSourceType,
    StopReason,
)
from kg_crag.retrieval.mock import MockRetriever
from kg_crag.workflow.artifacts import CheckpointStore, ContentAddressedCache
from kg_crag.workflow.corrective import build_corrective_graph, run_corrective_workflow
from kg_crag.workflow.nodes import WorkflowDependencies, assess_node
from kg_crag.workflow.state import agent_state_from_correction
from kg_crag.workflow.trace import append_trace


def _evidence() -> Evidence:
    return Evidence(
        evidence_id="e1",
        content="What architecture does Voyager use: a skill library architecture",
        source_type=EvidenceSourceType.CHUNK,
        source_id="chunk-1",
        paper_id="paper-1",
    )


def _state() -> CorrectionState:
    identity = build_run_identity(
        "What architecture does Voyager use?",
        [],
        [],
        config_hash="a" * 64,
        prompt_version="prompt-v1",
        model_revision="mock-v1",
        corpus_snapshot="corpus-v1",
        dense_version="dense-v1",
        sparse_version="sparse-v1",
        graph_version="graph-v1",
        rules_version="facet-rules-v1",
        coverage_version="coverage-v1",
        policy_version="policy-v1",
    )
    return CorrectionState(
        identity=identity,
        question_id="q1",
        question="What architecture does Voyager use?",
        budget=BudgetLedger(),
    )


def _deps(
    initial: tuple[Evidence, ...],
    corrective: list[Evidence],
    *,
    artifact_cache: ContentAddressedCache | None = None,
) -> WorkflowDependencies:
    retriever = MockRetriever(corrective)
    tools = {
        action: RetrieverTool(retriever)
        for action in (
            CorrectionAction.DENSE,
            CorrectionAction.SPARSE,
            CorrectionAction.HYBRID,
            CorrectionAction.REWRITE,
            CorrectionAction.DECOMPOSE,
            CorrectionAction.ADJUST,
        )
    }
    return WorkflowDependencies(
        config=CorrectiveWorkflowConfig(),
        executor=ActionExecutor(tools, max_candidates=20),
        initial_evidence=initial,
        artifact_cache=artifact_cache,
    )


def test_graph_compiles_only_on_explicit_call_and_has_fixed_nodes() -> None:
    compiled = build_corrective_graph(_deps((), []))
    drawing = compiled.get_graph()
    names = set(drawing.nodes)
    assert {
        "requirements",
        "initial_retrieve",
        "assess",
        "decide",
        "execute",
        "reassess",
        "finalize",
    } <= names


@pytest.mark.asyncio
async def test_first_retrieval_sufficient_stops_without_corrective_action() -> None:
    result = await run_corrective_workflow(_state(), _deps((_evidence(),), []))
    assert result.stop.reason is StopReason.SUFFICIENT
    assert result.state.retrieval_round == 1
    assert result.state.action_history == []


@pytest.mark.asyncio
async def test_one_correction_recovers_then_stops_at_two_rounds() -> None:
    result = await run_corrective_workflow(_state(), _deps((), [_evidence()]))
    assert result.stop.reason is StopReason.SUFFICIENT
    assert result.state.retrieval_round == 2
    assert len(result.state.action_history) == 1
    assert result.state.budget.used.retrieval_rounds == 2


@pytest.mark.asyncio
async def test_empty_correction_is_internal_knowledge_missing_and_checkpoint_replays(
    tmp_path: Path,
) -> None:
    store = CheckpointStore(tmp_path / "runs", workspace_root=tmp_path)
    cache = ContentAddressedCache(tmp_path / "cache", workspace_root=tmp_path)
    first = await run_corrective_workflow(
        _state(), _deps((), [], artifact_cache=cache), checkpoint_store=store
    )
    assert first.stop.reason is StopReason.INTERNAL_KNOWLEDGE_MISSING
    assert (tmp_path / "runs" / first.identity.run_id / "manifest.json").is_file()
    assert list((tmp_path / "cache" / "coverage").glob("*.json"))
    assert list((tmp_path / "cache" / "decisions").glob("*.json"))
    # 半写或损坏文件不会遮蔽最后一个合法检查点。
    (tmp_path / "runs" / first.identity.run_id / "999-finalize.json").write_text(
        "{", encoding="utf-8"
    )
    second = await run_corrective_workflow(
        _state(),
        _deps((), [_evidence()], artifact_cache=cache),
        checkpoint_store=store,
    )
    assert second.state == first.state
    assert second.stop == first.stop


def test_node_does_not_mutate_input_and_trace_rejects_sensitive_or_unbounded_values() -> None:
    state = _state()
    original = agent_state_from_correction(state)
    before = dict(original)
    # assess 节点在 facets 为空时仍产生一个合法空矩阵，不改变输入字典。
    output = assess_node(original, _deps((), []))
    assert original == before
    assert output is not original
    with pytest.raises(ValueError, match="forbidden"):
        append_trace([], trace_id="t", node="n", event="e", details={"api_key": "x"})
    with pytest.raises(ValueError, match="bounded scalar"):
        append_trace([], trace_id="t", node="n", event="e", details={"candidates": ["body"]})
