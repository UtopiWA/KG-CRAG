"""固定 LangGraph 拓扑及可恢复纠错工作流 runner。"""

from __future__ import annotations

import warnings
from datetime import UTC, datetime
from typing import Literal, cast

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

# LangGraph 0.2.x 的检查点模块会在导入时发出未来默认值告警；本项目没有使用
# 该序列化器，且已自行实现严格原子检查点，因此只在依赖导入边界屏蔽此告警。
with warnings.catch_warnings():
    warnings.simplefilter("ignore", LangChainPendingDeprecationWarning)
    from langgraph.graph import END, StateGraph
    from langgraph.graph.state import CompiledStateGraph

from kg_crag.models import (
    CorrectionRunResult,
    CorrectionState,
    CorrectiveStrategy,
    WorkflowStage,
)
from kg_crag.workflow.artifacts import CheckpointStore
from kg_crag.workflow.edges import route_after_assess, route_after_decide, route_after_execute
from kg_crag.workflow.nodes import (
    WorkflowDependencies,
    assess_node,
    decide_node,
    execute_node,
    finalize_node,
    initial_retrieve_node,
    reassess_node,
    requirements_node,
)
from kg_crag.workflow.state import (
    AgentState,
    agent_state_from_correction,
    correction_state_from_agent,
)


def build_corrective_graph(
    deps: WorkflowDependencies,
    *,
    entry_stage: WorkflowStage = WorkflowStage.REQUIREMENTS,
) -> CompiledStateGraph:
    """只在显式调用时导入/编译图；固定边不允许动态节点名。"""

    async def requirements(state: AgentState) -> AgentState:
        return await requirements_node(state, deps)

    async def initial_retrieve(state: AgentState) -> AgentState:
        return await initial_retrieve_node(state, deps)

    async def execute(state: AgentState) -> AgentState:
        return await execute_node(state, deps)

    graph = StateGraph(AgentState)
    graph.add_node("requirements", requirements)
    graph.add_node("initial_retrieve", initial_retrieve)
    graph.add_node("assess", lambda state: assess_node(state, deps))
    graph.add_node("decide", lambda state: decide_node(state, deps))
    graph.add_node("execute", execute)
    graph.add_node("reassess", lambda state: reassess_node(state, deps))
    graph.add_node("finalize", lambda state: finalize_node(state, deps))
    graph.set_entry_point(entry_stage.value)
    graph.add_edge("requirements", "initial_retrieve")
    graph.add_edge("initial_retrieve", "assess")
    graph.add_conditional_edges(
        "assess", route_after_assess, {"decide": "decide", "finalize": "finalize"}
    )
    graph.add_conditional_edges(
        "decide", route_after_decide, {"execute": "execute", "finalize": "finalize"}
    )
    graph.add_conditional_edges(
        "execute", route_after_execute, {"reassess": "reassess", "finalize": "finalize"}
    )
    graph.add_edge("reassess", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


async def run_corrective_workflow(
    initial: CorrectionState,
    deps: WorkflowDependencies,
    *,
    checkpoint_store: CheckpointStore | None = None,
) -> CorrectionRunResult:
    started = datetime.now(UTC)
    state = initial
    entry_stage = WorkflowStage.REQUIREMENTS
    if checkpoint_store is not None:
        checkpoint = checkpoint_store.load_latest(initial.identity)
        if checkpoint is not None:
            state = checkpoint.state
            if state.stop is not None:
                return CorrectionRunResult(
                    identity=state.identity,
                    state=state,
                    stop=state.stop,
                    started_at=started,
                    finished_at=datetime.now(UTC),
                )
            entry_stage = _next_stage(checkpoint.stage, agent_state_from_correction(state))
    compiled = build_corrective_graph(deps, entry_stage=entry_stage)
    current_agent = agent_state_from_correction(state)
    sequence = len(state.trace)
    async for update in compiled.astream(current_agent, stream_mode="updates"):
        if not isinstance(update, dict) or len(update) != 1:
            continue
        node_name, node_state = next(iter(update.items()))
        if node_name == "__interrupt__" or not isinstance(node_state, dict):
            continue
        current_agent = cast(AgentState, node_state)
        state = correction_state_from_agent(current_agent)
        if checkpoint_store is not None:
            stage = WorkflowStage(node_name)
            checkpoint_store.save(stage, state, sequence)
            sequence += 1
    final = correction_state_from_agent(current_agent)
    if final.stop is None:
        raise RuntimeError("corrective graph completed without a stop result")
    if checkpoint_store is not None:
        checkpoint_store.save_manifest(final.identity)
    return CorrectionRunResult(
        identity=final.identity,
        state=final,
        stop=final.stop,
        started_at=started,
        finished_at=datetime.now(UTC),
    )


def _next_stage(stage: WorkflowStage, state: AgentState) -> WorkflowStage:
    """从最后完整阶段计算固定的下一阶段，不重复外部调用或计费。"""

    if stage is WorkflowStage.REQUIREMENTS:
        return WorkflowStage.INITIAL_RETRIEVE
    if stage is WorkflowStage.INITIAL_RETRIEVE:
        return WorkflowStage.ASSESS
    if stage is WorkflowStage.ASSESS:
        return WorkflowStage(route_after_assess(state))
    if stage is WorkflowStage.DECIDE:
        return WorkflowStage(route_after_decide(state))
    if stage is WorkflowStage.EXECUTE:
        return WorkflowStage(route_after_execute(state))
    if stage is WorkflowStage.REASSESS:
        return WorkflowStage.FINALIZE
    return WorkflowStage.FINALIZE


def strategy_initial_action(
    strategy: CorrectiveStrategy, *, relationship_question: bool
) -> Literal["hybrid", "graph"]:
    if strategy is CorrectiveStrategy.TYPE_ROUTER and relationship_question:
        return "graph"
    return "hybrid"
