"""框架无关的纠错工作流节点；每个节点返回重新校验的新状态。"""

from __future__ import annotations

from dataclasses import dataclass

from kg_crag.correction.budget import BudgetManager
from kg_crag.correction.config import CorrectiveWorkflowConfig
from kg_crag.correction.coverage import FacetMatcher, SourceVerifier, build_coverage_matrix
from kg_crag.correction.executor import ActionExecutor, deduplicate_evidence
from kg_crag.correction.identity import stable_digest
from kg_crag.correction.policy import choose_action
from kg_crag.correction.requirements import (
    RequirementCache,
    RequirementProvider,
    analyze_question,
    generate_requirements,
)
from kg_crag.correction.sufficiency import assess_sufficiency
from kg_crag.models import (
    ActionStatus,
    BudgetUsage,
    CorrectionDecision,
    CorrectionState,
    CoverageMatrix,
    ErrorCode,
    ErrorDetail,
    Evidence,
    StopResult,
    TraceEvent,
)
from kg_crag.workflow.artifacts import ContentAddressedCache
from kg_crag.workflow.edges.corrective import stop_reason_from_state
from kg_crag.workflow.state import (
    AgentState,
    agent_state_from_correction,
    correction_state_from_agent,
)
from kg_crag.workflow.trace import append_trace


@dataclass(frozen=True)
class WorkflowDependencies:
    config: CorrectiveWorkflowConfig
    executor: ActionExecutor
    initial_evidence: tuple[Evidence, ...] = ()
    requirement_provider: RequirementProvider | None = None
    requirement_prompt: str = ""
    requirement_cache: RequirementCache | None = None
    artifact_cache: ContentAddressedCache | None = None
    matcher: FacetMatcher | None = None
    source_verifier: SourceVerifier | None = None


def _validated_update(state: CorrectionState, **updates: object) -> AgentState:
    payload = state.model_dump(mode="python")
    payload.update(updates)
    return agent_state_from_correction(CorrectionState.model_validate(payload))


def _trace(
    state: CorrectionState, node: str, event: str, details: dict[str, object]
) -> list[TraceEvent]:
    return append_trace(
        state.trace,
        trace_id=state.identity.run_id,
        node=node,
        event=event,
        details=details,
        max_events=100,
    )


async def requirements_node(state: AgentState, deps: WorkflowDependencies) -> AgentState:
    current = correction_state_from_agent(state)
    analysis = analyze_question(current.question_id, current.question)
    will_call_provider = bool(
        deps.config.facets.allow_llm
        and deps.requirement_provider is not None
        and (analysis.complex or analysis.confidence < deps.config.facets.confidence_threshold)
    )
    manager = BudgetManager(current.budget)
    reservation_token: str | None = None
    if will_call_provider:
        reservation_token = manager.reserve(
            BudgetUsage(
                llm_calls=1,
                input_tokens=4000,
                output_tokens=2000,
                context_chars=10_000,
                latency_ms=5000,
            )
        )
        if reservation_token is None:
            return finalize_invalid(current, "requirement provider exceeds remaining budget")
    facets, usage, source = await generate_requirements(
        current.question_id,
        current.question,
        deps.config.facets,
        provider=deps.requirement_provider,
        system_prompt=deps.requirement_prompt,
        cache=deps.requirement_cache,
    )
    if reservation_token is not None:
        if source == "rule_fallback":
            manager.settle(reservation_token, failed=True)
        else:
            manager.settle(reservation_token, usage)
    return _validated_update(
        current,
        facets=facets,
        budget=manager.ledger,
        trace=_trace(
            current,
            "requirements",
            "requirements_generated",
            {
                "count": len(facets),
                "source": source,
                "llm_calls": manager.ledger.used.llm_calls,
                "llm_input_count": manager.ledger.used.input_tokens,
                "llm_output_count": manager.ledger.used.output_tokens,
            },
        ),
    )


async def initial_retrieve_node(state: AgentState, deps: WorkflowDependencies) -> AgentState:
    current = correction_state_from_agent(state)
    # 离线 fixture 是冻结输入，不调用数据库；逻辑上仍记为首次内部检索轮次。
    evidence = deduplicate_evidence(
        deps.initial_evidence,
        limit=deps.config.max_total_evidence,
        max_content_chars=deps.config.coverage.max_evidence_chars,
    )
    manager = BudgetManager(current.budget)
    usage = BudgetUsage(retrieval_rounds=1, candidates=len(evidence))
    reservation = manager.reserve(usage)
    if reservation is None:
        return finalize_invalid(current, "initial retrieval exceeds remaining budget")
    manager.settle(reservation, usage)
    return _validated_update(
        current,
        candidates=evidence,
        retrieval_round=1,
        budget=manager.ledger,
        trace=_trace(
            current,
            "initial_retrieve",
            "retrieval_completed",
            {"candidates": len(evidence), "offline": True},
        ),
    )


def assess_node(state: AgentState, deps: WorkflowDependencies) -> AgentState:
    current = correction_state_from_agent(state)
    cache_key = stable_digest(
        {
            "identity": current.identity.model_dump(mode="json"),
            "facets": [item.model_dump(mode="json") for item in current.facets],
            "evidence": [item.model_dump(mode="json") for item in current.candidates],
            "coverage": deps.config.coverage.model_dump(mode="json"),
        }
    )
    matrix = (
        deps.artifact_cache.load("coverage", cache_key, CoverageMatrix)
        if deps.artifact_cache
        else None
    )
    cache_hit = matrix is not None
    if matrix is None:
        matrix = build_coverage_matrix(
            current.facets,
            current.candidates,
            rule_version=deps.config.coverage.version,
            matcher=deps.matcher,
            verifier=deps.source_verifier,
        )
        if deps.artifact_cache:
            deps.artifact_cache.save("coverage", cache_key, matrix)
    assessment = assess_sufficiency(
        current.facets,
        current.candidates,
        matrix,
        min_support=deps.config.coverage.min_support,
        min_sources=deps.config.coverage.min_sources,
        max_selected=deps.config.coverage.max_selected_evidence,
    )
    by_id = {item.evidence_id: item for item in current.candidates}
    selected = [by_id[item] for item in assessment.selected_evidence_ids]
    fingerprint = stable_digest(
        {
            "matrix": matrix.matrix_id,
            "evidence": sorted(matrix.evidence_ids),
            "actions": [item.request.action_id for item in current.action_history],
        }
    )
    return _validated_update(
        current,
        coverage_matrix=matrix,
        sufficiency=assessment,
        selected_evidence=selected,
        state_fingerprint=fingerprint,
        trace=_trace(
            current,
            "assess",
            "coverage_assessed",
            {
                "covered": len(assessment.covered_facet_ids),
                "missing": len(assessment.missing_required_facet_ids),
                "sufficient": assessment.sufficient,
                "cache_hit": cache_hit,
            },
        ),
    )


def decide_node(state: AgentState, deps: WorkflowDependencies) -> AgentState:
    current = correction_state_from_agent(state)
    if current.sufficiency is None:
        return finalize_invalid(current, "decision requires sufficiency assessment")
    history_ids = {item.request.action_id for item in current.action_history}
    cache_key = stable_digest(
        {
            "identity": current.identity.model_dump(mode="json"),
            "assessment": current.sufficiency.model_dump(mode="json"),
            "budget": current.budget.model_dump(mode="json"),
            "history": sorted(history_ids),
            "policy": deps.config.policy_version,
            "costs": deps.config.costs.model_dump(mode="json"),
            "actions": deps.config.actions.model_dump(mode="json"),
        }
    )
    decision = (
        deps.artifact_cache.load("decisions", cache_key, CorrectionDecision)
        if deps.artifact_cache
        else None
    )
    cache_hit = decision is not None
    if decision is None:
        decision = choose_action(
            current.question,
            current.facets,
            current.sufficiency,
            current.budget,
            history_ids,
            deps.config,
            state_improved=True,
        )
        if deps.artifact_cache:
            deps.artifact_cache.save("decisions", cache_key, decision)
    candidates = [] if decision.selected is None else [decision.selected]
    return _validated_update(
        current,
        candidate_actions=candidates,
        decision=decision,
        trace=_trace(
            current,
            "decide",
            "action_decided",
            {
                "selected": decision.selected.action.value if decision.selected else "stop",
                "utility": decision.utility,
                "cache_hit": cache_hit,
            },
        ),
    )


async def execute_node(state: AgentState, deps: WorkflowDependencies) -> AgentState:
    current = correction_state_from_agent(state)
    if (
        current.decision is None
        or current.decision.selected is None
        or current.decision.estimate is None
    ):
        return finalize_invalid(current, "execution requires a selected action")
    manager = BudgetManager(current.budget)
    token = manager.reserve(current.decision.estimate)
    if token is None:
        decision = current.decision.model_copy(
            update={
                "selected": None,
                "estimate": None,
                "utility": 0.0,
                "reason_code": "budget_exhausted",
                "stop_reason": "budget_exhausted",
            }
        )
        return _validated_update(current, decision=decision)
    result = await deps.executor.execute(current.decision.selected, current.decision.estimate)
    manager.settle(token, result.actual_usage, failed=result.status is ActionStatus.FAILED)
    merged = deduplicate_evidence(
        [*current.candidates, *result.evidence],
        limit=deps.config.max_total_evidence,
        max_content_chars=deps.config.coverage.max_evidence_chars,
    )
    errors = list(current.errors)
    if result.error is not None:
        errors.append(result.error)
    return _validated_update(
        current,
        candidates=merged,
        action_history=[*current.action_history, result],
        retrieval_round=current.retrieval_round + 1,
        budget=manager.ledger,
        errors=errors,
        trace=_trace(
            current,
            "execute",
            "action_executed",
            {
                "action": result.request.action.value,
                "status": result.status.value,
                "new_evidence": len(result.evidence),
                "retrieval_rounds": manager.ledger.used.retrieval_rounds,
                "candidates": manager.ledger.used.candidates,
                "error_code": result.error.code.value if result.error else "none",
            },
        ),
    )


def reassess_node(state: AgentState, deps: WorkflowDependencies) -> AgentState:
    return assess_node(state, deps)


def finalize_invalid(state: CorrectionState, message: str) -> AgentState:
    error = ErrorDetail(code=ErrorCode.VALIDATION, message=message, retryable=False)
    stop = StopResult(
        reason="invalid_input",
        message=message,
        missing_facet_ids=[],
        selected_evidence_ids=[item.evidence_id for item in state.selected_evidence],
        budget=state.budget,
    )
    return _validated_update(
        state,
        errors=[*state.errors, error],
        stop=stop,
        trace=_trace(
            state,
            "finalize",
            "workflow_failed",
            {"reason": "invalid_input", "error_code": error.code.value},
        ),
    )


def finalize_node(state: AgentState, deps: WorkflowDependencies) -> AgentState:
    current = correction_state_from_agent(state)
    reason = stop_reason_from_state(state)
    missing = (
        current.sufficiency.missing_required_facet_ids if current.sufficiency is not None else []
    )
    stop = StopResult(
        reason=reason,
        message=f"corrective workflow stopped: {reason.value}",
        missing_facet_ids=missing,
        selected_evidence_ids=[item.evidence_id for item in current.selected_evidence],
        budget=current.budget,
    )
    return _validated_update(
        current,
        stop=stop,
        trace=_trace(
            current,
            "finalize",
            "workflow_stopped",
            {"reason": reason.value, "missing": len(missing)},
        ),
    )
