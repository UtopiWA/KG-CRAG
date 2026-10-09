"""可恢复、单轮补救的回答反思与受控 Web 工作流。"""

from __future__ import annotations

from datetime import UTC, datetime

from kg_crag.answering import build_answer_identity
from kg_crag.answering.budget import AnswerBudgetManager
from kg_crag.answering.config import grounded_answer_config_hash
from kg_crag.correction.actions import action_catalog, build_action_request
from kg_crag.correction.budget import BudgetManager
from kg_crag.correction.coverage import build_coverage_matrix
from kg_crag.correction.executor import deduplicate_evidence
from kg_crag.correction.sufficiency import assess_sufficiency
from kg_crag.errors import KGCRAGError
from kg_crag.models import (
    ActionStatus,
    AnswerCheckCode,
    AnswerWorkflowStage,
    AnswerWorkflowState,
    CorrectionAction,
    CorrectionRunResult,
    ErrorCode,
    ErrorDetail,
    GroundedAnswerResult,
    ReflectionAction,
    StopReason,
    StopResult,
)
from kg_crag.web import (
    TrustedSourcePolicy,
    build_external_coverage,
    build_web_query,
    convert_web_results,
)
from kg_crag.workflow.answer_artifacts import AnswerCheckpointStore
from kg_crag.workflow.nodes.answer import (
    AnswerWorkflowDependencies,
    check_answer_node,
    decide_answer_node,
    finalize_answer_node,
    generate_answer_node,
    prepare_answer_node,
)
from kg_crag.workflow.trace import append_trace


def _update(state: AnswerWorkflowState, **updates: object) -> AnswerWorkflowState:
    payload = state.model_dump(mode="python")
    payload.update(updates)
    return AnswerWorkflowState.model_validate(payload)


def _can_retry_provider_generation(state: AnswerWorkflowState) -> bool:
    """只重试一次快速、可重试的 Provider 故障；超时不重复占用整段请求窗口。"""

    generation_events = [
        item
        for item in state.trace
        if item.event in {"answer_generated", "answer_generation_failed"}
    ]
    if not generation_events or generation_events[-1].event != "answer_generation_failed":
        return False
    latest = generation_events[-1]
    if latest.details.get("failure_category") != "provider_request":
        return False
    if not state.errors or not state.errors[-1].retryable:
        return False
    if state.errors[-1].code is ErrorCode.TIMEOUT:
        return False
    return state.budget.used.answer_calls < state.budget.limit.answer_calls


def _can_retry_remediation_generation(state: AnswerWorkflowState) -> bool:
    """Web/补检已完成后，允许用剩余生成预算修复一次结构化输出。"""

    if not state.remediation_used or state.candidate is not None:
        return False
    failures = [item for item in state.trace if item.event == "answer_generation_failed"]
    if not failures:
        return False
    category = failures[-1].details.get("failure_category")
    retryable_categories = {
        "invalid_json",
        "invalid_schema",
        "unknown_citation",
        "unknown_facet",
        "unsupported_facet_binding",
    }
    return bool(
        category in retryable_categories
        and state.budget.used.answer_calls < state.budget.limit.answer_calls
    )


def _trace(
    state: AnswerWorkflowState,
    deps: AnswerWorkflowDependencies,
    node: str,
    event: str,
    **details: object,
) -> AnswerWorkflowState:
    return _update(
        state,
        trace=append_trace(
            state.trace,
            trace_id=state.identity.run_id,
            node=node,
            event=event,
            details=details,
            max_events=deps.config.max_trace_events,
        ),
    )


def _error(state: AnswerWorkflowState, detail: ErrorDetail) -> AnswerWorkflowState:
    return _update(state, errors=[*state.errors, detail])


def _checkpoint(
    store: AnswerCheckpointStore | None,
    state: AnswerWorkflowState,
    sequence: int,
) -> int:
    if store is not None:
        store.save(state, sequence)
    return sequence + 1


def _can_reretrieve(state: AnswerWorkflowState, deps: AnswerWorkflowDependencies) -> bool:
    config = deps.correction_config
    return bool(
        config is not None
        and deps.action_executor is not None
        and CorrectionAction.HYBRID in config.actions.enabled
        and state.correction.state.budget.used.retrieval_rounds
        < state.correction.state.budget.limit.retrieval_rounds
        and state.correction.stop.missing_facet_ids
    )


def _charge_reflection(
    state: AnswerWorkflowState,
) -> AnswerWorkflowState | None:
    if state.decision is None:
        return None
    manager = AnswerBudgetManager(state.budget)
    token = manager.reserve(state.decision.estimated_usage)
    if token is None:
        return None
    manager.settle(token, state.decision.estimated_usage)
    return _update(state, budget=manager.ledger, remediation_used=True)


async def _web_remediation(
    state: AnswerWorkflowState,
    deps: AnswerWorkflowDependencies,
) -> AnswerWorkflowState:
    decision = state.decision
    charged = _charge_reflection(state)
    if decision is None or charged is None or deps.search_provider is None:
        return _error(
            state,
            ErrorDetail(
                code=ErrorCode.VALIDATION,
                message="Web remediation exceeds budget or lacks a provider",
                retryable=False,
            ),
        )
    state = charged
    by_id = {item.facet_id: item for item in state.correction.state.facets}
    targets = [by_id[item] for item in decision.target_facet_ids if item in by_id]
    query = build_web_query(state.correction.state.question, targets)
    try:
        results = await deps.search_provider.search(query, max_results=deps.config.web.max_results)
        trusted = TrustedSourcePolicy(deps.config.web).select(results)
        evidence = convert_web_results(
            trusted,
            provider=deps.config.web.provider,
            converter_version=deps.external_coverage_version,
            config=deps.config.web,
        )
        coverage = build_external_coverage(
            targets,
            evidence,
            rule_version=deps.external_coverage_version,
        )
    except Exception as error:
        detail = (
            error.detail
            if isinstance(error, KGCRAGError)
            else ErrorDetail(
                code=ErrorCode.EXTERNAL_SERVICE,
                message="Web search or validation failed",
                retryable=False,
            )
        )
        failed = _error(state, detail)
        return _trace(
            _update(failed, stage=AnswerWorkflowStage.REMEDIATE),
            deps,
            "remediate",
            "web_failed",
            search_calls=state.budget.used.web_calls,
        )
    remediated = _update(
        state,
        stage=AnswerWorkflowStage.REMEDIATE,
        external_evidence=evidence,
        external_coverage=coverage,
    )
    return _trace(
        remediated,
        deps,
        "remediate",
        "web_completed",
        search_calls=remediated.budget.used.web_calls,
        trusted_results=len(evidence),
        matched_facets=len({item.facet_id for item in coverage if item.matched}),
    )


async def _reretrieve(
    state: AnswerWorkflowState,
    deps: AnswerWorkflowDependencies,
) -> AnswerWorkflowState:
    decision = state.decision
    charged = _charge_reflection(state)
    config = deps.correction_config
    executor = deps.action_executor
    if decision is None or charged is None or config is None or executor is None:
        return _error(
            state,
            ErrorDetail(
                code=ErrorCode.CONFIGURATION,
                message="internal re-retrieval is unavailable",
                retryable=False,
            ),
        )
    state = charged
    by_id = {item.facet_id: item for item in state.correction.state.facets}
    targets = [by_id[item] for item in decision.target_facet_ids if item in by_id]
    request = build_action_request(
        CorrectionAction.HYBRID,
        state.correction.state.question,
        targets,
        config.actions,
    )
    estimate = action_catalog(config.actions)[CorrectionAction.HYBRID].worst_case
    manager = BudgetManager(state.correction.state.budget)
    token = manager.reserve(estimate)
    if token is None:
        return _error(
            state,
            ErrorDetail(
                code=ErrorCode.VALIDATION,
                message="internal re-retrieval exceeds remaining budget",
                retryable=False,
            ),
        )
    result = await executor.execute(request, estimate)
    manager.settle(token, result.actual_usage, failed=result.status is ActionStatus.FAILED)
    current = state.correction.state
    errors = list(current.errors)
    if result.error is not None:
        errors.append(result.error)
    candidates = deduplicate_evidence(
        [*current.candidates, *result.evidence],
        limit=config.max_total_evidence,
        max_content_chars=config.coverage.max_evidence_chars,
    )
    if result.status is ActionStatus.FAILED:
        new_stop = StopResult(
            reason=StopReason.EXECUTION_FAILED,
            message="answer-stage internal re-retrieval failed",
            missing_facet_ids=decision.target_facet_ids,
            selected_evidence_ids=[item.evidence_id for item in current.selected_evidence],
            budget=manager.ledger,
        )
        new_state = current.model_copy(
            update={
                "budget": manager.ledger,
                "errors": errors,
                "action_history": [*current.action_history, result],
                "retrieval_round": current.retrieval_round + 1,
                "stop": new_stop,
            }
        )
    else:
        matrix = build_coverage_matrix(
            current.facets,
            candidates,
            rule_version=config.coverage.version,
        )
        assessment = assess_sufficiency(
            current.facets,
            candidates,
            matrix,
            min_support=config.coverage.min_support,
            min_sources=config.coverage.min_sources,
            max_selected=config.coverage.max_selected_evidence,
        )
        evidence_by_id = {item.evidence_id: item for item in candidates}
        selected = [evidence_by_id[item] for item in assessment.selected_evidence_ids]
        reason = (
            StopReason.SUFFICIENT
            if assessment.sufficient
            else StopReason.INTERNAL_KNOWLEDGE_MISSING
        )
        new_stop = StopResult(
            reason=reason,
            message=f"answer-stage re-retrieval stopped: {reason.value}",
            missing_facet_ids=assessment.missing_required_facet_ids,
            selected_evidence_ids=assessment.selected_evidence_ids,
            budget=manager.ledger,
        )
        new_state = current.model_copy(
            update={
                "candidates": candidates,
                "selected_evidence": selected,
                "coverage_matrix": matrix,
                "sufficiency": assessment,
                "budget": manager.ledger,
                "errors": errors,
                "action_history": [*current.action_history, result],
                "retrieval_round": current.retrieval_round + 1,
                "stop": new_stop,
            }
        )
    now = datetime.now(UTC)
    correction = CorrectionRunResult(
        identity=state.correction.identity,
        state=new_state,
        stop=new_stop,
        started_at=state.correction.started_at,
        finished_at=now,
    )
    remediated = _update(
        state,
        stage=AnswerWorkflowStage.REMEDIATE,
        correction=correction,
        errors=[*state.errors, *([result.error] if result.error else [])],
    )
    return _trace(
        remediated,
        deps,
        "remediate",
        "internal_reretrieval_completed",
        status=result.status.value,
        retrieval_rounds=manager.ledger.used.retrieval_rounds,
        new_evidence=len(result.evidence),
    )


def _can_rewrite_remediation_answer(state: AnswerWorkflowState) -> bool:
    """补证后只允许用第二次生成修复拒答或遗漏，不再调用工具。"""

    return bool(
        state.remediation_used
        and state.candidate is not None
        and state.evaluation is not None
        and any(
            item.code in {AnswerCheckCode.ABSTENTION, AnswerCheckCode.INCOMPLETE}
            for item in state.evaluation.findings
        )
        and state.budget.used.answer_calls < state.budget.limit.answer_calls
    )


async def _complete_decision(
    state: AnswerWorkflowState,
    deps: AnswerWorkflowDependencies,
    *,
    allow_web: bool,
    checkpoint_store: AnswerCheckpointStore | None,
    sequence: int,
) -> tuple[AnswerWorkflowState, int]:
    if state.decision is None:
        return state, sequence
    action = state.decision.action
    if action in {ReflectionAction.ACCEPT, ReflectionAction.CONSERVATIVE_STOP}:
        return state, sequence
    if action is ReflectionAction.WEB_SEARCH:
        state = await _web_remediation(state, deps)
        sequence = _checkpoint(checkpoint_store, state, sequence)
        if not state.external_evidence:
            state = decide_answer_node(
                state,
                deps,
                allow_web=allow_web,
                internal_can_retrieve=False,
            )
            sequence = _checkpoint(checkpoint_store, state, sequence)
            return state, sequence
        state = _update(
            state,
            stage=AnswerWorkflowStage.GENERATE,
            candidate=None,
            evaluation=None,
        )
        sequence = _checkpoint(checkpoint_store, state, sequence)
        state = await generate_answer_node(state, deps, stage=AnswerWorkflowStage.REMEDIATE)
    elif action is ReflectionAction.RERETRIEVE:
        state = await _reretrieve(state, deps)
        sequence = _checkpoint(checkpoint_store, state, sequence)
        if state.correction.stop.reason is StopReason.EXECUTION_FAILED:
            return state, sequence
        if state.correction.stop.reason is not StopReason.SUFFICIENT:
            # 二次检索仍不足时不再浪费一次必然无法通过完整性检查的生成调用。
            state = decide_answer_node(
                state,
                deps,
                allow_web=allow_web,
                internal_can_retrieve=False,
            )
            sequence = _checkpoint(checkpoint_store, state, sequence)
            return state, sequence
        state = _update(
            state,
            stage=AnswerWorkflowStage.GENERATE,
            candidate=None,
            evaluation=None,
        )
        sequence = _checkpoint(checkpoint_store, state, sequence)
        state = await generate_answer_node(state, deps, stage=AnswerWorkflowStage.REMEDIATE)
    else:
        charged = _charge_reflection(state)
        if charged is None:
            return _error(
                state,
                ErrorDetail(
                    code=ErrorCode.VALIDATION,
                    message="regeneration exceeds remaining budget",
                    retryable=False,
                ),
            ), sequence
        state = _update(
            charged,
            stage=AnswerWorkflowStage.GENERATE,
            candidate=None,
            evaluation=None,
        )
        sequence = _checkpoint(checkpoint_store, state, sequence)
        state = await generate_answer_node(
            state,
            deps,
            constraints=(
                "只能使用当前上下文已经绑定的 Evidence 和 facet；可以把复合句拆成原子句，"
                "但不得引入新的来源、facet 或 Evidence 中不存在的事实。"
            ),
            stage=AnswerWorkflowStage.REMEDIATE,
        )
    if action in {
        ReflectionAction.WEB_SEARCH,
        ReflectionAction.RERETRIEVE,
    } and _can_retry_remediation_generation(state):
        failure = next(
            item for item in reversed(state.trace) if item.event == "answer_generation_failed"
        )
        state = _trace(
            state,
            deps,
            "remediate",
            "answer_generation_retry_scheduled",
            retry_number=1,
            retry_reason=failure.details.get("failure_category"),
        )
        sequence = _checkpoint(checkpoint_store, state, sequence)
        state = await generate_answer_node(
            state,
            deps,
            constraints=(
                "上次输出未通过结构校验；只返回单个合法 JSON 对象，严格复制现有 "
                "Evidence 编号与 facet_id，不要输出 Markdown 或额外说明。"
            ),
            stage=AnswerWorkflowStage.REMEDIATE,
        )
    sequence = _checkpoint(checkpoint_store, state, sequence)
    if state.candidate is None:
        return state, sequence
    state = _update(state, stage=AnswerWorkflowStage.CHECK)
    sequence = _checkpoint(checkpoint_store, state, sequence)
    state = await check_answer_node(
        state,
        deps,
        # 若首轮 Critic 已消费唯一语义检查预算，Web 后只执行确定性检查；
        # 初始内部不足直接走 Web 时，仍把 Critic 留给最终外部回答。
        use_critic=(action is ReflectionAction.WEB_SEARCH and state.budget.used.critic_calls == 0),
    )
    sequence = _checkpoint(checkpoint_store, state, sequence)
    if _can_rewrite_remediation_answer(state):
        state = _trace(
            state,
            deps,
            "remediate",
            "answer_generation_retry_scheduled",
            retry_number=1,
            retry_reason="abstention_or_incomplete",
        )
        state = _update(
            state,
            stage=AnswerWorkflowStage.GENERATE,
            candidate=None,
            evaluation=None,
        )
        sequence = _checkpoint(checkpoint_store, state, sequence)
        state = await generate_answer_node(
            state,
            deps,
            constraints=(
                "上次回答遗漏了已有 Evidence 能支持的内容或错误拒答。必须直接提取"
                "当前 Evidence 中能够回答问题的事实；只能使用现有 Evidence 与 facet，"
                "不得增加外部知识。"
            ),
            stage=AnswerWorkflowStage.REMEDIATE,
        )
        sequence = _checkpoint(checkpoint_store, state, sequence)
        if state.candidate is not None:
            state = _update(state, stage=AnswerWorkflowStage.CHECK)
            state = await check_answer_node(
                state,
                deps,
                use_critic=(
                    action is ReflectionAction.WEB_SEARCH and state.budget.used.critic_calls == 0
                ),
            )
            sequence = _checkpoint(checkpoint_store, state, sequence)
    state = decide_answer_node(
        state,
        deps,
        allow_web=allow_web,
        internal_can_retrieve=False,
    )
    sequence = _checkpoint(checkpoint_store, state, sequence)
    return state, sequence


async def run_grounded_answer_workflow(
    correction: CorrectionRunResult,
    deps: AnswerWorkflowDependencies,
    *,
    allow_web: bool = False,
    checkpoint_store: AnswerCheckpointStore | None = None,
) -> GroundedAnswerResult:
    """运行固定回答状态机；任何补救之后只允许接受或保守终止。"""

    started_at = datetime.now(UTC)
    identity = build_answer_identity(
        correction,
        config_hash=grounded_answer_config_hash(deps.config),
        answer_prompt_version=deps.answer_prompt.version,
        critic_prompt_version=(deps.critic.prompt.version if deps.critic else "0" * 64),
        checker_version=deps.config.prompts.checker_version,
        policy_version=deps.config.prompts.policy_version,
        model_revision=deps.model_revision,
        search_provider=deps.config.web.provider,
        search_provider_version=deps.config.web.provider_version,
    )
    if checkpoint_store is not None:
        cached = checkpoint_store.load_result(identity)
        if cached is not None:
            return cached
        latest = checkpoint_store.load_latest(identity)
    else:
        latest = None

    # 只恢复已经完整落盘的纯计算阶段；调用前检查点无法证明外部调用未发生，必须保守停止。
    if latest is None:
        state = prepare_answer_node(identity, correction, deps)
        sequence = _checkpoint(checkpoint_store, state, 0)
        if not deps.config.enabled:
            state = finalize_answer_node(
                state,
                deps,
                started_at=started_at,
                forced_reason="grounded_answer_disabled",
            )
            if state.result is None:
                raise RuntimeError("disabled grounded answer workflow has no result")
            if checkpoint_store is not None:
                checkpoint_store.save(state, sequence)
                checkpoint_store.save_result(state.result)
                checkpoint_store.save_manifest(identity)
            return state.result
        if correction.stop.reason is StopReason.SUFFICIENT:
            state = await generate_answer_node(state, deps)
            sequence = _checkpoint(checkpoint_store, state, sequence)
            if state.candidate is None and _can_retry_provider_generation(state):
                state = _trace(
                    state,
                    deps,
                    "generate",
                    "answer_generation_retry_scheduled",
                    retry_number=1,
                )
                sequence = _checkpoint(checkpoint_store, state, sequence)
                state = await generate_answer_node(
                    state,
                    deps,
                    stage=AnswerWorkflowStage.REMEDIATE,
                )
                sequence = _checkpoint(checkpoint_store, state, sequence)
            if state.candidate is not None:
                state = _update(state, stage=AnswerWorkflowStage.CHECK)
                sequence = _checkpoint(checkpoint_store, state, sequence)
                state = await check_answer_node(state, deps, use_critic=True)
                sequence = _checkpoint(checkpoint_store, state, sequence)
        state = decide_answer_node(
            state,
            deps,
            allow_web=allow_web,
            internal_can_retrieve=_can_reretrieve(state, deps),
        )
        sequence = _checkpoint(checkpoint_store, state, sequence)
        state, sequence = await _complete_decision(
            state,
            deps,
            allow_web=allow_web,
            checkpoint_store=checkpoint_store,
            sequence=sequence,
        )
        forced_reason = None
    else:
        state = latest.state
        sequence = latest.sequence + 1
        forced_reason = "recovery_uncertain_external_call"
        if (
            state.stage
            in {
                AnswerWorkflowStage.GENERATE,
                AnswerWorkflowStage.REMEDIATE,
            }
            and state.candidate is not None
        ):
            use_critic = bool(
                state.decision is None or state.decision.action is ReflectionAction.WEB_SEARCH
            )
            state = _update(state, stage=AnswerWorkflowStage.CHECK)
            sequence = _checkpoint(checkpoint_store, state, sequence)
            state = await check_answer_node(state, deps, use_critic=use_critic)
            sequence = _checkpoint(checkpoint_store, state, sequence)
            state = decide_answer_node(
                state,
                deps,
                allow_web=allow_web,
                internal_can_retrieve=_can_reretrieve(state, deps),
            )
            sequence = _checkpoint(checkpoint_store, state, sequence)
            decision = state.decision
            if decision is not None and decision.action in {
                ReflectionAction.ACCEPT,
                ReflectionAction.CONSERVATIVE_STOP,
            }:
                forced_reason = None
        elif state.stage is AnswerWorkflowStage.CHECK:
            state = decide_answer_node(
                state,
                deps,
                allow_web=allow_web,
                internal_can_retrieve=_can_reretrieve(state, deps),
            )
            sequence = _checkpoint(checkpoint_store, state, sequence)
            decision = state.decision
            if decision is not None and decision.action in {
                ReflectionAction.ACCEPT,
                ReflectionAction.CONSERVATIVE_STOP,
            }:
                forced_reason = None
        elif (
            state.stage is AnswerWorkflowStage.DECIDE
            and state.decision is not None
            and state.decision.action
            in {ReflectionAction.ACCEPT, ReflectionAction.CONSERVATIVE_STOP}
        ):
            forced_reason = None

    generation_events = [
        item
        for item in state.trace
        if item.event in {"answer_generated", "answer_generation_failed"}
    ]
    generation_failed = bool(
        generation_events and generation_events[-1].event == "answer_generation_failed"
    )
    if generation_failed:
        forced_reason = "answer_generation_failed"
    elif state.decision is None:
        forced_reason = "workflow_has_no_decision"
    elif state.decision.action not in {
        ReflectionAction.ACCEPT,
        ReflectionAction.CONSERVATIVE_STOP,
    }:
        forced_reason = "remediation_failed"
    state = finalize_answer_node(
        state,
        deps,
        started_at=started_at,
        forced_reason=forced_reason,
    )
    if state.result is None:
        raise RuntimeError("grounded answer workflow completed without a result")
    if checkpoint_store is not None:
        checkpoint_store.save(state, sequence)
        checkpoint_store.save_result(state.result)
        checkpoint_store.save_manifest(identity)
    return state.result
