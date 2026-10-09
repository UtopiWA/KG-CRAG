"""框架无关的回答生成、检查、决策和终止节点。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime

from kg_crag.answering.budget import AnswerBudgetManager
from kg_crag.answering.checker import deterministic_check, merge_critic_findings
from kg_crag.answering.config import GroundedAnswerConfig
from kg_crag.answering.conservative import build_conservative_candidate
from kg_crag.answering.context import (
    AnswerResponseError,
    BoundEvidence,
    build_answer_context,
    parse_grounded_answer,
)
from kg_crag.answering.critic import SemanticCritic
from kg_crag.answering.policy import decide_reflection
from kg_crag.answering.prompt import StructuredPrompt
from kg_crag.correction.config import CorrectiveWorkflowConfig
from kg_crag.correction.executor import ActionExecutor
from kg_crag.errors import KGCRAGError
from kg_crag.models import (
    AnswerBudgetUsage,
    AnswerCheckCode,
    AnswerFinding,
    AnswerRunIdentity,
    AnswerStopReason,
    AnswerWorkflowStage,
    AnswerWorkflowState,
    CombinedBudgetSummary,
    CorrectionRunResult,
    ErrorCode,
    ErrorDetail,
    FindingSeverity,
    GroundedAnswerCandidate,
    GroundedAnswerResult,
    ReflectionAction,
    RetrievalStrategy,
    StopReason,
    TraceEvent,
)
from kg_crag.providers import (
    LLMGeneration,
    LLMProvider,
    SearchProvider,
    UsageAwareLLMProvider,
)
from kg_crag.workflow.trace import append_trace


@dataclass(frozen=True)
class AnswerWorkflowDependencies:
    config: GroundedAnswerConfig
    llm: LLMProvider
    answer_prompt: StructuredPrompt
    critic: SemanticCritic | None = None
    search_provider: SearchProvider | None = None
    correction_config: CorrectiveWorkflowConfig | None = None
    action_executor: ActionExecutor | None = None
    model_revision: str = "unknown"
    external_coverage_version: str = "external-coverage-v4"


def _validated(state: AnswerWorkflowState, **updates: object) -> AnswerWorkflowState:
    payload = state.model_dump(mode="python")
    payload.update(updates)
    return AnswerWorkflowState.model_validate(payload)


def _trace(
    state: AnswerWorkflowState,
    node: str,
    event: str,
    details: dict[str, object],
    *,
    max_events: int,
) -> list[TraceEvent]:
    return append_trace(
        state.trace,
        trace_id=state.identity.run_id,
        node=node,
        event=event,
        details=details,
        max_events=max_events,
    )


def prepare_answer_node(
    identity: AnswerRunIdentity,
    correction: CorrectionRunResult,
    deps: AnswerWorkflowDependencies,
) -> AnswerWorkflowState:
    checked = CorrectionRunResult.model_validate(correction.model_dump(mode="json"))
    state = AnswerWorkflowState(identity=identity, correction=checked)
    return _validated(
        state,
        trace=_trace(
            state,
            "prepare",
            "internal_result_loaded",
            {
                "internal_stop": checked.stop.reason.value,
                "internal_evidence": len(checked.state.selected_evidence),
                "missing_facets": len(checked.stop.missing_facet_ids),
            },
            max_events=deps.config.max_trace_events,
        ),
    )


def render_generation_prompt(
    state: AnswerWorkflowState,
    deps: AnswerWorkflowDependencies,
    *,
    constraints: str,
) -> tuple[str, str, list[BoundEvidence]]:
    correction = state.correction.state
    evidence = [*correction.selected_evidence, *state.external_evidence]
    context_text, bindings = build_answer_context(
        evidence,
        internal_matrix=correction.coverage_matrix,
        external_coverage=state.external_coverage,
        config=deps.config.generation,
    )
    if not bindings:
        raise ValueError("answer generation requires selected Evidence")
    facets = json.dumps(
        [
            {"facet_id": item.facet_id, "description": item.description, "required": item.required}
            for item in correction.facets
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    prompt = deps.answer_prompt.render(
        max_chars=16_000,
        question=correction.question,
        facets=facets,
        evidence_context=context_text,
        constraints=constraints,
    )
    return prompt, context_text, bindings


async def _generate_with_usage(provider: LLMProvider, prompt: str) -> LLMGeneration:
    """优先获取逐请求实际用量，普通测试替身继续兼容文本接口。"""

    if isinstance(provider, UsageAwareLLMProvider):
        return await provider.generate_with_usage(prompt, temperature=0.0)
    return LLMGeneration(content=await provider.generate(prompt, temperature=0.0))


def _generation_usage(
    generation: LLMGeneration,
    *,
    prompt: str,
    reserved: AnswerBudgetUsage,
) -> tuple[AnswerBudgetUsage, bool]:
    """实际用量可安全结算时优先采用，否则使用有界字符估算。"""

    if (
        generation.has_actual_usage
        and (generation.input_tokens or 0) <= reserved.input_tokens
        and (generation.output_tokens or 0) <= reserved.output_tokens
    ):
        actual = AnswerBudgetUsage(
            answer_calls=1,
            input_tokens=generation.input_tokens or 0,
            output_tokens=generation.output_tokens or 0,
            context_chars=min(len(prompt), 16_000),
        )
        return actual, False
    estimated = AnswerBudgetUsage(
        answer_calls=1,
        input_tokens=min(reserved.input_tokens, max(1, (len(prompt) + 3) // 4)),
        output_tokens=min(
            reserved.output_tokens,
            max(1, (len(generation.content) + 3) // 4),
        ),
        context_chars=min(len(prompt), 16_000),
    )
    return estimated, True


async def generate_answer_node(
    state: AnswerWorkflowState,
    deps: AnswerWorkflowDependencies,
    *,
    constraints: str = "仅输出可由 Evidence 直接支持的结论。",
    stage: AnswerWorkflowStage = AnswerWorkflowStage.GENERATE,
) -> AnswerWorkflowState:
    prompt, _context, bindings = render_generation_prompt(state, deps, constraints=constraints)
    manager = AnswerBudgetManager(state.budget)
    provider_output_limit = getattr(deps.llm, "max_output_tokens", None)
    reserved_output_tokens = (
        provider_output_limit
        if isinstance(provider_output_limit, int) and not isinstance(provider_output_limit, bool)
        else 750
    )
    estimate = AnswerBudgetUsage(
        answer_calls=1,
        input_tokens=4000,
        output_tokens=reserved_output_tokens,
        context_chars=min(len(prompt), 16_000),
    )
    token = manager.reserve(estimate)
    if token is None:
        raise ValueError("answer generation exceeds remaining budget")
    try:
        generation = await _generate_with_usage(deps.llm, prompt)
    except Exception as error:
        manager.settle(token, failed=True)
        detail = (
            error.detail
            if isinstance(error, KGCRAGError)
            else ErrorDetail(
                code=ErrorCode.EXTERNAL_SERVICE,
                message="answer model request failed",
                retryable=False,
            )
        )
        failed = _validated(
            state,
            stage=stage,
            budget=manager.ledger,
            errors=[*state.errors, detail],
        )
        provider_status = detail.context.get("status")
        provider_error_type = detail.context.get("error_type")
        provider_finish_reason = detail.context.get("finish_reason")
        provider_analysis_chars = detail.context.get("analysis_chars")
        return _validated(
            failed,
            trace=_trace(
                failed,
                stage.value,
                "answer_generation_failed",
                {
                    "answer_calls": manager.ledger.used.answer_calls,
                    "failure_category": "provider_request",
                    "retryable": detail.retryable,
                    "provider_status": provider_status,
                    "provider_error_type": provider_error_type,
                    "provider_finish_reason": provider_finish_reason,
                    "provider_analysis_chars": provider_analysis_chars,
                    "input_tokens": estimate.input_tokens,
                    "output_tokens": estimate.output_tokens,
                    "usage_estimated": True,
                },
                max_events=deps.config.max_trace_events,
            ),
        )

    raw = generation.content
    usage, usage_estimated = _generation_usage(generation, prompt=prompt, reserved=estimate)
    try:
        candidate = parse_grounded_answer(
            raw,
            bindings,
            state.correction.state.facets,
            config=deps.config.generation,
        )
    except AnswerResponseError as error:
        manager.settle(token, usage)
        failed = _validated(
            state,
            stage=stage,
            budget=manager.ledger,
            errors=[
                *state.errors,
                ErrorDetail(
                    code=ErrorCode.DATA,
                    message="answer response validation failed",
                    retryable=False,
                    context={"failure_category": error.category.value},
                ),
            ],
        )
        return _validated(
            failed,
            trace=_trace(
                failed,
                stage.value,
                "answer_generation_failed",
                {
                    "answer_calls": manager.ledger.used.answer_calls,
                    "failure_category": error.category.value,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "usage_estimated": usage_estimated,
                },
                max_events=deps.config.max_trace_events,
            ),
        )
    except Exception:
        manager.settle(token, usage)
        failed = _validated(
            state,
            stage=stage,
            budget=manager.ledger,
            errors=[
                *state.errors,
                ErrorDetail(
                    code=ErrorCode.DATA,
                    message="answer response validation failed",
                    retryable=False,
                    context={"failure_category": "internal_validation"},
                ),
            ],
        )
        return _validated(
            failed,
            trace=_trace(
                failed,
                stage.value,
                "answer_generation_failed",
                {
                    "answer_calls": manager.ledger.used.answer_calls,
                    "failure_category": "internal_validation",
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "usage_estimated": usage_estimated,
                },
                max_events=deps.config.max_trace_events,
            ),
        )
    manager.settle(token, usage)
    generated = _validated(
        state,
        stage=stage,
        candidate=candidate,
        budget=manager.ledger,
    )
    return _validated(
        generated,
        trace=_trace(
            generated,
            stage.value,
            "answer_generated",
            {
                "claims": len(candidate.claims),
                "citations": len(candidate.citations),
                "answer_calls": manager.ledger.used.answer_calls,
                "input_tokens": manager.ledger.used.input_tokens,
                "output_tokens": manager.ledger.used.output_tokens,
                "usage_estimated": usage_estimated,
            },
            max_events=deps.config.max_trace_events,
        ),
    )


async def check_answer_node(
    state: AnswerWorkflowState,
    deps: AnswerWorkflowDependencies,
    *,
    use_critic: bool,
) -> AnswerWorkflowState:
    if state.candidate is None or state.correction.state.sufficiency is None:
        raise ValueError("answer check requires a candidate and sufficiency assessment")
    evaluation = deterministic_check(
        state.candidate,
        state.correction.state.facets,
        state.correction.state.sufficiency,
    )
    errors = list(state.errors)
    manager = AnswerBudgetManager(state.budget)
    if use_critic and deps.config.critic.enabled and evaluation.acceptable:
        if deps.critic is None:
            finding = AnswerFinding(
                code=AnswerCheckCode.CRITIC_FAILED,
                reason="critic is enabled but no provider is configured",
            )
            evaluation = merge_critic_findings(evaluation, [finding], critic_failed=True)
        else:
            estimate = AnswerBudgetUsage(
                critic_calls=1,
                input_tokens=1000,
                output_tokens=500,
                context_chars=deps.config.critic.max_context_chars,
            )
            token = manager.reserve(estimate)
            if token is None:
                finding = AnswerFinding(
                    code=AnswerCheckCode.CRITIC_FAILED,
                    reason="critic exceeds remaining answer budget",
                )
                evaluation = merge_critic_findings(evaluation, [finding], critic_failed=True)
            else:
                context_text, _bindings = build_answer_context(
                    [
                        *state.correction.state.selected_evidence,
                        *state.external_evidence,
                    ],
                    internal_matrix=state.correction.state.coverage_matrix,
                    external_coverage=state.external_coverage,
                    config=deps.config.generation,
                    citation_ids={item.citation_id for item in state.candidate.citations},
                )
                try:
                    findings = await deps.critic.evaluate(
                        question=state.correction.state.question,
                        candidate=state.candidate,
                        evidence_context=context_text,
                        facets=json.dumps(
                            [
                                {
                                    "facet_id": item.facet_id,
                                    "description": item.description,
                                }
                                for item in state.correction.state.facets
                            ],
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        conflicts=",".join(
                            item.conflict_id
                            for item in state.correction.state.sufficiency.conflicts
                        )
                        or "none",
                    )
                except Exception:
                    manager.settle(token, failed=True)
                    errors.append(
                        ErrorDetail(
                            code=ErrorCode.EXTERNAL_SERVICE,
                            message="semantic critic failed",
                            retryable=False,
                        )
                    )
                    finding = AnswerFinding(
                        code=AnswerCheckCode.CRITIC_FAILED,
                        reason="semantic critic failed",
                    )
                    evaluation = merge_critic_findings(evaluation, [finding], critic_failed=True)
                else:
                    manager.settle(
                        token,
                        AnswerBudgetUsage(
                            critic_calls=1,
                            input_tokens=min(1000, max(1, len(context_text[:4000]) // 4)),
                            output_tokens=min(500, max(1, len(findings) * 50)),
                            context_chars=min(
                                len(context_text), deps.config.critic.max_context_chars
                            ),
                        ),
                    )
                    evaluation = merge_critic_findings(evaluation, findings)
    checked = _validated(
        state,
        stage=AnswerWorkflowStage.CHECK,
        evaluation=evaluation,
        budget=manager.ledger,
        errors=errors,
    )
    return _validated(
        checked,
        trace=_trace(
            checked,
            "check",
            "answer_checked",
            {
                "acceptable": evaluation.acceptable,
                "findings": len(evaluation.findings),
                "finding_codes": ",".join(
                    sorted({item.code.value for item in evaluation.findings})
                ),
                "unlocalized_findings": sum(not item.claim_ids for item in evaluation.findings),
                "critic_used": evaluation.critic_used,
                "critic_calls": manager.ledger.used.critic_calls,
            },
            max_events=deps.config.max_trace_events,
        ),
    )


def decide_answer_node(
    state: AnswerWorkflowState,
    deps: AnswerWorkflowDependencies,
    *,
    allow_web: bool,
    internal_can_retrieve: bool,
) -> AnswerWorkflowState:
    missing = set(state.correction.stop.missing_facet_ids)
    if state.evaluation is not None:
        evidence_gap_codes = {
            AnswerCheckCode.INCOMPLETE,
            AnswerCheckCode.UNSUPPORTED,
            AnswerCheckCode.WRONG_ATTRIBUTION,
            AnswerCheckCode.SOURCE_MISMATCH,
            AnswerCheckCode.FACET_MISMATCH,
        }
        claims_by_id = {
            item.claim_id: set(item.facet_ids)
            for item in (state.candidate.claims if state.candidate else [])
        }
        candidate_facets = {value for values in claims_by_id.values() for value in values}
        for finding in state.evaluation.findings:
            if finding.code not in evidence_gap_codes:
                continue
            missing.update(finding.facet_ids)
            for claim_id in finding.claim_ids:
                missing.update(claims_by_id.get(claim_id, set()))
            if (
                finding.code is not AnswerCheckCode.INCOMPLETE
                and not finding.facet_ids
                and not finding.claim_ids
            ):
                # 部分 Provider 能识别证据问题却省略定位 ID；此时只回退到
                # 候选已声明的 facet，不把未作答维度无界加入 Web 查询。
                missing.update(candidate_facets)
    missing -= {item.facet_id for item in state.external_coverage if item.matched}
    decision = decide_reflection(
        internal_stop=state.correction.stop.reason,
        evaluation=state.evaluation,
        candidate=state.candidate,
        missing_facet_ids=sorted(missing),
        web_allowed=allow_web and deps.config.web.enabled,
        web_ready=deps.search_provider is not None,
        internal_can_retrieve=internal_can_retrieve,
        remediation_used=state.remediation_used,
    )
    decided = _validated(
        state,
        stage=AnswerWorkflowStage.DECIDE,
        decision=decision,
    )
    return _validated(
        decided,
        trace=_trace(
            decided,
            "decide",
            "reflection_decided",
            {
                "action": decision.action.value,
                "missing_facets": len(decision.target_facet_ids),
                "remediation_used": state.remediation_used,
            },
            max_events=deps.config.max_trace_events,
        ),
    )


def _safe_candidate(state: AnswerWorkflowState) -> GroundedAnswerCandidate | None:
    if state.candidate is None:
        return None
    bad_claims: set[str] = set()
    if state.evaluation is not None:
        bad_claims = {
            claim_id
            for finding in state.evaluation.findings
            if finding.severity is FindingSeverity.ERROR
            for claim_id in finding.claim_ids
        }
    claims = [item for item in state.candidate.claims if item.claim_id not in bad_claims]
    if not claims:
        return None
    used_citations = {value for item in claims for value in item.citation_ids}
    citations = [item for item in state.candidate.citations if item.citation_id in used_citations]
    answer = "\n".join(
        f"{item.text} {' '.join(f'[{value}]' for value in item.citation_ids)}".rstrip()
        for item in claims
    )
    return GroundedAnswerCandidate(
        answer=answer,
        claims=claims,
        citations=citations,
        confidence=min(state.candidate.confidence, 0.5),
    )


def finalize_answer_node(
    state: AnswerWorkflowState,
    deps: AnswerWorkflowDependencies,
    *,
    started_at: datetime,
    forced_reason: str | None = None,
) -> AnswerWorkflowState:
    external_covered = {item.facet_id for item in state.external_coverage if item.matched}
    missing = set(state.correction.stop.missing_facet_ids) - external_covered
    if state.evaluation is not None:
        missing.update(
            facet_id
            for item in state.evaluation.findings
            if item.code is AnswerCheckCode.INCOMPLETE
            for facet_id in item.facet_ids
        )
    candidate = state.candidate
    accepted = bool(
        forced_reason is None
        and state.decision is not None
        and state.decision.action is ReflectionAction.ACCEPT
        and state.evaluation is not None
        and state.evaluation.acceptable
        and candidate is not None
    )
    if accepted and candidate is not None:
        stop_reason = AnswerStopReason.ACCEPTED
    else:
        reason = forced_reason or (
            state.decision.reason_code if state.decision else "no_safe_answer"
        )
        candidate = build_conservative_candidate(
            supported=_safe_candidate(state),
            missing_facet_ids=sorted(missing),
            reason=reason,
            internal_evidence_count=len(state.correction.state.selected_evidence),
            external_evidence_count=len(state.external_evidence),
            budget_summary=CombinedBudgetSummary(
                internal=state.correction.state.budget,
                answer=state.budget,
            ),
        )
        if reason == "answer_generation_failed":
            stop_reason = AnswerStopReason.GENERATION_FAILED
        elif reason == "semantic critic failed":
            stop_reason = AnswerStopReason.CRITIC_FAILED
        elif "budget" in reason:
            stop_reason = AnswerStopReason.BUDGET_EXHAUSTED
        elif "search" in reason or "web" in reason:
            stop_reason = AnswerStopReason.SEARCH_FAILED
        elif state.correction.stop.reason is StopReason.EXECUTION_FAILED:
            stop_reason = AnswerStopReason.EXECUTION_FAILED
        else:
            stop_reason = AnswerStopReason.CONSERVATIVE
    retrieval_path = [RetrievalStrategy.HYBRID]
    for item in state.correction.state.action_history:
        value = item.request.action.value
        strategy = (
            RetrievalStrategy(value)
            if value in {"dense", "sparse", "graph", "hybrid"}
            else RetrievalStrategy.HYBRID
        )
        retrieval_path.append(strategy)
    if state.external_evidence:
        retrieval_path.append(RetrievalStrategy.WEB)
    final_trace = _trace(
        state,
        "finalize",
        "answer_stopped",
        {
            "stop_reason": stop_reason.value,
            "missing_facets": len(missing),
            "internal_evidence": len(state.correction.state.selected_evidence),
            "external_evidence": len(state.external_evidence),
        },
        max_events=deps.config.max_trace_events,
    )
    result = GroundedAnswerResult(
        identity=state.identity,
        trace_id=state.identity.run_id,
        question=state.correction.state.question,
        answer=candidate.answer,
        claims=candidate.claims,
        citations=candidate.citations,
        internal_evidence=state.correction.state.selected_evidence,
        external_evidence=state.external_evidence,
        external_coverage=state.external_coverage,
        evaluation=state.evaluation,
        requirements=state.correction.state.facets,
        facet_assessments=(
            state.correction.state.sufficiency.facets
            if state.correction.state.sufficiency is not None
            else []
        ),
        conflicts=(
            state.correction.state.sufficiency.conflicts
            if state.correction.state.sufficiency is not None
            else []
        ),
        internal_actions=state.correction.state.action_history,
        missing_required_facet_ids=sorted(missing),
        retrieval_path=retrieval_path,
        internal_stop_reason=state.correction.stop.reason,
        stop_reason=stop_reason,
        used_external=bool(state.external_evidence),
        confidence=candidate.confidence,
        budget=CombinedBudgetSummary(
            internal=state.correction.state.budget,
            answer=state.budget,
        ),
        trace=final_trace,
        errors=state.errors,
        started_at=started_at,
        finished_at=datetime.now(UTC),
    )
    return _validated(
        state,
        stage=AnswerWorkflowStage.FINALIZE,
        trace=final_trace,
        result=result,
    )
