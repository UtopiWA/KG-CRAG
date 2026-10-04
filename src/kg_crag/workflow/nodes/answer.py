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
    BoundEvidence,
    build_answer_context,
    parse_grounded_answer,
)
from kg_crag.answering.critic import SemanticCritic
from kg_crag.answering.policy import decide_reflection
from kg_crag.answering.prompt import StructuredPrompt
from kg_crag.correction.config import CorrectiveWorkflowConfig
from kg_crag.correction.executor import ActionExecutor
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
from kg_crag.providers import LLMProvider, SearchProvider
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
    external_coverage_version: str = "external-coverage-v1"


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


async def generate_answer_node(
    state: AnswerWorkflowState,
    deps: AnswerWorkflowDependencies,
    *,
    constraints: str = "仅输出可由 Evidence 直接支持的结论。",
    allowed_binding_sets: set[tuple[tuple[str, ...], tuple[str, ...]]] | None = None,
    stage: AnswerWorkflowStage = AnswerWorkflowStage.GENERATE,
) -> AnswerWorkflowState:
    prompt, _context, bindings = render_generation_prompt(state, deps, constraints=constraints)
    manager = AnswerBudgetManager(state.budget)
    estimate = AnswerBudgetUsage(
        answer_calls=1,
        input_tokens=4000,
        output_tokens=750,
        context_chars=min(len(prompt), 16_000),
    )
    token = manager.reserve(estimate)
    if token is None:
        raise ValueError("answer generation exceeds remaining budget")
    try:
        raw = await deps.llm.generate(prompt, temperature=0.0)
        candidate = parse_grounded_answer(
            raw,
            bindings,
            state.correction.state.facets,
            config=deps.config.generation,
            allowed_binding_sets=allowed_binding_sets,
        )
    except Exception:
        manager.settle(token, failed=True)
        failed = _validated(
            state,
            stage=stage,
            budget=manager.ledger,
            errors=[
                *state.errors,
                ErrorDetail(
                    code=ErrorCode.DATA,
                    message="answer generation or parsing failed",
                    retryable=False,
                ),
            ],
        )
        return _validated(
            failed,
            trace=_trace(
                failed,
                stage.value,
                "answer_generation_failed",
                {"answer_calls": manager.ledger.used.answer_calls},
                max_events=deps.config.max_trace_events,
            ),
        )
    manager.settle(
        token,
        AnswerBudgetUsage(
            answer_calls=1,
            input_tokens=min(4000, max(1, (len(prompt) + 3) // 4)),
            output_tokens=min(750, max(1, (len(raw) + 3) // 4)),
            context_chars=min(len(prompt), 16_000),
        ),
    )
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
    external_covered = {item.facet_id for item in state.external_coverage if item.matched}
    evaluation = deterministic_check(
        state.candidate,
        state.correction.state.facets,
        state.correction.state.sufficiency,
        externally_covered_facet_ids=external_covered,
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
                )
                try:
                    findings = await deps.critic.evaluate(
                        question=state.correction.state.question,
                        candidate=state.candidate,
                        evidence_context=context_text[: deps.config.critic.max_context_chars],
                        facets=",".join(item.facet_id for item in state.correction.state.facets),
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
        missing.update(
            facet_id
            for item in state.evaluation.findings
            if item.code is AnswerCheckCode.INCOMPLETE
            for facet_id in item.facet_ids
        )
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
        if reason == "semantic critic failed":
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
