"""首次检索失败压力集的离线三策略矩阵与资源指标。"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from kg_crag.correction.config import CorrectiveWorkflowConfig
from kg_crag.correction.coverage import build_coverage_matrix
from kg_crag.correction.executor import ActionExecutor, RetrieverTool
from kg_crag.correction.identity import build_run_identity, stable_digest
from kg_crag.correction.requirements import (
    RequirementCache,
    RequirementProvider,
    analyze_question,
)
from kg_crag.correction.sufficiency import assess_sufficiency
from kg_crag.models import (
    BudgetLedger,
    CorrectionAction,
    CorrectionState,
    CorrectiveEvaluationItem,
    CorrectiveEvaluationMetrics,
    CorrectiveEvaluationQuestion,
    CorrectiveEvaluationQuestionSet,
    CorrectiveEvaluationReport,
    CorrectiveStrategy,
    ErrorCode,
    ErrorDetail,
    EvidenceRequirement,
    StopReason,
    SufficiencyAssessment,
)
from kg_crag.retrieval.mock import MockRetriever
from kg_crag.workflow.artifacts import ContentAddressedCache, atomic_model_write
from kg_crag.workflow.corrective import run_corrective_workflow
from kg_crag.workflow.nodes import WorkflowDependencies


def load_corrective_questions(path: Path) -> CorrectiveEvaluationQuestionSet:
    questions = CorrectiveEvaluationQuestionSet.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    for item in questions.questions:
        generated = [
            facet.facet_id for facet in analyze_question(item.question_id, item.question).facets
        ]
        if generated != item.expected_facet_ids:
            raise ValueError(f"frozen facet truth drifted for {item.question_id}")
    return questions


def _initial_assessment(
    question: CorrectiveEvaluationQuestion, config: CorrectiveWorkflowConfig
) -> tuple[list[EvidenceRequirement], SufficiencyAssessment]:
    facets = list(analyze_question(question.question_id, question.question).facets)
    matrix = build_coverage_matrix(
        facets, question.initial_evidence, rule_version=config.coverage.version
    )
    assessment = assess_sufficiency(
        facets,
        question.initial_evidence,
        matrix,
        min_support=config.coverage.min_support,
        min_sources=config.coverage.min_sources,
        max_selected=config.coverage.max_selected_evidence,
    )
    return facets, assessment


def _baseline_item(
    question: CorrectiveEvaluationQuestion,
    strategy: CorrectiveStrategy,
    config: CorrectiveWorkflowConfig,
) -> CorrectiveEvaluationItem:
    facets, assessment = _initial_assessment(question, config)
    covered = len([item for item in assessment.facets if item.required and item.satisfied])
    return CorrectiveEvaluationItem(
        question_id=question.question_id,
        strategy=strategy,
        first_sufficient=assessment.sufficient,
        diagnosis_correct=[item.facet_id for item in facets] == question.expected_facet_ids,
        required_facets=sum(item.required for item in facets),
        covered_required_facets=covered,
        retrieval_rounds=1,
        tool_calls=1,
        stop_reason=(
            StopReason.SUFFICIENT if assessment.sufficient else StopReason.NO_POSITIVE_GAIN
        ),
    )


async def _corrective_item(
    question: CorrectiveEvaluationQuestion,
    config: CorrectiveWorkflowConfig,
    *,
    requirement_provider: RequirementProvider | None = None,
    requirement_prompt: str = "",
    requirement_cache: RequirementCache | None = None,
    artifact_cache: ContentAddressedCache | None = None,
) -> CorrectiveEvaluationItem:
    facets, initial_assessment = _initial_assessment(question, config)
    correction_evidence = [
        item
        for action in sorted(question.action_evidence, key=lambda value: value.value)
        for item in question.action_evidence[action]
    ]
    retriever = MockRetriever(correction_evidence)
    tools = {action: RetrieverTool(retriever) for action in CorrectionAction}
    executor = ActionExecutor(
        tools,
        max_candidates=config.actions.max_candidates_per_action,
        bounds=config.actions,
        max_evidence_chars=config.coverage.max_evidence_chars,
    )
    identity = build_run_identity(
        question.question,
        facets,
        question.initial_evidence,
        config_hash=stable_digest(config),
        prompt_version=config.facets.prompt_version,
        model_revision=(
            requirement_provider.revision if requirement_provider else "offline-mock-v1"
        ),
        corpus_snapshot=question.corpus_snapshot,
        dense_version=question.dense_version,
        sparse_version=question.sparse_version,
        graph_version=question.graph_version,
        rules_version=config.facets.version,
        coverage_version=config.coverage.version,
        policy_version=config.policy_version,
    )
    initial = CorrectionState(
        identity=identity,
        question_id=question.question_id,
        question=question.question,
        budget=BudgetLedger(limit=config.budget),
    )
    deps = WorkflowDependencies(
        config=config,
        executor=executor,
        initial_evidence=tuple(question.initial_evidence),
        requirement_provider=requirement_provider,
        requirement_prompt=requirement_prompt,
        requirement_cache=requirement_cache,
        artifact_cache=artifact_cache,
    )
    result = await run_corrective_workflow(initial, deps)
    final_assessment = result.state.sufficiency
    covered = (
        len([item for item in final_assessment.facets if item.required and item.satisfied])
        if final_assessment
        else 0
    )
    selected_actions = [item.request.action for item in result.state.action_history]
    meaningless = int(bool(selected_actions) and result.stop.reason is not StopReason.SUFFICIENT)
    diagnosis_correct = [item.facet_id for item in facets] == question.expected_facet_ids
    return CorrectiveEvaluationItem(
        question_id=question.question_id,
        strategy=CorrectiveStrategy.FACET_CORRECTIVE,
        first_sufficient=initial_assessment.sufficient,
        diagnosis_correct=diagnosis_correct,
        required_facets=sum(item.required for item in facets),
        covered_required_facets=covered,
        recovered=not initial_assessment.sufficient and result.stop.reason is StopReason.SUFFICIENT,
        correction_triggered=bool(selected_actions),
        meaningless_actions=meaningless,
        retrieval_rounds=result.state.retrieval_round,
        tool_calls=result.state.retrieval_round,
        llm_calls=result.state.budget.used.llm_calls,
        input_tokens=result.state.budget.used.input_tokens,
        output_tokens=result.state.budget.used.output_tokens,
        latency_ms=result.state.budget.used.latency_ms,
        stop_reason=result.stop.reason,
        error=(
            ErrorDetail(
                code=ErrorCode.DATA,
                message="offline result differs from frozen stop truth",
                retryable=False,
            )
            if result.stop.reason is not question.expected_stop
            else None
        ),
    )


def aggregate_metrics(
    items: list[CorrectiveEvaluationItem], strategy: CorrectiveStrategy
) -> CorrectiveEvaluationMetrics:
    selected = [item for item in items if item.strategy is strategy]
    if not selected:
        raise ValueError(f"no evaluation items for strategy: {strategy}")
    total = len(selected)
    failed = sum(item.error is not None for item in selected)
    required = sum(item.required_facets for item in selected)
    first_failures = sum(not item.first_sufficient for item in selected)
    triggered = sum(item.correction_triggered for item in selected)
    return CorrectiveEvaluationMetrics(
        total=total,
        failures=failed,
        first_sufficiency_rate=sum(item.first_sufficient for item in selected) / total,
        diagnosis_accuracy=sum(item.diagnosis_correct for item in selected) / total,
        required_coverage_rate=(
            sum(item.covered_required_facets for item in selected) / required if required else 1.0
        ),
        recovery_rate=(
            sum(item.recovered for item in selected) / first_failures if first_failures else 0.0
        ),
        correction_trigger_rate=triggered / total,
        meaningless_action_rate=(
            sum(item.meaningless_actions for item in selected) / triggered if triggered else 0.0
        ),
        average_retrieval_rounds=sum(item.retrieval_rounds for item in selected) / total,
        tool_calls=sum(item.tool_calls for item in selected),
        llm_calls=sum(item.llm_calls for item in selected),
        input_tokens=sum(item.input_tokens for item in selected),
        output_tokens=sum(item.output_tokens for item in selected),
        latency_ms=sum(item.latency_ms for item in selected),
        stop_reasons=dict(Counter(item.stop_reason for item in selected)),
    )


async def evaluate_offline_matrix(
    questions: CorrectiveEvaluationQuestionSet,
    config: CorrectiveWorkflowConfig,
    *,
    limit: int | None = None,
    offline: bool = True,
    requirement_provider: RequirementProvider | None = None,
    requirement_prompt: str = "",
    requirement_cache: RequirementCache | None = None,
    artifact_cache: ContentAddressedCache | None = None,
    results_root: Path | None = None,
    workspace_root: Path | None = None,
) -> CorrectiveEvaluationReport:
    selected_questions = questions.questions[:limit] if limit is not None else questions.questions
    identity_payload = {
        "questions": [item.model_dump(mode="json") for item in selected_questions],
        "config": config.model_dump(mode="json"),
        "strategies": [item.value for item in CorrectiveStrategy],
        "offline": offline,
        "provider": requirement_provider.revision if requirement_provider else None,
    }
    question_hash = stable_digest(selected_questions)
    evaluation_id = "eval-" + stable_digest(identity_payload)[:32]
    items: list[CorrectiveEvaluationItem] = []
    item_root = results_root / evaluation_id / "items" if results_root else None
    for question in selected_questions:
        for strategy in CorrectiveStrategy:
            checkpoint = (
                item_root / f"{question.question_id}-{strategy.value}.json" if item_root else None
            )
            if checkpoint and checkpoint.is_file():
                try:
                    items.append(
                        CorrectiveEvaluationItem.model_validate_json(
                            checkpoint.read_text(encoding="utf-8")
                        )
                    )
                    continue
                except ValueError:
                    pass
            try:
                item = (
                    await _corrective_item(
                        question,
                        config,
                        requirement_provider=requirement_provider,
                        requirement_prompt=requirement_prompt,
                        requirement_cache=requirement_cache,
                        artifact_cache=artifact_cache,
                    )
                    if strategy is CorrectiveStrategy.FACET_CORRECTIVE
                    else _baseline_item(question, strategy, config)
                )
            except Exception:
                item = CorrectiveEvaluationItem(
                    question_id=question.question_id,
                    strategy=strategy,
                    stop_reason=StopReason.EXECUTION_FAILED,
                    error=ErrorDetail(
                        code=ErrorCode.INTERNAL,
                        message="offline evaluation item failed",
                        retryable=False,
                    ),
                )
            items.append(item)
            if checkpoint and workspace_root:
                atomic_model_write(checkpoint, item, workspace_root=workspace_root)
    metrics = {strategy: aggregate_metrics(items, strategy) for strategy in CorrectiveStrategy}
    report = CorrectiveEvaluationReport(
        evaluation_id=evaluation_id,
        offline=offline,
        question_set_hash=question_hash,
        items=items,
        metrics=metrics,
        created_at=datetime.now(UTC),
    )
    if results_root and workspace_root:
        atomic_model_write(
            results_root / evaluation_id / "report.json", report, workspace_root=workspace_root
        )
    return report
