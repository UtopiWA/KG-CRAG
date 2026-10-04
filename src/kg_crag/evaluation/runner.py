"""统一、可恢复且默认离线的三策略评测 runner。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from kg_crag.evaluation.artifacts import UnifiedCheckpointStore
from kg_crag.evaluation.config import UnifiedEvaluationConfig
from kg_crag.evaluation.dataset import canonical_digest
from kg_crag.evaluation.judge import JudgeProvider, evaluate_with_judge, validate_judge_gate
from kg_crag.evaluation.metrics import aggregate_all_slices, evaluate_item
from kg_crag.models import (
    ErrorCode,
    ErrorDetail,
    EvaluationCheckpoint,
    EvaluationDatasetManifest,
    EvaluationFailureRecord,
    EvaluationFailureReport,
    EvaluationResourceUsage,
    EvaluationRunIdentity,
    EvaluationRunManifest,
    EvaluationStage,
    EvaluationStageStatus,
    EvaluationTracePhase,
    EvaluationTraceStatus,
    KnowledgeSufficiency,
    StrategyObservation,
    StrategyObservationSet,
    StressCategory,
    TraceResourceUsage,
    UnifiedEvaluationQuestion,
    UnifiedEvaluationReport,
    UnifiedQuestionSet,
    UnifiedStrategy,
)
from kg_crag.observability import append_evaluation_trace
from kg_crag.workflow.artifacts import ContentAddressedCache, atomic_model_write


class UnifiedStrategyAdapter(Protocol):
    strategy: UnifiedStrategy

    async def observe(self, question: UnifiedEvaluationQuestion) -> StrategyObservation: ...


class CallableStrategyAdapter:
    """把既有 evaluator 或工作流函数收敛为统一观察边界。"""

    def __init__(
        self,
        strategy: UnifiedStrategy,
        callback: Callable[[UnifiedEvaluationQuestion], Awaitable[StrategyObservation]],
    ) -> None:
        self.strategy = strategy
        self._callback = callback

    async def observe(self, question: UnifiedEvaluationQuestion) -> StrategyObservation:
        observation = await self._callback(question)
        if (
            observation.strategy is not self.strategy
            or observation.question_id != question.question_id
        ):
            raise ValueError("strategy adapter returned an observation for another input")
        return observation


class FixtureStrategyAdapter:
    """仅用于离线管线验收的确定性标注回放，不代表正式系统效果。"""

    def __init__(self, strategy: UnifiedStrategy) -> None:
        self.strategy = strategy

    async def observe(self, question: UnifiedEvaluationQuestion) -> StrategyObservation:
        required = [item.facet_id for item in question.facets if item.required]
        initial_success = question.stress_category is StressCategory.CONTROL
        recoverable_by_strategy: dict[UnifiedStrategy, set[StressCategory]] = {
            UnifiedStrategy.FIXED_HYBRID: {StressCategory.CONTROL},
            UnifiedStrategy.TYPE_ROUTER: {
                StressCategory.CONTROL,
                StressCategory.ENTITY_ALIAS,
                StressCategory.MULTI_HOP_GAP,
                StressCategory.TERMINOLOGY_MISMATCH,
            },
            UnifiedStrategy.FACET_CORRECTIVE: {category for category in StressCategory}
            - {StressCategory.CONFLICT, StressCategory.INTERNAL_MISSING},
        }
        recoverable = recoverable_by_strategy[self.strategy]
        final_success = question.stress_category in recoverable
        evidence = list(question.relevant_evidence) if final_success else []
        truth_available = question.knowledge_sufficiency is KnowledgeSufficiency.SUFFICIENT
        action = not initial_success and self.strategy is UnifiedStrategy.FACET_CORRECTIVE
        return StrategyObservation(
            question_id=question.question_id,
            strategy=self.strategy,
            status=EvaluationStageStatus.SUCCEEDED,
            ranked_evidence_ids=evidence,
            initial_covered_facet_ids=required if initial_success else [],
            final_covered_facet_ids=required if final_success and truth_available else [],
            predicted_sufficient=final_success and truth_available,
            action_ids=[f"fixture-action-{question.question_id}"] if action else [],
            route=("facet-gap" if action else self.strategy.value),
            loop_count=1 if action else 0,
            matched_answer_point_ids=[item.point_id for item in question.answer_points]
            if final_success and truth_available
            else [],
            cited_evidence_ids=evidence,
            usage=EvaluationResourceUsage(
                tool_calls=2 if action else 1,
                latency_ms=2 if action else 1,
            ),
        )


class RecordedStrategyAdapter:
    """读取由真实工作流预先发布的统一观察，不触发任何外部调用。"""

    def __init__(
        self,
        strategy: UnifiedStrategy,
        observations: Mapping[str, StrategyObservation],
    ) -> None:
        self.strategy = strategy
        self._observations = dict(observations)

    async def observe(self, question: UnifiedEvaluationQuestion) -> StrategyObservation:
        try:
            observation = self._observations[question.question_id]
        except KeyError as error:
            raise ValueError("recorded observations are incomplete") from error
        if observation.strategy is not self.strategy:
            raise ValueError("recorded observation strategy mismatch")
        return observation


def build_run_identity(
    manifest: EvaluationDatasetManifest,
    question_set: UnifiedQuestionSet,
    config: UnifiedEvaluationConfig,
    *,
    version_bindings: Mapping[str, str],
    fixture_mode: bool = False,
    selected_answer_strategy: UnifiedStrategy | None = None,
    judge_question_ids: set[str] | None = None,
) -> EvaluationRunIdentity:
    """把所有影响结果的版本绑定到单一运行身份。"""

    required = {"index", "strategy", "model", "prompt", "metric", "judge"}
    if missing := required - version_bindings.keys():
        raise ValueError(f"evaluation version bindings are missing: {sorted(missing)}")
    fields = {
        "dataset_hash": manifest.dataset_hash,
        "split_hash": canonical_digest(question_set),
        "corpus_hash": canonical_digest(manifest.corpus_snapshot),
        "evidence_hash": canonical_digest(manifest.evidence_version),
        "index_hash": canonical_digest(version_bindings["index"]),
        "config_hash": canonical_digest(config),
        "strategy_hash": canonical_digest(
            {
                "version": version_bindings["strategy"],
                "fixture_mode": fixture_mode,
                "selected_answer_strategy": (
                    selected_answer_strategy.value if selected_answer_strategy else None
                ),
            }
        ),
        "model_hash": canonical_digest(version_bindings["model"]),
        "prompt_hash": canonical_digest(version_bindings["prompt"]),
        "metric_hash": canonical_digest(version_bindings["metric"]),
        "judge_hash": canonical_digest(
            {
                "version": version_bindings["judge"],
                "question_ids": sorted(judge_question_ids or set()),
            }
        ),
        "seed": config.seed,
    }
    run_id = "unified-run-" + canonical_digest(fields)[:32]
    return EvaluationRunIdentity(run_id=run_id, **fields)


def _bounded_observation(
    observation: StrategyObservation,
    config: UnifiedEvaluationConfig,
) -> StrategyObservation:
    if len(observation.ranked_evidence_ids) > config.max_candidates_per_question:
        raise ValueError("strategy observation exceeds the candidate limit")
    if observation.loop_count > config.max_loops_per_question:
        raise ValueError("strategy observation exceeds the loop limit")
    if observation.usage.model_calls > config.max_model_calls:
        raise ValueError("strategy observation exceeds the model-call limit")
    return observation


def _without_answer(observation: StrategyObservation) -> StrategyObservation:
    return observation.model_copy(update={"matched_answer_point_ids": [], "cited_evidence_ids": []})


def _sum_usage(observations: Sequence[StrategyObservation]) -> EvaluationResourceUsage:
    costs = [item.usage.estimated_cost for item in observations]
    return EvaluationResourceUsage(
        tool_calls=sum(item.usage.tool_calls for item in observations),
        model_calls=sum(item.usage.model_calls for item in observations),
        judge_calls=sum(item.usage.judge_calls for item in observations),
        web_calls=sum(item.usage.web_calls for item in observations),
        input_tokens=sum(item.usage.input_tokens for item in observations),
        output_tokens=sum(item.usage.output_tokens for item in observations),
        latency_ms=sum(item.usage.latency_ms for item in observations),
        estimated_cost=(
            sum(item for item in costs if item is not None)
            if any(item is not None for item in costs)
            else None
        ),
    )


def _global_budget_exhausted(
    usage: EvaluationResourceUsage,
    config: UnifiedEvaluationConfig,
) -> bool:
    return any(
        (
            usage.model_calls >= config.max_model_calls,
            usage.input_tokens >= config.max_input_tokens,
            usage.output_tokens >= config.max_output_tokens,
            usage.latency_ms >= config.max_latency_ms,
            usage.estimated_cost is not None and usage.estimated_cost >= config.max_estimated_cost,
        )
    )


class UnifiedEvaluationRunner:
    def __init__(
        self,
        adapters: Mapping[UnifiedStrategy, UnifiedStrategyAdapter],
        config: UnifiedEvaluationConfig,
        *,
        results_root: Path,
        workspace_root: Path,
        fixture_mode: bool = False,
        judge_provider: JudgeProvider | None = None,
        judge_cache: ContentAddressedCache | None = None,
        confirm_judge_budget: bool = False,
    ) -> None:
        if set(adapters) != set(config.strategies):
            raise ValueError("runner adapters must match configured strategies")
        self.adapters = adapters
        self.config = config
        self.results_root = results_root
        self.workspace_root = workspace_root
        self.fixture_mode = fixture_mode
        self.judge_provider = judge_provider
        self.judge_cache = judge_cache
        self.confirm_judge_budget = confirm_judge_budget
        self.checkpoints = UnifiedCheckpointStore(results_root, workspace_root=workspace_root)

    async def run(
        self,
        manifest: EvaluationDatasetManifest,
        question_set: UnifiedQuestionSet,
        *,
        version_bindings: Mapping[str, str],
        selected_answer_strategy: UnifiedStrategy | None = None,
        judge_question_ids: set[str] | None = None,
    ) -> UnifiedEvaluationReport:
        if not self.fixture_mode and not manifest.reviewed:
            raise ValueError("real evaluation requires a human-reviewed dataset manifest")
        if len(question_set.questions) > min(self.config.max_questions, 100):
            raise ValueError("evaluation question count exceeds the configured hard limit")
        if selected_answer_strategy is not None and selected_answer_strategy not in self.adapters:
            raise ValueError("answer strategy is not part of the evaluation matrix")
        identity = build_run_identity(
            manifest,
            question_set,
            self.config,
            version_bindings=version_bindings,
            fixture_mode=self.fixture_mode,
            selected_answer_strategy=selected_answer_strategy,
            judge_question_ids=judge_question_ids,
        )
        trace_id = f"trace-{identity.run_id.removeprefix('unified-run-')}"
        trace = append_evaluation_trace(
            [],
            trace_id=trace_id,
            run_id=identity.run_id,
            phase=EvaluationTracePhase.VALIDATE,
            event="dataset_validated",
            status=EvaluationTraceStatus.SUCCEEDED,
            details={
                "split": question_set.split.value,
                "question_count": len(question_set.questions),
            },
            max_events=self.config.max_trace_events,
        )
        started_at = datetime.now(UTC)
        items = []
        observations: list[StrategyObservation] = []
        completed: list[str] = []
        checkpoint_hashes: dict[str, str] = {}
        failures: dict[str, int] = {}
        for question in question_set.questions:
            for strategy in self.config.strategies:
                key = f"{strategy.value}-{question.question_id}"
                checkpoint = self.checkpoints.load(identity, key)
                if checkpoint is not None:
                    items.append(checkpoint.item)
                    observations.append(checkpoint.item.observation)
                    completed.append(key)
                    checkpoint_hashes[key] = canonical_digest(checkpoint)
                    trace = append_evaluation_trace(
                        trace,
                        trace_id=trace_id,
                        run_id=identity.run_id,
                        question_id=question.question_id,
                        phase=EvaluationTracePhase.RETRIEVE,
                        event="checkpoint_reused",
                        status=EvaluationTraceStatus.SKIPPED,
                        details={"strategy": strategy.value},
                        max_events=self.config.max_trace_events,
                    )
                    continue
                start_sequence = len(trace)
                trace = append_evaluation_trace(
                    trace,
                    trace_id=trace_id,
                    run_id=identity.run_id,
                    question_id=question.question_id,
                    phase=EvaluationTracePhase.RETRIEVE,
                    event="strategy_started",
                    status=EvaluationTraceStatus.STARTED,
                    details={"strategy": strategy.value},
                    max_events=self.config.max_trace_events,
                )
                try:
                    if _global_budget_exhausted(_sum_usage(observations), self.config):
                        observation = StrategyObservation(
                            question_id=question.question_id,
                            strategy=strategy,
                            status=EvaluationStageStatus.BUDGET_STOPPED,
                            error=ErrorDetail(
                                code=ErrorCode.CONFIGURATION,
                                message="global evaluation budget exhausted",
                                retryable=False,
                            ),
                        )
                    else:
                        observation = _bounded_observation(
                            await self.adapters[strategy].observe(question), self.config
                        )
                    if selected_answer_strategy is None or strategy is not selected_answer_strategy:
                        observation = _without_answer(observation)
                    if selected_answer_strategy is None and observation.usage.model_calls:
                        raise ValueError("retrieval/coverage matrix must not call a model")
                except Exception as error:
                    observation = StrategyObservation(
                        question_id=question.question_id,
                        strategy=strategy,
                        status=EvaluationStageStatus.FAILED,
                        error=ErrorDetail(
                            code=ErrorCode.INTERNAL,
                            message=str(error)[:1000] or "strategy evaluation failed",
                            retryable=False,
                        ),
                    )
                observations.append(observation)
                if observation.error:
                    code = observation.error.code.value
                    failures[code] = failures.get(code, 0) + 1
                trace_status = (
                    EvaluationTraceStatus.SUCCEEDED
                    if observation.status is EvaluationStageStatus.SUCCEEDED
                    else EvaluationTraceStatus.FAILED
                )
                if strategy is selected_answer_strategy:
                    trace = append_evaluation_trace(
                        trace,
                        trace_id=trace_id,
                        run_id=identity.run_id,
                        question_id=question.question_id,
                        phase=EvaluationTracePhase.ANSWER,
                        event="answer_metrics_recorded",
                        status=trace_status,
                        details={"strategy": strategy.value},
                        max_events=self.config.max_trace_events,
                    )
                trace = append_evaluation_trace(
                    trace,
                    trace_id=trace_id,
                    run_id=identity.run_id,
                    question_id=question.question_id,
                    phase=EvaluationTracePhase.ASSESS,
                    event="strategy_finished",
                    status=trace_status,
                    details={
                        "strategy": strategy.value,
                        "evidence_count": len(observation.ranked_evidence_ids),
                    },
                    usage=TraceResourceUsage(
                        tool_calls=observation.usage.tool_calls,
                        model_calls=observation.usage.model_calls,
                        web_calls=observation.usage.web_calls,
                        input_tokens=observation.usage.input_tokens,
                        output_tokens=observation.usage.output_tokens,
                        latency_ms=observation.usage.latency_ms,
                    ),
                    max_events=self.config.max_trace_events,
                )
                item = evaluate_item(
                    question,
                    observation,
                    k=self.config.top_k,
                    trace_sequences=list(range(start_sequence, len(trace))),
                )
                item = item.model_copy(
                    update={"source_artifact_hash": canonical_digest(observation)}
                )
                checkpoint = EvaluationCheckpoint(
                    identity=identity,
                    stage=EvaluationStage.ASSESS,
                    item=item,
                    created_at=datetime.now(UTC),
                )
                self.checkpoints.save(checkpoint, key)
                items.append(item)
                completed.append(key)
                checkpoint_hashes[key] = canonical_digest(checkpoint)
        usage = _sum_usage(observations)
        judge_records = []
        if self.config.with_judge:
            if self.judge_provider is None or self.judge_cache is None:
                raise ValueError("Judge is enabled but no provider/cache was injected")
            requested = judge_question_ids or set()
            judge_items = [
                item
                for item in items
                if item.strategy is selected_answer_strategy and item.question_id in requested
            ]
            validate_judge_gate(
                self.config,
                judge_items,
                selected_strategy=selected_answer_strategy,
                confirmed_budget=self.confirm_judge_budget,
            )
            judge_records = await evaluate_with_judge(
                self.judge_provider,
                {item.question_id: item for item in question_set.questions},
                judge_items,
                prompt_hash=identity.prompt_hash,
                cache=self.judge_cache,
            )
            for record in judge_records:
                trace = append_evaluation_trace(
                    trace,
                    trace_id=trace_id,
                    run_id=identity.run_id,
                    question_id=record.question_id,
                    phase=EvaluationTracePhase.JUDGE,
                    event="judge_finished",
                    status=(
                        EvaluationTraceStatus.SUCCEEDED
                        if record.status is EvaluationStageStatus.SUCCEEDED
                        else EvaluationTraceStatus.FAILED
                    ),
                    details={"strategy": record.strategy.value},
                    max_events=self.config.max_trace_events,
                )
        slices = aggregate_all_slices(question_set.questions, items)
        trace = append_evaluation_trace(
            trace,
            trace_id=trace_id,
            run_id=identity.run_id,
            phase=EvaluationTracePhase.AGGREGATE,
            event="metrics_aggregated",
            status=EvaluationTraceStatus.SUCCEEDED,
            details={"item_count": len(items), "failure_count": sum(failures.values())},
            max_events=self.config.max_trace_events,
        )
        report = UnifiedEvaluationReport(
            identity=identity,
            split=question_set.split,
            offline=not self.config.online,
            fixture_mode=self.fixture_mode,
            selected_answer_strategy=selected_answer_strategy,
            items=items,
            slices=slices,
            trace=trace,
            judge_status=(
                "failed"
                if any(record.error is not None for record in judge_records)
                else "completed"
                if self.config.with_judge
                else "not_requested"
            ),
            judge_records=judge_records,
            created_at=datetime.now(UTC),
        )
        run_root = self.results_root / identity.run_id
        atomic_model_write(run_root / "report.json", report, workspace_root=self.workspace_root)
        failure_report = EvaluationFailureReport(
            run_id=identity.run_id,
            failures=[
                EvaluationFailureRecord(
                    question_id=item.question_id,
                    strategy=item.strategy,
                    status=item.observation.status,
                    error=item.observation.error,
                    usage=item.observation.usage,
                )
                for item in items
                if item.observation.error is not None
            ],
        )
        atomic_model_write(
            run_root / "failures.json", failure_report, workspace_root=self.workspace_root
        )
        run_manifest = EvaluationRunManifest(
            identity=identity,
            split=question_set.split,
            status=EvaluationStageStatus.SUCCEEDED,
            selected_answer_strategy=selected_answer_strategy,
            completed_item_keys=completed,
            checkpoint_hashes=checkpoint_hashes,
            budget=usage,
            failures=failures,
            started_at=started_at,
            finished_at=datetime.now(UTC),
        )
        atomic_model_write(
            run_root / "run-manifest.json", run_manifest, workspace_root=self.workspace_root
        )
        return report


def fixture_adapters() -> dict[UnifiedStrategy, FixtureStrategyAdapter]:
    return {strategy: FixtureStrategyAdapter(strategy) for strategy in UnifiedStrategy}


def recorded_adapters(
    observation_set: StrategyObservationSet,
    question_set: UnifiedQuestionSet,
    *,
    dataset_hash: str,
) -> dict[UnifiedStrategy, RecordedStrategyAdapter]:
    """严格校验记录覆盖和身份后，为三个策略构造适配器。"""

    if observation_set.dataset_hash != dataset_hash:
        raise ValueError("recorded observation dataset hash drifted")
    expected_split_hash = canonical_digest(question_set)
    if observation_set.split_hash != expected_split_hash:
        raise ValueError("recorded observation split hash drifted")
    expected = {
        (question.question_id, strategy)
        for question in question_set.questions
        for strategy in UnifiedStrategy
    }
    actual = {(item.question_id, item.strategy) for item in observation_set.observations}
    if actual != expected:
        missing = sorted(
            f"{strategy.value}:{question_id}" for question_id, strategy in expected - actual
        )
        extra = sorted(
            f"{strategy.value}:{question_id}" for question_id, strategy in actual - expected
        )
        raise ValueError(f"recorded observation coverage mismatch: missing={missing} extra={extra}")
    return {
        strategy: RecordedStrategyAdapter(
            strategy,
            {
                item.question_id: item
                for item in observation_set.observations
                if item.strategy is strategy
            },
        )
        for strategy in UnifiedStrategy
    }
