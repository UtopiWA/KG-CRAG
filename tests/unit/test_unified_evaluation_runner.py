"""统一 runner 的身份、失败隔离、恢复、预算与回答门禁测试。"""

from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from kg_crag.evaluation.config import UnifiedEvaluationConfig
from kg_crag.evaluation.dataset import canonical_digest
from kg_crag.evaluation.runner import (
    CallableStrategyAdapter,
    UnifiedEvaluationRunner,
    build_run_identity,
    recorded_adapters,
)
from kg_crag.models import (
    EvaluationAnswerPoint,
    EvaluationDatasetManifest,
    EvaluationFacetTarget,
    EvaluationResourceUsage,
    EvaluationSplit,
    EvaluationStageStatus,
    KnowledgeSufficiency,
    StrategyObservation,
    StrategyObservationSet,
    StressCategory,
    UnifiedEvaluationQuestion,
    UnifiedQuestionSet,
    UnifiedQuestionType,
    UnifiedStrategy,
)


def _question(index: int) -> UnifiedEvaluationQuestion:
    evidence_id = f"evidence-{index}"
    facet_id = f"facet-{index:016x}"
    point_id = f"point-{index:016x}"
    return UnifiedEvaluationQuestion(
        question_id=f"ueq-runner-{index:03d}",
        split=EvaluationSplit.DEV,
        question=f"What supports fixture {index}?",
        normalized_question=f"what supports fixture {index}?",
        question_types=[UnifiedQuestionType.FACTOID],
        stress_category=StressCategory.CONTROL,
        knowledge_sufficiency=KnowledgeSufficiency.SUFFICIENT,
        answer_points=[
            EvaluationAnswerPoint(
                point_id=point_id,
                description="Expected point",
                evidence_ids=[evidence_id],
            )
        ],
        facets=[
            EvaluationFacetTarget(
                facet_id=facet_id,
                description="Expected facet",
                target_evidence_ids=[evidence_id],
            )
        ],
        relevant_evidence={evidence_id: 3},
        minimum_sufficient_evidence_sets=[[evidence_id]],
        leakage_group_id=f"leak-runner-{index}",
        source_group_ids=[f"source-runner-{index}"],
        source_fixture=f"tests/fixtures/runner.json#{index}",
    )


def _question_set() -> UnifiedQuestionSet:
    return UnifiedQuestionSet(
        dataset_version="fixture-v1",
        split=EvaluationSplit.DEV,
        questions=[_question(1), _question(2)],
    )


def _manifest() -> EvaluationDatasetManifest:
    return EvaluationDatasetManifest(
        dataset_version="fixture-v1",
        corpus_snapshot="corpus-v1",
        evidence_version="evidence-v1",
        evidence_path="evidence.json",
        evidence_hash="0" * 64,
        question_sources_path="question-sources.json",
        question_sources_hash="a" * 64,
        dev_path="dev.json",
        test_path="test.json",
        dev_count=40,
        test_count=20,
        dev_hash="1" * 64,
        test_hash="2" * 64,
        dataset_hash="3" * 64,
        annotation_guide_version="guide-v1",
        reviewed=True,
        review_note="fixture review completed",
    )


def _versions(**updates: str) -> dict[str, str]:
    versions = {
        "index": "index-v1",
        "strategy": "strategy-v1",
        "model": "no-model",
        "prompt": "no-prompt",
        "metric": "metrics-v1",
        "judge": "judge-disabled",
    }
    versions.update(updates)
    return versions


def _adapters(
    calls: list[str],
    *,
    fail: tuple[UnifiedStrategy, str] | None = None,
    candidates: int = 1,
) -> dict[UnifiedStrategy, CallableStrategyAdapter]:
    output: dict[UnifiedStrategy, CallableStrategyAdapter] = {}
    for strategy in UnifiedStrategy:

        async def observe(
            question: UnifiedEvaluationQuestion,
            current: UnifiedStrategy = strategy,
        ) -> StrategyObservation:
            calls.append(f"{current.value}:{question.question_id}")
            if fail == (current, question.question_id):
                raise RuntimeError("fixture strategy failure")
            relevant = list(question.relevant_evidence)
            ranked = [*relevant, *[f"other-{index}" for index in range(candidates - 1)]]
            return StrategyObservation(
                question_id=question.question_id,
                strategy=current,
                status=EvaluationStageStatus.SUCCEEDED,
                ranked_evidence_ids=ranked,
                initial_covered_facet_ids=[],
                final_covered_facet_ids=[item.facet_id for item in question.facets],
                predicted_sufficient=True,
                matched_answer_point_ids=[item.point_id for item in question.answer_points],
                cited_evidence_ids=relevant,
                usage=EvaluationResourceUsage(tool_calls=1, latency_ms=1),
            )

        callback: Callable[[UnifiedEvaluationQuestion], Awaitable[StrategyObservation]] = observe
        output[strategy] = CallableStrategyAdapter(strategy, callback)
    return output


async def test_runner_isolates_failure_and_reuses_checkpoints(tmp_path: Path) -> None:
    calls: list[str] = []
    failing_key = (UnifiedStrategy.TYPE_ROUTER, "ueq-runner-002")
    runner = UnifiedEvaluationRunner(
        _adapters(calls, fail=failing_key),
        UnifiedEvaluationConfig(max_questions=10),
        results_root=tmp_path / "results",
        workspace_root=tmp_path,
    )
    first = await runner.run(_manifest(), _question_set(), version_bindings=_versions())
    second = await runner.run(_manifest(), _question_set(), version_bindings=_versions())
    assert len(calls) == 6
    assert sum(item.observation.error is not None for item in first.items) == 1
    assert first.slices == second.slices
    run_root = tmp_path / "results" / first.identity.run_id
    assert (run_root / "report.json").is_file()
    assert (run_root / "run-manifest.json").is_file()
    assert any(item.event == "checkpoint_reused" for item in second.trace)


async def test_real_runner_rejects_dataset_pending_human_review(tmp_path: Path) -> None:
    runner = UnifiedEvaluationRunner(
        _adapters([]),
        UnifiedEvaluationConfig(max_questions=10),
        results_root=tmp_path / "results",
        workspace_root=tmp_path,
    )
    manifest = _manifest().model_copy(update={"reviewed": False})
    with pytest.raises(ValueError, match="human-reviewed"):
        await runner.run(manifest, _question_set(), version_bindings=_versions())


def test_run_identity_changes_for_each_result_affecting_version() -> None:
    config = UnifiedEvaluationConfig(max_questions=10)
    base = build_run_identity(_manifest(), _question_set(), config, version_bindings=_versions())
    changed_index = build_run_identity(
        _manifest(), _question_set(), config, version_bindings=_versions(index="index-v2")
    )
    changed_config = build_run_identity(
        _manifest(),
        _question_set(),
        config.model_copy(update={"top_k": 11}),
        version_bindings=_versions(),
    )
    assert base.run_id != changed_index.run_id
    assert base.index_hash != changed_index.index_hash
    assert base.run_id != changed_config.run_id
    selected = build_run_identity(
        _manifest(),
        _question_set(),
        config,
        version_bindings=_versions(),
        selected_answer_strategy=UnifiedStrategy.FACET_CORRECTIVE,
    )
    assert base.run_id != selected.run_id


async def test_candidate_limit_becomes_per_item_failure(tmp_path: Path) -> None:
    calls: list[str] = []
    config = UnifiedEvaluationConfig(max_questions=10, max_candidates_per_question=1)
    runner = UnifiedEvaluationRunner(
        _adapters(calls, candidates=2),
        config,
        results_root=tmp_path / "results",
        workspace_root=tmp_path,
    )
    report = await runner.run(_manifest(), _question_set(), version_bindings=_versions())
    assert len(report.items) == 6
    assert all(item.observation.status is EvaluationStageStatus.FAILED for item in report.items)


async def test_only_selected_strategy_keeps_answer_fields(tmp_path: Path) -> None:
    calls: list[str] = []
    runner = UnifiedEvaluationRunner(
        _adapters(calls),
        UnifiedEvaluationConfig(max_questions=10),
        results_root=tmp_path / "results",
        workspace_root=tmp_path,
    )
    report = await runner.run(
        _manifest(),
        _question_set(),
        version_bindings=_versions(),
        selected_answer_strategy=UnifiedStrategy.FACET_CORRECTIVE,
    )
    selected = [item for item in report.items if item.strategy is UnifiedStrategy.FACET_CORRECTIVE]
    others = [
        item for item in report.items if item.strategy is not UnifiedStrategy.FACET_CORRECTIVE
    ]
    assert all(item.observation.matched_answer_point_ids for item in selected)
    assert all(not item.observation.matched_answer_point_ids for item in others)
    assert all("content" not in event.details for event in report.trace)


def test_recorded_adapters_require_exact_dataset_split_and_matrix() -> None:
    question_set = _question_set()
    observations = [
        StrategyObservation(
            question_id=question.question_id,
            strategy=strategy,
            status=EvaluationStageStatus.SUCCEEDED,
        )
        for question in question_set.questions
        for strategy in UnifiedStrategy
    ]
    observation_set = StrategyObservationSet(
        dataset_hash=_manifest().dataset_hash,
        split_hash=canonical_digest(question_set),
        source_versions={"retrieval": "fixture-v1"},
        observations=observations,
    )
    adapters = recorded_adapters(
        observation_set,
        question_set,
        dataset_hash=_manifest().dataset_hash,
    )
    assert set(adapters) == set(UnifiedStrategy)
    with pytest.raises(ValueError, match="coverage mismatch"):
        recorded_adapters(
            observation_set.model_copy(update={"observations": observations[:-1]}),
            question_set,
            dataset_hash=_manifest().dataset_hash,
        )
