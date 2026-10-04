"""统一冻结数据、哈希、泄漏与一次性测试门禁测试。"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from kg_crag.evaluation.artifacts import (
    ensure_formal_test_allowed,
    freeze_development_selection,
    load_test_selection,
    publish_formal_test_lock,
)
from kg_crag.evaluation.config import UnifiedEvaluationConfig
from kg_crag.evaluation.dataset import (
    canonical_digest,
    dataset_digest,
    load_unified_dataset,
    token_jaccard,
    validate_cross_split_leakage,
    validate_unified_dataset,
)
from kg_crag.models import (
    EvaluationDatasetManifest,
    EvaluationEvidenceCatalog,
    EvaluationItemResult,
    EvaluationMetricValue,
    EvaluationQuestionOrigin,
    EvaluationQuestionSourceCatalog,
    EvaluationResourceUsage,
    EvaluationRunIdentity,
    EvaluationSplit,
    EvaluationStageStatus,
    EvidenceMatchMode,
    StrategyObservation,
    StressCategory,
    UnifiedEvaluationReport,
    UnifiedQuestionType,
    UnifiedStrategy,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _identity() -> EvaluationRunIdentity:
    return EvaluationRunIdentity(
        run_id="unified-run-" + "1" * 32,
        dataset_hash="2" * 64,
        split_hash="3" * 64,
        corpus_hash="4" * 64,
        evidence_hash="5" * 64,
        index_hash="6" * 64,
        config_hash="7" * 64,
        strategy_hash="8" * 64,
        model_hash="9" * 64,
        prompt_hash="a" * 64,
        metric_hash="b" * 64,
        judge_hash="c" * 64,
        seed=42,
    )


def _report() -> UnifiedEvaluationReport:
    observation = StrategyObservation(
        question_id="ueq-fixture-001",
        strategy=UnifiedStrategy.FACET_CORRECTIVE,
        status=EvaluationStageStatus.SUCCEEDED,
        usage=EvaluationResourceUsage(),
    )
    item = EvaluationItemResult(
        question_id=observation.question_id,
        strategy=observation.strategy,
        observation=observation,
        metrics={
            "success": EvaluationMetricValue(
                value=1.0,
                numerator=1,
                denominator=1,
                question_ids=[observation.question_id],
            )
        },
    )
    return UnifiedEvaluationReport(
        identity=_identity(),
        split=EvaluationSplit.DEV,
        offline=True,
        fixture_mode=False,
        items=[item],
        slices={"facet_corrective": []},
        created_at=datetime.now(UTC),
    )


def test_frozen_dataset_loads_with_expected_counts_and_hashes() -> None:
    config = UnifiedEvaluationConfig()
    manifest, dev, test = load_unified_dataset(
        PROJECT_ROOT / config.manifest_path,
        workspace_root=PROJECT_ROOT,
        near_duplicate_threshold=config.near_duplicate_threshold,
    )
    assert manifest.reviewed
    assert "dev 40 题、test 20 题" in manifest.review_note
    assert len(dev.questions) == 40 and len(test.questions) == 20
    assert len(dev.questions) + len(test.questions) == 60
    conflict = next(
        item for item in dev.questions if item.question_id.endswith("corrective-dev-questions-q16")
    )
    assert [item.evidence_ids for item in conflict.answer_points[:3]] == [
        ["conflict-q16-a"],
        ["conflict-q16-b"],
        ["target-q16"],
    ]
    assert len(conflict.answer_points) == 3
    assert conflict.facets[-1].evidence_match is EvidenceMatchMode.ALL

    questions = {item.question_id: item for item in [*dev.questions, *test.questions]}
    relabeled_controls = {
        "ueq-literature-toolllm-problem",
        "ueq-literature-paperqa-motivation",
        "ueq-literature-graphrag-goal",
        "ueq-literature-tree-of-thoughts-tasks",
        "ueq-literature-agentbench-failures",
        "ueq-literature-lats-domains",
        "ueq-literature-hipporag-components",
        "ueq-literature-agentic-rag-design-patterns",
        "ueq-literature-lightrag-design",
    }
    assert all(
        questions[question_id].stress_category is StressCategory.CONTROL
        for question_id in relabeled_controls
    )
    assert questions["ueq-literature-hipporag-components"].question_types == [
        UnifiedQuestionType.SYNTHESIS
    ]
    assert questions["ueq-literature-lightrag-design"].question_types == [
        UnifiedQuestionType.SYNTHESIS
    ]
    environment_focus = questions["ueq-literature-agentbench-environment-focus"]
    assert [item.description for item in environment_focus.answer_points] == [
        "推理能力。",
        "决策能力。",
    ]
    assert all(item.required for item in environment_focus.facets)


def test_question_sources_distinguish_exact_migrations_from_chunk_derivations() -> None:
    manifest, _, _ = load_unified_dataset(
        PROJECT_ROOT / "data/evaluation/unified/manifest.json",
        workspace_root=PROJECT_ROOT,
    )
    catalog = EvaluationQuestionSourceCatalog.model_validate_json(
        (PROJECT_ROOT / manifest.question_sources_path).read_text(encoding="utf-8")
    )
    sources = {item.question_id: item for item in catalog.records}
    migrated = sources["ueq-literature-camel-roleplay"]
    assert migrated.origin is EvaluationQuestionOrigin.FIXTURE_MIGRATION
    assert migrated.origin_locator.endswith("#camel-roleplay")
    derived = sources["ueq-literature-generative-agents-architecture"]
    assert derived.origin is EvaluationQuestionOrigin.CHUNK_DERIVED
    assert derived.origin_locator.startswith("evidence:chunk-")


def test_canonical_hash_ignores_mapping_order_but_detects_content_change() -> None:
    assert canonical_digest({"a": 1, "b": 2}) == canonical_digest({"b": 2, "a": 1})
    assert canonical_digest({"a": 1}) != canonical_digest({"a": 2})


def test_dataset_rejects_an_unresolved_evidence_reference() -> None:
    manifest, dev, test = load_unified_dataset(
        PROJECT_ROOT / "data/evaluation/unified/manifest.json",
        workspace_root=PROJECT_ROOT,
    )
    catalog = EvaluationEvidenceCatalog.model_validate_json(
        (PROJECT_ROOT / manifest.evidence_path).read_text(encoding="utf-8")
    )
    question_sources = EvaluationQuestionSourceCatalog.model_validate_json(
        (PROJECT_ROOT / manifest.question_sources_path).read_text(encoding="utf-8")
    )
    referenced = next(iter(dev.questions[0].relevant_evidence))
    incomplete = catalog.model_copy(
        update={"records": [item for item in catalog.records if item.evidence_id != referenced]}
    )
    evidence_hash = canonical_digest(incomplete)
    incomplete_manifest = manifest.model_copy(
        update={
            "evidence_hash": evidence_hash,
            "dataset_hash": dataset_digest(
                dataset_version=manifest.dataset_version,
                corpus_snapshot=manifest.corpus_snapshot,
                evidence_version=manifest.evidence_version,
                evidence_hash=evidence_hash,
                question_sources_hash=manifest.question_sources_hash,
                dev_hash=manifest.dev_hash,
                test_hash=manifest.test_hash,
            ),
        }
    )
    with pytest.raises(ValueError, match="relevant_evidence is unresolved"):
        validate_unified_dataset(
            incomplete_manifest,
            dev,
            test,
            evidence_catalog=incomplete,
            question_source_catalog=question_sources,
            workspace_root=PROJECT_ROOT,
        )


def test_evidence_catalog_rejects_content_hash_drift() -> None:
    manifest, _, _ = load_unified_dataset(
        PROJECT_ROOT / "data/evaluation/unified/manifest.json",
        workspace_root=PROJECT_ROOT,
    )
    payload = (PROJECT_ROOT / manifest.evidence_path).read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="content hash drifted"):
        EvaluationEvidenceCatalog.model_validate_json(
            payload.replace('"content": "', '"content": "tampered ', 1)
        )


def test_cross_split_leakage_detects_groups_sources_and_near_duplicates() -> None:
    _, dev, test = load_unified_dataset(
        PROJECT_ROOT / "data/evaluation/unified/manifest.json",
        workspace_root=PROJECT_ROOT,
    )
    left = dev.questions[0]
    right = test.questions[0]
    validate_cross_split_leakage([left], [right], threshold=0.92)
    with pytest.raises(ValueError, match="leakage groups"):
        validate_cross_split_leakage(
            [left],
            [right.model_copy(update={"leakage_group_id": left.leakage_group_id})],
            threshold=0.92,
        )
    with pytest.raises(ValueError, match="answer source"):
        validate_cross_split_leakage(
            [left],
            [right.model_copy(update={"source_group_ids": left.source_group_ids})],
            threshold=0.92,
        )
    near = right.model_copy(
        update={
            "question": left.question,
            "normalized_question": left.normalized_question,
        }
    )
    with pytest.raises(ValueError, match="near duplicate"):
        validate_cross_split_leakage([left], [near], threshold=0.92)
    assert token_jaccard("Agent uses tools", "agent uses tools") == 1.0


def test_development_selection_and_test_lock_are_non_overwriting(tmp_path: Path) -> None:
    report = _report()
    manifest = EvaluationDatasetManifest(
        dataset_version="fixture-v1",
        corpus_snapshot="corpus-v1",
        evidence_version="evidence-v1",
        evidence_path="evidence.json",
        evidence_hash="f" * 64,
        question_sources_path="question-sources.json",
        question_sources_hash="a" * 64,
        dev_path="dev.json",
        test_path="test.json",
        dev_count=40,
        test_count=20,
        dev_hash="d" * 64,
        test_hash="e" * 64,
        dataset_hash="2" * 64,
        annotation_guide_version="guide-v1",
        reviewed=True,
        review_note="fixture review completed",
    )
    selection_path = tmp_path / "selection.json"
    selection = freeze_development_selection(
        selection_path,
        manifest=manifest,
        report=report,
        selected_strategy=UnifiedStrategy.FACET_CORRECTIVE,
        version_declarations={"model": "no-model", "prompt": "no-prompt"},
        thresholds={"top_k": 10},
        workspace_root=tmp_path,
    )
    assert load_test_selection(selection_path, manifest=manifest) == selection
    with pytest.raises(FileExistsError):
        freeze_development_selection(
            selection_path,
            manifest=manifest,
            report=report,
            selected_strategy=UnifiedStrategy.FACET_CORRECTIVE,
            version_declarations={"model": "no-model", "prompt": "no-prompt"},
            thresholds={"top_k": 10},
            workspace_root=tmp_path,
        )
    lock_path = tmp_path / "lock.json"
    with pytest.raises(ValueError, match="confirmation"):
        ensure_formal_test_allowed(lock_path, confirmed=False, tuning_requested=False)
    with pytest.raises(ValueError, match="tuning"):
        ensure_formal_test_allowed(lock_path, confirmed=True, tuning_requested=True)
    ensure_formal_test_allowed(lock_path, confirmed=True, tuning_requested=False)
    publish_formal_test_lock(
        lock_path,
        manifest=manifest,
        selection=selection,
        report=report,
        workspace_root=tmp_path,
    )
    with pytest.raises(FileExistsError, match="already"):
        ensure_formal_test_allowed(lock_path, confirmed=True, tuning_requested=False)
