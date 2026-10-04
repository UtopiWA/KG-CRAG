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
    load_unified_dataset,
    token_jaccard,
    validate_cross_split_leakage,
)
from kg_crag.models import (
    EvaluationDatasetManifest,
    EvaluationItemResult,
    EvaluationMetricValue,
    EvaluationResourceUsage,
    EvaluationRunIdentity,
    EvaluationSplit,
    EvaluationStageStatus,
    StrategyObservation,
    UnifiedEvaluationReport,
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
    assert not manifest.reviewed
    assert len(dev.questions) == 40 and len(test.questions) == 20
    assert len(dev.questions) + len(test.questions) == 60


def test_canonical_hash_ignores_mapping_order_but_detects_content_change() -> None:
    assert canonical_digest({"a": 1, "b": 2}) == canonical_digest({"b": 2, "a": 1})
    assert canonical_digest({"a": 1}) != canonical_digest({"a": 2})


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
        workspace_root=tmp_path,
    )
    assert load_test_selection(selection_path, manifest=manifest) == selection
    with pytest.raises(FileExistsError):
        freeze_development_selection(
            selection_path,
            manifest=manifest,
            report=report,
            selected_strategy=UnifiedStrategy.FACET_CORRECTIVE,
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
