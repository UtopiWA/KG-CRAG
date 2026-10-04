"""统一评测检查点、开发选择和正式测试锁。"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from kg_crag.evaluation.dataset import canonical_digest
from kg_crag.models import (
    DevelopmentSelection,
    EvaluationCheckpoint,
    EvaluationDatasetManifest,
    EvaluationRunIdentity,
    FormalTestLock,
    UnifiedEvaluationReport,
    UnifiedStrategy,
)
from kg_crag.workflow.artifacts import atomic_model_write


class UnifiedCheckpointStore:
    """按完整运行身份隔离逐题逐策略检查点。"""

    def __init__(self, root: Path, *, workspace_root: Path) -> None:
        self.root = root
        self.workspace_root = workspace_root

    def path_for(self, identity: EvaluationRunIdentity, key: str) -> Path:
        return self.root / identity.run_id / "items" / f"{key}.json"

    def load(self, identity: EvaluationRunIdentity, key: str) -> EvaluationCheckpoint | None:
        path = self.path_for(identity, key)
        try:
            checkpoint = EvaluationCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return checkpoint if checkpoint.identity == identity else None

    def save(self, checkpoint: EvaluationCheckpoint, key: str) -> Path:
        path = self.path_for(checkpoint.identity, key)
        atomic_model_write(path, checkpoint, workspace_root=self.workspace_root)
        return path


def freeze_development_selection(
    path: Path,
    *,
    manifest: EvaluationDatasetManifest,
    report: UnifiedEvaluationReport,
    selected_strategy: UnifiedStrategy,
    version_declarations: Mapping[str, str],
    thresholds: Mapping[str, str | int | float | bool],
    workspace_root: Path,
) -> DevelopmentSelection:
    """一次性写入开发集选择，避免正式测试前继续静默调参。"""

    if report.split.value != "dev":
        raise ValueError("development selection requires a dev report")
    if report.fixture_mode:
        raise ValueError("fixture-mode reports cannot freeze a development selection")
    if path.exists():
        raise FileExistsError("development selection is already frozen")
    selection = DevelopmentSelection(
        dataset_version=manifest.dataset_version,
        dataset_hash=manifest.dataset_hash,
        selected_strategy=selected_strategy,
        config_hash=report.identity.config_hash,
        prompt_hash=report.identity.prompt_hash,
        model_hash=report.identity.model_hash,
        report_hash=canonical_digest(report),
        version_declarations=dict(version_declarations),
        thresholds=dict(thresholds),
        frozen_at=datetime.now(UTC),
    )
    atomic_model_write(path, selection, workspace_root=workspace_root)
    return selection


def load_test_selection(
    path: Path,
    *,
    manifest: EvaluationDatasetManifest,
) -> DevelopmentSelection:
    try:
        selection = DevelopmentSelection.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError("formal test requires a frozen development selection") from error
    if (
        selection.dataset_version != manifest.dataset_version
        or selection.dataset_hash != manifest.dataset_hash
    ):
        raise ValueError("development selection does not match the frozen dataset")
    return selection


def ensure_formal_test_allowed(
    lock_path: Path,
    *,
    confirmed: bool,
    tuning_requested: bool,
) -> None:
    """在调用被评系统前拒绝未确认、调参或同版本二次运行。"""

    if not confirmed:
        raise ValueError("formal test requires explicit confirmation")
    if tuning_requested:
        raise ValueError("formal test cannot run with tuning or strategy selection")
    if lock_path.exists():
        raise FileExistsError("this frozen test dataset version already has a formal run")


def publish_formal_test_lock(
    path: Path,
    *,
    manifest: EvaluationDatasetManifest,
    selection: DevelopmentSelection,
    report: UnifiedEvaluationReport,
    workspace_root: Path,
) -> FormalTestLock:
    if report.fixture_mode:
        raise ValueError("fixture-mode reports cannot publish a formal test lock")
    if path.exists():
        raise FileExistsError("formal test lock already exists")
    lock = FormalTestLock(
        dataset_version=manifest.dataset_version,
        dataset_hash=manifest.dataset_hash,
        selection_hash=canonical_digest(selection),
        run_id=report.identity.run_id,
        report_hash=canonical_digest(report),
        created_at=datetime.now(UTC),
    )
    atomic_model_write(path, lock, workspace_root=workspace_root)
    return lock
