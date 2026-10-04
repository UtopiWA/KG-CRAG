"""回答工作流的原子检查点、最终结果和运行清单。"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from kg_crag.models import (
    AnswerCheckpoint,
    AnswerRunIdentity,
    AnswerWorkflowStage,
    AnswerWorkflowState,
    GroundedAnswerResult,
)
from kg_crag.models.domain import StrictModel
from kg_crag.workflow.artifacts import _contained, atomic_model_write


class AnswerRunManifest(StrictModel):
    identity: AnswerRunIdentity
    completed_stages: list[AnswerWorkflowStage]
    checkpoint_paths: list[str]
    result_path: str | None = None
    created_at: datetime


class AnswerCheckpointStore:
    def __init__(self, root: Path, *, workspace_root: Path) -> None:
        self.root = _contained(workspace_root, root)
        self.workspace_root = workspace_root

    def save(self, state: AnswerWorkflowState, sequence: int) -> Path:
        checkpoint = AnswerCheckpoint(
            identity=state.identity,
            stage=state.stage,
            sequence=sequence,
            state=state,
            created_at=datetime.now(UTC),
        )
        path = self.root / state.identity.run_id / f"{sequence:03d}-{state.stage.value}.json"
        atomic_model_write(path, checkpoint, workspace_root=self.workspace_root)
        return path

    def load_latest(self, identity: AnswerRunIdentity) -> AnswerCheckpoint | None:
        directory = _contained(self.workspace_root, self.root / identity.run_id)
        if not directory.is_dir():
            return None
        by_sequence: dict[int, AnswerCheckpoint] = {}
        for path in sorted(directory.glob("[0-9][0-9][0-9]-*.json")):
            try:
                checkpoint = AnswerCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if checkpoint.identity == identity:
                by_sequence[checkpoint.sequence] = checkpoint
        latest: AnswerCheckpoint | None = None
        for expected in range(len(by_sequence) + 1):
            item = by_sequence.get(expected)
            if item is None:
                break
            latest = item
        return latest

    def save_result(self, result: GroundedAnswerResult) -> Path:
        path = self.root / result.identity.run_id / "result.json"
        atomic_model_write(path, result, workspace_root=self.workspace_root)
        return path

    def load_result(self, identity: AnswerRunIdentity) -> GroundedAnswerResult | None:
        path = _contained(self.workspace_root, self.root / identity.run_id / "result.json")
        try:
            result = GroundedAnswerResult.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return result if result.identity == identity else None

    def save_manifest(self, identity: AnswerRunIdentity) -> Path:
        directory = _contained(self.workspace_root, self.root / identity.run_id)
        checkpoints: list[tuple[Path, AnswerCheckpoint]] = []
        for path in sorted(directory.glob("[0-9][0-9][0-9]-*.json")):
            try:
                checkpoint = AnswerCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if checkpoint.identity == identity:
                checkpoints.append((path, checkpoint))
        result_path = directory / "result.json"
        relative_result = (
            result_path.resolve().relative_to(self.workspace_root.resolve()).as_posix()
            if result_path.is_file()
            else None
        )
        manifest = AnswerRunManifest(
            identity=identity,
            completed_stages=[item.stage for _path, item in checkpoints],
            checkpoint_paths=[
                path.resolve().relative_to(self.workspace_root.resolve()).as_posix()
                for path, _item in checkpoints
            ],
            result_path=relative_result,
            created_at=datetime.now(UTC),
        )
        path = directory / "manifest.json"
        atomic_model_write(path, manifest, workspace_root=self.workspace_root)
        return path
