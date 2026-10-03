"""内容寻址缓存、逐阶段原子检查点和运行清单。"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from kg_crag.ingestion.storage import stable_json_bytes
from kg_crag.models import CorrectionCheckpoint, CorrectionState, RunIdentity, WorkflowStage
from kg_crag.models.domain import StrictModel

T = TypeVar("T", bound=BaseModel)


def _contained(root: Path, path: Path) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError("artifact path escapes the configured workspace root")
    return resolved


def atomic_model_write(path: Path, value: BaseModel, *, workspace_root: Path) -> None:
    destination = _contained(workspace_root, path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        data = stable_json_bytes(value)
        temporary.write_bytes(data)
        if temporary.read_bytes() != data:
            raise OSError("atomic artifact verification failed")
        json.loads(temporary.read_text(encoding="utf-8"))
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


class ContentAddressedCache:
    def __init__(self, root: Path, *, workspace_root: Path) -> None:
        self.root = _contained(workspace_root, root)
        self.workspace_root = workspace_root

    def load(self, namespace: str, key: str, model: type[T]) -> T | None:
        path = _contained(self.workspace_root, self.root / namespace / f"{key}.json")
        try:
            return model.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def save(self, namespace: str, key: str, value: BaseModel) -> None:
        atomic_model_write(
            self.root / namespace / f"{key}.json", value, workspace_root=self.workspace_root
        )


class CheckpointStore:
    def __init__(self, root: Path, *, workspace_root: Path) -> None:
        self.root = _contained(workspace_root, root)
        self.workspace_root = workspace_root

    def save(self, stage: WorkflowStage, state: CorrectionState, sequence: int) -> Path:
        checkpoint = CorrectionCheckpoint(
            identity=state.identity,
            stage=stage,
            sequence=sequence,
            state=state,
            created_at=datetime.now(UTC),
        )
        path = self.root / state.identity.run_id / f"{sequence:03d}-{stage.value}.json"
        atomic_model_write(path, checkpoint, workspace_root=self.workspace_root)
        return path

    def load_latest(self, identity: RunIdentity) -> CorrectionCheckpoint | None:
        directory = _contained(self.workspace_root, self.root / identity.run_id)
        if not directory.is_dir():
            return None
        for path in sorted(directory.glob("*.json"), reverse=True):
            try:
                checkpoint = CorrectionCheckpoint.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                continue
            if checkpoint.identity == identity:
                return checkpoint
        return None

    def save_manifest(self, identity: RunIdentity) -> Path:
        """只收录已通过严格回读校验的阶段检查点。"""

        directory = _contained(self.workspace_root, self.root / identity.run_id)
        checkpoints: list[tuple[Path, CorrectionCheckpoint]] = []
        for path in sorted(directory.glob("[0-9][0-9][0-9]-*.json")):
            try:
                checkpoint = CorrectionCheckpoint.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                continue
            if checkpoint.identity == identity:
                checkpoints.append((path, checkpoint))
        manifest = RunManifest(
            identity=identity,
            completed_stages=[item.stage for _path, item in checkpoints],
            checkpoint_paths=[
                path.resolve().relative_to(self.workspace_root.resolve()).as_posix()
                for path, _item in checkpoints
            ],
            created_at=datetime.now(UTC),
        )
        path = directory / "manifest.json"
        atomic_model_write(path, manifest, workspace_root=self.workspace_root)
        return path


class RunManifest(StrictModel):
    identity: RunIdentity
    completed_stages: list[WorkflowStage]
    checkpoint_paths: list[str]
    created_at: datetime
