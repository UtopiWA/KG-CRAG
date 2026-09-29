"""摄取产物的稳定序列化、哈希校验与原子版本发布。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from kg_crag.models import (
    Chunk,
    CleanedDocument,
    IngestionRunManifest,
    Paper,
    ParsedDocument,
    QualityReport,
)


def paper_key(paper_id: str) -> str:
    return "paper-" + hashlib.sha256(paper_id.encode()).hexdigest()[:16]


def stable_json_bytes(value: BaseModel | dict[str, Any]) -> bytes:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def stable_jsonl_bytes(chunks: list[Chunk]) -> bytes:
    lines = [
        json.dumps(
            chunk.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        for chunk in sorted(chunks, key=lambda item: item.ordinal)
    ]
    return (("\n".join(lines) + "\n") if lines else "").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class ArtifactStore:
    """按 Paper 键和处理版本发布与复用可再生产物。"""

    def __init__(
        self,
        interim_root: Path,
        processed_root: Path,
        *,
        before_publish: Callable[[Path], None] | None = None,
    ) -> None:
        self.interim_root = interim_root
        self.processed_root = processed_root
        self.before_publish = before_publish

    def version_directories(self, paper_id: str, version: str) -> tuple[Path, Path]:
        key = paper_key(paper_id)
        return self.interim_root / key / version, self.processed_root / key / version

    def _artifact_paths(self, paper_id: str, version: str) -> dict[str, Path]:
        interim, processed = self.version_directories(paper_id, version)
        return {
            "parsed_document.json": interim / "parsed_document.json",
            "cleaned_document.json": interim / "cleaned_document.json",
            "paper.json": processed / "paper.json",
            "chunks.jsonl": processed / "chunks.jsonl",
        }

    def validate_existing(self, paper_id: str, version: str) -> dict[str, str] | None:
        paths = self._artifact_paths(paper_id, version)
        _, processed = self.version_directories(paper_id, version)
        quality_path = processed / "quality_report.json"
        try:
            # 质量报告是复用入口，只有身份、门禁和全部产物哈希一致才算缓存命中。
            quality = QualityReport.model_validate_json(quality_path.read_text(encoding="utf-8"))
            if (
                quality.paper_id != paper_id
                or quality.processing_version != version
                or not quality.passed
            ):
                return None
            for name, expected in quality.artifact_hashes.items():
                path = paths.get(name)
                if (
                    path is None
                    or not path.is_file()
                    or sha256_bytes(path.read_bytes()) != expected
                ):
                    return None
            if set(quality.artifact_hashes) != set(paths):
                return None
        except (OSError, ValueError):
            return None
        return dict(quality.artifact_hashes)

    @staticmethod
    def _write_staged(directory: Path, files: dict[str, bytes]) -> None:
        directory.mkdir(parents=True, exist_ok=False)
        for name, data in files.items():
            path = directory / name
            path.write_bytes(data)
            if path.read_bytes() != data:
                raise OSError(f"staged artifact could not be verified: {name}")

    def publish(
        self,
        paper: Paper,
        parsed: ParsedDocument,
        cleaned: CleanedDocument,
        chunks: list[Chunk],
        quality: QualityReport,
    ) -> dict[str, str]:
        """先完整暂存两层目录，再以可回滚的目录替换发布。"""

        version = quality.processing_version
        interim_final, processed_final = self.version_directories(paper.paper_id, version)
        token = uuid.uuid4().hex
        interim_tmp = interim_final.parent / f".{version}.tmp-{token}"
        processed_tmp = processed_final.parent / f".{version}.tmp-{token}"
        core = {
            "parsed_document.json": stable_json_bytes(parsed),
            "cleaned_document.json": stable_json_bytes(cleaned),
            "paper.json": stable_json_bytes(paper),
            "chunks.jsonl": stable_jsonl_bytes(chunks),
        }
        hashes = {name: sha256_bytes(data) for name, data in core.items()}
        quality = quality.model_copy(update={"artifact_hashes": hashes})
        interim_files = {
            name: core[name] for name in ("parsed_document.json", "cleaned_document.json")
        }
        processed_files = {
            "paper.json": core["paper.json"],
            "chunks.jsonl": core["chunks.jsonl"],
            "quality_report.json": stable_json_bytes(quality),
        }
        backups: list[tuple[Path, Path]] = []
        published: list[Path] = []
        try:
            interim_final.parent.mkdir(parents=True, exist_ok=True)
            processed_final.parent.mkdir(parents=True, exist_ok=True)
            # 两层产物先在隐藏目录完整写入并回读，正式目录始终保持可读状态。
            self._write_staged(interim_tmp, interim_files)
            self._write_staged(processed_tmp, processed_files)
            if self.before_publish is not None:
                self.before_publish(processed_tmp)
            for final in (interim_final, processed_final):
                if final.exists():
                    backup = final.parent / f".{version}.bak-{token}-{len(backups)}"
                    os.replace(final, backup)
                    backups.append((final, backup))
            # 目录替换失败时，except 会删除本次发布并按相反顺序恢复旧版本。
            os.replace(interim_tmp, interim_final)
            published.append(interim_final)
            os.replace(processed_tmp, processed_final)
            published.append(processed_final)
        except Exception:
            for path in reversed(published):
                shutil.rmtree(path, ignore_errors=True)
            for final, backup in reversed(backups):
                if backup.exists():
                    os.replace(backup, final)
            raise
        finally:
            shutil.rmtree(interim_tmp, ignore_errors=True)
            shutil.rmtree(processed_tmp, ignore_errors=True)
            for _, backup in backups:
                shutil.rmtree(backup, ignore_errors=True)
        return hashes

    def read_quality(self, paper_id: str, version: str) -> QualityReport | None:
        _, processed = self.version_directories(paper_id, version)
        try:
            return QualityReport.model_validate_json(
                (processed / "quality_report.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return None

    def write_run_manifest(self, manifest: IngestionRunManifest) -> Path:
        destination = self.processed_root / "runs" / manifest.run_id / "manifest.json"
        destination.parent.mkdir(parents=True, exist_ok=False)
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_bytes(stable_json_bytes(manifest))
        os.replace(temporary, destination)
        return destination
