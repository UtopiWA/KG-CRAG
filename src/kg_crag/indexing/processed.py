"""只读发现和校验已发布 Chunk 产物。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from kg_crag.models import Chunk, PilotManifest, QualityReport


@dataclass(frozen=True)
class ProcessedPaper:
    paper_id: str
    processing_version: str
    chunks: tuple[Chunk, ...]
    directory: Path


def _contained_file(root: Path, path: Path) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
        raise ValueError("processed artifact escapes the configured root or is not a file")
    return resolved


def discover_processed(processed_root: Path) -> list[ProcessedPaper]:
    root = processed_root.resolve(strict=True)
    discovered: dict[str, list[ProcessedPaper]] = {}
    for quality_path in sorted(root.glob("paper-*/*/quality_report.json")):
        quality_file = _contained_file(root, quality_path)
        quality = QualityReport.model_validate_json(quality_file.read_text(encoding="utf-8"))
        if not quality.passed:
            continue
        chunks_path = _contained_file(root, quality_file.parent / "chunks.jsonl")
        expected = quality.artifact_hashes.get("chunks.jsonl")
        actual = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
        if expected != actual:
            raise ValueError(f"processed chunks hash mismatch for {quality.paper_id}")
        chunks: list[Chunk] = []
        for line_number, line in enumerate(chunks_path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                chunk = Chunk.model_validate(json.loads(line))
            except (ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid processed chunk at line {line_number} for {quality.paper_id}"
                ) from error
            if (
                chunk.paper_id != quality.paper_id
                or chunk.processing_version != quality.processing_version
            ):
                raise ValueError(f"processed chunk identity mismatch for {quality.paper_id}")
            chunks.append(chunk)
        if not chunks:
            raise ValueError(f"processed paper contains no chunks: {quality.paper_id}")
        chunks.sort(key=lambda item: (item.ordinal, item.chunk_id))
        if len({item.chunk_id for item in chunks}) != len(chunks):
            raise ValueError(f"duplicate chunk ID for {quality.paper_id}")
        discovered.setdefault(quality.paper_id, []).append(
            ProcessedPaper(
                paper_id=quality.paper_id,
                processing_version=quality.processing_version,
                chunks=tuple(chunks),
                directory=quality_file.parent,
            )
        )
    selected: list[ProcessedPaper] = []
    for _paper_id, versions in discovered.items():
        versions.sort(key=lambda item: item.processing_version)
        selected.append(versions[-1])
    return sorted(selected, key=lambda item: item.paper_id)


def select_processed(
    papers: list[ProcessedPaper],
    *,
    pilot_manifest: Path | None = None,
    paper_ids: list[str] | None = None,
    select_all: bool = False,
    limit: int | None = None,
    max_papers: int,
) -> list[ProcessedPaper]:
    modes = int(pilot_manifest is not None) + int(bool(paper_ids)) + int(select_all)
    if modes != 1:
        raise ValueError("choose exactly one of pilot, explicit paper IDs, or all")
    by_id = {paper.paper_id: paper for paper in papers}
    if pilot_manifest is not None:
        manifest = PilotManifest.model_validate_json(pilot_manifest.read_text(encoding="utf-8"))
        requested = [item.paper_id for item in manifest.papers]
    elif paper_ids:
        if len(paper_ids) != len(set(paper_ids)):
            raise ValueError("paper IDs must be unique")
        requested = list(paper_ids)
    else:
        if limit is None:
            raise ValueError("all mode requires an explicit limit")
        requested = sorted(by_id)[:limit]
    if limit is not None and (limit <= 0 or limit > max_papers):
        raise ValueError("limit is outside the configured indexing boundary")
    if len(requested) > max_papers:
        raise ValueError("selected paper count exceeds the configured indexing boundary")
    missing = [paper_id for paper_id in requested if paper_id not in by_id]
    if missing:
        raise ValueError(f"processed papers are unavailable: {missing}")
    return [by_id[paper_id] for paper_id in requested[:limit]]
