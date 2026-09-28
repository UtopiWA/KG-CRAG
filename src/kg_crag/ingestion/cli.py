"""本地论文语料摄取命令行的参数校验与薄编排。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal

from kg_crag.errors import KGCRAGError
from kg_crag.ingestion.config import load_ingestion_config
from kg_crag.ingestion.pdf import PyMuPDFParser
from kg_crag.ingestion.pilot import (
    load_pilot_manifest,
    load_pilot_review,
    manifest_hash,
    resolve_pilot_inputs,
    validate_pilot_gate,
)
from kg_crag.ingestion.pipeline import IngestionPipeline
from kg_crag.ingestion.raw import RawPaperInput, discover_raw_inputs
from kg_crag.ingestion.storage import ArtifactStore
from kg_crag.models import IngestionStatus

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把本地 raw 论文转换为版本化摄取产物")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--pilot", action="store_true", help="处理固定 15 篇试点 (默认)")
    selection.add_argument("--paper-id", action="append", help="处理指定 Paper ID，可重复")
    selection.add_argument("--all", action="store_true", help="门禁通过后处理完整语料")
    parser.add_argument("--limit", type=int, help="本次最多处理的论文数")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/default.yaml")
    parser.add_argument("--raw-root", type=Path, help="覆盖 raw 根目录")
    parser.add_argument("--interim-root", type=Path, help="覆盖 interim 输出根目录")
    parser.add_argument("--processed-root", type=Path, help="覆盖 processed 输出根目录")
    parser.add_argument("--dry-run", action="store_true", help="仅校验并打印选择，不创建产物")
    parser.add_argument("--force", action="store_true", help="重建已存在且哈希有效的处理版本")
    return parser.parse_args(argv)


def _workspace_path(value: str) -> Path:
    return PROJECT_ROOT / value


def _select_by_ids(inputs: list[RawPaperInput], ids: list[str]) -> list[RawPaperInput]:
    by_id = {item.paper_id: item for item in inputs}
    unknown = [paper_id for paper_id in ids if paper_id not in by_id]
    if unknown:
        raise ValueError(f"unknown paper IDs: {', '.join(unknown)}")
    if len(ids) != len(set(ids)):
        raise ValueError("paper IDs must be unique")
    return [by_id[paper_id] for paper_id in ids]


def _validate_full_corpus_gate(
    pipeline: IngestionPipeline,
    store: ArtifactStore,
    pilot_inputs: list[RawPaperInput],
    manifest_path: Path,
    review_path: Path,
) -> str:
    manifest = load_pilot_manifest(manifest_path)
    review = load_pilot_review(review_path)
    versions = {raw.paper_id: pipeline.version_for(raw) for raw in pilot_inputs}
    validate_pilot_gate(review, manifest, versions)
    for raw in pilot_inputs:
        version = versions[raw.paper_id]
        report = store.read_quality(raw.paper_id, version)
        if (
            report is None
            or not report.passed
            or store.validate_existing(raw.paper_id, version) is None
        ):
            raise ValueError(f"pilot automatic quality artifacts are not current: {raw.paper_id}")
    return manifest_hash(manifest)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_ingestion_config(args.config)
        if args.limit is not None and not 1 <= args.limit <= config.limits.max_papers:
            raise ValueError(f"--limit must be between 1 and {config.limits.max_papers}")
        raw_root = args.raw_root or _workspace_path(config.paths.raw_root)
        inputs = discover_raw_inputs(
            raw_root,
            max_pdf_bytes=config.limits.max_pdf_bytes,
            max_papers=config.limits.max_papers,
        )
        manifest_path = _workspace_path(config.paths.pilot_manifest)
        review_path = _workspace_path(config.paths.pilot_review)
        pilot_manifest = load_pilot_manifest(manifest_path)
        interim_root = args.interim_root or _workspace_path(config.paths.interim_root)
        processed_root = args.processed_root or _workspace_path(config.paths.processed_root)
        store = ArtifactStore(interim_root, processed_root)
        pipeline = IngestionPipeline(config, PyMuPDFParser(config.parsing), store)
        pilot_inputs: list[RawPaperInput] | None = None
        pilot_manifest_hash: str | None = None
        mode: Literal["pilot", "selected", "all", "dry-run"]
        if args.paper_id:
            selected = _select_by_ids(inputs, args.paper_id)
            mode = "selected"
        elif args.all:
            pilot_inputs = resolve_pilot_inputs(pilot_manifest, inputs)
            pilot_manifest_hash = _validate_full_corpus_gate(
                pipeline,
                store,
                pilot_inputs,
                manifest_path,
                review_path,
            )
            selected = inputs
            mode = "all"
        else:
            pilot_inputs = resolve_pilot_inputs(pilot_manifest, inputs)
            selected = pilot_inputs
            pilot_manifest_hash = manifest_hash(pilot_manifest)
            mode = "pilot"
        if args.limit is not None:
            selected = selected[: args.limit]
        if not selected:
            raise ValueError("selection produced no papers")
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "count": len(selected),
                        "paper_ids": [item.paper_id for item in selected],
                        "config_hash": pipeline.configuration_hash,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        result = pipeline.run_batch(
            selected,
            mode=mode,
            force=args.force,
            pilot_manifest_hash=pilot_manifest_hash,
        )
        counts = {
            status.value: sum(item.status is status for item in result.items)
            for status in IngestionStatus
        }
        print(json.dumps({"run_id": result.run_id, "counts": counts}, sort_keys=True))
        return 2 if counts[IngestionStatus.FAILED.value] else 0
    except (KGCRAGError, OSError, ValueError) as error:
        print(f"ingestion validation failed: {error}", file=sys.stderr)
        return 1
