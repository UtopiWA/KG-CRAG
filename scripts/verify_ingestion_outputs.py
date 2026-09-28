#!/usr/bin/env python3
"""重读并验证 pilot 或完整语料的版本化摄取产物。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kg_crag.ingestion.config import load_ingestion_config
from kg_crag.ingestion.pdf import PyMuPDFParser
from kg_crag.ingestion.pilot import load_pilot_manifest, resolve_pilot_inputs
from kg_crag.ingestion.pipeline import IngestionPipeline
from kg_crag.ingestion.raw import discover_raw_inputs
from kg_crag.ingestion.storage import ArtifactStore
from kg_crag.models import Chunk, CleanedDocument, Paper, ParsedDocument

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--pilot", action="store_true", help="验证固定 pilot (默认)")
    selection.add_argument("--all", action="store_true", help="验证完整 raw 语料")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/default.yaml")
    parser.add_argument("--interim-root", type=Path)
    parser.add_argument("--processed-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_ingestion_config(args.config)
    raw = discover_raw_inputs(
        PROJECT_ROOT / config.paths.raw_root,
        max_pdf_bytes=config.limits.max_pdf_bytes,
        max_papers=config.limits.max_papers,
    )
    if args.all:
        selected = raw
    else:
        selected = resolve_pilot_inputs(
            load_pilot_manifest(PROJECT_ROOT / config.paths.pilot_manifest), raw
        )
    store = ArtifactStore(
        args.interim_root or PROJECT_ROOT / config.paths.interim_root,
        args.processed_root or PROJECT_ROOT / config.paths.processed_root,
    )
    pipeline = IngestionPipeline(config, PyMuPDFParser(config.parsing), store)
    failures: list[str] = []
    totals = {"papers": 0, "pages": 0, "blocks": 0, "chunks": 0, "warnings": 0}
    for item in selected:
        version = pipeline.version_for(item)
        hashes = store.validate_existing(item.paper_id, version)
        if hashes is None:
            failures.append(f"{item.paper_id}: missing or invalid artifact hash")
            continue
        interim, processed = store.version_directories(item.paper_id, version)
        try:
            paper = Paper.model_validate_json(
                (processed / "paper.json").read_text(encoding="utf-8")
            )
            parsed = ParsedDocument.model_validate_json(
                (interim / "parsed_document.json").read_text(encoding="utf-8")
            )
            cleaned = CleanedDocument.model_validate_json(
                (interim / "cleaned_document.json").read_text(encoding="utf-8")
            )
            chunks = [
                Chunk.model_validate_json(line)
                for line in (processed / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            quality = store.read_quality(item.paper_id, version)
            if quality is None or not quality.passed or not chunks:
                raise ValueError("quality report or chunks are incomplete")
            if not (
                paper.paper_id == parsed.paper_id == cleaned.paper_id == item.paper_id
                and all(chunk.paper_id == item.paper_id for chunk in chunks)
            ):
                raise ValueError("paper identity is inconsistent across artifacts")
            if any(
                chunk.page_start is None
                or chunk.page_end is None
                or chunk.page_start > chunk.page_end
                or chunk.page_end > parsed.page_count
                for chunk in chunks
            ):
                raise ValueError("chunk page range is invalid")
            totals["papers"] += 1
            totals["pages"] += parsed.page_count
            totals["blocks"] += quality.block_count
            totals["chunks"] += len(chunks)
            totals["warnings"] += sum(quality.warning_counts.values())
        except (OSError, ValueError) as error:
            failures.append(f"{item.paper_id}: {error}")
    print(json.dumps({"totals": totals, "failures": failures}, ensure_ascii=False, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
