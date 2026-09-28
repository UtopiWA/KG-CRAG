"""Dense 索引命令行的参数校验与薄编排。"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from kg_crag.errors import KGCRAGError
from kg_crag.indexing.pipeline import DenseIndexPipeline
from kg_crag.indexing.processed import discover_processed, select_processed
from kg_crag.retrieval import load_dense_rag_config
from kg_crag.runtime import build_embedding_service, build_vector_store
from kg_crag.settings import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从已发布 Chunk 构建版本化 Dense 索引")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--pilot", action="store_true", help="索引固定 pilot (默认)")
    selection.add_argument("--paper-id", action="append", help="索引指定 Paper ID，可重复")
    selection.add_argument("--all", action="store_true", help="在显式上限内索引全部论文")
    parser.add_argument("--limit", type=int, help="本次最多处理的论文数")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/retrieval.yaml")
    parser.add_argument("--processed-root", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument(
        "--pilot-manifest",
        type=Path,
        default=PROJECT_ROOT / "configs/pilot_corpus.json",
    )
    parser.add_argument("--dry-run", action="store_true", help="只读取并输出差异计划")
    parser.add_argument("--rebuild", action="store_true", help="显式重建版本化集合")
    return parser.parse_args(argv)


async def _run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_dense_rag_config(args.config)
    papers = discover_processed(args.processed_root)
    selected = select_processed(
        papers,
        pilot_manifest=args.pilot_manifest if not args.paper_id and not args.all else None,
        paper_ids=args.paper_id,
        select_all=args.all,
        limit=args.limit,
        max_papers=config.indexing.max_papers,
    )
    if not selected:
        raise ValueError("selection produced no processed papers")
    settings = Settings()
    store = build_vector_store(config, settings)
    try:
        pipeline = DenseIndexPipeline(
            config,
            build_embedding_service(config, workspace_root=PROJECT_ROOT),
            store,
            workspace_root=PROJECT_ROOT,
        )
        manifest = await pipeline.run(selected, dry_run=args.dry_run, rebuild=args.rebuild)
        print(manifest.model_dump_json())
        return 2 if any(item.status.value == "failed" for item in manifest.items) else 0
    finally:
        await store.close()


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_run(argv))
    except (KGCRAGError, OSError, ValueError) as error:
        print(f"dense indexing failed: {error}", file=sys.stderr)
        return 1
