"""Sparse 索引命令行的参数校验与薄编排。"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from kg_crag.errors import KGCRAGError
from kg_crag.indexing.processed import discover_processed, select_processed
from kg_crag.indexing.sparse import SparseIndexPipeline
from kg_crag.retrieval import load_hybrid_retrieval_config
from kg_crag.sparse_store import SQLiteSparseStore, build_sqlite_index_identity
from kg_crag.vector_store import corpus_snapshot_hash

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从已发布 Chunk 构建版本化 Sparse FTS5 索引")
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
    parser.add_argument(
        "--rebuild-index-version",
        help="显式重建时必须精确填写预览中的 64 位 index_version",
    )
    return parser.parse_args(argv)


def validate_cli_bounds(args: argparse.Namespace) -> None:
    """在读取语料和打开索引前拒绝无界或重复目标。"""

    if args.all and args.limit is None:
        raise ValueError("all mode requires an explicit limit")
    if args.paper_id and len(args.paper_id) != len(set(args.paper_id)):
        raise ValueError("paper IDs must be unique")


async def _run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_cli_bounds(args)
    config = load_hybrid_retrieval_config(args.config)
    papers = discover_processed(args.processed_root)
    selected = select_processed(
        papers,
        pilot_manifest=args.pilot_manifest if not args.paper_id and not args.all else None,
        paper_ids=args.paper_id,
        select_all=args.all,
        limit=args.limit,
        max_papers=config.dense.indexing.max_papers,
    )
    if not selected:
        raise ValueError("selection produced no processed papers")
    snapshot = corpus_snapshot_hash([chunk for paper in selected for chunk in paper.chunks])
    identity = build_sqlite_index_identity(
        schema_version=config.sparse.schema_version,
        tokenizer=config.sparse.tokenizer,
        bm25_version=config.sparse.bm25_version,
        corpus_snapshot_hash=snapshot,
    )
    rebuild = args.rebuild_index_version is not None
    if rebuild and args.rebuild_index_version != identity.index_version:
        raise ValueError("rebuild index version does not match the exact planned target")
    db_path = PROJECT_ROOT / config.sparse.index_root / identity.index_version / "index.sqlite3"
    store = SQLiteSparseStore(
        db_path,
        identity,
        max_top_k=config.sparse.max_top_k,
        max_candidates=config.sparse.max_candidates,
        max_query_chars=config.sparse.max_query_chars,
        max_query_tokens=config.sparse.max_query_tokens,
    )
    pipeline = SparseIndexPipeline(config, store, workspace_root=PROJECT_ROOT)
    manifest = await pipeline.run(selected, dry_run=args.dry_run, rebuild=rebuild)
    print(manifest.model_dump_json())
    return 2 if any(item.status.value == "failed" for item in manifest.items) else 0


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_run(argv))
    except (KGCRAGError, OSError, ValueError) as error:
        print(f"sparse indexing failed: {error}", file=sys.stderr)
        return 1
