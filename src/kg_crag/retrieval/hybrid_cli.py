"""Hybrid 检索命令行的安全参数覆盖与运行时装配。"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from kg_crag.errors import KGCRAGError
from kg_crag.retrieval import (
    CrossEncoderReranker,
    HybridRetrievalConfig,
    HybridRetrievalService,
    SparseRetriever,
    load_hybrid_retrieval_config,
)
from kg_crag.retrieval.dense import DenseRetriever
from kg_crag.runtime import build_embedding_service, build_vector_store
from kg_crag.settings import Settings
from kg_crag.sparse_store import SQLiteSparseStore, load_sqlite_index_identity

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="执行一次零 LLM 的 Dense/Sparse Hybrid 检索")
    parser.add_argument("question", help="检索问题")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/retrieval.yaml")
    parser.add_argument("--sparse-index-version", help="已构建的精确 Sparse index_version")
    parser.add_argument("--method", choices=["rrf", "weighted"])
    parser.add_argument("--dense-weight", type=float)
    parser.add_argument("--sparse-weight", type=float)
    parser.add_argument("--filter", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--deduplicate-by-paper", action=argparse.BooleanOptionalAction)
    parser.add_argument("--reranker", action=argparse.BooleanOptionalAction)
    parser.add_argument("--allow-partial", action=argparse.BooleanOptionalAction)
    parser.add_argument("--output-root", help="工作区内的结果目录覆盖")
    parser.add_argument("--dry-run", action="store_true", help="只显示配置和目标，不访问后端")
    return parser.parse_args(argv)


def apply_overrides(
    config: HybridRetrievalConfig, args: argparse.Namespace
) -> HybridRetrievalConfig:
    payload = config.model_dump(mode="python")
    if args.method is not None:
        payload["fusion"]["method"] = args.method
    if args.dense_weight is not None:
        payload["fusion"]["dense_weight"] = args.dense_weight
    if args.sparse_weight is not None:
        payload["fusion"]["sparse_weight"] = args.sparse_weight
    if args.deduplicate_by_paper is not None:
        payload["hybrid"]["deduplicate_by_paper"] = args.deduplicate_by_paper
    if args.reranker is not None:
        payload["reranker"]["enabled"] = args.reranker
    if args.allow_partial is not None:
        payload["hybrid"]["allow_partial"] = args.allow_partial
    if args.output_root is not None:
        payload["hybrid"]["output_root"] = args.output_root
    return HybridRetrievalConfig.model_validate(payload)


def parse_filters(values: list[str]) -> dict[str, str | int | bool]:
    parsed: dict[str, str | int | bool] = {}
    for item in values:
        key, separator, value = item.partition("=")
        if not separator or not key or not value:
            raise ValueError("filters must use non-empty KEY=VALUE syntax")
        if key in parsed:
            raise ValueError("filter keys must be unique")
        parsed[key] = value
    return parsed


async def _run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = apply_overrides(load_hybrid_retrieval_config(args.config), args)
    filters = parse_filters(args.filter)
    unknown = set(filters) - set(config.sparse.allowed_filters)
    if unknown:
        raise ValueError(f"unsupported filter field: {sorted(unknown)[0]}")
    preview = {
        "question": args.question,
        "method": config.fusion.method,
        "weights": [config.fusion.dense_weight, config.fusion.sparse_weight],
        "reranker": config.reranker.enabled,
        "allow_partial": config.hybrid.allow_partial,
        "deduplicate_by_paper": config.hybrid.deduplicate_by_paper,
        "sparse_index_version": args.sparse_index_version,
        "filters": filters,
        "llm_calls": 0,
    }
    if args.dry_run:
        print(json.dumps(preview, ensure_ascii=False, sort_keys=True))
        return 0
    if not args.sparse_index_version:
        raise ValueError("actual Hybrid retrieval requires --sparse-index-version")

    db_path = PROJECT_ROOT / config.sparse.index_root / args.sparse_index_version / "index.sqlite3"
    identity = load_sqlite_index_identity(db_path)
    if identity.index_version != args.sparse_index_version:
        raise ValueError("Sparse manifest does not match the requested index version")
    sparse_store = SQLiteSparseStore(
        db_path,
        identity,
        max_top_k=config.sparse.max_top_k,
        max_candidates=config.sparse.max_candidates,
        max_query_chars=config.sparse.max_query_chars,
        max_query_tokens=config.sparse.max_query_tokens,
    )
    await sparse_store.ensure_index(identity)
    settings = Settings()
    vector_store = build_vector_store(config.dense, settings)
    embedding = build_embedding_service(config.dense, workspace_root=PROJECT_ROOT)
    reranker = (
        CrossEncoderReranker(config.reranker, workspace_root=PROJECT_ROOT)
        if config.reranker.enabled
        else None
    )
    service = HybridRetrievalService(
        DenseRetriever(embedding, vector_store, config.dense),
        SparseRetriever(sparse_store, config.sparse),
        config,
        collection_version=vector_store.collection_version,
        sparse_index_version=identity.index_version,
        corpus_snapshot_hash=identity.corpus_snapshot_hash,
        reranker=reranker,
        workspace_root=PROJECT_ROOT,
    )
    try:
        result = await service.retrieve(args.question, filters=filters, persist=True)
        print(result.model_dump_json())
        return 0
    finally:
        await vector_store.close()


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_run(argv))
    except (KGCRAGError, OSError, ValidationError, ValueError) as error:
        print(f"hybrid retrieval failed: {error}", file=sys.stderr)
        return 1
