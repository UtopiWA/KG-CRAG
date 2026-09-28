"""单次 Dense RAG 查询命令行。"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from kg_crag.errors import KGCRAGError
from kg_crag.generation import load_prompt
from kg_crag.retrieval import dense_config_hash, load_dense_rag_config
from kg_crag.runtime import (
    build_dense_rag_service,
    build_embedding_service,
    build_vector_store,
)
from kg_crag.settings import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="执行一次可追溯的 Dense RAG 查询")
    parser.add_argument("question", help="要查询的问题")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--filter", action="append", default=[], help="白名单过滤 key=value")
    parser.add_argument("--deduplicate-by-paper", action="store_true")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/retrieval.yaml")
    parser.add_argument("--dry-run", action="store_true", help="只校验参数、配置和 Prompt")
    parser.add_argument("--no-persist", action="store_true", help="不保存本次结果")
    return parser.parse_args(argv)


def _filters(values: list[str]) -> dict[str, str | int | bool]:
    parsed: dict[str, str | int | bool] = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key or not item or key in parsed:
            raise ValueError("--filter must use unique non-empty key=value pairs")
        parsed[key] = item
    return parsed


async def _run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_dense_rag_config(args.config)
    prompt = load_prompt(PROJECT_ROOT / config.generation.prompt_path)
    top_k = args.top_k or config.dense.top_k
    filters = _filters(args.filter)
    if top_k <= 0 or top_k > config.dense.max_top_k:
        raise ValueError("--top-k is outside the configured boundary")
    unknown = set(filters) - set(config.collection.allowed_filters)
    if unknown:
        raise ValueError(f"unsupported filter: {sorted(unknown)[0]}")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "config_hash": dense_config_hash(config),
                    "prompt_version": prompt.version,
                    "top_k": top_k,
                    "filters": filters,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    settings = Settings()
    store = build_vector_store(config, settings)
    try:
        await store.ensure_collection()
        embedding = build_embedding_service(config, workspace_root=PROJECT_ROOT)
        service = build_dense_rag_service(
            config,
            settings,
            workspace_root=PROJECT_ROOT,
            store=store,
            embedding=embedding,
        )
        result = await service.ask(
            args.question,
            top_k=top_k,
            filters=filters,
            deduplicate_by_paper=args.deduplicate_by_paper,
            persist=not args.no_persist,
        )
        print(result.model_dump_json())
        return 0
    finally:
        await store.close()


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_run(argv))
    except (KGCRAGError, OSError, ValueError) as error:
        print(f"dense query failed: {error}", file=sys.stderr)
        return 1
