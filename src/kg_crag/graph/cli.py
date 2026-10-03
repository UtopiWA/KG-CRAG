"""知识图谱基础构建和事实抽取 CLI 的参数装配。"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from kg_crag.graph.config import load_graph_config
from kg_crag.graph.extraction import (
    GraphExtractionProvider,
    LLMGraphExtractionProvider,
    MockGraphExtractionProvider,
)
from kg_crag.graph.metadata import make_graph_identity, plan_metadata_graph
from kg_crag.graph.pipeline import GraphExtractionPipeline, MetadataGraphPipeline
from kg_crag.graph_store import GraphStore, InMemoryGraphStore, Neo4jGraphStore
from kg_crag.indexing.processed import ProcessedPaper, discover_processed, select_processed
from kg_crag.providers import OpenAICompatibleLLMProvider
from kg_crag.settings import get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _manifest_has_failures(manifest: dict[str, object]) -> bool:
    """只检查结构有效的条目，避免 CLI 因异常清单形状再次崩溃。"""

    for key in ("items", "bundles"):
        values = manifest.get(key)
        if isinstance(values, list) and any(
            isinstance(item, dict) and item.get("status") == "failed" for item in values
        ):
            return True
    return False


def _selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--processed-root", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument(
        "--pilot-manifest",
        type=Path,
        default=PROJECT_ROOT / "configs/pilot_corpus.json",
    )
    parser.add_argument("--paper-id", action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int)


def _select(args: argparse.Namespace, max_papers: int) -> list[ProcessedPaper]:
    papers = discover_processed(args.processed_root)
    return select_processed(
        papers,
        pilot_manifest=args.pilot_manifest if not args.paper_id and not args.all else None,
        paper_ids=args.paper_id,
        select_all=args.all,
        limit=args.limit,
        max_papers=max_papers,
    )


def metadata_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="构建无 LLM 的可溯源基础图")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/retrieval.yaml")
    parser.add_argument("--backend", choices=("memory", "neo4j"), default="memory")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    _selection_arguments(parser)
    return parser


async def metadata_main_async(args: argparse.Namespace) -> int:
    config = load_graph_config(args.config)
    settings = get_settings()
    papers = _select(args, config.selection.max_papers)
    # 基础图不依赖 LLM，固定模型标识可避免仅因 .env 切换模型而改变图身份。
    plan = plan_metadata_graph(papers, config, model="offline-metadata")
    if args.dry_run or args.backend == "memory":
        store: GraphStore = InMemoryGraphStore()
    else:
        store = Neo4jGraphStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    try:
        manifest = await MetadataGraphPipeline(config, store, workspace_root=PROJECT_ROOT).run(
            plan, dry_run=args.dry_run, rebuild=args.rebuild
        )
        print(__import__("json").dumps(manifest, ensure_ascii=False, sort_keys=True))
        return 2 if _manifest_has_failures(manifest) else 0
    finally:
        await store.close()


def extraction_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="有预算地抽取 pilot 正文图事实")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/retrieval.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--online", action="store_true", help="显式允许真实 LLM 调用")
    parser.add_argument("--confirm-budget", action="store_true")
    parser.add_argument("--backend", choices=("memory", "neo4j"), default="memory")
    parser.add_argument("--max-chunks", type=int, choices=range(1, 6))
    _selection_arguments(parser)
    return parser


async def extraction_main_async(args: argparse.Namespace) -> int:
    config = load_graph_config(args.config)
    if args.max_chunks is not None:
        config = config.model_copy(
            update={
                "selection": config.selection.model_copy(
                    update={
                        "min_chunks_per_paper": min(
                            config.selection.min_chunks_per_paper, args.max_chunks
                        ),
                        "max_chunks_per_paper": args.max_chunks,
                    }
                )
            }
        )
    settings = get_settings()
    papers = _select(args, config.selection.max_papers)
    model = settings.llm_model if args.online else "mock-graph-v1"
    identity = make_graph_identity(papers, config, model=model)
    if args.online:
        if not args.confirm_budget:
            raise ValueError("online graph extraction requires --confirm-budget")
        config = config.model_copy(
            update={"extraction": config.extraction.model_copy(update={"enabled": True})}
        )
        llm = OpenAICompatibleLLMProvider(
            settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout_seconds=settings.llm_timeout_seconds,
        )
        provider: GraphExtractionProvider = LLMGraphExtractionProvider(
            llm, max_output_chars=config.extraction.max_output_chars
        )
    else:
        provider = MockGraphExtractionProvider()
    store: GraphStore = (
        Neo4jGraphStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
        if args.online and not args.dry_run and args.backend == "neo4j"
        else InMemoryGraphStore()
    )
    try:
        manifest = await GraphExtractionPipeline(
            config,
            provider,
            workspace_root=PROJECT_ROOT,
            model=model,
            store=store,
        ).run(papers, identity, dry_run=args.dry_run)
        print(__import__("json").dumps(manifest, ensure_ascii=False, sort_keys=True))
        return 2 if _manifest_has_failures(manifest) else 0
    finally:
        await store.close()


def metadata_main() -> int:
    return asyncio.run(metadata_main_async(metadata_parser().parse_args()))


def extraction_main() -> int:
    return asyncio.run(extraction_main_async(extraction_parser().parse_args()))
