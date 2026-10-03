"""离线 Graph Retriever 查询入口。"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from kg_crag.graph.config import load_graph_config
from kg_crag.graph.metadata import plan_metadata_graph
from kg_crag.graph_store import InMemoryGraphStore
from kg_crag.indexing.processed import discover_processed, select_processed
from kg_crag.models import GraphQueryRequest, GraphQueryTemplate
from kg_crag.retrieval.graph import GraphRetriever, ProcessedSourceResolver

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="查询版本化基础图并输出统一 Evidence")
    value.add_argument("query", help="实体规范名")
    value.add_argument(
        "--template",
        choices=[item.value for item in GraphQueryTemplate],
        default="entity_neighbors",
    )
    value.add_argument("--max-hops", type=int, default=1)
    value.add_argument("--top-k", type=int, default=10)
    value.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/retrieval.yaml")
    value.add_argument("--processed-root", type=Path, default=PROJECT_ROOT / "data/processed")
    value.add_argument(
        "--pilot-manifest",
        type=Path,
        default=PROJECT_ROOT / "configs/pilot_corpus.json",
    )
    return value


async def main_async(args: argparse.Namespace) -> int:
    config = load_graph_config(args.config)
    papers = select_processed(
        discover_processed(args.processed_root),
        pilot_manifest=args.pilot_manifest,
        max_papers=config.selection.max_papers,
    )
    plan = plan_metadata_graph(papers, config, model="offline-metadata")
    store = InMemoryGraphStore()
    await store.ensure_graph(plan.identity)
    for bundle in plan.bundles:
        await store.sync_paper(bundle)
    request = GraphQueryRequest(
        graph_version=plan.identity.graph_version,
        template=GraphQueryTemplate(args.template),
        entity_names=[args.query],
        max_hops=args.max_hops,
        top_k=args.top_k,
        max_candidates=config.query.max_candidates,
        timeout_seconds=config.query.timeout_seconds,
    )
    result = await GraphRetriever(store, ProcessedSourceResolver(papers)).retrieve(request)
    payload = {
        "evidence": [item.model_dump(mode="json") for item in result.evidence],
        "diagnostics": {
            "paths": len(result.paths),
            "rejected_paths": result.rejected_paths,
            "database_calls": result.database_calls,
        },
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def main() -> int:
    return asyncio.run(main_async(parser().parse_args()))
