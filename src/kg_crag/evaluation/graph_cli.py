"""固定开发集的零 LLM Graph Retriever 评测入口。"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from kg_crag.evaluation.graph import GraphEvaluator
from kg_crag.graph.config import load_graph_config
from kg_crag.graph.metadata import plan_metadata_graph
from kg_crag.graph_store import InMemoryGraphStore
from kg_crag.indexing.processed import discover_processed, select_processed
from kg_crag.models import GraphEvaluationQuestionSet
from kg_crag.retrieval.graph import GraphRetriever, ProcessedSourceResolver

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="运行零在线调用的图检索开发集评测")
    value.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/retrieval.yaml")
    value.add_argument("--processed-root", type=Path, default=PROJECT_ROOT / "data/processed")
    value.add_argument(
        "--pilot-manifest",
        type=Path,
        default=PROJECT_ROOT / "configs/pilot_corpus.json",
    )
    value.add_argument("--questions", type=Path)
    value.add_argument("--results-root", type=Path)
    return value


async def main_async(args: argparse.Namespace) -> int:
    config = load_graph_config(args.config)
    papers = select_processed(
        discover_processed(args.processed_root),
        pilot_manifest=args.pilot_manifest,
        max_papers=config.selection.max_papers,
    )
    plan = plan_metadata_graph(papers, config, model="offline-metadata")
    questions_path = args.questions or PROJECT_ROOT / config.paths.evaluation_questions
    questions = GraphEvaluationQuestionSet.model_validate_json(
        questions_path.read_text(encoding="utf-8")
    )
    if (
        questions.questions[0].corpus_snapshot != plan.identity.corpus_snapshot
        or questions.questions[0].graph_version != plan.identity.graph_version
    ):
        raise ValueError("graph evaluation set identity has drifted")
    store = InMemoryGraphStore()
    await store.ensure_graph(plan.identity)
    for bundle in plan.bundles:
        await store.sync_paper(bundle)
    results_root = args.results_root or PROJECT_ROOT / config.paths.evaluation_results_root
    report = await GraphEvaluator(
        GraphRetriever(store, ProcessedSourceResolver(papers)), results_root=results_root
    ).evaluate(questions)
    print(json.dumps(report["metrics"], ensure_ascii=False, sort_keys=True))
    return 0


def main() -> int:
    return asyncio.run(main_async(parser().parse_args()))
