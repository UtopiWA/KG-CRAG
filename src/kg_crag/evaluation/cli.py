"""Dense RAG pilot 基线评测命令行。"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from kg_crag.errors import KGCRAGError
from kg_crag.evaluation import DenseEvaluationRunner, load_question_set
from kg_crag.indexing import discover_processed, select_processed
from kg_crag.models import DenseRAGResult
from kg_crag.retrieval import (
    dense_config_hash,
    load_dense_evaluation_config,
    load_dense_rag_config,
)
from kg_crag.runtime import (
    build_dense_rag_service,
    build_embedding_service,
    build_vector_store,
)
from kg_crag.settings import Settings
from kg_crag.vector_store import corpus_snapshot_hash

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行固定 pilot 的 Dense RAG 基线评测")
    parser.add_argument(
        "--retrieval-config",
        type=Path,
        default=PROJECT_ROOT / "configs/retrieval.yaml",
    )
    parser.add_argument(
        "--evaluation-config",
        type=Path,
        default=PROJECT_ROOT / "configs/evaluation.yaml",
    )
    parser.add_argument("--processed-root", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument(
        "--pilot-manifest",
        type=Path,
        default=PROJECT_ROOT / "configs/pilot_corpus.json",
    )
    parser.add_argument("--limit", "--max-questions", dest="limit", type=int)
    parser.add_argument("--questions", type=Path, help="覆盖冻结问题清单路径")
    parser.add_argument("--dry-run", action="store_true", help="只校验语料快照与问题集")
    return parser.parse_args(argv)


async def _run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rag_config = load_dense_rag_config(args.retrieval_config)
    eval_config = load_dense_evaluation_config(args.evaluation_config)
    selected = select_processed(
        discover_processed(args.processed_root),
        pilot_manifest=args.pilot_manifest,
        max_papers=rag_config.indexing.max_papers,
    )
    chunks = [chunk for paper in selected for chunk in paper.chunks]
    snapshot = corpus_snapshot_hash(chunks)
    questions = load_question_set(
        args.questions or PROJECT_ROOT / eval_config.questions_path,
        available_chunk_ids={chunk.chunk_id for chunk in chunks},
        corpus_snapshot_hash=snapshot,
    )
    limit = args.limit or min(len(questions.questions), eval_config.max_questions)
    if limit < 20 or limit > eval_config.max_questions:
        raise ValueError("--limit must be between 20 and the configured maximum")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "corpus_snapshot_hash": snapshot,
                    "paper_count": len(selected),
                    "question_count": limit,
                },
                sort_keys=True,
            )
        )
        return 0
    settings = Settings()
    store = build_vector_store(rag_config, settings)
    try:
        await store.ensure_collection()
        embedding = build_embedding_service(rag_config, workspace_root=PROJECT_ROOT)
        service = build_dense_rag_service(
            rag_config,
            settings,
            workspace_root=PROJECT_ROOT,
            store=store,
            embedding=embedding,
        )

        async def query(question: str) -> DenseRAGResult:
            return await service.ask(
                question,
                top_k=max(eval_config.k_values),
                persist=False,
            )

        runner = DenseEvaluationRunner(
            query=query,
            config_hash=dense_config_hash(rag_config),
            prompt_version=service.prompt.version,
            collection_version=store.collection_version,
            k_values=eval_config.k_values,
            results_root=PROJECT_ROOT / eval_config.results_root,
        )
        report = await runner.run(questions, limit=limit)
        print(report.model_dump_json())
        return 2 if report.metrics.failure_count else 0
    finally:
        await store.close()


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_run(argv))
    except (KGCRAGError, OSError, ValueError) as error:
        print(f"dense evaluation failed: {error}", file=sys.stderr)
        return 1
