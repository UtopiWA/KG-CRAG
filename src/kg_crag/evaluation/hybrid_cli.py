"""Hybrid 检索矩阵与唯一入选配置回答验证命令行。"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

from kg_crag.errors import KGCRAGError
from kg_crag.evaluation.dense import load_question_set
from kg_crag.evaluation.hybrid import HybridEvaluationRunner, HybridStrategyOutput
from kg_crag.generation import HybridQueryService, load_prompt
from kg_crag.indexing import discover_processed, select_processed
from kg_crag.ingestion.storage import stable_json_bytes
from kg_crag.models import HybridQueryResult, HybridStrategy
from kg_crag.providers import LLMProvider, MockLLMProvider, OpenAICompatibleLLMProvider
from kg_crag.retrieval import (
    CrossEncoderReranker,
    HybridRetrievalConfig,
    HybridRetrievalService,
    SparseRetriever,
    dense_config_hash,
    fusion_version,
    load_hybrid_evaluation_config,
    load_hybrid_retrieval_config,
)
from kg_crag.retrieval.dense import DenseRetriever
from kg_crag.runtime import build_embedding_service, build_vector_store
from kg_crag.settings import Settings
from kg_crag.sparse_store import SQLiteSparseStore, load_sqlite_index_identity
from kg_crag.vector_store import corpus_snapshot_hash

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行零 LLM Hybrid 检索开发集矩阵")
    parser.add_argument(
        "--retrieval-config", type=Path, default=PROJECT_ROOT / "configs/retrieval.yaml"
    )
    parser.add_argument(
        "--evaluation-config", type=Path, default=PROJECT_ROOT / "configs/evaluation.yaml"
    )
    parser.add_argument("--processed-root", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument(
        "--pilot-manifest", type=Path, default=PROJECT_ROOT / "configs/pilot_corpus.json"
    )
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--sparse-index-version")
    parser.add_argument(
        "--strategy",
        action="append",
        choices=["dense", "sparse", "rrf", "weighted", "fusion_rerank"],
    )
    parser.add_argument("--smoke", action="store_true", help="只运行配置规定的 5 题 smoke")
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--with-answer", action="store_true")
    parser.add_argument("--selected-strategy", choices=["fusion_rerank"])
    parser.add_argument("--answer-budget", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def validate_cli_args(args: argparse.Namespace, *, max_questions: int, smoke: int) -> int:
    """在创建模型、数据库或网络客户端之前固定运行边界。"""

    limit = smoke if args.smoke else (args.max_questions or max_questions)
    if limit <= 0 or limit > max_questions:
        raise ValueError("question limit is outside the configured boundary")
    if args.with_answer:
        if args.selected_strategy != "fusion_rerank":
            raise ValueError("answer validation requires the selected fusion_rerank strategy")
        if args.answer_budget != limit:
            raise ValueError("--answer-budget must exactly match the bounded question count")
    elif args.selected_strategy is not None or args.answer_budget != 0:
        raise ValueError("answer selection and budget require --with-answer")
    return limit


async def _run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_hybrid_retrieval_config(args.retrieval_config)
    evaluation = load_hybrid_evaluation_config(args.evaluation_config)
    limit = validate_cli_args(
        args,
        max_questions=evaluation.max_questions,
        smoke=evaluation.smoke_questions,
    )
    strategies: list[HybridStrategy] = args.strategy or list(evaluation.strategies)
    if len(strategies) != len(set(strategies)):
        raise ValueError("evaluation strategies must be unique")
    selected = select_processed(
        discover_processed(args.processed_root),
        pilot_manifest=args.pilot_manifest,
        max_papers=config.dense.indexing.max_papers,
    )
    chunks = [chunk for paper in selected for chunk in paper.chunks]
    snapshot = corpus_snapshot_hash(chunks)
    questions = load_question_set(
        args.questions or PROJECT_ROOT / evaluation.questions_path,
        available_chunk_ids={chunk.chunk_id for chunk in chunks},
        corpus_snapshot_hash=snapshot,
    )
    limit = min(limit, len(questions.questions))
    preview = {
        "corpus_snapshot_hash": snapshot,
        "paper_count": len(selected),
        "question_count": limit,
        "strategies": strategies,
        "with_answer": args.with_answer,
        "answer_budget": args.answer_budget,
        "default_llm_calls": 0 if not args.with_answer else limit,
    }
    if args.dry_run:
        print(json.dumps(preview, ensure_ascii=False, sort_keys=True))
        return 0
    if not args.sparse_index_version:
        raise ValueError("actual Hybrid evaluation requires --sparse-index-version")
    return await _execute(args, config, evaluation, questions, strategies, limit, snapshot)


async def _execute(
    args: argparse.Namespace,
    config: HybridRetrievalConfig,
    evaluation: object,
    questions: object,
    strategies: list[HybridStrategy],
    limit: int,
    snapshot: str,
) -> int:
    from kg_crag.models import PilotQuestionSet
    from kg_crag.retrieval import HybridEvaluationConfig

    eval_config = HybridEvaluationConfig.model_validate(evaluation)
    question_set = PilotQuestionSet.model_validate(questions)
    db_path = PROJECT_ROOT / config.sparse.index_root / args.sparse_index_version / "index.sqlite3"
    identity = load_sqlite_index_identity(db_path)
    if identity.index_version != args.sparse_index_version:
        raise ValueError("Sparse manifest does not match the requested index version")
    if identity.corpus_snapshot_hash != snapshot:
        raise ValueError("Sparse index and evaluation corpus snapshots do not match")
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
    try:
        await vector_store.ensure_collection()
        embedding = build_embedding_service(
            config.dense,
            workspace_root=PROJECT_ROOT,
            settings=settings,
        )
        dense = DenseRetriever(embedding, vector_store, config.dense)
        sparse = SparseRetriever(sparse_store, config.sparse)
        runners: dict[HybridStrategy, object] = {}
        services: dict[HybridStrategy, HybridRetrievalService] = {}
        fusion_versions: dict[HybridStrategy, str] = {}

        async def dense_run(question: str) -> HybridStrategyOutput:
            started = time.perf_counter()
            evidence = await dense.retrieve(question, top_k=max(eval_config.k_values))
            return HybridStrategyOutput(
                tuple(evidence),
                {"dense": (time.perf_counter() - started) * 1000},
                {"dense": 1},
            )

        async def sparse_run(question: str) -> HybridStrategyOutput:
            started = time.perf_counter()
            evidence = await sparse.retrieve(question, top_k=max(eval_config.k_values))
            return HybridStrategyOutput(
                tuple(evidence),
                {"sparse": (time.perf_counter() - started) * 1000},
                {"sparse": 1},
            )

        if "dense" in strategies:
            runners["dense"] = dense_run
        if "sparse" in strategies:
            runners["sparse"] = sparse_run
        for strategy in ("rrf", "weighted", "fusion_rerank"):
            if strategy not in strategies:
                continue
            method = "weighted" if strategy == "weighted" else "rrf"
            strategy_config = config.model_copy(
                update={
                    "fusion": config.fusion.model_copy(update={"method": method}),
                    "reranker": config.reranker.model_copy(
                        update={"enabled": strategy == "fusion_rerank"}
                    ),
                }
            )
            reranker = (
                CrossEncoderReranker(
                    strategy_config.reranker,
                    workspace_root=PROJECT_ROOT,
                    local_files_only=settings.model_local_files_only,
                )
                if strategy == "fusion_rerank"
                else None
            )
            service = HybridRetrievalService(
                dense,
                sparse,
                strategy_config,
                collection_version=vector_store.collection_version,
                sparse_index_version=identity.index_version,
                corpus_snapshot_hash=snapshot,
                reranker=reranker,
                workspace_root=PROJECT_ROOT,
            )
            services[strategy] = service
            fusion_versions[strategy] = fusion_version(strategy_config.fusion)

            async def fused_run(
                question: str, *, selected_service: HybridRetrievalService = service
            ) -> HybridStrategyOutput:
                result = await selected_service.retrieve(question)
                return HybridStrategyOutput(
                    tuple(result.evidence),
                    {stage.stage: stage.latency_ms for stage in result.stages},
                    {stage.stage: stage.call_count for stage in result.stages},
                )

            runners[strategy] = fused_run
        typed_runners = {key: value for key, value in runners.items()}
        runner = HybridEvaluationRunner(
            runners=typed_runners,  # type: ignore[arg-type]
            config_hash=dense_config_hash(config),
            collection_version=vector_store.collection_version,
            sparse_index_version=identity.index_version,
            fusion_versions=fusion_versions,
            reranker_version=(
                f"{config.reranker.model}@{config.reranker.revision}"
                if "fusion_rerank" in strategies
                else None
            ),
            k_values=eval_config.k_values,
            results_root=PROJECT_ROOT / eval_config.results_root,
        )
        report = await runner.run(
            question_set,
            limit=limit,
            answer_validation_enabled=args.with_answer,
        )
        if args.with_answer:
            await _validate_answers(
                report.evaluation_version,
                question_set,
                limit,
                services["fusion_rerank"],
                config,
                settings,
                PROJECT_ROOT / eval_config.results_root,
            )
        print(report.model_dump_json())
        return 2 if any(metric.failure_count for metric in report.metrics.values()) else 0
    finally:
        await vector_store.close()


async def _validate_answers(
    evaluation_version: str,
    questions: object,
    limit: int,
    service: HybridRetrievalService,
    config: HybridRetrievalConfig,
    settings: Settings,
    results_root: Path,
) -> None:
    from kg_crag.models import PilotQuestionSet

    question_set = PilotQuestionSet.model_validate(questions)
    prompt = load_prompt(PROJECT_ROOT / config.dense.generation.prompt_path)
    llm = _build_llm(settings)
    answer_version = hashlib.sha256(
        f"{evaluation_version}\n{prompt.version}\n{settings.llm_provider}\n{settings.llm_model}".encode()
    ).hexdigest()
    query = HybridQueryService(service, llm, config, prompt, workspace_root=PROJECT_ROOT)
    for question in question_set.questions[:limit]:
        path = (
            results_root
            / evaluation_version
            / "answers"
            / answer_version
            / f"{question.question_id}.json"
        )
        if path.exists():
            HybridQueryResult.model_validate_json(path.read_text(encoding="utf-8"))
            continue
        result = await query.ask(question.question, with_answer=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
        try:
            temporary.write_bytes(stable_json_bytes(result))
            HybridQueryResult.model_validate_json(temporary.read_text(encoding="utf-8"))
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _build_llm(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "mock":
        return MockLLMProvider(
            '{"claims":[{"text":"离线 Mock 回答","claim_type":"fact",'
            '"citation_ids":["E1"]}],"confidence":0.5}'
        )
    if settings.llm_provider == "openai-compatible":
        return OpenAICompatibleLLMProvider(
            settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    raise ValueError(f"unsupported LLM provider: {settings.llm_provider}")


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_run(argv))
    except (KGCRAGError, OSError, ValueError) as error:
        print(f"hybrid evaluation failed: {error}", file=sys.stderr)
        return 1
