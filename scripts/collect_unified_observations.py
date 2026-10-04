"""生成统一开发集的零 LLM 三策略观察集。"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

from pydantic import ValidationError

from kg_crag.evaluation.artifacts import ensure_formal_test_allowed, load_test_selection
from kg_crag.evaluation.config import load_unified_evaluation_config
from kg_crag.evaluation.corrective import load_corrective_questions
from kg_crag.evaluation.dataset import canonical_digest, load_unified_dataset, write_json_atomic
from kg_crag.evaluation.observation_collection import (
    LocalRetrievalCallbacks,
    collect_strategy_observations,
    prepare_extractive_answer_validation,
)
from kg_crag.graph_store import InMemoryGraphStore
from kg_crag.indexing.processed import ProcessedPaper, discover_processed, select_processed
from kg_crag.models import EvaluationSplit
from kg_crag.providers import EmbeddingService, SentenceTransformerEmbeddingProvider
from kg_crag.retrieval import SparseRetriever, load_hybrid_retrieval_config
from kg_crag.retrieval.dense import DenseRetriever
from kg_crag.retrieval.hybrid import HybridRetrievalService
from kg_crag.sparse_store import (
    SQLiteSparseStore,
    load_sqlite_index_identity,
)
from kg_crag.vector_store import InMemoryVectorStore, corpus_snapshot_hash

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TERM = re.compile(r"[A-Za-z][A-Za-z0-9+.-]{3,}")
_STOP_TERMS = {
    "what",
    "which",
    "where",
    "when",
    "does",
    "paper",
    "method",
    "agent",
    "system",
    "reported",
    "required",
    "evidence",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/retrieval.yaml")
    parser.add_argument(
        "--evaluation-config",
        type=Path,
        default=PROJECT_ROOT / "configs/evaluation.yaml",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data/evaluation/unified/manifest.json",
    )
    parser.add_argument(
        "--corrective-questions",
        type=Path,
        default=PROJECT_ROOT / "data/evaluation/corrective_dev_questions.json",
    )
    parser.add_argument("--processed-root", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument(
        "--pilot-manifest", type=Path, default=PROJECT_ROOT / "configs/pilot_corpus.json"
    )
    parser.add_argument("--sparse-index-version")
    parser.add_argument("--split", choices=[item.value for item in EvaluationSplit], default="dev")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--confirm-test-observation", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _resolve_sparse_index(config_root: Path, requested: str | None) -> Path:
    if requested:
        candidate = config_root / requested / "index.sqlite3"
        if not candidate.is_file():
            raise ValueError(f"sparse index is unavailable: {requested}")
        return candidate
    candidates = sorted(config_root.glob("*/index.sqlite3"))
    if len(candidates) != 1:
        raise ValueError("select --sparse-index-version when zero or multiple indices exist")
    return candidates[0]


def _graph_terms(query: str) -> list[str]:
    """只从问题文本提取显式实体词，不读取真值 Evidence。"""

    return list(
        dict.fromkeys(
            term
            for term in _TERM.findall(query)
            if term.casefold() not in _STOP_TERMS and not term.casefold().startswith("facet")
        )
    )[:8]


async def _build_callbacks(
    args: argparse.Namespace,
    papers: list[ProcessedPaper],
    *,
    snapshot: str,
) -> tuple[LocalRetrievalCallbacks, dict[str, str]]:
    config = load_hybrid_retrieval_config(args.config)
    # 开发矩阵禁用重排器，避免下载第二个模型并保持零 LLM、低资源边界。
    config = config.model_copy(
        update={"reranker": config.reranker.model_copy(update={"enabled": False})}
    )
    sparse_path = _resolve_sparse_index(
        PROJECT_ROOT / config.sparse.index_root, args.sparse_index_version
    )
    sparse_identity = load_sqlite_index_identity(sparse_path)
    if sparse_identity.corpus_snapshot_hash != snapshot:
        raise ValueError("sparse index and selected processed corpus have different snapshots")
    sparse_store = SQLiteSparseStore(
        sparse_path,
        sparse_identity,
        max_top_k=config.sparse.max_top_k,
        max_candidates=config.sparse.max_candidates,
        max_query_chars=config.sparse.max_query_chars,
        max_query_tokens=config.sparse.max_query_tokens,
    )
    await sparse_store.ensure_index(sparse_identity)

    embedding = EmbeddingService(
        SentenceTransformerEmbeddingProvider(
            config.dense.embedding.model,
            config.dense.embedding.revision,
            dimensions=config.dense.embedding.dimensions,
            normalize=config.dense.embedding.normalize,
            local_files_only=True,
        ),
        config.dense.embedding,
        workspace_root=PROJECT_ROOT,
    )
    collection_version = f"memory-{embedding.namespace[:16]}-{snapshot[:16]}"
    vector_store = InMemoryVectorStore(collection_version=collection_version)
    await vector_store.ensure_collection()
    for paper in papers:
        chunks = list(paper.chunks)
        embedded = await embedding.embed([item.text for item in chunks], input_type="document")
        for offset in range(0, len(chunks), config.dense.indexing.upsert_batch_size):
            end = offset + config.dense.indexing.upsert_batch_size
            await vector_store.upsert(
                list(zip(chunks[offset:end], embedded.vectors[offset:end], strict=True))
            )

    dense = DenseRetriever(embedding, vector_store, config.dense)
    sparse = SparseRetriever(sparse_store, config.sparse)
    hybrid = HybridRetrievalService(
        dense,
        sparse,
        config,
        collection_version=collection_version,
        sparse_index_version=sparse_identity.index_version,
        corpus_snapshot_hash=snapshot,
    )
    graph_store = InMemoryGraphStore()
    for paper in papers:
        if paper.paper is None:
            raise ValueError(f"processed paper metadata is unavailable: {paper.paper_id}")
        await graph_store.upsert_paper(paper.paper, list(paper.chunks))

    async def dense_ids(query: str) -> list[str]:
        return [item.source_id for item in await dense.retrieve(query, top_k=10)]

    async def sparse_ids(query: str) -> list[str]:
        return [item.source_id for item in await sparse.retrieve(query, top_k=10)]

    async def hybrid_ids(query: str) -> list[str]:
        result = await hybrid.retrieve(query)
        return [item.source_id for item in result.evidence]

    async def graph_ids(query: str) -> list[str]:
        found: list[str] = []
        for term in _graph_terms(query):
            evidence = await graph_store.retrieve([term], max_hops=2, top_k=10)
            found.extend(item.source_id for item in evidence)
            if len(dict.fromkeys(found)) >= 10:
                break
        return list(dict.fromkeys(found))[:10]

    versions = {
        "observation_mode": "mixed-local-index-and-controlled-replay-v1",
        "dense": f"{config.dense.embedding.model}@{config.dense.embedding.revision}",
        "dense_collection": collection_version,
        "sparse": sparse_identity.index_version,
        "graph": "processed-in-memory-substring-v1",
        "corrective": "corrective-dev-v1-controlled-replay",
        "corpus_snapshot": snapshot,
        "reranker": "disabled",
        "llm": "disabled",
        "web": "disabled",
    }
    return (
        LocalRetrievalCallbacks(
            dense=dense_ids,
            sparse=sparse_ids,
            hybrid=hybrid_ids,
            graph=graph_ids,
        ),
        versions,
    )


async def _run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest, dev, test = load_unified_dataset(
        args.manifest,
        workspace_root=PROJECT_ROOT,
    )
    split = EvaluationSplit(args.split)
    question_set = dev if split is EvaluationSplit.DEV else test
    dataset_label = manifest.dataset_version.rsplit("-", maxsplit=1)[-1]
    output = args.output or (
        PROJECT_ROOT
        / "data/evaluation/results/unified/observations"
        / f"{split.value}-{dataset_label}-local.json"
    )
    selection = None
    if split is EvaluationSplit.TEST:
        evaluation_config = load_unified_evaluation_config(args.evaluation_config)
        selection = load_test_selection(
            PROJECT_ROOT / evaluation_config.selection_path,
            manifest=manifest,
        )
        ensure_formal_test_allowed(
            PROJECT_ROOT / evaluation_config.test_lock_root / f"{manifest.dataset_version}.json",
            confirmed=args.confirm_test_observation,
            tuning_requested=False,
        )
    config = load_hybrid_retrieval_config(args.config)
    papers = select_processed(
        discover_processed(args.processed_root),
        pilot_manifest=args.pilot_manifest,
        max_papers=config.dense.indexing.max_papers,
    )
    chunks = [chunk for paper in papers for chunk in paper.chunks]
    snapshot = corpus_snapshot_hash(chunks)
    sparse_path = _resolve_sparse_index(
        PROJECT_ROOT / config.sparse.index_root, args.sparse_index_version
    )
    sparse_identity = load_sqlite_index_identity(sparse_path)
    if sparse_identity.corpus_snapshot_hash != snapshot:
        raise ValueError("sparse index and selected processed corpus have different snapshots")
    print(
        f"validated dataset={manifest.dataset_version} split={split.value} "
        f"questions={len(question_set.questions)} "
        f"papers={len(papers)} chunks={len(chunks)} snapshot={snapshot}"
    )
    print(
        "mode=mixed-local-index-and-controlled-replay-v1 "
        f"sparse_index={sparse_identity.index_version} llm=disabled web=disabled"
    )
    if args.dry_run:
        print("dry-run: embedding model not loaded and no observation artifact written")
        return 0
    if output.exists() and not args.overwrite:
        raise ValueError("observation output exists; pass --overwrite to replace the exact file")

    callbacks, versions = await _build_callbacks(args, papers, snapshot=snapshot)
    corrective = load_corrective_questions(args.corrective_questions)
    observation_set = await collect_strategy_observations(
        question_set,
        dataset_hash=manifest.dataset_hash,
        split_hash=canonical_digest(question_set),
        callbacks=callbacks,
        corrective_questions={item.question_id: item for item in corrective.questions},
        source_versions=versions,
    )
    if selection is not None:
        observation_set = prepare_extractive_answer_validation(
            observation_set,
            question_set,
            selected_strategy=selection.selected_strategy,
        )
    write_json_atomic(output, observation_set)
    failures = sum(item.error is not None for item in observation_set.observations)
    print(
        f"observations={len(observation_set.observations)} failures={failures} "
        f"hash={canonical_digest(observation_set)}"
    )
    print(f"output={output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_run(argv))
    except (OSError, RuntimeError, ValidationError, ValueError) as error:
        print(f"observation collection failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
