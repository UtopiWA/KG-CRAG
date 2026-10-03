"""默认零在线调用的关系与多跳图检索评测。"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from kg_crag.graph.artifacts import atomic_write_json
from kg_crag.models import GraphEvaluationQuestion, GraphEvaluationQuestionSet, GraphPathHit
from kg_crag.retrieval.graph import GraphRetriever


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, item in enumerate(retrieved, 1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 1.0 if not retrieved[:k] else 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 1.0 if not retrieved[:k] else 0.0
    seen: set[str] = set()
    dcg = 0.0
    for rank, item in enumerate(retrieved[:k], 1):
        if item in relevant and item not in seen:
            dcg += 1.0 / math.log2(rank + 1)
            seen.add(item)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal if ideal else 0.0


def _path_hit(question: GraphEvaluationQuestion, paths: Sequence[GraphPathHit]) -> float:
    if not question.relevant_fact_ids and not question.relevant_path_ids:
        return 1.0 if not paths else 0.0
    for path in paths:
        if path.path_id in question.relevant_path_ids:
            return 1.0
        if set(question.relevant_fact_ids).issubset(path.fact_ids):
            return 1.0
    return 0.0


class GraphEvaluator:
    def __init__(self, retriever: GraphRetriever, *, results_root: Path) -> None:
        self.retriever = retriever
        self.results_root = results_root

    async def evaluate(self, questions: GraphEvaluationQuestionSet) -> dict[str, object]:
        started = datetime.now(UTC)
        items: list[dict[str, object]] = []
        path_hits: list[float] = []
        recalls: list[float] = []
        reciprocal_ranks: list[float] = []
        ndcgs: list[float] = []
        provenance_scores: list[float] = []
        failures = 0
        database_calls = 0
        for question in questions.questions:
            item_started = time.perf_counter()
            checkpoint = (
                self.results_root
                / question.graph_version
                / "checkpoints"
                / f"{question.question_id}.json"
            )
            if checkpoint.is_file():
                cached = json.loads(checkpoint.read_text(encoding="utf-8"))
                if (
                    cached.get("question_id") != question.question_id
                    or cached.get("graph_version") != question.graph_version
                    or cached.get("corpus_snapshot") != question.corpus_snapshot
                ):
                    raise ValueError("graph evaluation checkpoint identity has drifted")
                items.append(cached)
                path_hits.append(float(cached["path_hit"]))
                recalls.append(float(cached["recall_at_k"]))
                reciprocal_ranks.append(float(cached["mrr"]))
                ndcgs.append(float(cached["ndcg_at_k"]))
                provenance_scores.append(float(cached["provenance_completeness"]))
                failures += int(cached["status"] == "failed")
                database_calls += int(cached["database_calls"])
                continue
            try:
                result = await self.retriever.retrieve(question.request)
                database_calls += result.database_calls
                source_ids = [item.source_id for item in result.evidence]
                relevant = set(question.relevant_chunk_ids)
                path_score = _path_hit(question, result.paths)
                recall = recall_at_k(source_ids, relevant, question.k)
                rr = reciprocal_rank(source_ids, relevant)
                ndcg = ndcg_at_k(source_ids, relevant, question.k)
                provenance = (
                    sum(
                        1
                        for item in result.evidence
                        if bool(item.metadata.get("path_id"))
                        and bool(item.metadata.get("fact_ids"))
                        and item.external is False
                    )
                    / len(result.evidence)
                    if result.evidence
                    else (1.0 if not relevant else 0.0)
                )
                path_hits.append(path_score)
                recalls.append(recall)
                reciprocal_ranks.append(rr)
                ndcgs.append(ndcg)
                provenance_scores.append(provenance)
                item = {
                    "question_id": question.question_id,
                    "status": "succeeded",
                    "path_ids": [path.path_id for path in result.paths],
                    "evidence_ids": [evidence.evidence_id for evidence in result.evidence],
                    "source_ids": source_ids,
                    "path_hit": path_score,
                    "recall_at_k": recall,
                    "mrr": rr,
                    "ndcg_at_k": ndcg,
                    "provenance_completeness": provenance,
                    "rejected_paths": result.rejected_paths,
                    "database_calls": result.database_calls,
                }
            except Exception as error:
                failures += 1
                path_hits.append(0.0)
                recalls.append(0.0)
                reciprocal_ranks.append(0.0)
                ndcgs.append(0.0)
                provenance_scores.append(0.0)
                item = {
                    "question_id": question.question_id,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "database_calls": 0,
                }
            item["latency_ms"] = (time.perf_counter() - item_started) * 1000
            item["graph_version"] = question.graph_version
            item["corpus_snapshot"] = question.corpus_snapshot
            items.append(item)
            atomic_write_json(checkpoint, item)
        total = len(questions.questions)
        latency_ms = 0.0
        for item in items:
            value = item.get("latency_ms", 0.0)
            if isinstance(value, int | float):
                latency_ms += float(value)
        report: dict[str, object] = {
            "schema_version": "v1",
            "corpus_snapshot": questions.questions[0].corpus_snapshot,
            "graph_version": questions.questions[0].graph_version,
            "started_at": started.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "metrics": {
                "path_hit_rate": sum(path_hits) / total,
                "evidence_recall_at_k": sum(recalls) / total,
                "mrr": sum(reciprocal_ranks) / total,
                "ndcg_at_k": sum(ndcgs) / total,
                "provenance_completeness": sum(provenance_scores) / total,
                "failure_count": failures,
                "database_calls": database_calls,
                "latency_ms": latency_ms,
            },
            "items": items,
        }
        destination = self.results_root / questions.questions[0].graph_version / "report.json"
        atomic_write_json(destination, report)
        return report
