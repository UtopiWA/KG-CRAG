"""零 LLM 的 Hybrid 检索矩阵、指标和可恢复逐题检查点。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from kg_crag.errors import KGCRAGError
from kg_crag.ingestion.storage import stable_json_bytes
from kg_crag.models import (
    ErrorCode,
    ErrorDetail,
    EvaluationStatus,
    Evidence,
    HybridEvaluationItem,
    HybridEvaluationMetrics,
    HybridEvaluationReport,
    HybridStrategy,
    PilotQuestion,
    PilotQuestionSet,
)


@dataclass(frozen=True)
class HybridStrategyOutput:
    """策略适配器向评测器返回的最小、后端无关结果。"""

    evidence: tuple[Evidence, ...]
    stage_latency_ms: Mapping[str, float]
    stage_call_counts: Mapping[str, int] = field(default_factory=dict)


HybridStrategyRunner = Callable[[str], Awaitable[HybridStrategyOutput]]


def compute_hybrid_metrics(
    questions: Sequence[PilotQuestion],
    items: Sequence[HybridEvaluationItem],
    k_values: Sequence[int],
) -> HybridEvaluationMetrics:
    """以全部题目为分母计算检索指标，失败题不会被静默排除。"""

    by_id = {item.question_id: item for item in items}
    recall_totals = {value: 0.0 for value in k_values}
    ndcg_totals = {value: 0.0 for value in k_values}
    reciprocal_ranks = 0.0
    covered_targets = 0
    target_total = 0
    failure_count = 0
    latency_total = 0.0
    stage_totals: dict[str, float] = {}
    for question in questions:
        targets = set(question.target_chunk_ids)
        target_total += len(targets)
        item = by_id.get(question.question_id)
        if item is None or item.status is EvaluationStatus.FAILED:
            failure_count += 1
            continue
        latency_total += item.latency_ms
        for stage, latency in item.stage_latency_ms.items():
            stage_totals[stage] = stage_totals.get(stage, 0.0) + latency
        retrieved = list(dict.fromkeys(item.retrieved_chunk_ids))
        relevant_ranks = [rank for rank, chunk_id in enumerate(retrieved, 1) if chunk_id in targets]
        if relevant_ranks:
            reciprocal_ranks += 1.0 / relevant_ranks[0]
        covered_targets += len(set(retrieved) & targets)
        for k in k_values:
            top = retrieved[:k]
            recall_totals[k] += len(set(top) & targets) / len(targets)
            dcg = sum(
                1.0 / math.log2(rank + 1)
                for rank, chunk_id in enumerate(top, 1)
                if chunk_id in targets
            )
            ideal_count = min(len(targets), k)
            ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
            ndcg_totals[k] += dcg / ideal if ideal else 0.0
    denominator = len(questions)
    if not denominator:
        raise ValueError("hybrid evaluation requires at least one question")
    return HybridEvaluationMetrics(
        recall_at_k={str(k): recall_totals[k] / denominator for k in k_values},
        mrr=reciprocal_ranks / denominator,
        ndcg_at_k={str(k): ndcg_totals[k] / denominator for k in k_values},
        evidence_coverage=covered_targets / target_total if target_total else 0.0,
        failure_count=failure_count,
        mean_latency_ms=latency_total / denominator,
        mean_stage_latency_ms={key: value / denominator for key, value in stage_totals.items()},
    )


class HybridEvaluationRunner:
    """按策略和题目隔离失败，并复用严格匹配版本的检查点。"""

    def __init__(
        self,
        *,
        runners: Mapping[HybridStrategy, HybridStrategyRunner],
        config_hash: str,
        collection_version: str,
        sparse_index_version: str,
        fusion_versions: Mapping[HybridStrategy, str],
        reranker_version: str | None,
        k_values: Sequence[int],
        results_root: Path,
    ) -> None:
        if not runners:
            raise ValueError("hybrid evaluation requires at least one strategy")
        self.runners = dict(runners)
        self.config_hash = config_hash
        self.collection_version = collection_version
        self.sparse_index_version = sparse_index_version
        self.fusion_versions = dict(fusion_versions)
        self.reranker_version = reranker_version
        self.k_values = list(k_values)
        self.results_root = results_root.resolve()

    async def run(
        self,
        question_set: PilotQuestionSet,
        *,
        limit: int | None = None,
        answer_validation_enabled: bool = False,
    ) -> HybridEvaluationReport:
        if limit is not None and limit <= 0:
            raise ValueError("evaluation limit must be positive")
        questions = question_set.questions[:limit]
        version = self._version(question_set)
        started = datetime.now(UTC)
        all_items: list[HybridEvaluationItem] = []
        metrics: dict[HybridStrategy, HybridEvaluationMetrics] = {}
        for strategy, runner in self.runners.items():
            strategy_items: list[HybridEvaluationItem] = []
            for question in questions:
                cached = self._load_checkpoint(version, strategy, question.question_id)
                if cached is not None:
                    strategy_items.append(cached)
                    continue
                item = await self._run_item(strategy, runner, question)
                self._write_checkpoint(version, item)
                strategy_items.append(item)
            metrics[strategy] = compute_hybrid_metrics(questions, strategy_items, self.k_values)
            all_items.extend(strategy_items)
        report = HybridEvaluationReport(
            run_id=f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{version[:12]}-{uuid.uuid4().hex[:8]}",
            evaluation_version=version,
            corpus_snapshot_hash=question_set.corpus_snapshot_hash,
            config_hash=self.config_hash,
            collection_version=self.collection_version,
            sparse_index_version=self.sparse_index_version,
            fusion_versions=self.fusion_versions,
            reranker_version=self.reranker_version,
            started_at=started,
            finished_at=datetime.now(UTC),
            answer_validation_enabled=answer_validation_enabled,
            metrics=metrics,
            items=all_items,
        )
        self._write_atomic(self.results_root / version / "report.json", report)
        return report

    async def _run_item(
        self,
        strategy: HybridStrategy,
        runner: HybridStrategyRunner,
        question: PilotQuestion,
    ) -> HybridEvaluationItem:
        started = time.perf_counter()
        try:
            output = await runner(question.question)
            return HybridEvaluationItem(
                question_id=question.question_id,
                strategy=strategy,
                status=EvaluationStatus.SUCCEEDED,
                target_chunk_ids=question.target_chunk_ids,
                retrieved_chunk_ids=[item.source_id for item in output.evidence],
                latency_ms=(time.perf_counter() - started) * 1000,
                stage_latency_ms=dict(output.stage_latency_ms),
                stage_call_counts=dict(output.stage_call_counts),
            )
        except Exception as error:
            return HybridEvaluationItem(
                question_id=question.question_id,
                strategy=strategy,
                status=EvaluationStatus.FAILED,
                target_chunk_ids=question.target_chunk_ids,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=_safe_error(error),
            )

    def _version(self, question_set: PilotQuestionSet) -> str:
        payload = {
            "config_hash": self.config_hash,
            "corpus_snapshot_hash": question_set.corpus_snapshot_hash,
            "questions": [
                {
                    "question_id": item.question_id,
                    "question": item.question,
                    "target_chunk_ids": item.target_chunk_ids,
                    "rationale": item.rationale,
                }
                for item in question_set.questions
            ],
            "collection_version": self.collection_version,
            "sparse_index_version": self.sparse_index_version,
            "fusion_versions": self.fusion_versions,
            "reranker_version": self.reranker_version,
            "strategies": list(self.runners),
            "k_values": self.k_values,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def _load_checkpoint(
        self,
        version: str,
        strategy: HybridStrategy,
        question_id: str,
    ) -> HybridEvaluationItem | None:
        path = self._checkpoint_path(version, strategy, question_id)
        if not path.exists():
            return None
        item = HybridEvaluationItem.model_validate_json(path.read_text(encoding="utf-8"))
        if item.strategy != strategy or item.question_id != question_id:
            raise ValueError("hybrid evaluation checkpoint version drift")
        return item

    def _write_checkpoint(self, version: str, item: HybridEvaluationItem) -> None:
        self._write_atomic(self._checkpoint_path(version, item.strategy, item.question_id), item)

    def _checkpoint_path(
        self,
        version: str,
        strategy: HybridStrategy,
        question_id: str,
    ) -> Path:
        return self.results_root / version / "items" / strategy / f"{question_id}.json"

    @staticmethod
    def _write_atomic(path: Path, model: HybridEvaluationItem | HybridEvaluationReport) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
        try:
            temporary.write_bytes(stable_json_bytes(model))
            if isinstance(model, HybridEvaluationItem):
                HybridEvaluationItem.model_validate_json(temporary.read_text(encoding="utf-8"))
            else:
                HybridEvaluationReport.model_validate_json(temporary.read_text(encoding="utf-8"))
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _safe_error(error: Exception) -> ErrorDetail:
    if isinstance(error, KGCRAGError):
        return error.detail
    return ErrorDetail(
        code=ErrorCode.INTERNAL,
        message="Hybrid evaluation item failed",
        retryable=False,
        context={"error_type": type(error).__name__},
    )
