"""Dense RAG pilot 加载、指标与逐题失败隔离。"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

from kg_crag.errors import KGCRAGError
from kg_crag.ingestion.storage import stable_json_bytes
from kg_crag.models import (
    DenseEvaluationItem,
    DenseEvaluationMetrics,
    DenseEvaluationReport,
    DenseRAGResult,
    ErrorCode,
    ErrorDetail,
    EvaluationStatus,
    PilotQuestion,
    PilotQuestionSet,
)


def load_question_set(
    path: Path,
    *,
    available_chunk_ids: set[str],
    corpus_snapshot_hash: str,
) -> PilotQuestionSet:
    questions = PilotQuestionSet.model_validate_json(path.read_text(encoding="utf-8"))
    if questions.corpus_snapshot_hash != corpus_snapshot_hash:
        raise ValueError("pilot question corpus snapshot does not match the selected corpus")
    missing = sorted(
        {
            chunk_id
            for question in questions.questions
            for chunk_id in question.target_chunk_ids
            if chunk_id not in available_chunk_ids
        }
    )
    if missing:
        raise ValueError(f"pilot question targets are unavailable: {missing[:5]}")
    return questions


def compute_metrics(
    questions: list[PilotQuestion],
    items: list[DenseEvaluationItem],
    k_values: list[int],
) -> DenseEvaluationMetrics:
    by_id = {item.question_id: item for item in items}
    recall_totals = {value: 0.0 for value in k_values}
    ndcg_totals = {value: 0.0 for value in k_values}
    reciprocal_ranks = 0.0
    cited_total = 0
    cited_relevant = 0
    target_total = 0
    failures = 0
    for question in questions:
        item = by_id.get(question.question_id)
        targets = set(question.target_chunk_ids)
        target_total += len(targets)
        if item is None or item.status is EvaluationStatus.FAILED:
            # 失败题保留在统一分母中，避免只统计成功请求而夸大指标。
            failures += 1
            continue
        retrieved = list(dict.fromkeys(item.retrieved_chunk_ids))
        relevant_ranks = [
            index for index, chunk_id in enumerate(retrieved, 1) if chunk_id in targets
        ]
        if relevant_ranks:
            reciprocal_ranks += 1.0 / relevant_ranks[0]
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
        cited = set(item.cited_chunk_ids)
        cited_total += len(cited)
        cited_relevant += len(cited & targets)
    denominator = len(questions)
    return DenseEvaluationMetrics(
        recall_at_k={str(k): recall_totals[k] / denominator for k in k_values},
        mrr=reciprocal_ranks / denominator,
        ndcg_at_k={str(k): ndcg_totals[k] / denominator for k in k_values},
        citation_precision=cited_relevant / cited_total if cited_total else 0.0,
        citation_recall=cited_relevant / target_total if target_total else 0.0,
        failure_count=failures,
    )


class DenseEvaluationRunner:
    def __init__(
        self,
        *,
        query: Callable[[str], Awaitable[DenseRAGResult]],
        config_hash: str,
        prompt_version: str,
        collection_version: str,
        k_values: list[int],
        results_root: Path,
    ) -> None:
        self.query = query
        self.config_hash = config_hash
        self.prompt_version = prompt_version
        self.collection_version = collection_version
        self.k_values = k_values
        self.results_root = results_root.resolve()

    async def run(
        self,
        question_set: PilotQuestionSet,
        *,
        limit: int | None = None,
    ) -> DenseEvaluationReport:
        questions = question_set.questions[:limit]
        if len(questions) < 20:
            raise ValueError("Dense pilot evaluation requires at least 20 questions")
        baseline_id = self._baseline_id(question_set.corpus_snapshot_hash)
        run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{baseline_id[:12]}-{uuid.uuid4().hex[:8]}"
        started = datetime.now(UTC)
        items: list[DenseEvaluationItem] = []
        for question in questions:
            item_started = time.perf_counter()
            try:
                # 单题是最小故障隔离单元；异常会记录到该题，但不会中断整批评测。
                result = await self.query(question.question)
                items.append(
                    DenseEvaluationItem(
                        question_id=question.question_id,
                        status=EvaluationStatus.SUCCEEDED,
                        retrieved_chunk_ids=[item.source_id for item in result.evidence],
                        cited_chunk_ids=[item.source_id for item in result.citations],
                        latency_ms=(time.perf_counter() - item_started) * 1000,
                        result=result,
                    )
                )
            except Exception as error:
                items.append(
                    DenseEvaluationItem(
                        question_id=question.question_id,
                        status=EvaluationStatus.FAILED,
                        latency_ms=(time.perf_counter() - item_started) * 1000,
                        error=self._safe_error(error),
                    )
                )
        report = DenseEvaluationReport(
            baseline_id=baseline_id,
            run_id=run_id,
            corpus_snapshot_hash=question_set.corpus_snapshot_hash,
            config_hash=self.config_hash,
            prompt_version=self.prompt_version,
            collection_version=self.collection_version,
            started_at=started,
            finished_at=datetime.now(UTC),
            metrics=compute_metrics(questions, items, self.k_values),
            items=items,
        )
        self._publish(report)
        return report

    def _baseline_id(self, corpus_snapshot_hash: str) -> str:
        # 基线标识绑定数据、配置、Prompt、集合和指标口径，任一变化都会产生新版本。
        payload = "\n".join(
            [
                corpus_snapshot_hash,
                self.config_hash,
                self.prompt_version,
                self.collection_version,
                ",".join(str(value) for value in self.k_values),
            ]
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _publish(self, report: DenseEvaluationReport) -> Path:
        destination = self.results_root / report.run_id
        temporary = self.results_root / f".{report.run_id}.tmp-{uuid.uuid4().hex}"
        if destination.exists():
            raise FileExistsError(f"evaluation result already exists: {report.run_id}")
        self.results_root.mkdir(parents=True, exist_ok=True)
        try:
            temporary.mkdir(exist_ok=False)
            data = stable_json_bytes(report)
            path = temporary / "report.json"
            path.write_bytes(data)
            DenseEvaluationReport.model_validate_json(path.read_text(encoding="utf-8"))
            os.replace(temporary, destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return destination / "report.json"

    @staticmethod
    def _safe_error(error: Exception) -> ErrorDetail:
        if isinstance(error, KGCRAGError):
            return error.detail
        return ErrorDetail(
            code=ErrorCode.INTERNAL,
            message="Dense evaluation item failed",
            retryable=False,
            context={"error_type": type(error).__name__},
        )
