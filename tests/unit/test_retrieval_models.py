"""Sparse/Hybrid 公共模型的状态组合和安全边界测试。"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from kg_crag.models import (
    ErrorCode,
    ErrorDetail,
    EvaluationStatus,
    HybridEvaluationItem,
    HybridRetrievalResult,
    RetrievalCostSummary,
    RetrievalStageStatus,
    RetrievalStageSummary,
    SparseIndexIdentity,
    SparseIndexItemResult,
    SparseIndexRunManifest,
)

HASH = "a" * 64


def _error() -> ErrorDetail:
    return ErrorDetail(code=ErrorCode.INTERNAL, message="backend failed", retryable=False)


def _identity() -> SparseIndexIdentity:
    return SparseIndexIdentity(
        schema_version="v1",
        tokenizer="scientific-unicode-v1",
        bm25_version="sqlite-fts5-bm25-v1",
        python_version="3.11.9",
        sqlite_version="3.45.1",
        fts5_enabled=True,
        fts5_version="sqlite-builtin-3.45.1",
        corpus_snapshot_hash=HASH,
        index_version="b" * 64,
    )


def test_sparse_manifest_validates_status_time_and_snapshot() -> None:
    now = datetime.now(UTC)
    failed = SparseIndexItemResult(paper_id="p1", status="failed", error=_error())
    manifest = SparseIndexRunManifest(
        run_id="run-1",
        dry_run=False,
        config_hash=HASH,
        corpus_snapshot_hash=HASH,
        index=_identity(),
        started_at=now,
        finished_at=now,
        items=[failed],
    )
    assert manifest.items[0].error is not None

    with pytest.raises(ValidationError, match="requires an error"):
        SparseIndexItemResult(paper_id="p1", status="failed")
    with pytest.raises(ValidationError, match="snapshot hashes"):
        manifest.model_copy(update={"corpus_snapshot_hash": "c" * 64}).__class__.model_validate(
            {**manifest.model_dump(), "corpus_snapshot_hash": "c" * 64}
        )


def test_retrieval_stage_and_cost_reject_invalid_combinations() -> None:
    failed = RetrievalStageSummary(
        stage="sparse", status=RetrievalStageStatus.FAILED, call_count=1, error=_error()
    )
    assert failed.latency_ms == 0.0
    with pytest.raises(ValidationError, match="requires an error"):
        RetrievalStageSummary(stage="dense", status="failed")
    with pytest.raises(ValidationError, match="output candidates"):
        RetrievalStageSummary(stage="rerank", status="skipped", output_candidates=1)
    with pytest.raises(ValidationError):
        RetrievalCostSummary(reranker_calls=-1)
    with pytest.raises(ValidationError):
        RetrievalCostSummary(llm_calls=1)


def test_hybrid_result_tracks_degradation_and_rejects_unknown_fields() -> None:
    result = HybridRetrievalResult(
        run_id="run-1",
        query_id="q1",
        query="What is GPT-4?",
        config_hash=HASH,
        corpus_snapshot_hash=HASH,
        collection_version="dense-v1",
        sparse_index_version="sparse-v1",
        fusion_version="rrf-v1",
        degraded=True,
        degraded_stages=["sparse"],
    )
    assert result.degraded_stages == ["sparse"]
    with pytest.raises(ValidationError, match="degraded must match"):
        result.__class__.model_validate({**result.model_dump(), "degraded": False})
    with pytest.raises(ValidationError, match="extra"):
        result.__class__.model_validate({**result.model_dump(), "secret": "x"})


def test_evaluation_failure_requires_safe_error() -> None:
    item = HybridEvaluationItem(
        question_id="q1",
        strategy="rrf",
        status=EvaluationStatus.FAILED,
        latency_ms=1.0,
        error=_error(),
    )
    assert item.error is not None
    with pytest.raises(ValidationError, match="sensitive context"):
        ErrorDetail(
            code=ErrorCode.INTERNAL,
            message="safe",
            context={"api_key": "must-not-appear"},
        )
