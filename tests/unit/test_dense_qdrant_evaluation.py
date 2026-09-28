"""Qdrant 生命周期和 Dense 评测指标的离线测试。"""

import warnings
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient

from kg_crag.errors import KGCRAGError
from kg_crag.evaluation import compute_metrics
from kg_crag.models import (
    Chunk,
    DenseEvaluationItem,
    ErrorCode,
    ErrorDetail,
    EvaluationStatus,
    PilotQuestion,
)
from kg_crag.retrieval.config import load_dense_rag_config
from kg_crag.vector_store import QdrantVectorStore, collection_identity

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _chunk(chunk_id: str, paper_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        section="Results",
        page_start=2,
        page_end=2,
        text=f"result {chunk_id}",
        token_count=2,
        content_hash=f"content-{chunk_id}",
        processing_version="pv1",
    )


async def test_qdrant_adapter_lifecycle_in_memory() -> None:
    config = load_dense_rag_config(PROJECT_ROOT / "configs/retrieval.yaml")
    config = config.model_copy(
        update={"embedding": config.embedding.model_copy(update={"dimensions": 2})}
    )
    identity = collection_identity(config)
    client = AsyncQdrantClient(location=":memory:")
    store = QdrantVectorStore(identity, client=client)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Payload indexes have no effect.*")
        await store.ensure_collection()
    await store.upsert([(_chunk("c1", "p1"), [1.0, 0.0]), (_chunk("c2", "p1"), [0.0, 1.0])])

    results = await store.search([1.0, 0.0], top_k=2, filters={"paper_id": "p1"})
    assert [item.source_id for item in results] == ["c1", "c2"]
    assert results[0].ranks.dense == 1
    assert await store.delete_stale("p1", {"c1"}) == 1
    assert set(await store.record_state("p1")) == {"c1"}

    incompatible = identity.model_copy(update={"collection_version": "f" * 64})
    mismatch = QdrantVectorStore(incompatible, client=client)
    with pytest.raises(KGCRAGError, match="incompatible"):
        await mismatch.ensure_collection()
    await store.close()


def test_evaluation_failures_remain_in_metric_denominator() -> None:
    questions = [
        PilotQuestion(
            question_id=f"question-{index}",
            question=f"Question number {index}?",
            target_chunk_ids=[f"c{index}"],
            rationale="fixed target",
        )
        for index in range(20)
    ]
    items = [
        DenseEvaluationItem(
            question_id="question-0",
            status=EvaluationStatus.SUCCEEDED,
            retrieved_chunk_ids=["c0"],
            cited_chunk_ids=["c0"],
            latency_ms=1,
        ),
        DenseEvaluationItem(
            question_id="question-1",
            status=EvaluationStatus.FAILED,
            latency_ms=1,
            error=ErrorDetail(code=ErrorCode.INTERNAL, message="failed", retryable=False),
        ),
    ]

    metrics = compute_metrics(questions, items, [1])
    assert metrics.recall_at_k["1"] == pytest.approx(0.05)
    assert metrics.mrr == pytest.approx(0.05)
    assert metrics.citation_precision == pytest.approx(1.0)
    assert metrics.citation_recall == pytest.approx(0.05)
    assert metrics.failure_count == 19
