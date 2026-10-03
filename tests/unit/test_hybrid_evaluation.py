"""Hybrid 检索指标、失败隔离和检查点复用测试。"""

from pathlib import Path

import pytest

from kg_crag.evaluation import (
    HybridEvaluationRunner,
    HybridStrategyOutput,
    compute_hybrid_metrics,
)
from kg_crag.models import (
    EvaluationStatus,
    Evidence,
    EvidenceSourceType,
    HybridEvaluationItem,
    PilotQuestion,
    PilotQuestionSet,
)


def _evidence(source_id: str) -> Evidence:
    return Evidence(
        evidence_id=f"fixture:{source_id}",
        content=f"content {source_id}",
        source_type=EvidenceSourceType.CHUNK,
        source_id=source_id,
        paper_id="paper-1",
    )


def _questions() -> PilotQuestionSet:
    return PilotQuestionSet(
        corpus_snapshot_hash="a" * 64,
        questions=[
            PilotQuestion(
                question_id=f"question-{index:02d}",
                question=f"What does fixture {index} explain?",
                target_chunk_ids=[f"target-{index:02d}"],
                rationale="用于稳定的离线评测测试。",
            )
            for index in range(20)
        ],
    )


def test_metrics_deduplicate_hits_and_keep_failures_in_denominator() -> None:
    questions = _questions().questions[:2]
    items = [
        HybridEvaluationItem(
            question_id="question-00",
            strategy="dense",
            status=EvaluationStatus.SUCCEEDED,
            target_chunk_ids=["target-00"],
            retrieved_chunk_ids=["wrong", "target-00", "target-00"],
            latency_ms=10,
            stage_latency_ms={"dense": 8},
        ),
        HybridEvaluationItem(
            question_id="question-01",
            strategy="dense",
            status=EvaluationStatus.FAILED,
            target_chunk_ids=["target-01"],
            latency_ms=5,
            error={
                "code": "internal_error",
                "message": "fixture failed",
                "retryable": False,
            },
        ),
    ]

    metrics = compute_hybrid_metrics(questions, items, [1, 2])
    assert metrics.recall_at_k == {"1": 0.0, "2": 0.5}
    assert metrics.mrr == pytest.approx(0.25)
    assert metrics.evidence_coverage == 0.5
    assert metrics.failure_count == 1
    assert metrics.mean_stage_latency_ms == {"dense": 4.0}


async def test_runner_isolates_failure_and_reuses_same_version_checkpoints(tmp_path: Path) -> None:
    calls: list[str] = []

    async def runner(question: str) -> HybridStrategyOutput:
        calls.append(question)
        index = int(question.split()[3])
        if index == 3:
            raise RuntimeError("fixture failure")
        return HybridStrategyOutput((_evidence(f"target-{index:02d}"),), {"dense": 1.0})

    evaluation = HybridEvaluationRunner(
        runners={"dense": runner},
        config_hash="b" * 64,
        collection_version="dense-v1",
        sparse_index_version="sparse-v1",
        fusion_versions={},
        reranker_version=None,
        k_values=[1, 5],
        results_root=tmp_path / "results",
    )
    first = await evaluation.run(_questions())
    second = await evaluation.run(_questions())

    assert len(calls) == 20
    assert first.metrics["dense"].failure_count == 1
    assert first.metrics["dense"].recall_at_k["1"] == pytest.approx(19 / 20)
    assert second.evaluation_version == first.evaluation_version
    assert (tmp_path / "results" / first.evaluation_version / "report.json").is_file()


async def test_checkpoint_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    async def runner(_question: str) -> HybridStrategyOutput:
        return HybridStrategyOutput((_evidence("target-00"),), {})

    evaluation = HybridEvaluationRunner(
        runners={"dense": runner},
        config_hash="b" * 64,
        collection_version="dense-v1",
        sparse_index_version="sparse-v1",
        fusion_versions={},
        reranker_version=None,
        k_values=[1],
        results_root=tmp_path / "results",
    )
    first = await evaluation.run(_questions(), limit=1)
    checkpoint = (
        tmp_path / "results" / first.evaluation_version / "items" / "dense" / "question-00.json"
    )
    payload = checkpoint.read_text(encoding="utf-8").replace("question-00", "question-99")
    checkpoint.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="version drift"):
        await evaluation.run(_questions(), limit=1)
