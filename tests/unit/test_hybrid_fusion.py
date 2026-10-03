"""RRF、Weighted Fusion 和稳定去重的手算测试。"""

import pytest

from kg_crag.models import (
    Evidence,
    EvidenceRanks,
    EvidenceScores,
    EvidenceSourceType,
)
from kg_crag.retrieval import FusionConfig, deduplicate_evidence_by_paper, fuse_evidence


def _evidence(
    source_id: str,
    paper_id: str,
    *,
    dense: float | None = None,
    sparse: float | None = None,
    dense_rank: int | None = None,
    sparse_rank: int | None = None,
) -> Evidence:
    return Evidence(
        evidence_id=f"source:{source_id}",
        content=f"content {source_id}",
        source_type=EvidenceSourceType.CHUNK,
        source_id=source_id,
        paper_id=paper_id,
        scores=EvidenceScores(dense=dense, sparse=sparse),
        ranks=EvidenceRanks(dense=dense_rank, sparse=sparse_rank),
        metadata={"content_hash": f"hash-{source_id}"},
    )


def test_rrf_merges_overlap_preserves_sources_and_does_not_mutate_inputs() -> None:
    dense = [
        _evidence("c1", "p1", dense=0.9, dense_rank=1),
        _evidence("c2", "p2", dense=0.8, dense_rank=2),
    ]
    sparse = [
        _evidence("c1", "p1", sparse=3.0, sparse_rank=1),
        _evidence("c3", "p3", sparse=2.0, sparse_rank=2),
    ]
    original_dense = [item.model_copy(deep=True) for item in dense]
    results = fuse_evidence(dense, sparse, FusionConfig(method="rrf", rrf_k=60))

    assert [item.source_id for item in results] == ["c1", "c2", "c3"]
    assert results[0].scores.fusion == pytest.approx(2 / 61)
    assert results[0].scores.dense == 0.9
    assert results[0].scores.sparse == 3.0
    assert results[0].ranks.dense == 1
    assert results[0].ranks.sparse == 1
    assert results[0].metadata["dense_evidence_id"] == "source:c1"
    assert results[0].metadata["sparse_evidence_id"] == "source:c1"
    assert results[1].scores.sparse is None
    assert dense == original_dense


def test_weighted_fusion_normalizes_scales_and_resolves_ties_stably() -> None:
    dense = [
        _evidence("c1", "p1", dense=0.2, dense_rank=1),
        _evidence("c2", "p2", dense=0.8, dense_rank=2),
    ]
    sparse = [
        _evidence("c1", "p1", sparse=10.0, sparse_rank=1),
        _evidence("c3", "p3", sparse=20.0, sparse_rank=2),
    ]
    config = FusionConfig(method="weighted", dense_weight=0.5, sparse_weight=0.5)
    results = fuse_evidence(dense, sparse, config)

    assert [item.source_id for item in results] == ["c2", "c3", "c1"]
    assert results[0].scores.fusion == pytest.approx(0.5)
    assert results[1].scores.fusion == pytest.approx(0.5)
    equal = fuse_evidence([_evidence("only", "p", dense=7.0)], [], config)
    assert equal[0].scores.fusion == pytest.approx(0.5)
    assert fuse_evidence([], [], config) == []


def test_paper_dedup_keeps_best_current_rank_without_refill_loop() -> None:
    candidates = [
        _evidence("c1", "p1", dense=1.0),
        _evidence("c2", "p1", dense=0.9),
        _evidence("c3", "p2", dense=0.8),
    ]
    selected = deduplicate_evidence_by_paper(candidates, max_results=3)
    assert [item.source_id for item in selected] == ["c1", "c3"]
    assert selected[0] is not candidates[0]
