"""Hybrid 编排、显式降级、重排和原子发布测试。"""

from pathlib import Path

import pytest

from kg_crag.errors import KGCRAGError
from kg_crag.models import (
    ErrorCode,
    ErrorDetail,
    Evidence,
    EvidenceRanks,
    EvidenceScores,
    EvidenceSourceType,
    RetrievalStageStatus,
)
from kg_crag.retrieval import (
    HybridRetrievalConfig,
    HybridRetrievalService,
    MockReranker,
    MockRetriever,
    load_hybrid_retrieval_config,
    publish_hybrid_result,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HASH = "a" * 64


def _config(*, reranker: bool, allow_partial: bool = True) -> HybridRetrievalConfig:
    config = load_hybrid_retrieval_config(PROJECT_ROOT / "configs/retrieval.yaml")
    return config.model_copy(
        update={
            "reranker": config.reranker.model_copy(update={"enabled": reranker}),
            "hybrid": config.hybrid.model_copy(update={"allow_partial": allow_partial}),
        }
    )


def _evidence(source_id: str, paper_id: str, route: str, score: float, rank: int) -> Evidence:
    scores = EvidenceScores(**{route: score})
    ranks = EvidenceRanks(**{route: rank})
    return Evidence(
        evidence_id=f"{route}:{source_id}",
        content=f"content {source_id}",
        source_type=EvidenceSourceType.CHUNK,
        source_id=source_id,
        paper_id=paper_id,
        scores=scores,
        ranks=ranks,
    )


def _service(
    dense: MockRetriever,
    sparse: MockRetriever,
    config: HybridRetrievalConfig,
    *,
    reranker: MockReranker | None = None,
    workspace_root: Path | None = None,
) -> HybridRetrievalService:
    return HybridRetrievalService(
        dense,
        sparse,
        config,
        collection_version="dense-v1",
        sparse_index_version="sparse-v1",
        corpus_snapshot_hash=HASH,
        reranker=reranker,
        workspace_root=workspace_root,
    )


async def test_hybrid_calls_each_route_once_and_replays_stably() -> None:
    dense = MockRetriever([_evidence("c1", "p1", "dense", 0.9, 1)])
    sparse = MockRetriever([_evidence("c1", "p1", "sparse", 3.0, 1)])
    service = _service(dense, sparse, _config(reranker=False))

    first = await service.retrieve("query")
    second = await service.retrieve("query")
    assert [item.source_id for item in first.evidence] == ["c1"]
    assert [item.source_id for item in second.evidence] == ["c1"]
    assert len(dense.calls) == len(sparse.calls) == 2
    assert first.degraded is False
    assert [stage.status for stage in first.stages] == [
        RetrievalStageStatus.SUCCEEDED,
        RetrievalStageStatus.SUCCEEDED,
        RetrievalStageStatus.SUCCEEDED,
        RetrievalStageStatus.SKIPPED,
    ]
    assert first.costs.llm_calls == 0


async def test_hybrid_empty_routes_and_single_route_degradation() -> None:
    empty = await _service(MockRetriever([]), MockRetriever([]), _config(reranker=False)).retrieve(
        "query"
    )
    assert empty.evidence == []
    assert empty.degraded is False

    class FailingRetriever(MockRetriever):
        async def retrieve(
            self,
            query: str,
            *,
            top_k: int,
            filters: dict[str, str | int | bool] | None = None,
        ) -> list[Evidence]:
            raise KGCRAGError(
                ErrorDetail(
                    code=ErrorCode.EXTERNAL_SERVICE,
                    message="dense unavailable",
                    retryable=True,
                )
            )

    sparse = MockRetriever([_evidence("c2", "p2", "sparse", 2.0, 1)])
    degraded = await _service(FailingRetriever([]), sparse, _config(reranker=False)).retrieve(
        "query"
    )
    assert degraded.degraded_stages == ["dense"]
    assert [item.source_id for item in degraded.evidence] == ["c2"]
    assert degraded.stages[0].error is not None

    with pytest.raises(KGCRAGError, match="degradation policy"):
        await _service(
            FailingRetriever([]), sparse, _config(reranker=False, allow_partial=False)
        ).retrieve("query")
    with pytest.raises(KGCRAGError, match="degradation policy"):
        await _service(
            FailingRetriever([]), FailingRetriever([]), _config(reranker=False)
        ).retrieve("query")


async def test_hybrid_reranks_bounded_candidates_and_falls_back_explicitly() -> None:
    dense = MockRetriever(
        [
            _evidence("c1", "p1", "dense", 0.9, 1),
            _evidence("c2", "p2", "dense", 0.8, 2),
        ]
    )
    sparse = MockRetriever([])
    reranker = MockReranker()
    service = _service(dense, sparse, _config(reranker=True), reranker=reranker)
    result = await service.retrieve("query")
    assert len(reranker.calls) == 1
    assert reranker.calls[0].top_k == 12
    assert all(item.ranks.rerank is not None for item in result.evidence)

    class FailingReranker(MockReranker):
        async def rerank(
            self, query: str, candidates: list[Evidence], *, top_k: int
        ) -> list[Evidence]:
            raise RuntimeError("fixture failure")

    fallback = await _service(
        dense, sparse, _config(reranker=True), reranker=FailingReranker()
    ).retrieve("query")
    assert fallback.degraded_stages == ["rerank"]
    assert fallback.stages[-1].status is RetrievalStageStatus.FAILED

    strict_config = _config(reranker=True).model_copy(
        update={
            "hybrid": _config(reranker=True).hybrid.model_copy(
                update={"allow_rerank_fallback": False}
            )
        }
    )
    with pytest.raises(KGCRAGError, match="reranker failed"):
        await _service(dense, sparse, strict_config, reranker=FailingReranker()).retrieve("query")


async def test_hybrid_publishes_atomically_and_cleans_failed_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(
        MockRetriever([_evidence("c1", "p1", "dense", 1.0, 1)]),
        MockRetriever([]),
        _config(reranker=False),
        workspace_root=tmp_path,
    )
    result = await service.retrieve("query", persist=True)
    published = tmp_path / service.config.hybrid.output_root / result.run_id / "result.json"
    assert published.is_file()
    payload = published.read_text(encoding="utf-8")
    assert "api_key" not in payload
    assert "prompt" not in payload.casefold()

    from kg_crag.retrieval import hybrid as hybrid_module

    monkeypatch.setattr(
        hybrid_module.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError())
    )
    failed = result.model_copy(update={"run_id": "failed-run"})
    with pytest.raises(OSError):
        publish_hybrid_result(
            failed,
            workspace_root=tmp_path,
            output_root=service.config.hybrid.output_root,
        )
    assert not list((tmp_path / service.config.hybrid.output_root).glob(".failed-run.tmp-*"))
