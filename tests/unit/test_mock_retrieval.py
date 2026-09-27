"""确定性离线 Retriever 与 Reranker 测试。"""

import pytest

from kg_crag.errors import KGCRAGError
from kg_crag.models import Evidence, EvidenceLocation, EvidenceSourceType
from kg_crag.retrieval import MockReranker, MockRetriever


def _evidence(
    evidence_id: str,
    *,
    paper_id: str,
    section: str,
    topic: str,
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        content=f"content for {evidence_id}",
        source_type=EvidenceSourceType.CHUNK,
        source_id=f"chunk-{evidence_id}",
        paper_id=paper_id,
        location=EvidenceLocation(section=section, page=2),
        metadata={"topic": topic},
    )


async def test_mock_retriever_filters_truncates_and_records_immutable_snapshot() -> None:
    candidates = [
        _evidence("e1", paper_id="p1", section="Methods", topic="rag"),
        _evidence("e2", paper_id="p1", section="Results", topic="rag"),
        _evidence("e3", paper_id="p2", section="Methods", topic="graph"),
    ]
    retriever = MockRetriever(candidates)

    first = await retriever.retrieve("query", top_k=1, filters={"topic": "rag"})
    second = await retriever.retrieve("query", top_k=1, filters={"topic": "rag"})

    assert [item.evidence_id for item in first] == ["e1"]
    assert first == second
    assert retriever.calls[0].query == "query"
    assert retriever.calls[0].filters == (("topic", "rag"),)
    assert await retriever.retrieve("query", top_k=3, filters={"unknown": "value"}) == []


async def test_mock_reranker_is_stable_and_preserves_provenance() -> None:
    candidates = [
        _evidence("e1", paper_id="p1", section="Methods", topic="rag"),
        _evidence("e2", paper_id="p2", section="Results", topic="graph"),
        _evidence("e3", paper_id="p3", section="Discussion", topic="agent"),
    ]
    reranker = MockReranker({"e1": 0.5, "e2": 0.9, "e3": 0.5})

    ranked = await reranker.rerank("query", candidates, top_k=2)

    assert [item.evidence_id for item in ranked] == ["e2", "e1"]
    assert ranked[0].source_id == candidates[1].source_id
    assert ranked[0].location == candidates[1].location
    assert ranked[0].metadata == candidates[1].metadata
    assert ranked[0].scores.rerank == pytest.approx(0.9)
    assert candidates[1].scores.rerank is None
    assert reranker.calls[0].candidate_ids == ("e1", "e2", "e3")


@pytest.mark.parametrize("implementation", ["retriever", "reranker"])
async def test_mock_retrieval_rejects_non_positive_top_k(implementation: str) -> None:
    candidate = _evidence("e1", paper_id="p1", section="Methods", topic="rag")
    with pytest.raises(KGCRAGError) as caught:
        if implementation == "retriever":
            await MockRetriever([candidate]).retrieve("query", top_k=0)
        else:
            await MockReranker().rerank("query", [candidate], top_k=-1)

    assert caught.value.detail.retryable is False
    assert caught.value.detail.code.value == "validation_error"
