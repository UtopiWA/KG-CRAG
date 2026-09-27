"""跨模块数据模式不变量测试。"""

import pytest
from pydantic import ValidationError

from kg_crag.models import (
    Chunk,
    QueryType,
    RetrievalStrategy,
    RetrievalWeights,
    RouteDecision,
)


def test_chunk_rejects_reversed_page_range() -> None:
    with pytest.raises(ValidationError, match="page_end"):
        Chunk(
            chunk_id="paper:methods:0",
            paper_id="paper",
            section="Methods",
            page_start=4,
            page_end=3,
            text="A test passage.",
            token_count=4,
            content_hash="abc12345",
        )


def test_retrieval_weights_must_sum_to_one() -> None:
    with pytest.raises(ValidationError, match="sum to 1.0"):
        RetrievalWeights(dense=0.4, sparse=0.4, graph=0.4)


def test_route_decision_accepts_normalized_weights() -> None:
    decision = RouteDecision(
        query_types=[QueryType.COMPARATIVE, QueryType.MULTI_HOP],
        strategy=RetrievalStrategy.HYBRID,
        weights=RetrievalWeights(dense=0.3, sparse=0.2, graph=0.5),
        reason="The question compares methods across papers.",
    )
    assert decision.weights.graph == pytest.approx(0.5)
