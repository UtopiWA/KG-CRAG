"""图检索指标和失败分母测试。"""

import pytest

from kg_crag.evaluation.graph import ndcg_at_k, recall_at_k, reciprocal_rank


def test_graph_metrics_match_hand_calculation() -> None:
    retrieved = ["x", "a", "b"]
    relevant = {"a", "b"}
    assert recall_at_k(retrieved, relevant, 2) == 0.5
    assert reciprocal_rank(retrieved, relevant) == 0.5
    assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(
        (1 / __import__("math").log2(3) + 1 / __import__("math").log2(4))
        / (1 + 1 / __import__("math").log2(3))
    )


def test_no_path_question_scores_only_an_empty_result() -> None:
    assert recall_at_k([], set(), 10) == 1.0
    assert ndcg_at_k([], set(), 10) == 1.0
    assert recall_at_k(["unexpected"], set(), 10) == 0.0
