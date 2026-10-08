"""Hybrid 配置的兼容性、资源边界与稳定哈希测试。"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from kg_crag.retrieval import (
    FusionConfig,
    RerankerConfig,
    SparseRetrievalConfig,
    dense_config_hash,
    load_dense_rag_config,
    load_hybrid_evaluation_config,
    load_hybrid_retrieval_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "retrieval.yaml"


def test_hybrid_config_preserves_dense_loader_and_pins_resources() -> None:
    dense = load_dense_rag_config(CONFIG_PATH)
    hybrid = load_hybrid_retrieval_config(CONFIG_PATH)

    assert hybrid.dense == dense
    assert hybrid.sparse.tokenizer == "scientific-unicode-v1"
    assert hybrid.reranker.model == "BAAI/bge-reranker-base"
    assert hybrid.reranker.device == "cpu"
    assert hybrid.reranker.max_candidates == 20
    assert hybrid.fusion.final_top_k == 16
    assert hybrid.reranker.top_k == 12
    assert hybrid.hybrid.max_output == 12
    evaluation = load_hybrid_evaluation_config(PROJECT_ROOT / "configs" / "evaluation.yaml")
    assert evaluation.strategies == ["dense", "sparse", "rrf", "weighted", "fusion_rerank"]
    assert evaluation.smoke_questions == 5


def test_hybrid_config_rejects_invalid_boundaries() -> None:
    with pytest.raises(ValidationError, match="sum to 1.0"):
        FusionConfig(dense_weight=0.8, sparse_weight=0.3)
    with pytest.raises(ValidationError, match="less than or equal to 20"):
        RerankerConfig(max_candidates=21)
    with pytest.raises(ValidationError, match="workspace-relative"):
        SparseRetrievalConfig(index_root="../outside")
    with pytest.raises(ValidationError, match="extra"):
        SparseRetrievalConfig.model_validate({"unknown": True})


def test_hybrid_hash_tracks_algorithm_but_ignores_machine_cache_path() -> None:
    config = load_hybrid_retrieval_config(CONFIG_PATH)
    changes = [
        config.model_copy(
            update={"fusion": config.fusion.model_copy(update={"method": "weighted"})}
        ),
        config.model_copy(
            update={
                "sparse": config.sparse.model_copy(update={"tokenizer": "scientific-unicode-v2"})
            }
        ),
        config.model_copy(
            update={"reranker": config.reranker.model_copy(update={"revision": "new-revision"})}
        ),
        config.model_copy(update={"hybrid": config.hybrid.model_copy(update={"max_output": 7})}),
    ]
    moved_cache = config.model_copy(
        update={"reranker": config.reranker.model_copy(update={"cache_root": "D:/models/cache"})}
    )

    assert all(dense_config_hash(changed) != dense_config_hash(config) for changed in changes)
    assert dense_config_hash(moved_cache) == dense_config_hash(config)
