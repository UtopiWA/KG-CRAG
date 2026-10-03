"""检索契约。"""

from kg_crag.retrieval.base import Reranker, Retriever
from kg_crag.retrieval.config import (
    DenseEvaluationConfig,
    DenseRAGConfig,
    FusionConfig,
    HybridEvaluationConfig,
    HybridRetrievalConfig,
    HybridRuntimeConfig,
    RerankerConfig,
    SparseRetrievalConfig,
    dense_config_hash,
    load_dense_evaluation_config,
    load_dense_rag_config,
    load_hybrid_evaluation_config,
    load_hybrid_retrieval_config,
)
from kg_crag.retrieval.cross_encoder import CrossEncoderReranker
from kg_crag.retrieval.fusion import (
    deduplicate_evidence_by_paper,
    fuse_evidence,
    fusion_version,
)
from kg_crag.retrieval.hybrid import HybridRetrievalService, publish_hybrid_result
from kg_crag.retrieval.mock import MockReranker, MockRetriever, RerankCall, RetrievalCall
from kg_crag.retrieval.sparse import SparseRetriever

__all__ = [
    "CrossEncoderReranker",
    "DenseEvaluationConfig",
    "DenseRAGConfig",
    "FusionConfig",
    "HybridEvaluationConfig",
    "HybridRetrievalConfig",
    "HybridRetrievalService",
    "HybridRuntimeConfig",
    "MockReranker",
    "MockRetriever",
    "RerankCall",
    "Reranker",
    "RerankerConfig",
    "RetrievalCall",
    "Retriever",
    "SparseRetrievalConfig",
    "SparseRetriever",
    "deduplicate_evidence_by_paper",
    "dense_config_hash",
    "fuse_evidence",
    "fusion_version",
    "load_dense_evaluation_config",
    "load_dense_rag_config",
    "load_hybrid_evaluation_config",
    "load_hybrid_retrieval_config",
    "publish_hybrid_result",
]
