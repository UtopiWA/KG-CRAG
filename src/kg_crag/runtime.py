"""Dense RAG 命令行共享的运行时装配。"""

from __future__ import annotations

from pathlib import Path

from kg_crag.generation import DenseRAGService, load_prompt
from kg_crag.providers import (
    EmbeddingService,
    LLMProvider,
    MockLLMProvider,
    OpenAICompatibleLLMProvider,
    SentenceTransformerEmbeddingProvider,
)
from kg_crag.retrieval import DenseRAGConfig
from kg_crag.retrieval.dense import DenseRetriever
from kg_crag.settings import Settings
from kg_crag.vector_store import QdrantVectorStore, collection_identity


def build_embedding_service(
    config: DenseRAGConfig,
    *,
    workspace_root: Path,
) -> EmbeddingService:
    """按锁定的模型与 revision 创建惰性 Embedding 服务。"""

    if config.embedding.provider != "sentence-transformers":
        raise ValueError(f"unsupported embedding provider: {config.embedding.provider}")
    provider = SentenceTransformerEmbeddingProvider(
        config.embedding.model,
        config.embedding.revision,
        dimensions=config.embedding.dimensions,
        normalize=config.embedding.normalize,
    )
    return EmbeddingService(provider, config.embedding, workspace_root=workspace_root)


def build_vector_store(config: DenseRAGConfig, settings: Settings) -> QdrantVectorStore:
    identity = collection_identity(config)
    return QdrantVectorStore(
        identity,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        allowed_filters=set(config.collection.allowed_filters),
    )


def build_dense_rag_service(
    config: DenseRAGConfig,
    settings: Settings,
    *,
    workspace_root: Path,
    store: QdrantVectorStore,
    embedding: EmbeddingService,
) -> DenseRAGService:
    """装配一次检索、至多一次生成的在线服务。"""

    if settings.llm_provider == "mock":
        llm: LLMProvider = MockLLMProvider(
            '{"claims":[{"text":"离线 Mock 回答",'
            '"claim_type":"fact","citation_ids":["E1"]}],"confidence":0.5}'
        )
    elif settings.llm_provider == "openai-compatible":
        llm = OpenAICompatibleLLMProvider(
            settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    else:
        raise ValueError(f"unsupported LLM provider: {settings.llm_provider}")
    prompt = load_prompt(workspace_root / config.generation.prompt_path)
    return DenseRAGService(
        DenseRetriever(embedding, store, config),
        llm,
        config,
        prompt,
        collection_version=store.collection_version,
        workspace_root=workspace_root,
    )
