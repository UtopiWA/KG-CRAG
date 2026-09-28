"""Dense 索引、检索与引用回答的离线闭环测试。"""

from pathlib import Path

import pytest

from kg_crag.errors import KGCRAGError
from kg_crag.generation import DenseRAGService, VersionedPrompt
from kg_crag.indexing import DenseIndexPipeline, ProcessedPaper
from kg_crag.models import Chunk, IndexItemStatus
from kg_crag.providers import EmbeddingService, MockLLMProvider
from kg_crag.retrieval.config import DenseRAGConfig, load_dense_rag_config
from kg_crag.retrieval.dense import DenseRetriever
from kg_crag.vector_store import InMemoryVectorStore, collection_identity

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FixedEmbedding:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[1.0, 0.0] for _ in texts]


def _config(tmp_path: Path) -> DenseRAGConfig:
    config = load_dense_rag_config(PROJECT_ROOT / "configs/retrieval.yaml")
    return config.model_copy(
        update={
            "embedding": config.embedding.model_copy(
                update={
                    "provider": "test",
                    "model": "fixed",
                    "revision": "r1",
                    "dimensions": 2,
                    "cache_root": "embedding-cache",
                }
            ),
            "indexing": config.indexing.model_copy(
                update={"manifest_root": "index-runs", "upsert_batch_size": 1}
            ),
            "generation": config.generation.model_copy(
                update={"output_root": "query-runs", "max_context_chars": 1000}
            ),
        }
    )


def _chunk(chunk_id: str, paper_id: str, content_hash: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        section="Methods",
        page_start=1,
        page_end=1,
        text=f"Evidence for {paper_id}",
        token_count=3,
        content_hash=content_hash,
        processing_version="pv1",
    )


async def test_incremental_pipeline_is_idempotent_and_persists_manifest(tmp_path: Path) -> None:
    config = _config(tmp_path)
    provider = FixedEmbedding()
    embedding = EmbeddingService(provider, config.embedding, workspace_root=tmp_path)
    store = InMemoryVectorStore(collection_version=collection_identity(config).collection_version)
    pipeline = DenseIndexPipeline(config, embedding, store, workspace_root=tmp_path)
    papers = [ProcessedPaper("paper-1", "pv1", (_chunk("c1", "paper-1", "hash-one"),), tmp_path)]

    dry_run = await pipeline.run(papers, dry_run=True)
    assert dry_run.items[0].status is IndexItemStatus.PLANNED
    assert store.record_count == 0

    first = await pipeline.run(papers, dry_run=False)
    second = await pipeline.run(papers, dry_run=False)
    assert first.items[0].status is IndexItemStatus.SUCCEEDED
    assert second.items[0].status is IndexItemStatus.SKIPPED
    assert len(provider.calls) == 1
    assert (tmp_path / first.manifest_path).is_file()  # type: ignore[arg-type]


async def test_retriever_and_generation_use_one_call_and_validate_citations(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    provider = FixedEmbedding()
    embedding = EmbeddingService(provider, config.embedding, workspace_root=tmp_path)
    store = InMemoryVectorStore(collection_version=collection_identity(config).collection_version)
    await store.upsert(
        [
            (_chunk("c1", "paper-1", "hash-one"), [1.0, 0.0]),
            (_chunk("c2", "paper-1", "hash-two"), [1.0, 0.0]),
            (_chunk("c3", "paper-2", "hash-three"), [1.0, 0.0]),
        ]
    )
    retriever = DenseRetriever(embedding, store, config)
    deduplicated = await retriever.retrieve("question", top_k=2, deduplicate_by_paper=True)
    assert [item.paper_id for item in deduplicated] == ["paper-1", "paper-2"]

    llm = MockLLMProvider(
        '{"claims":[{"text":"Supported", "claim_type":"fact",'
        '"citation_ids":["E1"]}],"confidence":0.8}'
    )
    prompt = VersionedPrompt("Q={question}\nE={evidence_context}", "a" * 64)
    service = DenseRAGService(
        retriever,
        llm,
        config,
        prompt,
        collection_version=store.collection_version,
        workspace_root=tmp_path,
    )
    result = await service.ask("question", top_k=1, persist=True)
    assert result.answer == "Supported [E1]"
    assert result.citations[0].source_id == "c1"
    assert len(llm.calls) == 1
    assert (tmp_path / "query-runs" / result.run_id / "result.json").is_file()

    invalid = MockLLMProvider(
        '{"claims":[{"text":"Bad", "claim_type":"fact","citation_ids":["E9"]}],"confidence":0.2}'
    )
    bad_service = DenseRAGService(
        retriever,
        invalid,
        config,
        prompt,
        collection_version=store.collection_version,
        workspace_root=tmp_path,
    )
    with pytest.raises(KGCRAGError, match="unknown citation"):
        await bad_service.ask("question", top_k=1, persist=False)
