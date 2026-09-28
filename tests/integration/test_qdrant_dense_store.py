"""显式启用的真实 Qdrant Dense Store 集成测试。"""

import os
import uuid
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient

from kg_crag.generation import DenseRAGService, VersionedPrompt
from kg_crag.indexing import DenseIndexPipeline, ProcessedPaper
from kg_crag.models import Chunk
from kg_crag.providers import EmbeddingService, MockLLMProvider
from kg_crag.retrieval.config import load_dense_rag_config
from kg_crag.retrieval.dense import DenseRetriever
from kg_crag.vector_store import QdrantVectorStore, collection_identity

PROJECT_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    os.getenv("KG_CRAG_RUN_QDRANT_TESTS") != "1",
    reason="set KG_CRAG_RUN_QDRANT_TESTS=1 to connect to an explicit test Qdrant",
)


def _chunk(chunk_id: str, paper_id: str, text: str, content_hash: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        section="Methods",
        page_start=1,
        page_end=1,
        text=text,
        token_count=1,
        content_hash=content_hash,
        processing_version="test-v1",
    )


class FixedEmbedding:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, float(index % 2)] for index, _ in enumerate(texts)]


async def test_real_qdrant_schema_upsert_filter_search_and_exact_cleanup(
    tmp_path: Path,
) -> None:
    url = os.getenv("KG_CRAG_QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("KG_CRAG_QDRANT_API_KEY")
    config = load_dense_rag_config(PROJECT_ROOT / "configs/retrieval.yaml")
    config = config.model_copy(
        update={
            "embedding": config.embedding.model_copy(
                update={
                    "provider": "test",
                    "model": "fixed",
                    "revision": "test-r1",
                    "dimensions": 2,
                    "cache_root": "embedding-cache",
                }
            ),
            "collection": config.collection.model_copy(
                update={"prefix": f"qtest_{uuid.uuid4().hex[:12]}"}
            ),
            "indexing": config.indexing.model_copy(update={"manifest_root": "index-runs"}),
            "generation": config.generation.model_copy(update={"output_root": "query-runs"}),
        }
    )
    identity = collection_identity(config)
    client = AsyncQdrantClient(url=url, api_key=api_key)
    store = QdrantVectorStore(identity, client=client)
    try:
        await store.ensure_collection()
        first = _chunk("c1", "p1", "old", "hash-old")
        replacement = _chunk("c1", "p1", "new", "hash-new")
        same_paper = _chunk("c2", "p1", "same paper", "hash-same")
        other = _chunk("c3", "p2", "other", "hash-other")
        embedding = EmbeddingService(FixedEmbedding(), config.embedding, workspace_root=tmp_path)
        pipeline = DenseIndexPipeline(config, embedding, store, workspace_root=tmp_path)
        papers = [
            ProcessedPaper("p1", "test-v1", (first, same_paper), tmp_path),
            ProcessedPaper("p2", "test-v1", (other,), tmp_path),
        ]

        dry_run = await pipeline.run(papers, dry_run=True)
        assert sum(item.added for item in dry_run.items) == 3
        initial = await pipeline.run(papers, dry_run=False)
        repeated = await pipeline.run(papers, dry_run=False)
        assert all(item.status.value == "succeeded" for item in initial.items)
        assert all(item.status.value == "skipped" for item in repeated.items)

        updated_papers = [
            ProcessedPaper("p1", "test-v1", (replacement, same_paper), tmp_path),
            papers[1],
        ]
        updated = await pipeline.run(updated_papers, dry_run=False)
        assert updated.items[0].updated == 1

        filtered = await store.search([1.0, 0.0], top_k=5, filters={"paper_id": "p1"})
        assert {item.source_id for item in filtered} == {"c1", "c2"}
        assert next(item for item in filtered if item.source_id == "c1").content == "new"
        assert (await store.record_state("p1"))["c1"].content_hash == "hash-new"

        retriever = DenseRetriever(embedding, store, config)
        deduplicated = await retriever.retrieve(
            "query",
            top_k=2,
            deduplicate_by_paper=True,
        )
        assert {item.paper_id for item in deduplicated} == {"p1", "p2"}
        service = DenseRAGService(
            retriever,
            MockLLMProvider(
                '{"claims":[{"text":"answer","claim_type":"fact",'
                '"citation_ids":["E1"]}],"confidence":0.8}'
            ),
            config,
            VersionedPrompt("Q={question}\nE={evidence_context}", "c" * 64),
            collection_version=identity.collection_version,
            workspace_root=tmp_path,
        )
        answer = await service.ask("query", top_k=2, persist=False)
        assert answer.citations[0].source_id in {"c1", "c2", "c3"}
        assert answer.used_external is False

        assert await store.delete_stale("p2", set()) == 1
        assert await store.record_state("p2") == {}
        await store.ensure_collection(rebuild=True)
        assert await store.record_state("p1") == {}
    finally:
        if await client.collection_exists(identity.collection_name):
            await client.delete_collection(identity.collection_name)
        await client.close()
