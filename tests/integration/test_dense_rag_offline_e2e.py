"""从内存索引到基线报告的默认零网络闭环。"""

from pathlib import Path

from kg_crag.evaluation import DenseEvaluationRunner
from kg_crag.generation import DenseRAGService, VersionedPrompt
from kg_crag.indexing import DenseIndexPipeline, ProcessedPaper
from kg_crag.models import Chunk, DenseRAGResult, PilotQuestion, PilotQuestionSet
from kg_crag.providers import EmbeddingService, MockLLMProvider
from kg_crag.retrieval import dense_config_hash, load_dense_rag_config
from kg_crag.retrieval.dense import DenseRetriever
from kg_crag.vector_store import InMemoryVectorStore, collection_identity, corpus_snapshot_hash

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FixedEmbedding:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


async def test_dense_rag_offline_end_to_end_with_isolated_failure(tmp_path: Path) -> None:
    config = load_dense_rag_config(PROJECT_ROOT / "configs/retrieval.yaml")
    config = config.model_copy(
        update={
            "embedding": config.embedding.model_copy(
                update={
                    "provider": "test",
                    "model": "fixed",
                    "revision": "r1",
                    "dimensions": 2,
                    "cache_root": "cache",
                }
            ),
            "indexing": config.indexing.model_copy(update={"manifest_root": "index-runs"}),
            "generation": config.generation.model_copy(update={"output_root": "query-runs"}),
        }
    )
    chunk = Chunk(
        chunk_id="target-chunk",
        paper_id="paper-1",
        section="Abstract",
        page_start=1,
        page_end=1,
        text="The agent uses a deterministic retrieval method.",
        token_count=7,
        content_hash="target-hash",
        processing_version="pv1",
    )
    papers = [ProcessedPaper("paper-1", "pv1", (chunk,), tmp_path)]
    identity = collection_identity(config)
    store = InMemoryVectorStore(collection_version=identity.collection_version)
    embedding = EmbeddingService(FixedEmbedding(), config.embedding, workspace_root=tmp_path)
    pipeline = DenseIndexPipeline(config, embedding, store, workspace_root=tmp_path)
    first = await pipeline.run(papers, dry_run=False)
    second = await pipeline.run(papers, dry_run=False)
    assert first.items[0].status.value == "succeeded"
    assert second.items[0].status.value == "skipped"

    service = DenseRAGService(
        DenseRetriever(embedding, store, config),
        MockLLMProvider(
            '{"claims":[{"text":"Answer","claim_type":"fact",'
            '"citation_ids":["E1"]}],"confidence":0.9}'
        ),
        config,
        VersionedPrompt("Q={question}\nE={evidence_context}", "b" * 64),
        collection_version=identity.collection_version,
        workspace_root=tmp_path,
    )
    questions = PilotQuestionSet(
        corpus_snapshot_hash=corpus_snapshot_hash([chunk]),
        questions=[
            PilotQuestion(
                question_id=f"offline-{index}",
                question=f"What method is used in case {index}?",
                target_chunk_ids=["target-chunk"],
                rationale="fixed offline fixture",
            )
            for index in range(20)
        ],
    )

    async def query(question: str) -> DenseRAGResult:
        if question == "What method is used in case 7?":
            raise RuntimeError("isolated fixture failure")
        return await service.ask(question, top_k=1, persist=False)

    runner = DenseEvaluationRunner(
        query=query,
        config_hash=dense_config_hash(config),
        prompt_version=service.prompt.version,
        collection_version=identity.collection_version,
        k_values=[1],
        results_root=tmp_path / "results",
    )
    report = await runner.run(questions)
    assert report.metrics.failure_count == 1
    assert report.metrics.recall_at_k["1"] == 0.95
    assert (tmp_path / "results" / report.run_id / "report.json").is_file()

    empty_store = InMemoryVectorStore(collection_version=identity.collection_version)
    insufficient = DenseRAGService(
        DenseRetriever(embedding, empty_store, config),
        MockLLMProvider(),
        config,
        service.prompt,
        collection_version=identity.collection_version,
        workspace_root=tmp_path,
    )
    no_evidence = await insufficient.ask("missing evidence", top_k=1, persist=False)
    assert no_evidence.insufficient_evidence is True
