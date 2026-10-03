"""SQLite Sparse、融合、Mock 重排到评测报告的零网络闭环。"""

from pathlib import Path

from kg_crag.evaluation import HybridEvaluationRunner, HybridStrategyOutput
from kg_crag.indexing.processed import ProcessedPaper
from kg_crag.indexing.sparse import SparseIndexPipeline
from kg_crag.models import (
    Chunk,
    Evidence,
    EvidenceRanks,
    EvidenceScores,
    EvidenceSourceType,
    IndexItemStatus,
    PilotQuestion,
    PilotQuestionSet,
)
from kg_crag.retrieval import (
    HybridRetrievalService,
    MockReranker,
    MockRetriever,
    SparseRetriever,
    dense_config_hash,
    load_hybrid_retrieval_config,
)
from kg_crag.sparse_store import SQLiteSparseStore, build_sqlite_index_identity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = "a" * 64


def _chunk() -> Chunk:
    return Chunk(
        chunk_id="c1",
        paper_id="p1",
        section="Methods",
        page_start=1,
        page_end=1,
        text="CAMEL role-playing agents coordinate scientific tasks.",
        token_count=7,
        content_hash="content-c1",
        processing_version="v1",
    )


def _dense_evidence() -> Evidence:
    return Evidence(
        evidence_id="dense:c1",
        content=_chunk().text,
        source_type=EvidenceSourceType.CHUNK,
        source_id="c1",
        paper_id="p1",
        scores=EvidenceScores(dense=0.9),
        ranks=EvidenceRanks(dense=1),
    )


async def test_offline_hybrid_pipeline_is_idempotent_degradable_and_resumable(
    tmp_path: Path,
) -> None:
    config = load_hybrid_retrieval_config(PROJECT_ROOT / "configs/retrieval.yaml")
    config = config.model_copy(
        update={
            "sparse": config.sparse.model_copy(
                update={"manifest_root": "sparse-runs", "index_root": "sparse-index"}
            ),
            "hybrid": config.hybrid.model_copy(update={"output_root": "hybrid-runs"}),
        }
    )
    identity = build_sqlite_index_identity(
        schema_version=config.sparse.schema_version,
        tokenizer=config.sparse.tokenizer,
        bm25_version=config.sparse.bm25_version,
        corpus_snapshot_hash=SNAPSHOT,
    )
    store = SQLiteSparseStore(tmp_path / "index.sqlite3", identity)
    paper = ProcessedPaper("p1", "v1", (_chunk(),), tmp_path)
    pipeline = SparseIndexPipeline(config, store, workspace_root=tmp_path)
    first = await pipeline.run([paper], dry_run=False)
    repeated = await pipeline.run([paper], dry_run=False)
    assert first.items[0].status is IndexItemStatus.SUCCEEDED
    assert repeated.items[0].status is IndexItemStatus.SKIPPED

    service = HybridRetrievalService(
        MockRetriever([_dense_evidence()]),
        SparseRetriever(store, config.sparse),
        config,
        collection_version="dense-v1",
        sparse_index_version=identity.index_version,
        corpus_snapshot_hash=SNAPSHOT,
        reranker=MockReranker(),
        workspace_root=tmp_path,
    )
    calls = 0

    async def strategy(question: str) -> HybridStrategyOutput:
        nonlocal calls
        calls += 1
        result = await service.retrieve(question)
        return HybridStrategyOutput(
            tuple(result.evidence),
            {stage.stage: stage.latency_ms for stage in result.stages},
            {stage.stage: stage.call_count for stage in result.stages},
        )

    questions = PilotQuestionSet(
        corpus_snapshot_hash=SNAPSHOT,
        questions=[
            PilotQuestion(
                question_id=f"camel-{index:02d}",
                question="What is the CAMEL role-playing method?",
                target_chunk_ids=["c1"],
                rationale="离线闭环目标。",
            )
            for index in range(20)
        ],
    )
    runner = HybridEvaluationRunner(
        runners={"fusion_rerank": strategy},
        config_hash=dense_config_hash(config),
        collection_version="dense-v1",
        sparse_index_version=identity.index_version,
        fusion_versions={"fusion_rerank": "fixture-fusion-v1"},
        reranker_version="mock-v1",
        k_values=[1, 5],
        results_root=tmp_path / "evaluation",
    )
    report = await runner.run(questions, limit=2)
    replay = await runner.run(questions, limit=2)
    assert report.metrics["fusion_rerank"].recall_at_k["1"] == 1.0
    assert report.metrics["fusion_rerank"].failure_count == 0
    assert report.items[0].stage_call_counts["rerank"] == 1
    assert replay.evaluation_version == report.evaluation_version
    assert calls == 2

    class FailingDense(MockRetriever):
        async def retrieve(
            self,
            query: str,
            *,
            top_k: int,
            filters: dict[str, str | int | bool] | None = None,
        ) -> list[Evidence]:
            raise RuntimeError("offline dense failure")

    degraded = HybridRetrievalService(
        FailingDense([]),
        SparseRetriever(store, config.sparse),
        config,
        collection_version="dense-v1",
        sparse_index_version=identity.index_version,
        corpus_snapshot_hash=SNAPSHOT,
        reranker=MockReranker(),
    )
    degraded_result = await degraded.retrieve("CAMEL")
    assert degraded_result.degraded_stages == ["dense"]
    assert degraded_result.evidence

    class FailingReranker(MockReranker):
        async def rerank(
            self, query: str, candidates: list[Evidence], *, top_k: int
        ) -> list[Evidence]:
            raise RuntimeError("offline reranker failure")

    fallback = HybridRetrievalService(
        MockRetriever([_dense_evidence()]),
        SparseRetriever(store, config.sparse),
        config,
        collection_version="dense-v1",
        sparse_index_version=identity.index_version,
        corpus_snapshot_hash=SNAPSHOT,
        reranker=FailingReranker(),
    )
    fallback_result = await fallback.retrieve("CAMEL")
    assert fallback_result.degraded_stages == ["rerank"]
    assert fallback_result.evidence
