"""显式启用的真实 BGE Cross-Encoder 资源 smoke。"""

import os
import time
import tracemalloc
from pathlib import Path

import pytest

from kg_crag.models import Evidence, EvidenceRanks, EvidenceSourceType
from kg_crag.retrieval import CrossEncoderReranker, RerankerConfig


@pytest.mark.skipif(
    os.getenv("KG_CRAG_RUN_RERANKER_TESTS") != "1",
    reason="set KG_CRAG_RUN_RERANKER_TESTS=1 to load the pinned local Cross-Encoder",
)
async def test_real_cross_encoder_smoke_is_bounded() -> None:
    cache_root = os.getenv("KG_CRAG_MODEL_CACHE_ROOT", "data/processed/model-cache")
    config = RerankerConfig(cache_root=cache_root, max_candidates=20, top_k=8, batch_size=4)
    reranker = CrossEncoderReranker(config, workspace_root=Path.cwd())
    candidates = [
        Evidence(
            evidence_id=f"hybrid:c{index}",
            content=f"Evidence about retrieval augmented generation number {index}.",
            source_type=EvidenceSourceType.CHUNK,
            source_id=f"c{index}",
            paper_id=f"p{index}",
            ranks=EvidenceRanks(fusion=index + 1),
        )
        for index in range(20)
    ]
    tracemalloc.start()
    started = time.perf_counter()
    results = await reranker.rerank("What is retrieval augmented generation?", candidates, top_k=8)
    latency_ms = (time.perf_counter() - started) * 1000
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert len(results) == 8
    assert all(item.scores.rerank is not None for item in results)
    print(
        {
            "model": config.model,
            "revision": config.revision,
            "device": config.device,
            "cache_root": cache_root,
            "latency_ms": latency_ms,
            "python_peak_bytes": peak_bytes,
        }
    )
