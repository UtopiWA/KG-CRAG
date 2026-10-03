"""共享证据约束和显式 Hybrid 回答生成测试。"""

from pathlib import Path

import pytest

from kg_crag.errors import KGCRAGError
from kg_crag.generation import HybridQueryService, VersionedPrompt
from kg_crag.models import Evidence, EvidenceRanks, EvidenceScores, EvidenceSourceType
from kg_crag.providers import MockLLMProvider
from kg_crag.retrieval import (
    HybridRetrievalConfig,
    HybridRetrievalService,
    MockRetriever,
    load_hybrid_retrieval_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _evidence() -> Evidence:
    return Evidence(
        evidence_id="dense:c1",
        content="CAMEL uses role-playing agents.",
        source_type=EvidenceSourceType.CHUNK,
        source_id="c1",
        paper_id="p1",
        scores=EvidenceScores(dense=0.9),
        ranks=EvidenceRanks(dense=1),
    )


def _config(tmp_path: Path) -> HybridRetrievalConfig:
    config = load_hybrid_retrieval_config(PROJECT_ROOT / "configs/retrieval.yaml")
    return config.model_copy(
        update={
            "reranker": config.reranker.model_copy(update={"enabled": False}),
            "hybrid": config.hybrid.model_copy(update={"output_root": "hybrid-runs"}),
            "dense": config.dense.model_copy(
                update={
                    "generation": config.dense.generation.model_copy(
                        update={"min_score": 0.0, "min_evidence": 1}
                    )
                }
            ),
        }
    )


def _service(
    tmp_path: Path,
    llm: MockLLMProvider,
    *,
    evidence: list[Evidence] | None = None,
) -> HybridQueryService:
    config = _config(tmp_path)
    retrieval = HybridRetrievalService(
        MockRetriever(evidence if evidence is not None else [_evidence()]),
        MockRetriever([]),
        config,
        collection_version="dense-v1",
        sparse_index_version="sparse-v1",
        corpus_snapshot_hash="a" * 64,
        workspace_root=tmp_path,
    )
    return HybridQueryService(
        retrieval,
        llm,
        config,
        VersionedPrompt("Q={question}\nE={evidence_context}", "b" * 64),
        workspace_root=tmp_path,
    )


async def test_answer_is_disabled_by_default_and_persists_without_llm(tmp_path: Path) -> None:
    llm = MockLLMProvider()
    result = await _service(tmp_path, llm).ask("What is CAMEL?", persist=True)

    assert result.with_answer is False
    assert result.llm_calls == 0
    assert result.retrieval_path == ["dense", "sparse"]
    assert llm.calls == []
    path = tmp_path / "hybrid-runs" / result.run_id / "query-result.json"
    assert path.is_file()


async def test_insufficient_evidence_returns_fixed_answer_without_llm(tmp_path: Path) -> None:
    llm = MockLLMProvider()
    result = await _service(tmp_path, llm, evidence=[]).ask("What is CAMEL?", with_answer=True)

    assert result.insufficient_evidence is True
    assert result.llm_calls == 0
    assert llm.calls == []


async def test_answer_uses_one_call_and_validates_citations(tmp_path: Path) -> None:
    llm = MockLLMProvider(
        '{"claims":[{"text":"Role play","claim_type":"fact",'
        '"citation_ids":["E1"]}],"confidence":0.8}'
    )
    result = await _service(tmp_path, llm).ask("What is CAMEL?", with_answer=True)

    assert result.answer == "Role play [E1]"
    assert result.retrieval.fusion_version
    assert result.citations[0].source_id == "c1"
    assert result.llm_calls == 1
    assert len(llm.calls) == 1

    unknown = MockLLMProvider(
        '{"claims":[{"text":"Bad","claim_type":"fact","citation_ids":["E9"]}],"confidence":0.1}'
    )
    with pytest.raises(KGCRAGError, match="unknown citation"):
        await _service(tmp_path, unknown).ask("What is CAMEL?", with_answer=True)


async def test_provider_failure_is_not_retried(tmp_path: Path) -> None:
    class FailingProvider(MockLLMProvider):
        async def generate(
            self,
            prompt: str,
            *,
            system_prompt: str | None = None,
            temperature: float = 0.0,
        ) -> str:
            self.calls.append(
                {
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "temperature": temperature,
                }
            )
            raise RuntimeError("provider unavailable")

    provider = FailingProvider()
    with pytest.raises(RuntimeError, match="provider unavailable"):
        await _service(tmp_path, provider).ask("What is CAMEL?", with_answer=True)
    assert len(provider.calls) == 1
