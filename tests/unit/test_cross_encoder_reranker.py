"""Cross-Encoder 的惰性加载、整批校验和稳定排序测试。"""

from pathlib import Path

import pytest

from kg_crag.errors import KGCRAGError
from kg_crag.models import Evidence, EvidenceRanks, EvidenceSourceType
from kg_crag.retrieval import CrossEncoderReranker, RerankerConfig


def _evidence(source_id: str, fusion_rank: int) -> Evidence:
    return Evidence(
        evidence_id=f"hybrid:{source_id}",
        content=f"content {source_id}",
        source_type=EvidenceSourceType.CHUNK,
        source_id=source_id,
        paper_id=f"paper-{source_id}",
        ranks=EvidenceRanks(fusion=fusion_rank),
        metadata={"content_hash": f"hash-{source_id}"},
    )


class FakeCrossEncoder:
    def __init__(self, scores: object) -> None:
        self.scores = scores
        self.calls: list[tuple[tuple[tuple[str, str], ...], int]] = []

    def predict(
        self,
        sentences: list[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> object:
        self.calls.append((tuple(sentences), batch_size))
        return self.scores


async def test_cross_encoder_is_lazy_bounded_stable_and_preserves_input(tmp_path: Path) -> None:
    created: list[dict[str, object]] = []
    fake = FakeCrossEncoder([0.5, 0.9, 0.5])

    def factory(_model: str, **kwargs: object) -> FakeCrossEncoder:
        created.append(kwargs)
        return fake

    config = RerankerConfig(cache_root="cache", batch_size=3, max_candidates=3, top_k=3)
    reranker = CrossEncoderReranker(config, workspace_root=tmp_path, model_factory=factory)
    candidates = [_evidence("c2", 2), _evidence("c1", 1), _evidence("c3", 3)]
    original = [item.model_copy(deep=True) for item in candidates]

    assert reranker.model_loaded is False
    assert await reranker.rerank("query", [], top_k=1) == []
    assert created == []
    ranked = await reranker.rerank("query", candidates, top_k=3)

    assert reranker.model_loaded is True
    assert [item.source_id for item in ranked] == ["c1", "c2", "c3"]
    assert [item.ranks.rerank for item in ranked] == [1, 2, 3]
    assert candidates == original
    assert len(fake.calls[0][0]) == 3
    assert created[0]["revision"] == config.revision
    assert created[0]["device"] == "cpu"
    assert Path(str(created[0]["cache_dir"])).is_relative_to(tmp_path)


@pytest.mark.parametrize("scores", [[0.1], [0.1, float("nan")], [[0.1, 0.2], [0.3]]])
async def test_cross_encoder_rejects_incomplete_or_invalid_batch(
    tmp_path: Path, scores: object
) -> None:
    fake = FakeCrossEncoder(scores)
    reranker = CrossEncoderReranker(
        RerankerConfig(max_candidates=2, top_k=2),
        workspace_root=tmp_path,
        model_factory=lambda *_args, **_kwargs: fake,
    )
    with pytest.raises(KGCRAGError):
        await reranker.rerank("query", [_evidence("c1", 1), _evidence("c2", 2)], top_k=2)


async def test_cross_encoder_sanitizes_loading_and_inference_errors(tmp_path: Path) -> None:
    def broken_factory(*_args: object, **_kwargs: object) -> FakeCrossEncoder:
        raise OSError(f"secret path {tmp_path}")

    reranker = CrossEncoderReranker(
        RerankerConfig(), workspace_root=tmp_path, model_factory=broken_factory
    )
    with pytest.raises(KGCRAGError, match="could not be loaded") as captured:
        await reranker.rerank("private query", [_evidence("c1", 1)], top_k=1)
    rendered = str(captured.value)
    assert "private query" not in rendered
    assert str(tmp_path) not in rendered
