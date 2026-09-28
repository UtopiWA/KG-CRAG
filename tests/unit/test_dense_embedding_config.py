"""Dense 配置、Embedding 校验与缓存行为。"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from kg_crag.errors import KGCRAGError
from kg_crag.generation import load_prompt
from kg_crag.providers import EmbeddingService
from kg_crag.retrieval.config import (
    DenseEmbeddingConfig,
    dense_config_hash,
    load_dense_rag_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class StaticEmbedding:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors
        self.calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return self.vectors[: len(texts)]


def _config(cache_root: str = "cache") -> DenseEmbeddingConfig:
    return DenseEmbeddingConfig(
        provider="test",
        model="fixed",
        revision="r1",
        dimensions=2,
        batch_size=2,
        max_batch_size=2,
        cache_root=cache_root,
    )


def test_dense_config_is_strict_and_pinned() -> None:
    config = load_dense_rag_config(PROJECT_ROOT / "configs/retrieval.yaml")
    assert config.embedding.revision == "5617a9f61b028005a4858fdac845db406aefb181"
    with pytest.raises(ValidationError, match="extra"):
        DenseEmbeddingConfig.model_validate({**config.embedding.model_dump(), "typo": 1})
    with pytest.raises(ValidationError, match="batch_size"):
        _config().model_copy(update={"batch_size": 3}).model_validate(
            {**_config().model_dump(), "batch_size": 3}
        )


def test_config_hash_tracks_retrieval_but_not_credentials() -> None:
    config = load_dense_rag_config(PROJECT_ROOT / "configs/retrieval.yaml")
    changed = config.model_copy(
        update={"dense": config.dense.model_copy(update={"top_k": config.dense.top_k + 1})}
    )
    assert dense_config_hash(changed) != dense_config_hash(config)
    assert "api_key" not in config.model_dump_json()

    with pytest.raises(ValidationError, match="max_candidates"):
        config.dense.__class__.model_validate(
            {**config.dense.model_dump(), "max_top_k": 100, "max_candidates": 99}
        )
    with pytest.raises(ValidationError, match="max_context_chars"):
        config.generation.__class__.model_validate(
            {
                **config.generation.model_dump(),
                "max_chars_per_evidence": 1000,
                "max_context_chars": 999,
            }
        )


def test_prompt_placeholders_and_version_are_stable(tmp_path: Path) -> None:
    path = tmp_path / "prompt.txt"
    path.write_text("Q={question}\nE={evidence_context}", encoding="utf-8")
    first = load_prompt(path)
    assert first.render(question=" q ", evidence_context="[E1] evidence") == (
        "Q=q\nE=[E1] evidence"
    )
    path.write_text("Q={question}\nEvidence={evidence_context}", encoding="utf-8")
    assert load_prompt(path).version != first.version
    path.write_text("Q={question}", encoding="utf-8")
    with pytest.raises(ValueError, match="placeholders"):
        load_prompt(path)


async def test_embedding_cache_is_reused_and_normalized(tmp_path: Path) -> None:
    provider = StaticEmbedding([[3.0, 4.0], [0.0, 2.0]])
    service = EmbeddingService(provider, _config(), workspace_root=tmp_path)
    first = await service.embed([" a ", "b"], input_type="document")
    second = await service.embed(["a", "b"], input_type="document")

    assert first.cache_misses == 2
    assert second.cache_hits == 2
    assert provider.calls == 1
    assert first.vectors[0] == pytest.approx([0.6, 0.8])
    assert first.vectors[1] == pytest.approx([0.0, 1.0])


async def test_invalid_embedding_batch_publishes_no_cache(tmp_path: Path) -> None:
    provider = StaticEmbedding([[1.0]])
    service = EmbeddingService(provider, _config(), workspace_root=tmp_path)

    with pytest.raises(KGCRAGError, match="dimension"):
        await service.embed(["a"], input_type="document")
    assert not service.cache_root.exists()
