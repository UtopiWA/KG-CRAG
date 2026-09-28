"""真实 Provider 边界的离线替身测试。"""

from types import SimpleNamespace

import pytest

from kg_crag.errors import KGCRAGError
from kg_crag.providers import (
    OpenAICompatibleLLMProvider,
    SentenceTransformerEmbeddingProvider,
)


class FakeSentenceModel:
    def __init__(self) -> None:
        self.inputs: list[list[str]] = []

    def encode(
        self,
        sentences: list[str],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> list[list[float]]:
        self.inputs.append(sentences)
        return [[float(index), 1.0] for index, _ in enumerate(sentences)]


class FakeCompletions:
    def __init__(self, response: object = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


async def test_sentence_transformer_is_lazy_and_preserves_order() -> None:
    created: list[tuple[str, str]] = []
    model = FakeSentenceModel()

    def factory(name: str, *, revision: str) -> FakeSentenceModel:
        created.append((name, revision))
        return model

    provider = SentenceTransformerEmbeddingProvider(
        "model",
        "revision",
        dimensions=2,
        normalize=True,
        model_factory=factory,
    )
    assert created == []
    assert await provider.embed([]) == []
    assert created == []
    vectors = await provider.embed(["first", "second"])
    assert created == [("model", "revision")]
    assert vectors == [[0.0, 1.0], [1.0, 1.0]]
    assert model.inputs == [["first", "second"]]


async def test_sentence_transformer_maps_load_and_dimension_failures() -> None:
    def broken_factory(name: str, *, revision: str) -> FakeSentenceModel:
        raise RuntimeError(f"cannot load {name}@{revision}")

    broken = SentenceTransformerEmbeddingProvider(
        "secret-model",
        "revision",
        dimensions=2,
        normalize=True,
        model_factory=broken_factory,
    )
    with pytest.raises(KGCRAGError, match="could not be loaded") as load_error:
        await broken.embed(["private input"])
    assert "private input" not in str(load_error.value.detail)

    wrong = SentenceTransformerEmbeddingProvider(
        "model",
        "revision",
        dimensions=3,
        normalize=True,
        model_factory=lambda *args, **kwargs: FakeSentenceModel(),
    )
    with pytest.raises(KGCRAGError, match="dimension"):
        await wrong.embed(["input"])


async def test_openai_compatible_provider_success_and_safe_errors() -> None:
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))])
    completions = FakeCompletions(response=response)
    provider = OpenAICompatibleLLMProvider(
        "model",
        api_key=None,
        client=FakeClient(completions),
    )
    assert await provider.generate("private prompt") == "answer"
    assert completions.calls[0]["temperature"] == 0.0

    timeout = TimeoutError("leaked bearer token and private prompt")
    failing = OpenAICompatibleLLMProvider(
        "model",
        api_key=None,
        client=FakeClient(FakeCompletions(error=timeout)),
    )
    with pytest.raises(KGCRAGError, match="request failed") as error:
        await failing.generate("private prompt")
    rendered = str(error.value.detail)
    assert error.value.detail.retryable is True
    assert "bearer" not in rendered
    assert "private prompt" not in rendered


async def test_openai_compatible_provider_rejects_empty_response() -> None:
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=""))])
    provider = OpenAICompatibleLLMProvider(
        "model",
        api_key=None,
        client=FakeClient(FakeCompletions(response=response)),
    )
    with pytest.raises(KGCRAGError, match="request failed") as error:
        await provider.generate("prompt")
    assert error.value.detail.retryable is False
