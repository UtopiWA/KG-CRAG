"""确定性离线 Provider 行为测试。"""

from kg_crag.providers import MockEmbeddingProvider, MockLLMProvider


async def test_mock_llm_records_calls() -> None:
    provider = MockLLMProvider(response="grounded answer")
    result = await provider.generate("question", system_prompt="cite evidence")
    assert result == "grounded answer"
    assert provider.calls[0]["prompt"] == "question"


async def test_mock_embedding_is_deterministic_and_ordered() -> None:
    provider = MockEmbeddingProvider(dimensions=4)
    first = await provider.embed(["alpha", "beta"])
    second = await provider.embed(["alpha", "beta"])
    assert first == second
    assert first[0] != first[1]
    assert all(len(vector) == 4 for vector in first)
