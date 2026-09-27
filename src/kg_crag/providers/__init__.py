"""Provider 接口与测试替身。"""

from kg_crag.providers.base import EmbeddingProvider, LLMProvider
from kg_crag.providers.mock import MockEmbeddingProvider, MockLLMProvider

__all__ = [
    "EmbeddingProvider",
    "LLMProvider",
    "MockEmbeddingProvider",
    "MockLLMProvider",
]
