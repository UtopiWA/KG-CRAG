"""供单元测试和工作流测试使用的确定性 Provider。"""

import hashlib
import json
from pathlib import Path

from pydantic import TypeAdapter

from kg_crag.models.answer import SearchResult


class MockLLMProvider:
    """无需联网即可返回预先配置的响应。"""

    def __init__(self, response: str = "mock response") -> None:
        self.response = response
        self.calls: list[dict[str, str | float | None]] = []

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        """记录调用并返回确定性响应。"""

        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
            }
        )
        return self.response


class MockEmbeddingProvider:
    """根据 SHA-256 字节生成稳定且近似归一化的向量。"""

    def __init__(self, dimensions: int = 8) -> None:
        if dimensions < 1 or dimensions > 32:
            raise ValueError("dimensions must be between 1 and 32")
        self.dimensions = dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """无需下载模型，将每个字符串映射为确定性向量。"""

        vectors: list[list[float]] = []
        for value in texts:
            digest = hashlib.sha256(value.encode("utf-8")).digest()
            vectors.append([digest[index] / 255.0 for index in range(self.dimensions)])
        return vectors


class MockSearchProvider:
    """返回预置结果并记录调用，不进行网络访问。"""

    def __init__(self, results: list[SearchResult] | None = None) -> None:
        self.results = [item.model_copy(deep=True) for item in results or []]
        self.calls: list[dict[str, str | int]] = []

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        normalized = " ".join(query.split())
        if not normalized or len(normalized) > 2000:
            raise ValueError("search query must be non-empty and bounded")
        if max_results < 1 or max_results > 5:
            raise ValueError("search max_results must be between 1 and 5")
        self.calls.append({"query": normalized, "max_results": max_results})
        return [item.model_copy(deep=True) for item in self.results[:max_results]]


class RecordedSearchProvider:
    """从版本化 JSON fixture 读取响应，默认测试绝不联网。"""

    def __init__(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("recorded search fixture must be a mapping")
        adapter = TypeAdapter(list[SearchResult])
        self._responses = {
            str(query): adapter.validate_python(results) for query, results in payload.items()
        }
        self.calls: list[dict[str, str | int]] = []

    @property
    def response_count(self) -> int:
        """返回 fixture 中已经严格解析的查询数量。"""

        return len(self._responses)

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        normalized = " ".join(query.split())
        if not normalized or len(normalized) > 2000:
            raise ValueError("search query must be non-empty and bounded")
        if max_results < 1 or max_results > 5:
            raise ValueError("search max_results must be between 1 and 5")
        self.calls.append({"query": normalized, "max_results": max_results})
        return [
            item.model_copy(deep=True) for item in self._responses.get(normalized, [])[:max_results]
        ]
