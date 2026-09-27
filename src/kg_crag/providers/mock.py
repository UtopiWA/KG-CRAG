"""供单元测试和工作流测试使用的确定性 Provider。"""

import hashlib


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
