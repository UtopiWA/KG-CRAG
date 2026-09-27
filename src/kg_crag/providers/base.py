"""模型 Provider 接口；具体 SDK 不进入业务模块。"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """根据版本化系统 Prompt 和用户输入生成文本。"""

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> str: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """按输入顺序为一批文本生成向量。"""

    async def embed(self, texts: list[str]) -> list[list[float]]: ...
