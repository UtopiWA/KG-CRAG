"""模型与搜索 Provider 接口；具体 SDK 不进入业务模块。"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from kg_crag.models.answer import SearchResult


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


@dataclass(frozen=True)
class LLMGeneration:
    """一次模型响应及 Provider 明确返回的实际 Token 用量。"""

    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def has_actual_usage(self) -> bool:
        return self.input_tokens is not None and self.output_tokens is not None


@runtime_checkable
class UsageAwareLLMProvider(Protocol):
    """可在并发调用中直接返回逐请求用量，避免读取共享的 last_usage。"""

    async def generate_with_usage(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> LLMGeneration: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """按输入顺序为一批文本生成向量。"""

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class SearchProvider(Protocol):
    """执行一次有界搜索，并返回不超过请求数量的严格结果。"""

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]: ...
