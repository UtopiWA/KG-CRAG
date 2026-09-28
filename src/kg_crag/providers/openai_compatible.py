"""OpenAI-compatible LLM 适配器。"""

from __future__ import annotations

from typing import Protocol, cast

from kg_crag.errors import KGCRAGError
from kg_crag.models import ErrorCode, ErrorDetail


class _Message(Protocol):
    content: str | None


class _Choice(Protocol):
    message: _Message


class _Response(Protocol):
    choices: list[_Choice]


class _Completions(Protocol):
    async def create(self, **kwargs: object) -> _Response: ...


class _Chat(Protocol):
    completions: _Completions


class _OpenAIClient(Protocol):
    chat: _Chat


class OpenAICompatibleLLMProvider:
    """隔离 SDK 类型，并把外部异常转换为安全错误。"""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None,
        base_url: str | None = None,
        timeout_seconds: float = 60.0,
        client: object | None = None,
    ) -> None:
        if client is None and not api_key:
            raise KGCRAGError(
                ErrorDetail(
                    code=ErrorCode.CONFIGURATION,
                    message="LLM API key is required for the OpenAI-compatible provider",
                    retryable=False,
                )
            )
        self.model = model
        self._client: _OpenAIClient | None = cast(_OpenAIClient | None, client)
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = cast(
                _OpenAIClient,
                AsyncOpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout_seconds,
                ),
            )

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        client = self._client
        if client is None:  # pragma: no cover - constructor always initializes it
            raise RuntimeError("LLM client is not initialized")
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )
            content = response.choices[0].message.content
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty response")
            return content
        except Exception as error:
            status = getattr(error, "status_code", None)
            retryable = bool(
                status == 429
                or (isinstance(status, int) and status >= 500)
                or "timeout" in type(error).__name__.casefold()
                or "ratelimit" in type(error).__name__.casefold()
            )
            raise KGCRAGError(
                ErrorDetail(
                    code=ErrorCode.EXTERNAL_SERVICE,
                    message="LLM provider request failed",
                    retryable=retryable,
                    context={"provider": "openai-compatible", "status": status},
                )
            ) from error
