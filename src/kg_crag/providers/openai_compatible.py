"""OpenAI-compatible LLM 适配器。"""

from __future__ import annotations

from typing import Literal, Protocol, cast

from kg_crag.errors import KGCRAGError
from kg_crag.models import ErrorCode, ErrorDetail
from kg_crag.providers.base import LLMGeneration


class _Message(Protocol):
    content: str | None
    reasoning_content: str | None


class _Choice(Protocol):
    message: _Message
    finish_reason: str | None


class _Response(Protocol):
    choices: list[_Choice]
    usage: object | None


class _Completions(Protocol):
    async def create(self, **kwargs: object) -> _Response: ...


class _Chat(Protocol):
    completions: _Completions


class _OpenAIClient(Protocol):
    chat: _Chat


class EmptyResponseError(ValueError):
    """保留空响应的有界元数据，不携带思考或正文内容。"""

    def __init__(
        self,
        *,
        finish_reason: object,
        analysis_chars: int,
        input_tokens: object,
        output_tokens: object,
    ) -> None:
        super().__init__("empty response")
        self.finish_reason = finish_reason
        self.analysis_chars = analysis_chars
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class OpenAICompatibleLLMProvider:
    """隔离 SDK 类型，并把外部异常转换为安全错误。"""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None,
        base_url: str | None = None,
        timeout_seconds: float = 60.0,
        max_output_tokens: int | None = None,
        reasoning_effort: Literal["provider-default", "low", "high", "max"] = ("provider-default"),
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
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self._client: _OpenAIClient | None = cast(_OpenAIClient | None, client)
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = cast(
                _OpenAIClient,
                AsyncOpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout_seconds,
                    # 重试由回答工作流统一控制，确保预算与 Trace 都能观察到每次调用。
                    max_retries=0,
                ),
            )

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        result = await self.generate_with_usage(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
        )
        return result.content

    async def generate_with_usage(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> LLMGeneration:
        """返回正文和响应自带用量；缺失 usage 时不伪造实际值。"""

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        client = self._client
        if client is None:  # pragma: no cover - constructor always initializes it
            raise RuntimeError("LLM client is not initialized")
        try:
            arguments: dict[str, object] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }
            if self.max_output_tokens is not None:
                arguments["max_tokens"] = self.max_output_tokens
            if self.reasoning_effort != "provider-default":
                arguments["reasoning_effort"] = self.reasoning_effort
            response = await client.chat.completions.create(**arguments)
            choice = response.choices[0]
            content = choice.message.content
            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "prompt_tokens", None)
            output_tokens = getattr(usage, "completion_tokens", None)
            if not isinstance(content, str) or not content.strip():
                analysis = getattr(choice.message, "reasoning_content", None)
                raise EmptyResponseError(
                    finish_reason=getattr(choice, "finish_reason", None),
                    analysis_chars=len(analysis) if isinstance(analysis, str) else 0,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            valid_usage = (
                isinstance(input_tokens, int)
                and not isinstance(input_tokens, bool)
                and input_tokens >= 0
                and isinstance(output_tokens, int)
                and not isinstance(output_tokens, bool)
                and output_tokens >= 0
            )
            return LLMGeneration(
                content=content,
                input_tokens=input_tokens if valid_usage else None,
                output_tokens=output_tokens if valid_usage else None,
            )
        except Exception as error:
            status = getattr(error, "status_code", None)
            error_type = type(error).__name__
            timed_out = "timeout" in error_type.casefold()
            empty_error = error if isinstance(error, EmptyResponseError) else None
            empty_response = empty_error is not None
            finish_reason = empty_error.finish_reason if empty_error is not None else None
            retryable = bool(
                status == 429
                or (isinstance(status, int) and status >= 500)
                or timed_out
                or "ratelimit" in type(error).__name__.casefold()
                or (empty_response and finish_reason != "length")
            )
            raise KGCRAGError(
                ErrorDetail(
                    code=ErrorCode.TIMEOUT if timed_out else ErrorCode.EXTERNAL_SERVICE,
                    message="LLM provider request failed",
                    retryable=retryable,
                    context={
                        "provider": "openai-compatible",
                        "status": status,
                        "error_type": error_type,
                        "finish_reason": (
                            finish_reason if isinstance(finish_reason, str) else None
                        ),
                        "analysis_chars": (
                            empty_error.analysis_chars if empty_error is not None else None
                        ),
                        "input_tokens": (
                            empty_error.input_tokens if empty_error is not None else None
                        ),
                        "output_tokens": (
                            empty_error.output_tokens if empty_error is not None else None
                        ),
                    },
                )
            ) from error
