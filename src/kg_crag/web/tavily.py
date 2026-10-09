"""固定端点、单次请求且无隐藏重试的 Tavily Search 适配器。"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Protocol
from urllib.parse import urlsplit

import anyio

from kg_crag.correction.identity import stable_digest
from kg_crag.errors import KGCRAGError
from kg_crag.models import ErrorCode, ErrorDetail, SearchResult, WebSourceType

_TAVILY_ENDPOINT = "https://api.tavily.com/search"


class JSONSearchTransport(Protocol):
    async def post(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class UrllibTavilyTransport:
    def __init__(self, api_key: str, *, timeout_seconds: float) -> None:
        if not api_key.strip():
            raise ValueError("Tavily API key is required")
        self._api_key = api_key
        self._timeout = timeout_seconds

    async def post(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_payload = {**payload, "api_key": self._api_key}
        return await anyio.to_thread.run_sync(self._send, request_payload)

    def _send(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(  # noqa: S310 - 端点是模块内固定 HTTPS 常量
            _TAVILY_ENDPOINT,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:  # noqa: S310
                raw = response.read(1_000_001)
        except (TimeoutError, urllib.error.URLError) as error:
            raise _search_error("Tavily request failed", retryable=True) from error
        if len(raw) > 1_000_000:
            raise _search_error("Tavily response exceeds the size limit")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise _search_error("Tavily response is not valid JSON") from error
        if not isinstance(payload, dict):
            raise _search_error("Tavily response root must be an object")
        return payload


def _source_type(url: str) -> WebSourceType:
    host = (urlsplit(url).hostname or "").casefold()
    if host == "arxiv.org" or host.endswith(".arxiv.org"):
        return WebSourceType.ARXIV
    if any(value in host for value in ("openalex", "semanticscholar")):
        return WebSourceType.ACADEMIC_DATABASE
    if host == "github.com" or host.endswith(".github.io"):
        return WebSourceType.PROJECT_PAGE
    return WebSourceType.PAPER_SITE


class TavilySearchProvider:
    provider_version = "tavily-json-v3"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout_seconds: float = 15.0,
        max_excerpt_chars: int = 2000,
        include_domains: Sequence[str] | None = None,
        transport: JSONSearchTransport | None = None,
    ) -> None:
        if max_excerpt_chars < 100 or max_excerpt_chars > 2000:
            raise ValueError("Tavily excerpt limit must be between 100 and 2000")
        self._transport = transport or UrllibTavilyTransport(
            api_key or "", timeout_seconds=timeout_seconds
        )
        self._max_excerpt_chars = max_excerpt_chars
        self._include_domains = tuple(include_domains or ())
        self.calls = 0

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        normalized = " ".join(query.split())
        if not normalized or len(normalized) > 2000:
            raise ValueError("search query must be non-empty and bounded")
        if max_results < 1 or max_results > 5:
            raise ValueError("search max_results must be between 1 and 5")
        self.calls += 1
        try:
            request_payload: dict[str, Any] = {
                "query": normalized,
                "max_results": max_results,
                "search_depth": "basic",
                "chunks_per_source": 3,
                "include_answer": False,
                "include_raw_content": "text",
            }
            if self._include_domains:
                # 在服务端先限制可信学术域名，客户端仍会执行最终 URL 白名单复核。
                request_payload.update(
                    {
                        "include_domains": list(self._include_domains),
                        "include_domains_mode": "restrict",
                    }
                )
            payload = await self._transport.post(request_payload)
        except KGCRAGError:
            raise
        except TimeoutError as error:
            raise _search_error("Tavily request timed out", retryable=True) from error
        except Exception as error:
            raise _search_error("Tavily transport failed", retryable=True) from error
        raw_results = payload.get("results")
        if not isinstance(raw_results, list):
            raise _search_error("Tavily response is missing results")
        accessed_at = datetime.now(UTC)
        results: list[SearchResult] = []
        for raw in raw_results[:max_results]:
            if not isinstance(raw, dict):
                raise _search_error("Tavily result must be an object")
            title, url, content = raw.get("title"), raw.get("url"), raw.get("content")
            if not all(isinstance(value, str) and value.strip() for value in (title, url, content)):
                raise _search_error("Tavily result fields are incomplete")
            raw_content = raw.get("raw_content")
            if raw_content is not None and not isinstance(raw_content, str):
                raise _search_error("Tavily raw content is invalid")
            excerpt = _evidence_excerpt(
                str(content),
                raw_content,
                query=normalized,
                max_chars=self._max_excerpt_chars,
            )
            score = raw.get("score", 0.0)
            if not isinstance(score, int | float) or not 0.0 <= float(score) <= 1.0:
                raise _search_error("Tavily result score is invalid")
            result_id = str(raw.get("id") or stable_digest({"url": url, "title": title})[:24])
            results.append(
                SearchResult(
                    provider_result_id=result_id,
                    title=str(title).strip(),
                    url=str(url),
                    final_url=str(raw.get("final_url") or url),
                    accessed_at=accessed_at,
                    excerpt=excerpt,
                    source_type=_source_type(str(url)),
                    score=float(score),
                    requires_auth=bool(raw.get("requires_auth", False)),
                )
            )
        return results


_QUERY_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_.+-]{3,}")


def _normalized_text(value: str) -> str:
    return " ".join(value.split())


def _evidence_excerpt(
    content: str,
    raw_content: str | None,
    *,
    query: str,
    max_chars: int,
) -> str:
    """优先截取目标实体附近的网页正文，并保留搜索服务给出的相关片段。"""

    snippet = _normalized_text(content)
    raw_text = _normalized_text(raw_content or "")
    if not raw_text:
        return snippet[:max_chars].rstrip()
    focus_tokens = list(dict.fromkeys(token.casefold() for token in _QUERY_TOKEN.findall(query)))
    folded = raw_text.casefold()
    positions = [folded.find(token) for token in focus_tokens if folded.find(token) >= 0]
    start = max(0, min(positions, default=0) - 200)
    raw_limit = max_chars if not snippet else max(400, max_chars * 3 // 4)
    raw_window = raw_text[start : start + raw_limit].rstrip()
    if not snippet or snippet in raw_window:
        return raw_window[:max_chars].rstrip()
    remaining = max_chars - len(raw_window) - 2
    return f"{raw_window}\n\n{snippet[:remaining]}".rstrip() if remaining > 0 else raw_window


def _search_error(message: str, *, retryable: bool = False) -> KGCRAGError:
    return KGCRAGError(
        ErrorDetail(
            code=ErrorCode.EXTERNAL_SERVICE,
            message=message,
            retryable=retryable,
        )
    )
