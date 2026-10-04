"""缺失 facet 查询构造和可信来源白名单策略。"""

from __future__ import annotations

from collections.abc import Sequence
from urllib.parse import urlsplit, urlunsplit

from kg_crag.answering.config import WebSearchConfig
from kg_crag.models import EvidenceRequirement, SearchResult, WebSourceType

_BLOCKED_PATH_PARTS = ("/login", "/signin", "/subscribe", "/paywall", "/account")
_SOURCE_PRIORITY = {
    WebSourceType.ARXIV: 0,
    WebSourceType.PAPER_SITE: 1,
    WebSourceType.ACADEMIC_DATABASE: 2,
    WebSourceType.PROJECT_PAGE: 3,
}


def build_web_query(
    question: str,
    facets: Sequence[EvidenceRequirement],
    *,
    max_chars: int = 2000,
) -> str:
    if not question.strip() or not facets:
        raise ValueError("Web query requires a question and missing facets")
    descriptions = " ".join(
        item.description for item in sorted(facets, key=lambda item: item.facet_id)
    )
    query = " ".join(f"{question.strip()} {descriptions}".split())
    return query[:max_chars].rstrip()


def canonical_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme.casefold() != "https" or not parsed.hostname:
        raise ValueError("trusted Web results require HTTPS URLs with a host")
    if parsed.username or parsed.password or parsed.port not in {None, 443}:
        raise ValueError("trusted Web result URL contains forbidden authority fields")
    host = parsed.hostname.rstrip(".").casefold().encode("idna").decode("ascii")
    path = parsed.path or "/"
    if any(marker in path.casefold() for marker in _BLOCKED_PATH_PARTS):
        raise ValueError("authentication or paywall result paths are forbidden")
    netloc = host if parsed.port is None else f"{host}:{parsed.port}"
    return urlunsplit(("https", netloc, path, parsed.query, ""))


class TrustedSourcePolicy:
    def __init__(self, config: WebSearchConfig) -> None:
        self.config = config
        self._domains = frozenset(config.allowed_domains)
        self._types = frozenset(config.allowed_source_types)

    def _allowed_host(self, url: str) -> bool:
        host = urlsplit(url).hostname
        if host is None:
            return False
        normalized = host.rstrip(".").casefold().encode("idna").decode("ascii")
        return any(
            normalized == domain or normalized.endswith(f".{domain}") for domain in self._domains
        )

    def validate(self, result: SearchResult) -> SearchResult:
        if result.requires_auth or result.source_type not in self._types:
            raise ValueError("search result source type or access mode is not trusted")
        normalized = canonical_url(str(result.url))
        final = canonical_url(str(result.final_url or result.url))
        if not self._allowed_host(normalized) or not self._allowed_host(final):
            raise ValueError("search result domain is not trusted")
        return result.model_copy(update={"url": normalized, "final_url": final})

    def select(self, results: Sequence[SearchResult]) -> list[SearchResult]:
        accepted: list[SearchResult] = []
        for result in results:
            try:
                accepted.append(self.validate(result))
            except ValueError:
                continue
        accepted.sort(
            key=lambda item: (
                _SOURCE_PRIORITY[item.source_type],
                -item.score,
                str(item.final_url or item.url),
                item.provider_result_id,
            )
        )
        return accepted[: self.config.max_results]
