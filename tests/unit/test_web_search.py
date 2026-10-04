"""受控 Web Provider、可信来源和外部 Evidence 测试。"""

import json
import socket
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

import pytest

from kg_crag.answering.config import WebSearchConfig
from kg_crag.errors import KGCRAGError
from kg_crag.models import (
    ConditionKind,
    EvidenceRequirement,
    EvidenceSourceType,
    FacetKind,
    SatisfactionCondition,
    SearchResult,
    WebSourceType,
)
from kg_crag.providers import MockSearchProvider, RecordedSearchProvider, SearchProvider
from kg_crag.web import (
    TavilySearchProvider,
    TrustedSourcePolicy,
    build_external_coverage,
    build_web_query,
    convert_web_results,
)


def _result(
    *,
    url: str = "https://arxiv.org/abs/1234.5678",
    excerpt: str = "missing metric is 42 percent",
) -> SearchResult:
    return SearchResult(
        provider_result_id="r1",
        title="Paper",
        url=url,
        final_url=url,
        accessed_at=datetime(2026, 1, 1, tzinfo=UTC),
        excerpt=excerpt,
        source_type=WebSourceType.ARXIV,
        score=0.8,
    )


def _facet() -> EvidenceRequirement:
    return EvidenceRequirement(
        facet_id="facet-" + "a" * 16,
        question_id="q1",
        kind=FacetKind.METRIC,
        description="missing metric",
        expected_evidence_types=[EvidenceSourceType.CHUNK],
        condition=SatisfactionCondition(
            kind=ConditionKind.TERMS,
            terms=["missing", "metric"],
            min_term_matches=2,
        ),
    )


@pytest.mark.asyncio
async def test_mock_and_recorded_search_conform_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def deny_network(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise AssertionError("offline search providers must not access network")

    monkeypatch.setattr(socket, "create_connection", deny_network)
    mock = MockSearchProvider([_result()])
    assert isinstance(mock, SearchProvider)
    assert len(await mock.search("query", max_results=1)) == 1
    fixture = tmp_path / "recorded.json"
    fixture.write_text(
        json.dumps({"query": [_result().model_dump(mode="json")]}, ensure_ascii=False),
        encoding="utf-8",
    )
    recorded = RecordedSearchProvider(fixture)
    assert isinstance(recorded, SearchProvider)
    assert len(await recorded.search("query", max_results=1)) == 1
    with pytest.raises(ValueError, match="between 1 and 5"):
        await mock.search("query", max_results=6)


def test_query_and_trust_policy_reject_untrusted_or_authenticated_results() -> None:
    query = build_web_query("What is missing?", [_facet()])
    assert "missing metric" in query
    config = WebSearchConfig(enabled=True, provider="mock")
    policy = TrustedSourcePolicy(config)
    assert policy.select([_result()])[0].provider_result_id == "r1"
    assert policy.select([_result(url="http://arxiv.org/abs/1234")]) == []
    assert policy.select([_result(url="https://evil.example/paper")]) == []
    assert policy.select([_result(url="https://arxiv.org/login")]) == []
    authenticated = _result().model_copy(update={"requires_auth": True})
    assert policy.select([authenticated]) == []


def test_web_conversion_is_stable_bounded_and_does_not_mutate_internal_state() -> None:
    config = WebSearchConfig(enabled=True, provider="mock")
    source = _result()
    internal_marker = {"matrix_id": "unchanged"}
    first = convert_web_results(
        [source], provider="mock", converter_version="web-v1", config=config
    )
    second = convert_web_results(
        [source], provider="mock", converter_version="web-v1", config=config
    )
    assert first == second
    assert first[0].external is True and first[0].paper_id is None
    assert first[0].location.url is not None
    coverage = build_external_coverage([_facet()], first, rule_version="external-v1")
    assert coverage[0].matched
    assert internal_marker == {"matrix_id": "unchanged"}
    many = convert_web_results(
        [
            _result(excerpt="x" * 2000).model_copy(update={"provider_result_id": f"r{i}"})
            for i in range(5)
        ],
        provider="mock",
        converter_version="web-v1",
        config=config,
    )
    assert sum(len(item.content) for item in many) <= 8000


class _FakeTransport:
    def __init__(
        self, payload: dict[str, Any] | None = None, error: Exception | None = None
    ) -> None:
        self.payload = payload or {}
        self.error = error
        self.calls = 0

    async def post(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        assert "api_key" not in payload
        if self.error:
            raise self.error
        return self.payload


@pytest.mark.asyncio
async def test_tavily_adapter_uses_one_bounded_request_and_no_hidden_retry() -> None:
    transport = _FakeTransport(
        {
            "results": [
                {
                    "id": "r1",
                    "title": "Paper",
                    "url": "https://arxiv.org/abs/1234.5678",
                    "content": "abstract",
                    "score": 0.9,
                }
            ]
        }
    )
    provider = TavilySearchProvider(transport=transport)
    results = await provider.search("agent paper", max_results=1)
    assert len(results) == 1
    assert transport.calls == provider.calls == 1
    malformed = TavilySearchProvider(transport=_FakeTransport({"unexpected": []}))
    with pytest.raises(KGCRAGError, match="missing results"):
        await malformed.search("agent paper", max_results=1)
    assert malformed.calls == 1
    timed_out = TavilySearchProvider(transport=_FakeTransport(error=TimeoutError()))
    with pytest.raises(KGCRAGError, match="timed out"):
        await timed_out.search("agent paper", max_results=1)
    assert timed_out.calls == 1
