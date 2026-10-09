"""受控 Web Provider、可信来源和外部 Evidence 测试。"""

import json
import socket
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

import pytest

from kg_crag.answering import build_answer_context
from kg_crag.answering.config import AnswerGenerationConfig, WebSearchConfig
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


def test_web_query_prioritizes_target_entities_and_evidence_terms() -> None:
    facet = EvidenceRequirement(
        facet_id="facet-" + "d" * 16,
        question_id="mlr-query",
        kind=FacetKind.CONTENT,
        description="MLR-Bench 的组成与研究阶段",
        expected_evidence_types=[EvidenceSourceType.CHUNK],
        condition=SatisfactionCondition(
            kind=ConditionKind.TERMS,
            terms=["components", "research stages", "reliability"],
            min_term_matches=2,
        ),
    )

    query = build_web_query("MLR-Bench 包含什么?", [facet])

    assert query.startswith("mlr-bench components research stages reliability")


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


def test_external_coverage_accepts_bilingual_abstract_with_distinctive_anchors() -> None:
    """中文 facet 与英文摘要可通过论文实体锚点建立受控绑定。"""

    facet = EvidenceRequirement(
        facet_id="facet-" + "b" * 16,
        question_id="alama",
        kind=FacetKind.RELATIONSHIP,
        description="ALAMA 如何在 UniAct 中把机制激活接入底层模型?",
        expected_evidence_types=[EvidenceSourceType.CHUNK],
        condition=SatisfactionCondition(
            kind=ConditionKind.TERMS,
            terms=["机制激活", "底层模型"],
            min_term_matches=2,
        ),
    )
    result = _result(
        excerpt=(
            "ALAMA introduces adaptive mechanism activation in UniAct. The controller "
            "selects heterogeneous reasoning experts before invoking the underlying "
            "foundation model, allowing task-dependent agent behavior."
        )
    ).model_copy(update={"title": "Towards Adaptive Mechanism Activation"})
    converted = convert_web_results(
        [result],
        provider="mock",
        converter_version="external-coverage-v3",
        config=WebSearchConfig(enabled=True, provider="mock"),
    )

    coverage = build_external_coverage([facet], converted, rule_version="external-coverage-v3")

    assert coverage[0].matched
    assert {"anchor:alama", "anchor:uniact"} <= set(coverage[0].basis)


def test_external_coverage_uses_primary_long_project_name_for_compound_question() -> None:
    """复合问题不要求一条搜索摘要重复所有子组件名。"""

    facet = EvidenceRequirement(
        facet_id="facet-" + "c" * 16,
        question_id="mlr-bench",
        kind=FacetKind.CONTENT,
        description=("MLR-Bench 有哪些核心部分? MLR-Agent 包含哪些阶段并发现了什么可靠性问题?"),
        expected_evidence_types=[EvidenceSourceType.CHUNK],
        condition=SatisfactionCondition(
            kind=ConditionKind.TERMS,
            terms=["核心部分", "研究阶段", "可靠性问题"],
            min_term_matches=2,
        ),
    )
    result = _result(
        excerpt=(
            "This benchmark evaluates open-ended machine learning research. It includes "
            "research tasks, an automated judge, and a modular agent workflow."
        )
    ).model_copy(update={"title": "MLR-Bench: Evaluating AI Agents"})
    converted = convert_web_results(
        [result],
        provider="mock",
        converter_version="external-coverage-v3",
        config=WebSearchConfig(enabled=True, provider="mock"),
    )

    coverage = build_external_coverage([facet], converted, rule_version="external-coverage-v3")

    assert coverage[0].matched
    assert "anchor:mlr-bench" in coverage[0].basis


def test_primary_academic_source_suppresses_matching_project_page() -> None:
    """同一 facet 有论文正文时，代码仓库等次要页面不能获得回答绑定。"""

    facet = EvidenceRequirement(
        facet_id="facet-" + "e" * 16,
        question_id="mlr-authority",
        kind=FacetKind.CONTENT,
        description="MLR-Bench 由哪三个核心部分组成?",
        expected_evidence_types=[EvidenceSourceType.CHUNK],
        condition=SatisfactionCondition(
            kind=ConditionKind.TERMS,
            terms=["MLR-Bench", "components"],
            min_term_matches=2,
        ),
    )
    arxiv = _result(
        excerpt="MLR-Bench consists of three components for machine learning research."
    ).model_copy(update={"title": "MLR-Bench", "provider_result_id": "arxiv"})
    github = _result(
        url="https://github.com/example/mlr-bench",
        excerpt="MLR-Bench consists of three components for machine learning research.",
    ).model_copy(
        update={
            "title": "MLR-Bench notes",
            "provider_result_id": "github",
            "source_type": WebSourceType.PROJECT_PAGE,
        }
    )
    converted = convert_web_results(
        [arxiv, github],
        provider="mock",
        converter_version="external-coverage-v4",
        config=WebSearchConfig(enabled=True, provider="mock"),
    )

    coverage = build_external_coverage([facet], converted, rule_version="external-coverage-v4")
    by_evidence = {item.evidence_id: item for item in coverage}

    assert by_evidence[converted[0].evidence_id].matched
    assert not by_evidence[converted[1].evidence_id].matched
    assert converted[0].metadata["search_rank"] == 1
    assert converted[1].metadata["search_score"] == 0.8
    context, bindings = build_answer_context(
        converted,
        internal_matrix=None,
        external_coverage=coverage,
        config=AnswerGenerationConfig(),
    )
    assert "github.com" not in context
    assert [item.evidence.evidence_id for item in bindings] == [converted[0].evidence_id]


class _FakeTransport:
    def __init__(
        self, payload: dict[str, Any] | None = None, error: Exception | None = None
    ) -> None:
        self.payload = payload or {}
        self.error = error
        self.calls = 0
        self.request_payloads: list[dict[str, Any]] = []

    async def post(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        self.request_payloads.append(payload)
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
    provider = TavilySearchProvider(
        transport=transport,
        include_domains=["arxiv.org", "openreview.net"],
    )
    assert provider.provider_version == "tavily-json-v3"
    results = await provider.search("agent paper", max_results=1)
    assert len(results) == 1
    assert transport.calls == provider.calls == 1
    assert transport.request_payloads[0]["include_domains"] == [
        "arxiv.org",
        "openreview.net",
    ]
    assert transport.request_payloads[0]["include_domains_mode"] == "restrict"
    assert transport.request_payloads[0]["search_depth"] == "basic"
    assert transport.request_payloads[0]["chunks_per_source"] == 3
    assert transport.request_payloads[0]["include_raw_content"] == "text"
    malformed = TavilySearchProvider(transport=_FakeTransport({"unexpected": []}))
    with pytest.raises(KGCRAGError, match="missing results"):
        await malformed.search("agent paper", max_results=1)
    assert malformed.calls == 1
    timed_out = TavilySearchProvider(transport=_FakeTransport(error=TimeoutError()))
    with pytest.raises(KGCRAGError, match="timed out"):
        await timed_out.search("agent paper", max_results=1)
    assert timed_out.calls == 1


@pytest.mark.asyncio
async def test_tavily_adapter_extracts_target_window_from_raw_page() -> None:
    raw = "preface " * 100 + "MLR-Bench has three components and four research stages. "
    raw += "Current agents may fabricate invalid experimental results. " + "tail " * 500
    transport = _FakeTransport(
        {
            "results": [
                {
                    "id": "mlr",
                    "title": "MLR-Bench",
                    "url": "https://arxiv.org/abs/2505.19955",
                    "content": "MLR-Bench paper metadata.",
                    "raw_content": raw,
                    "score": 0.9,
                }
            ]
        }
    )
    provider = TavilySearchProvider(transport=transport, max_excerpt_chars=500)

    results = await provider.search("MLR-Bench components stages", max_results=1)

    assert "three components" in results[0].excerpt
    assert "fabricate invalid experimental results" in results[0].excerpt
    assert len(results[0].excerpt) <= 500
