"""Facet 生成、来源复核、覆盖矩阵、冲突与充分性测试。"""

import json
from pathlib import Path

import pytest

from kg_crag.correction.config import FacetRulesConfig
from kg_crag.correction.coverage import CatalogSourceVerifier, build_coverage_matrix
from kg_crag.correction.requirements import (
    LLMRequirementProvider,
    MockRequirementProvider,
    RequirementBatch,
    RequirementCache,
    RequirementCandidate,
    RequirementProviderResponse,
    analyze_question,
    generate_budgeted_requirements,
    generate_requirements,
)
from kg_crag.correction.sufficiency import assess_sufficiency
from kg_crag.models import (
    BudgetLedger,
    BudgetUsage,
    ConditionKind,
    ConflictKind,
    Evidence,
    EvidenceRanks,
    EvidenceSourceType,
    FacetKind,
    SatisfactionCondition,
)
from kg_crag.providers import LLMGeneration


def _evidence(
    evidence_id: str,
    content: str,
    *,
    source_id: str | None = None,
    paper_id: str = "paper-1",
    external: bool = False,
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        content=content,
        source_type=EvidenceSourceType.CHUNK,
        source_id=source_id or f"chunk-{evidence_id}",
        paper_id=paper_id,
        external=external,
        metadata={"content_hash": f"hash-{evidence_id}"},
    )


def test_rule_analysis_is_stable_for_simple_comparison_relationship_and_metric() -> None:
    simple = analyze_question("q1", "What architecture does Voyager use?")
    assert len(simple.facets) == 1
    assert not simple.complex
    comparison = analyze_question("q2", "比较 Voyager 与 ReAct 的架构和准确率")
    assert comparison.complex
    assert len(comparison.facets) >= 4
    assert all(item.required for item in comparison.facets)
    assert (
        comparison.facets == analyze_question("q2", "比较 Voyager 与 ReAct 的架构和准确率").facets
    )
    relationship = analyze_question("q3", "A 与 B 的引用关系是什么?")
    assert relationship.facets[0].kind is FacetKind.RELATIONSHIP


@pytest.mark.asyncio
async def test_llm_boundary_skips_simple_uses_one_call_and_falls_back_atomically(
    tmp_path: Path,
) -> None:
    candidate = RequirementCandidate(
        kind=FacetKind.COMPARISON,
        description="A architecture",
        expected_evidence_types=[EvidenceSourceType.CHUNK],
        condition=SatisfactionCondition(kind=ConditionKind.TERMS, terms=["A", "architecture"]),
        confidence=0.9,
    )
    provider = MockRequirementProvider(
        RequirementProviderResponse(
            batch=RequirementBatch(candidates=[candidate]),
            usage=BudgetUsage(llm_calls=1, input_tokens=10, output_tokens=5),
        )
    )
    config = FacetRulesConfig(allow_llm=True)
    facets, usage, source = await generate_requirements(
        "q1", "What is Voyager?", config, provider=provider
    )
    assert source == "rule" and usage.llm_calls == 0 and provider.calls == 0 and facets
    cache = RequirementCache(tmp_path / "requirements")
    facets, usage, source = await generate_requirements(
        "q2", "比较 A 与 B 的架构", config, provider=provider, cache=cache
    )
    assert source == "llm" and usage.llm_calls == 1 and provider.calls == 1
    cached, usage, source = await generate_requirements(
        "q2", "比较 A 与 B 的架构", config, provider=provider, cache=cache
    )
    assert cached == facets
    assert source == "llm_cache" and usage.llm_calls == 0 and provider.calls == 1
    # 重复候选整批非法，不能让合法前缀部分进入状态。
    bad = MockRequirementProvider(
        RequirementProviderResponse(
            batch=RequirementBatch(candidates=[candidate, candidate]),
            usage=BudgetUsage(llm_calls=1),
        )
    )
    fallback, _, source = await generate_requirements(
        "q2", "比较 A 与 B 的架构", config, provider=bad
    )
    assert source == "rule_fallback"
    assert all(item.source.value == "rule" for item in fallback)
    assert bad.calls == 1

    budgeted_provider = MockRequirementProvider(
        RequirementProviderResponse(
            batch=RequirementBatch(candidates=[candidate]),
            usage=BudgetUsage(llm_calls=1, input_tokens=12, output_tokens=6),
        )
    )
    _, ledger, source = await generate_budgeted_requirements(
        "q3",
        "比较 A 与 B 的架构",
        config,
        BudgetLedger(),
        provider=budgeted_provider,
    )
    assert source == "llm"
    assert ledger.used.llm_calls == 1
    assert ledger.used.input_tokens == 12
    assert ledger.used.output_tokens == 6
    assert ledger.reserved == BudgetUsage()


@pytest.mark.asyncio
async def test_llm_requirement_provider_supplies_schema_accepts_fence_and_tracks_usage() -> None:
    candidate = RequirementCandidate(
        kind=FacetKind.MULTI_HOP,
        description="方法、数据集与结果之间的关系链",
        expected_evidence_types=[EvidenceSourceType.CHUNK, EvidenceSourceType.GRAPH],
        condition=SatisfactionCondition(
            kind=ConditionKind.GRAPH_PATH,
            terms=["方法", "数据集", "结果"],
            max_hops=3,
        ),
        confidence=0.9,
    )

    class _UsageLLM:
        def __init__(self) -> None:
            self.system_prompt = ""

        async def generate(
            self,
            prompt: str,
            *,
            system_prompt: str | None = None,
            temperature: float = 0.0,
        ) -> str:
            raise AssertionError((prompt, system_prompt, temperature))

        async def generate_with_usage(
            self,
            prompt: str,
            *,
            system_prompt: str | None = None,
            temperature: float = 0.0,
        ) -> LLMGeneration:
            del prompt, temperature
            self.system_prompt = system_prompt or ""
            payload = RequirementBatch(candidates=[candidate]).model_dump(mode="json")
            return LLMGeneration(
                content="```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```",
                input_tokens=123,
                output_tokens=45,
            )

    llm = _UsageLLM()
    provider = LLMRequirementProvider(llm, revision="facet-live-v1")  # type: ignore[arg-type]
    response = await provider.propose("比较方法并说明关系链", system_prompt="只返回 JSON。")

    assert response.batch.candidates == [candidate]
    assert response.usage.input_tokens == 123
    assert response.usage.output_tokens == 45
    assert response.usage_estimated is False
    assert "JSON Schema" in llm.system_prompt
    assert "expected_evidence_types" in llm.system_prompt


def test_coverage_is_order_independent_and_excludes_drift_and_external() -> None:
    facet = analyze_question("q1", "Voyager architecture").facets[0]
    valid = _evidence("e1", "Voyager architecture uses a skill library")
    drift = _evidence("e2", "Voyager architecture", paper_id="wrong")
    external = _evidence("e3", "Voyager architecture", external=True)
    catalog = {
        valid.source_id: (valid.paper_id, "hash-e1"),
        drift.source_id: ("paper-1", "hash-e2"),
        external.source_id: (external.paper_id, "hash-e3"),
    }
    verifier = CatalogSourceVerifier(catalog)
    first = build_coverage_matrix(
        [facet], [valid, drift, external], rule_version="v1", verifier=verifier
    )
    second = build_coverage_matrix(
        [facet], [external, drift, valid], rule_version="v1", verifier=verifier
    )
    assert first == second
    assert sum(item.matched for item in first.entries) == 1
    assert {item.reason for item in first.entries if not item.matched} == {
        "paper_mismatch",
        "external_evidence",
    }


def test_sufficiency_locates_missing_side_and_blocking_numeric_conflict() -> None:
    facets = list(analyze_question("q2", "比较 A 与 B 的架构").facets)
    a_only = _evidence("e1", f"{facets[0].description} architecture")
    matrix = build_coverage_matrix(facets, [a_only], rule_version="v1")
    assessment = assess_sufficiency(
        facets, [a_only], matrix, min_support=0.3, min_sources=1, max_selected=4
    )
    assert not assessment.sufficient
    assert assessment.missing_required_facet_ids

    metric = analyze_question("q3", "What accuracy metric is reported?").facets[0]
    # 共享的实体编号不能遮蔽后续真正冲突的指标值。
    left = _evidence("e2", "Agent 16 accuracy metric is 80%")
    right = _evidence("e3", "Agent 16 accuracy metric is 90%")
    matrix = build_coverage_matrix([metric], [left, right], rule_version="v1")
    assessment = assess_sufficiency(
        [metric], [left, right], matrix, min_support=0.3, min_sources=1, max_selected=4
    )
    assert not assessment.sufficient
    assert assessment.conflicts[0].blocking
    assert all(item.conflict for item in matrix.entries)


def test_conflict_reason_does_not_duplicate_long_evidence_ids() -> None:
    metric = analyze_question("q-long", "What accuracy metric is reported?").facets[0]
    left = _evidence("dense:" + "a" * 180, "accuracy metric is 80%")
    right = _evidence("sparse:" + "b" * 180, "accuracy metric is 90%")

    matrix = build_coverage_matrix([metric], [left, right], rule_version="v1")
    assessment = assess_sufficiency(
        [metric], [left, right], matrix, min_support=0.3, min_sources=1, max_selected=4
    )

    assert assessment.conflicts
    assert len(assessment.conflicts[0].reason) <= 300


def test_content_facet_ignores_unrelated_numbers_and_negation_words() -> None:
    question = "Voyager 在 Minecraft 中持续学习所依赖的三个核心组件是什么?"
    facet = analyze_question("voyager-components", question).facets[0]
    abstract = _evidence(
        "voyager-abstract",
        "VOYAGER is a lifelong learning agent in Minecraft without human intervention. "
        "It has three key components and reports 3.3x more items and 15.3x faster progress.",
    )
    context = _evidence(
        "voyager-context",
        "Voyager in Minecraft was evaluated on 5 tasks in 2024.",
    )

    matrix = build_coverage_matrix([facet], [abstract, context], rule_version="v1")
    assessment = assess_sufficiency(
        [facet], [abstract, context], matrix, min_support=0.3, min_sources=1, max_selected=4
    )

    assert assessment.sufficient
    assert assessment.conflicts == []
    assert not any(item.conflict for item in matrix.entries)


def test_paper_anchor_keeps_answer_bearing_chunk_and_selection_follows_retrieval_rank() -> None:
    facet = analyze_question(
        "camel-roleplay",
        "CAMEL 如何通过角色扮演促进多个智能体协作?",
    ).facets[0]
    decoy = _evidence(
        "camel-checklist",
        "CAMEL appendix checklist requirements.",
        paper_id="paper-camel",
    ).model_copy(update={"ranks": EvidenceRanks(rerank=2, fusion=1)})
    abstract = _evidence(
        "camel-abstract",
        "Role-playing with inception prompting guides communicative agents toward task "
        "completion while maintaining consistency with human intentions.",
        paper_id="paper-camel",
    ).model_copy(update={"ranks": EvidenceRanks(rerank=1, dense=1)})
    unrelated = _evidence(
        "unrelated",
        "An unrelated method uses role assignment.",
        paper_id="paper-other",
    ).model_copy(update={"ranks": EvidenceRanks(rerank=3)})

    evidence = [decoy, abstract, unrelated]
    matrix = build_coverage_matrix([facet], evidence, rule_version="v1")
    assessment = assess_sufficiency(
        [facet], evidence, matrix, min_support=0.3, min_sources=1, max_selected=3
    )

    abstract_coverage = next(
        item for item in matrix.entries if item.evidence_id == abstract.evidence_id
    )
    assert abstract_coverage.matched
    assert abstract_coverage.basis == ["paper_scope:paper-camel"]
    assert assessment.selected_evidence_ids[:2] == [abstract.evidence_id, decoy.evidence_id]


def test_discrete_conflict_and_independent_source_count_are_conservative() -> None:
    metric = analyze_question("q4", "What mode is reported?").facets[0]
    left = _evidence("e4", "What mode is reported? mode is enabled").model_copy(
        update={
            "metadata": {
                "normalized_value_key": "reported-mode",
                "normalized_value": "enabled",
            }
        }
    )
    right = _evidence("e5", "What mode is reported? mode is disabled").model_copy(
        update={
            "metadata": {
                "normalized_value_key": "reported-mode",
                "normalized_value": "disabled",
            }
        }
    )
    matrix = build_coverage_matrix([metric], [left, right], rule_version="v1")
    assessment = assess_sufficiency(
        [metric], [left, right], matrix, min_support=0.3, min_sources=1, max_selected=4
    )
    assert assessment.conflicts[0].kind is ConflictKind.DISCRETE
    assert not assessment.sufficient

    unrelated = [
        left,
        right.model_copy(
            update={
                "metadata": {
                    "normalized_value_key": "deployment-mode",
                    "normalized_value": "disabled",
                }
            }
        ),
    ]
    matrix = build_coverage_matrix([metric], unrelated, rule_version="v1")
    assessment = assess_sufficiency(
        [metric], unrelated, matrix, min_support=0.3, min_sources=1, max_selected=4
    )
    assert assessment.sufficient
    assert assessment.conflicts == []

    duplicate_publication = [
        _evidence("e6", "What mode is reported? mode", paper_id="same-paper"),
        _evidence("e7", "What mode is reported? mode", paper_id="same-paper"),
    ]
    matrix = build_coverage_matrix([metric], duplicate_publication, rule_version="v1")
    assessment = assess_sufficiency(
        [metric], duplicate_publication, matrix, min_support=0.3, min_sources=2, max_selected=4
    )
    assert not assessment.sufficient


def test_graph_without_provenance_cannot_cover_a_facet() -> None:
    facet = analyze_question("q5", "What relationship is reported?").facets[0]
    graph = _evidence("e8", "What relationship is reported?").model_copy(
        update={"source_type": EvidenceSourceType.GRAPH, "metadata": {}}
    )
    matrix = build_coverage_matrix([facet], [graph], rule_version="v1")
    assert matrix.entries[0].reason == "graph_provenance_missing"
    assert not matrix.entries[0].matched
