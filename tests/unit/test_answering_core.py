"""有界回答上下文、检查、Critic 和决策表测试。"""

import json

import pytest

from kg_crag.answering import (
    SemanticCritic,
    build_answer_context,
    build_conservative_candidate,
    decide_reflection,
    deterministic_check,
    merge_critic_findings,
    parse_grounded_answer,
)
from kg_crag.answering.config import AnswerGenerationConfig
from kg_crag.answering.prompt import load_structured_prompt
from kg_crag.correction.coverage import build_coverage_matrix
from kg_crag.correction.sufficiency import assess_sufficiency
from kg_crag.models import (
    AnswerCheckCode,
    AnswerFinding,
    ConditionKind,
    Evidence,
    EvidenceRequirement,
    EvidenceSourceType,
    FacetKind,
    FindingSeverity,
    GroundedAnswerCandidate,
    GroundedClaim,
    ReflectionAction,
    SatisfactionCondition,
    StopReason,
)
from kg_crag.providers import MockLLMProvider


def _facet(name: str = "architecture") -> EvidenceRequirement:
    return EvidenceRequirement(
        facet_id="facet-" + name.encode().hex()[:16].ljust(16, "0"),
        question_id="q1",
        kind=FacetKind.CONTENT,
        description=name,
        expected_evidence_types=[EvidenceSourceType.CHUNK],
        condition=SatisfactionCondition(kind=ConditionKind.TERMS, terms=[name]),
    )


def _evidence() -> Evidence:
    return Evidence(
        evidence_id="ev-1",
        content="The architecture uses a skill library.",
        source_type=EvidenceSourceType.CHUNK,
        source_id="chunk-1",
        paper_id="paper-1",
    )


def _coverage() -> tuple[list[EvidenceRequirement], object, object]:
    facets = [_facet()]
    evidence = [_evidence()]
    matrix = build_coverage_matrix(facets, evidence, rule_version="v1")
    sufficiency = assess_sufficiency(
        facets,
        evidence,
        matrix,
        min_support=0.1,
        min_sources=1,
        max_selected=8,
    )
    return facets, matrix, sufficiency


def _candidate() -> tuple[
    GroundedAnswerCandidate, str, list[object], list[EvidenceRequirement], object
]:
    facets, matrix, sufficiency = _coverage()
    context, bindings = build_answer_context(
        [_evidence()],
        internal_matrix=matrix,
        external_coverage=[],
        config=AnswerGenerationConfig(),
    )
    raw = json.dumps(
        {
            "claims": [
                {
                    "text": "It uses a skill library.",
                    "claim_type": "fact",
                    "citation_ids": ["E1"],
                    "facet_ids": [facets[0].facet_id],
                }
            ],
            "confidence": 0.9,
        }
    )
    candidate = parse_grounded_answer(raw, bindings, facets, config=AnswerGenerationConfig())
    return candidate, context, bindings, facets, sufficiency


def test_context_and_parser_bind_claims_to_evidence_and_facets() -> None:
    candidate, context, bindings, facets, _ = _candidate()
    assert "[E1]" in context
    assert candidate.citations[0].evidence_id == "ev-1"
    assert candidate.claims[0].facet_ids == [facets[0].facet_id]
    raw = json.dumps(
        {
            "claims": [
                {
                    "text": "unknown",
                    "claim_type": "fact",
                    "citation_ids": ["E9"],
                    "facet_ids": [facets[0].facet_id],
                }
            ],
            "confidence": 0.1,
        }
    )
    with pytest.raises(ValueError, match="unknown citation"):
        parse_grounded_answer(raw, bindings, facets, config=AnswerGenerationConfig())


def test_critic_context_keeps_only_cited_evidence_with_original_numbering() -> None:
    facets, matrix, _ = _coverage()
    first = _evidence()
    second = first.model_copy(
        update={
            "evidence_id": "ev-2",
            "source_id": "chunk-2",
            "content": "The second excerpt contains the detailed mechanism.",
        }
    )
    context, bindings = build_answer_context(
        [first, second],
        internal_matrix=matrix,
        external_coverage=[],
        config=AnswerGenerationConfig(),
        citation_ids={"E2"},
    )

    assert context.startswith("[E2]")
    assert "[E1]" not in context
    assert [item.citation_id for item in bindings] == ["E2"]


def test_parser_accepts_only_a_single_markdown_json_fence_wrapper() -> None:
    candidate, _, bindings, facets, _ = _candidate()
    fenced = (
        "```json\n"
        + json.dumps(
            {
                "claims": [
                    {
                        "text": candidate.claims[0].text,
                        "claim_type": "fact",
                        "citation_ids": ["E1"],
                        "facet_ids": [facets[0].facet_id],
                    }
                ],
                "confidence": 0.8,
            }
        )
        + "\n```"
    )

    parsed = parse_grounded_answer(fenced, bindings, facets, config=AnswerGenerationConfig())

    assert parsed.claims[0].citation_ids == ["E1"]
    with pytest.raises(ValueError, match="valid JSON"):
        parse_grounded_answer(
            "说明如下:" + fenced,
            bindings,
            facets,
            config=AnswerGenerationConfig(),
        )


def test_parser_allows_uncertain_boundary_without_fabricated_bindings() -> None:
    _candidate_value, _, bindings, facets, _ = _candidate()
    raw = json.dumps(
        {
            "claims": [
                {
                    "text": "uncertain",
                    "claim_type": "uncertain",
                    "citation_ids": [],
                    "facet_ids": [],
                }
            ],
            "confidence": 0.2,
        }
    )
    parsed = parse_grounded_answer(
        raw,
        bindings,
        facets,
        config=AnswerGenerationConfig(),
    )

    assert parsed.claims[0].claim_type.value == "uncertain"
    assert parsed.claims[0].citation_ids == []
    assert parsed.claims[0].facet_ids == []


def test_parser_allows_splitting_claims_with_supported_evidence_facet_bindings() -> None:
    """复合结论可拆成原子句，逐条 Evidence—facet 支持仍是硬约束。"""

    first = _facet("architecture")
    second = _facet("results")
    evidence = Evidence(
        evidence_id="ev-compound",
        content="The architecture uses planning, and the results show improved accuracy.",
        source_type=EvidenceSourceType.CHUNK,
        source_id="chunk-compound",
        paper_id="paper-compound",
    )
    matrix = build_coverage_matrix([first, second], [evidence], rule_version="v1")
    _, bindings = build_answer_context(
        [evidence],
        internal_matrix=matrix,
        external_coverage=[],
        config=AnswerGenerationConfig(),
    )
    split_raw = json.dumps(
        {
            "claims": [
                {
                    "text": "It uses planning.",
                    "claim_type": "fact",
                    "citation_ids": ["E1"],
                    "facet_ids": [first.facet_id],
                },
                {
                    "text": "It reports improved accuracy.",
                    "claim_type": "fact",
                    "citation_ids": ["E1"],
                    "facet_ids": [second.facet_id],
                },
            ],
            "confidence": 0.8,
        }
    )

    regenerated = parse_grounded_answer(
        split_raw,
        bindings,
        [first, second],
        config=AnswerGenerationConfig(),
    )

    assert len(regenerated.claims) == 2


def test_deterministic_checker_detects_missing_facet_and_uncited_number() -> None:
    facets, _, sufficiency = _coverage()
    second = _facet("result")
    numeric = GroundedClaim(
        claim_id="claim-" + "a" * 16,
        text="The score is 42%.",
        claim_type="uncertain",
    )
    candidate = GroundedAnswerCandidate(answer=numeric.text, claims=[numeric], confidence=0.1)
    evaluation = deterministic_check(candidate, [*facets, second], sufficiency)
    codes = {item.code for item in evaluation.findings}
    assert AnswerCheckCode.UNCITED_NUMBER in codes
    assert AnswerCheckCode.INCOMPLETE in codes
    assert not evaluation.acceptable


def test_deterministic_checker_rejects_abstention_bound_as_a_supported_answer() -> None:
    candidate, _, _, facets, sufficiency = _candidate()
    refusal = candidate.model_copy(
        update={
            "answer": "暂无可安全陈述的结论。",
            "claims": [candidate.claims[0].model_copy(update={"text": "暂无可安全陈述的结论。"})],
        }
    )

    evaluation = deterministic_check(refusal, facets, sufficiency)

    codes = {item.code for item in evaluation.findings}
    assert AnswerCheckCode.ABSTENTION in codes
    assert AnswerCheckCode.INCOMPLETE in codes
    assert not evaluation.faithful
    assert not evaluation.acceptable


def test_deterministic_checker_rejects_evidence_metadata_refusal() -> None:
    candidate, _, _, facets, sufficiency = _candidate()
    refusal_text = "提供的 Evidence 只有页面元数据，未包含回答所需的具体信息。"
    refusal = candidate.model_copy(
        update={
            "answer": refusal_text,
            "claims": [candidate.claims[0].model_copy(update={"text": refusal_text})],
        }
    )

    evaluation = deterministic_check(refusal, facets, sufficiency)

    assert {item.code for item in evaluation.findings} >= {
        AnswerCheckCode.ABSTENTION,
        AnswerCheckCode.INCOMPLETE,
    }
    assert not evaluation.acceptable


@pytest.mark.asyncio
async def test_critic_identifies_unsupported_and_rejects_unknown_ids() -> None:
    candidate, context, _, facets, sufficiency = _candidate()
    finding = AnswerFinding(
        code=AnswerCheckCode.UNSUPPORTED,
        severity=FindingSeverity.ERROR,
        reason="excerpt does not support the claim",
        claim_ids=[candidate.claims[0].claim_id],
        citation_ids=["E1"],
        facet_ids=[facets[0].facet_id],
    )
    provider = MockLLMProvider(json.dumps({"findings": [finding.model_dump(mode="json")]}))
    critic = SemanticCritic(
        provider,
        load_structured_prompt(
            __import__("pathlib").Path("prompts/answer-critic-v2.txt"),
            placeholders={"question", "claims", "evidence_context", "facets", "conflicts"},
        ),
        max_prompt_chars=10_000,
        max_response_chars=6000,
    )
    findings = await critic.evaluate(
        question="What architecture?",
        candidate=candidate,
        evidence_context=context,
        facets=facets[0].description,
        conflicts="none",
    )
    merged = merge_critic_findings(deterministic_check(candidate, facets, sufficiency), findings)
    assert merged.critic_used and not merged.faithful and not merged.acceptable
    provider.response = "```json\n" + json.dumps({"findings": []}) + "\n```"
    assert (
        await critic.evaluate(
            question="What architecture?",
            candidate=candidate,
            evidence_context=context,
            facets=facets[0].description,
            conflicts="none",
        )
        == []
    )
    await critic.evaluate(
        question="What architecture?",
        candidate=candidate,
        evidence_context="evidence " * 5000,
        facets=facets[0].description,
        conflicts="none",
    )
    assert len(str(provider.calls[-1]["prompt"])) <= 10_000
    provider.response = json.dumps(
        {
            "findings": [
                {
                    **finding.model_dump(mode="json"),
                    "claim_ids": ["claim-" + "f" * 16],
                }
            ]
        }
    )
    with pytest.raises(ValueError, match="unknown claim"):
        await critic.evaluate(
            question="What architecture?",
            candidate=candidate,
            evidence_context=context,
            facets=facets[0].description,
            conflicts="none",
        )

    provider.response = "not-json"
    with pytest.raises(ValueError, match="structured output"):
        await critic.evaluate(
            question="What architecture?",
            candidate=candidate,
            evidence_context=context,
            facets=facets[0].description,
            conflicts="none",
        )


@pytest.mark.asyncio
async def test_critic_provider_failure_cannot_be_treated_as_pass() -> None:
    class FailingProvider:
        async def generate(
            self,
            prompt: str,
            *,
            system_prompt: str | None = None,
            temperature: float = 0.0,
        ) -> str:
            del prompt, system_prompt, temperature
            raise TimeoutError("bounded timeout")

    candidate, context, _, facets, sufficiency = _candidate()
    critic = SemanticCritic(
        FailingProvider(),
        load_structured_prompt(
            __import__("pathlib").Path("prompts/answer-critic-v2.txt"),
            placeholders={"question", "claims", "evidence_context", "facets", "conflicts"},
        ),
        max_prompt_chars=10_000,
        max_response_chars=6000,
    )
    with pytest.raises(TimeoutError):
        await critic.evaluate(
            question="What architecture?",
            candidate=candidate,
            evidence_context=context,
            facets=facets[0].description,
            conflicts="none",
        )
    failed = AnswerFinding(
        code=AnswerCheckCode.CRITIC_FAILED,
        reason="critic provider failed",
    )
    evaluation = merge_critic_findings(
        deterministic_check(candidate, facets, sufficiency),
        [failed],
        critic_failed=True,
    )
    assert evaluation.critic_used and not evaluation.acceptable


def test_decision_table_and_conservative_answer_are_bounded() -> None:
    candidate, _, _, facets, sufficiency = _candidate()
    passing = deterministic_check(candidate, facets, sufficiency)
    accepted = decide_reflection(
        internal_stop=StopReason.SUFFICIENT,
        evaluation=passing,
        candidate=candidate,
        missing_facet_ids=[],
        web_allowed=True,
        web_ready=True,
        internal_can_retrieve=False,
        remediation_used=False,
    )
    assert accepted.action is ReflectionAction.ACCEPT
    web = decide_reflection(
        internal_stop=StopReason.INTERNAL_KNOWLEDGE_MISSING,
        evaluation=None,
        candidate=None,
        missing_facet_ids=[facets[0].facet_id],
        web_allowed=True,
        web_ready=True,
        internal_can_retrieve=False,
        remediation_used=False,
    )
    assert web.action is ReflectionAction.WEB_SEARCH
    critic_gap = merge_critic_findings(
        passing,
        [
            AnswerFinding(
                code=AnswerCheckCode.UNSUPPORTED,
                reason="the selected paper does not support this claim",
                claim_ids=[candidate.claims[0].claim_id],
                facet_ids=[facets[0].facet_id],
            )
        ],
    )
    critic_web = decide_reflection(
        internal_stop=StopReason.SUFFICIENT,
        evaluation=critic_gap,
        candidate=candidate,
        missing_facet_ids=[facets[0].facet_id],
        web_allowed=True,
        web_ready=True,
        internal_can_retrieve=False,
        remediation_used=False,
    )
    assert critic_web.action is ReflectionAction.WEB_SEARCH
    for internal_stop in (StopReason.NO_POSITIVE_GAIN, StopReason.BUDGET_EXHAUSTED):
        exhausted_internal = decide_reflection(
            internal_stop=internal_stop,
            evaluation=None,
            candidate=None,
            missing_facet_ids=[facets[0].facet_id],
            web_allowed=True,
            web_ready=True,
            internal_can_retrieve=True,
            remediation_used=False,
        )
        assert exhausted_internal.action is ReflectionAction.WEB_SEARCH
    blocked = decide_reflection(
        internal_stop=StopReason.EXECUTION_FAILED,
        evaluation=None,
        candidate=None,
        missing_facet_ids=[facets[0].facet_id],
        web_allowed=True,
        web_ready=True,
        internal_can_retrieve=True,
        remediation_used=False,
    )
    assert blocked.action is ReflectionAction.CONSERVATIVE_STOP
    conservative = build_conservative_candidate(
        supported=candidate,
        missing_facet_ids=[facets[0].facet_id],
        reason=blocked.reason_code,
    )
    assert "证据边界" in conservative.answer
    assert conservative.claims == candidate.claims
