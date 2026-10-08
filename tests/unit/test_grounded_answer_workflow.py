"""回答反思工作流的离线分支、预算与恢复测试。"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from kg_crag.answering import build_answer_identity
from kg_crag.answering.config import (
    GroundedAnswerConfig,
    WebSearchConfig,
    grounded_answer_config_hash,
)
from kg_crag.answering.prompt import load_structured_prompt
from kg_crag.correction.config import CorrectiveWorkflowConfig
from kg_crag.correction.coverage import build_coverage_matrix
from kg_crag.correction.executor import ActionExecutor, RetrieverTool
from kg_crag.correction.identity import build_run_identity
from kg_crag.correction.sufficiency import assess_sufficiency
from kg_crag.errors import KGCRAGError
from kg_crag.models import (
    AnswerCheckCode,
    AnswerFinding,
    AnswerStopReason,
    BudgetLedger,
    BudgetUsage,
    ConditionKind,
    CorrectionAction,
    CorrectionRunResult,
    CorrectionState,
    ErrorCode,
    ErrorDetail,
    Evidence,
    EvidenceRequirement,
    EvidenceSourceType,
    FacetKind,
    SatisfactionCondition,
    SearchResult,
    StopReason,
    StopResult,
    WebSourceType,
)
from kg_crag.providers import LLMGeneration, MockSearchProvider
from kg_crag.retrieval.mock import MockRetriever
from kg_crag.workflow.answer import run_grounded_answer_workflow
from kg_crag.workflow.answer_artifacts import AnswerCheckpointStore
from kg_crag.workflow.nodes import AnswerWorkflowDependencies, prepare_answer_node


class QueueLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        del system_prompt, temperature
        self.calls.append(prompt)
        if not self.responses:
            raise AssertionError("unexpected LLM call")
        return self.responses.pop(0)


class UsageQueueLLM(QueueLLM):
    async def generate_with_usage(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> LLMGeneration:
        content = await self.generate(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
        )
        return LLMGeneration(content=content, input_tokens=120, output_tokens=40)


class RetryableQueueLLM(QueueLLM):
    def __init__(self, responses: list[str]) -> None:
        super().__init__(responses)
        self.failed_once = False
        # 与实时 GLM Provider 的单次输出上限一致，防止测试绕过真实预算约束。
        self.max_output_tokens = 1600

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        if not self.failed_once:
            self.failed_once = True
            self.calls.append(prompt)
            raise KGCRAGError(
                ErrorDetail(
                    code=ErrorCode.EXTERNAL_SERVICE,
                    message="temporary provider failure",
                    retryable=True,
                    context={"status": 503, "error_type": "ServiceUnavailable"},
                )
            )
        return await super().generate(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
        )


class StubCritic:
    def __init__(self) -> None:
        self.prompt = load_structured_prompt(
            Path("prompts/answer-critic-v1.txt"),
            placeholders={"question", "claims", "evidence_context", "facets", "conflicts"},
        )
        self.calls = 0

    async def evaluate(self, **values: object) -> list[AnswerFinding]:
        self.calls += 1
        candidate = values["candidate"]
        claim = candidate.claims[0]  # type: ignore[union-attr]
        return [
            AnswerFinding(
                code=AnswerCheckCode.UNSUPPORTED,
                reason="claim is broader than excerpt",
                claim_ids=[claim.claim_id],
                citation_ids=claim.citation_ids,
                facet_ids=claim.facet_ids,
            )
        ]


class FailingSearch:
    def __init__(self) -> None:
        self.calls = 0

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        del query, max_results
        self.calls += 1
        raise TimeoutError("bounded timeout")


def _facet() -> EvidenceRequirement:
    return EvidenceRequirement(
        facet_id="facet-" + "a" * 16,
        question_id="q1",
        kind=FacetKind.CONTENT,
        description="skill library architecture",
        expected_evidence_types=[EvidenceSourceType.CHUNK],
        condition=SatisfactionCondition(
            kind=ConditionKind.TERMS,
            terms=["skill", "library"],
            min_term_matches=2,
        ),
    )


def _evidence() -> Evidence:
    return Evidence(
        evidence_id="ev-1",
        content="The architecture uses a skill library.",
        source_type=EvidenceSourceType.CHUNK,
        source_id="chunk-1",
        paper_id="paper-1",
    )


def _correction(
    reason: StopReason,
    *,
    evidence: list[Evidence] | None = None,
    retrieval_round: int = 1,
) -> CorrectionRunResult:
    facets = [_facet()]
    candidates = evidence or []
    matrix = build_coverage_matrix(facets, candidates, rule_version="coverage-v1")
    sufficiency = assess_sufficiency(
        facets,
        candidates,
        matrix,
        min_support=0.1,
        min_sources=1,
        max_selected=8,
    )
    by_id = {item.evidence_id: item for item in candidates}
    selected = [by_id[item] for item in sufficiency.selected_evidence_ids]
    identity = build_run_identity(
        "What architecture is used?",
        facets,
        candidates,
        config_hash="a" * 64,
        prompt_version="prompt-v1",
        model_revision="model-v1",
        corpus_snapshot="corpus-v1",
        dense_version="dense-v1",
        sparse_version="sparse-v1",
        graph_version="graph-v1",
        rules_version="rules-v1",
        coverage_version="coverage-v1",
        policy_version="policy-v1",
    )
    budget = BudgetLedger(used=BudgetUsage(retrieval_rounds=retrieval_round))
    stop = StopResult(
        reason=reason,
        message=reason.value,
        missing_facet_ids=sufficiency.missing_required_facet_ids,
        selected_evidence_ids=sufficiency.selected_evidence_ids,
        budget=budget,
    )
    state = CorrectionState(
        identity=identity,
        question_id="q1",
        question="What architecture is used?",
        facets=facets,
        candidates=candidates,
        selected_evidence=selected,
        coverage_matrix=matrix,
        sufficiency=sufficiency,
        retrieval_round=retrieval_round,
        budget=budget,
        stop=stop,
    )
    now = datetime.now(UTC)
    return CorrectionRunResult(
        identity=identity,
        state=state,
        stop=stop,
        started_at=now,
        finished_at=now,
    )


def _answer_json(text: str = "It uses a skill library.") -> str:
    return json.dumps(
        {
            "claims": [
                {
                    "text": text,
                    "claim_type": "fact",
                    "citation_ids": ["E1"],
                    "facet_ids": [_facet().facet_id],
                }
            ],
            "confidence": 0.8,
        }
    )


def _two_claim_answer_json() -> str:
    payload = json.loads(_answer_json("Broad unsupported claim."))
    payload["claims"].append(
        {
            "text": "It uses a skill library.",
            "claim_type": "fact",
            "citation_ids": ["E1"],
            "facet_ids": [_facet().facet_id],
        }
    )
    return json.dumps(payload)


def _deps(
    llm: QueueLLM,
    *,
    config: GroundedAnswerConfig | None = None,
    search: object | None = None,
    critic: object | None = None,
    corrective: CorrectiveWorkflowConfig | None = None,
    executor: ActionExecutor | None = None,
) -> AnswerWorkflowDependencies:
    selected = config or GroundedAnswerConfig(
        enabled=True,
        critic={"enabled": False},
        web={"enabled": False, "provider": "disabled"},
    )
    return AnswerWorkflowDependencies(
        config=selected,
        llm=llm,
        answer_prompt=load_structured_prompt(
            Path("prompts/grounded-answer-v1.txt"),
            placeholders={"question", "facets", "evidence_context", "constraints"},
        ),
        critic=critic,  # type: ignore[arg-type]
        search_provider=search,  # type: ignore[arg-type]
        correction_config=corrective,
        action_executor=executor,
        model_revision="mock-v1",
    )


@pytest.mark.asyncio
async def test_internal_sufficient_answers_without_web_and_nodes_do_not_mutate() -> None:
    correction = _correction(StopReason.SUFFICIENT, evidence=[_evidence()])
    llm = QueueLLM([_answer_json()])
    search = MockSearchProvider([])
    deps = _deps(llm, search=search)
    identity = __import__("kg_crag.answering", fromlist=["build_answer_identity"])
    answer_identity = identity.build_answer_identity(
        correction,
        config_hash="b" * 64,
        answer_prompt_version=deps.answer_prompt.version,
        critic_prompt_version="0" * 64,
        checker_version=deps.config.prompts.checker_version,
        policy_version=deps.config.prompts.policy_version,
        model_revision="mock-v1",
        search_provider="disabled",
        search_provider_version="disabled-v1",
    )
    prepared = prepare_answer_node(answer_identity, correction, deps)
    before = prepared.model_dump(mode="json")
    result = await run_grounded_answer_workflow(correction, deps, allow_web=True)
    assert prepared.model_dump(mode="json") == before
    assert result.stop_reason is AnswerStopReason.ACCEPTED
    assert result.used_external is False
    assert search.calls == []
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_disabled_workflow_returns_conservative_result_without_calls() -> None:
    correction = _correction(StopReason.SUFFICIENT, evidence=[_evidence()])
    llm = QueueLLM([])
    result = await run_grounded_answer_workflow(
        correction,
        _deps(
            llm,
            config=GroundedAnswerConfig(
                enabled=False,
                critic={"enabled": False},
                web={"enabled": False, "provider": "disabled"},
            ),
        ),
    )
    assert result.stop_reason is AnswerStopReason.CONSERVATIVE
    assert not llm.calls


@pytest.mark.asyncio
async def test_generation_failure_is_classified_without_exposing_raw_response() -> None:
    correction = _correction(StopReason.SUFFICIENT, evidence=[_evidence()])
    result = await run_grounded_answer_workflow(correction, _deps(QueueLLM(["not-json"])))

    assert result.stop_reason is AnswerStopReason.GENERATION_FAILED
    assert result.errors[-1].context == {"failure_category": "invalid_json"}
    failure = next(item for item in result.trace if item.event == "answer_generation_failed")
    assert failure.details["failure_category"] == "invalid_json"
    assert failure.details["usage_estimated"] is True
    assert "not-json" not in result.model_dump_json()


@pytest.mark.asyncio
async def test_retryable_provider_failure_gets_one_traceable_retry() -> None:
    correction = _correction(StopReason.SUFFICIENT, evidence=[_evidence()])
    llm = RetryableQueueLLM([_answer_json()])

    result = await run_grounded_answer_workflow(correction, _deps(llm))

    assert result.stop_reason is AnswerStopReason.ACCEPTED
    assert len(llm.calls) == 2
    assert any(item.event == "answer_generation_retry_scheduled" for item in result.trace)
    failures = [item for item in result.trace if item.event == "answer_generation_failed"]
    assert failures[0].details["provider_status"] == 503
    assert failures[0].details["retryable"] is True


@pytest.mark.asyncio
async def test_abstention_claim_is_regenerated_instead_of_accepted() -> None:
    correction = _correction(StopReason.SUFFICIENT, evidence=[_evidence()])
    refusal = _answer_json("暂无可安全陈述的结论。")
    llm = QueueLLM([refusal, _answer_json()])

    result = await run_grounded_answer_workflow(correction, _deps(llm))

    assert result.stop_reason is AnswerStopReason.ACCEPTED
    assert "skill library" in result.answer
    assert len(llm.calls) == 2
    decisions = [
        item.details.get("action") for item in result.trace if item.event == "reflection_decided"
    ]
    assert decisions == ["regenerate", "accept"]


@pytest.mark.asyncio
async def test_provider_actual_usage_replaces_character_estimate() -> None:
    correction = _correction(StopReason.SUFFICIENT, evidence=[_evidence()])
    result = await run_grounded_answer_workflow(
        correction,
        _deps(UsageQueueLLM([_answer_json()])),
    )

    assert result.stop_reason is AnswerStopReason.ACCEPTED
    assert result.budget.answer.used.input_tokens == 120
    assert result.budget.answer.used.output_tokens == 40
    generated = next(item for item in result.trace if item.event == "answer_generated")
    assert generated.details["usage_estimated"] is False


@pytest.mark.asyncio
async def test_web_only_triggers_for_internal_knowledge_missing() -> None:
    correction = _correction(StopReason.INTERNAL_KNOWLEDGE_MISSING)
    llm = QueueLLM([_answer_json()])
    result = SearchResult(
        provider_result_id="r1",
        title="Paper",
        url="https://arxiv.org/abs/1234.5678",
        accessed_at=datetime.now(UTC),
        excerpt="The architecture uses a skill library.",
        source_type=WebSourceType.ARXIV,
        score=0.9,
    )
    search = MockSearchProvider([result])
    config = GroundedAnswerConfig(
        enabled=True,
        critic={"enabled": False},
        web=WebSearchConfig(enabled=True, provider="mock"),
    )
    answer = await run_grounded_answer_workflow(
        correction, _deps(llm, config=config, search=search), allow_web=True
    )
    assert answer.stop_reason is AnswerStopReason.ACCEPTED
    assert answer.used_external is True
    assert answer.external_evidence[0].external is True
    assert len(search.calls) == 1
    assert answer.budget.answer.used.web_calls == 1


@pytest.mark.asyncio
async def test_web_disabled_and_failed_return_explicit_conservative_boundary() -> None:
    correction = _correction(StopReason.INTERNAL_KNOWLEDGE_MISSING)
    disabled_llm = QueueLLM([])
    disabled = await run_grounded_answer_workflow(correction, _deps(disabled_llm), allow_web=False)
    assert disabled.stop_reason is AnswerStopReason.SEARCH_FAILED
    assert disabled.missing_required_facet_ids == [_facet().facet_id]
    assert not disabled_llm.calls

    failing = FailingSearch()
    config = GroundedAnswerConfig(
        enabled=True,
        critic={"enabled": False},
        web=WebSearchConfig(enabled=True, provider="mock"),
    )
    failed = await run_grounded_answer_workflow(
        correction,
        _deps(QueueLLM([]), config=config, search=failing),
        allow_web=True,
    )
    assert failed.stop_reason in {AnswerStopReason.CONSERVATIVE, AnswerStopReason.SEARCH_FAILED}
    assert failing.calls == 1
    assert failed.errors


@pytest.mark.asyncio
async def test_critic_can_trigger_exactly_one_bounded_regeneration() -> None:
    correction = _correction(StopReason.SUFFICIENT, evidence=[_evidence()])
    llm = QueueLLM([_two_claim_answer_json(), _answer_json("Skill library.")])
    critic = StubCritic()
    config = GroundedAnswerConfig(enabled=True, web={"enabled": False, "provider": "disabled"})
    result = await run_grounded_answer_workflow(
        correction, _deps(llm, config=config, critic=critic)
    )
    assert result.stop_reason is AnswerStopReason.ACCEPTED
    assert len(llm.calls) == 2
    assert critic.calls == 1
    assert result.budget.answer.used.answer_calls == 2
    assert result.budget.answer.used.critic_calls == 1
    assert result.budget.answer.used.reflection_rounds == 1


@pytest.mark.asyncio
async def test_reretrieval_reuses_internal_executor_and_original_budget() -> None:
    correction = _correction(StopReason.NO_POSITIVE_GAIN, retrieval_round=1)
    retriever = MockRetriever([_evidence()])
    corrective = CorrectiveWorkflowConfig()
    executor = ActionExecutor(
        {CorrectionAction.HYBRID: RetrieverTool(retriever)},
        max_candidates=20,
        bounds=corrective.actions,
    )
    result = await run_grounded_answer_workflow(
        correction,
        _deps(
            QueueLLM([_answer_json()]),
            corrective=corrective,
            executor=executor,
        ),
    )
    assert result.stop_reason is AnswerStopReason.ACCEPTED
    assert result.budget.internal.used.retrieval_rounds == 2
    assert len(retriever.calls) == 1


@pytest.mark.asyncio
async def test_unsuccessful_reretrieval_does_not_spend_answer_generation_call() -> None:
    correction = _correction(StopReason.NO_POSITIVE_GAIN, retrieval_round=1)
    irrelevant = _evidence().model_copy(
        update={"evidence_id": "ev-irrelevant", "content": "Unrelated content."}
    )
    retriever = MockRetriever([irrelevant])
    corrective = CorrectiveWorkflowConfig()
    executor = ActionExecutor(
        {CorrectionAction.HYBRID: RetrieverTool(retriever)},
        max_candidates=20,
        bounds=corrective.actions,
    )
    llm = QueueLLM([])

    result = await run_grounded_answer_workflow(
        correction,
        _deps(llm, corrective=corrective, executor=executor),
    )

    assert result.stop_reason is AnswerStopReason.CONSERVATIVE
    assert result.budget.answer.used.answer_calls == 0
    assert not llm.calls


@pytest.mark.asyncio
async def test_checkpointed_final_result_prevents_repeated_provider_calls(
    tmp_path: Path,
) -> None:
    correction = _correction(StopReason.SUFFICIENT, evidence=[_evidence()])
    llm = QueueLLM([_answer_json()])
    deps = _deps(llm)
    store = AnswerCheckpointStore(tmp_path / "answer-runs", workspace_root=tmp_path)
    first = await run_grounded_answer_workflow(correction, deps, checkpoint_store=store)
    second = await run_grounded_answer_workflow(correction, deps, checkpoint_store=store)
    assert second == first
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_recovery_from_call_boundary_never_replays_llm(tmp_path: Path) -> None:
    correction = _correction(StopReason.SUFFICIENT, evidence=[_evidence()])
    llm = QueueLLM([])
    deps = _deps(llm)
    identity = build_answer_identity(
        correction,
        config_hash=grounded_answer_config_hash(deps.config),
        answer_prompt_version=deps.answer_prompt.version,
        critic_prompt_version=(deps.critic.prompt.version if deps.critic else "0" * 64),
        checker_version=deps.config.prompts.checker_version,
        policy_version=deps.config.prompts.policy_version,
        model_revision=deps.model_revision,
        search_provider=deps.config.web.provider,
        search_provider_version=deps.config.web.provider_version,
    )
    store = AnswerCheckpointStore(tmp_path / "answer-runs", workspace_root=tmp_path)
    store.save(prepare_answer_node(identity, correction, deps), 0)

    result = await run_grounded_answer_workflow(correction, deps, checkpoint_store=store)

    assert result.stop_reason is AnswerStopReason.CONSERVATIVE
    assert not llm.calls
