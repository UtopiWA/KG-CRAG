"""实时应用模型预热的就绪与快速失败测试。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path

import pytest

from kg_crag.answering.config import GroundedAnswerConfig
from kg_crag.answering.prompt import load_structured_prompt
from kg_crag.application.live import LiveQueryApplicationService, LiveQueryRuntime
from kg_crag.application.services import ApplicationServiceError, GroundedAnswerQueryAdapter
from kg_crag.correction.config import CorrectiveWorkflowConfig
from kg_crag.correction.executor import ActionExecutor
from kg_crag.correction.requirements import (
    MockRequirementProvider,
    RequirementBatch,
    RequirementCandidate,
    RequirementProviderResponse,
    analyze_question,
)
from kg_crag.models import (
    ApplicationMode,
    BudgetUsage,
    ComponentStatus,
    ConditionKind,
    Evidence,
    EvidenceSourceType,
    FacetKind,
    FacetStatus,
    HybridRetrievalResult,
    QueryRequest,
    RuntimeIdentitySummary,
    SatisfactionCondition,
    TokenUsageSource,
)
from kg_crag.providers import MockLLMProvider
from kg_crag.workflow.nodes.answer import AnswerWorkflowDependencies
from kg_crag.workflow.nodes.corrective import WorkflowDependencies


class _WarmupRuntime:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def warmup_models(self) -> None:
        self.started.set()
        await self.release.wait()

    async def run(self, request: QueryRequest) -> object:
        raise AssertionError(request)


class _StaticHybrid:
    def __init__(self, evidence: list[Evidence]) -> None:
        self.evidence = evidence

    async def retrieve(self, question: str) -> HybridRetrievalResult:
        return HybridRetrievalResult(
            run_id="hybrid-voyager",
            query_id="query-voyager",
            query=question,
            config_hash="a" * 64,
            corpus_snapshot_hash="b" * 64,
            collection_version="dense-v1",
            sparse_index_version="sparse-v1",
            fusion_version="fusion-v1",
            evidence=self.evidence,
        )


class _UnusedEmbedding:
    async def embed(self, *args: object, **kwargs: object) -> object:
        raise AssertionError((args, kwargs))


class _FacetAwareAnswerLLM:
    """从回答 Prompt 复制稳定 facet ID，避免测试硬编码运行标识。"""

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        del system_prompt, temperature
        facet_id = re.search(r"facet-[a-f0-9]{16}", prompt)
        assert facet_id is not None
        return json.dumps(
            {
                "claims": [
                    {
                        "text": "Voyager 与 ReAct 使用不同的智能体架构。",
                        "claim_type": "synthesis",
                        "citation_ids": ["E1"],
                        "facet_ids": [facet_id.group()],
                    }
                ],
                "confidence": 0.8,
            },
            ensure_ascii=False,
        )


class _StaticPaperContext:
    def __init__(self, evidence: list[Evidence]) -> None:
        self.evidence = evidence
        self.calls: list[tuple[str, int]] = []

    async def paper_context(self, paper_id: str, *, limit: int) -> list[Evidence]:
        self.calls.append((paper_id, limit))
        return [
            item.model_copy(
                update={
                    "metadata": {
                        **item.metadata,
                        "paper_context_expansion": True,
                    }
                },
                deep=True,
            )
            for item in self.evidence[:limit]
            if item.paper_id == paper_id
        ]


@pytest.mark.asyncio
async def test_preload_reports_warming_and_rejects_live_query_without_waiting() -> None:
    runtime = _WarmupRuntime()

    async def factory() -> object:
        return runtime

    service = LiveQueryApplicationService(factory, preload_on_start=True)  # type: ignore[arg-type]
    service.start_preload()
    await runtime.started.wait()

    readiness = service.readiness()
    assert readiness.status is ComponentStatus.WARMING
    with pytest.raises(ApplicationServiceError, match="正在预热") as captured:
        await service.query(
            QueryRequest(question="实时问题", mode=ApplicationMode.LIVE),
            request_id="request-warming",
        )
    assert captured.value.retryable is True

    runtime.release.set()
    assert service._preload_task is not None
    await service._preload_task
    assert service.readiness().status is ComponentStatus.READY


@pytest.mark.asyncio
async def test_preload_reports_dependency_failure_without_retry() -> None:
    calls = 0

    async def factory() -> object:
        nonlocal calls
        calls += 1
        raise RuntimeError("storage unavailable")

    service = LiveQueryApplicationService(
        factory,  # type: ignore[arg-type]
        preload_on_start=True,
    )
    service.start_preload()
    assert service._preload_task is not None
    await service._preload_task

    readiness = service.readiness()
    assert calls == 1
    assert readiness.status is ComponentStatus.DEGRADED
    assert "索引、存储与本地模型" in readiness.message


@pytest.mark.asyncio
async def test_voyager_live_question_returns_grounded_answer_without_false_conflict() -> None:
    question = "Voyager 在 Minecraft 中持续学习所依赖的三个核心组件是什么?"
    question_id = f"live-{hashlib.sha256(question.encode()).hexdigest()[:24]}"
    facet = analyze_question(question_id, question).facets[0]
    answer = {
        "claims": [
            {
                "text": (
                    "三个组件是自动课程、持续增长的可执行代码技能库，以及结合环境反馈、"
                    "执行错误和自我验证的迭代提示机制。"
                ),
                "claim_type": "fact",
                "citation_ids": ["E1"],
                "facet_ids": [facet.facet_id],
            }
        ],
        "confidence": 0.95,
    }
    evidence = [
        Evidence(
            evidence_id="a-voyager-abstract",
            content=(
                "VOYAGER is a lifelong learning agent in Minecraft without human intervention. "
                "It consists of three key components: an automatic curriculum, an ever-growing "
                "skill library of executable code, and an iterative prompting mechanism using "
                "environment feedback, execution errors, and self-verification. It obtains 3.3x "
                "more unique items and unlocks milestones 15.3x faster."
            ),
            source_type=EvidenceSourceType.CHUNK,
            source_id="chunk-voyager-abstract",
            paper_id="paper-voyager",
        ),
        Evidence(
            evidence_id="z-voyager-context",
            content="Voyager in Minecraft was evaluated on 5 tasks in 2024.",
            source_type=EvidenceSourceType.CHUNK,
            source_id="chunk-voyager-context",
            paper_id="paper-voyager",
        ),
    ]
    corrective = CorrectiveWorkflowConfig()
    executor = ActionExecutor({}, max_candidates=20, bounds=corrective.actions)
    llm = MockLLMProvider("```json\n" + json.dumps(answer, ensure_ascii=False) + "\n```")
    paper_context = _StaticPaperContext([evidence[0]])
    runtime = LiveQueryRuntime(
        hybrid=_StaticHybrid(evidence[1:]),  # type: ignore[arg-type]
        correction_config=corrective,
        correction_dependencies=WorkflowDependencies(config=corrective, executor=executor),
        answer_dependencies=AnswerWorkflowDependencies(
            config=GroundedAnswerConfig(
                enabled=True,
                critic={"enabled": False},
                web={"enabled": False, "provider": "disabled"},
            ),
            llm=llm,
            answer_prompt=load_structured_prompt(
                Path("prompts/grounded-answer-v1.txt"),
                placeholders={"question", "facets", "evidence_context", "constraints"},
            ),
            correction_config=corrective,
            action_executor=executor,
            model_revision="mock-voyager-v1",
        ),
        identity=RuntimeIdentitySummary(
            corpus_snapshot="c" * 64,
            dense_version="d" * 64,
            sparse_version="e" * 64,
        ),
        embedding=_UnusedEmbedding(),  # type: ignore[arg-type]
        reranker=None,
        paper_context_store=paper_context,  # type: ignore[arg-type]
    )
    response = await GroundedAnswerQueryAdapter(runtime.run).query(
        QueryRequest(question=question, mode=ApplicationMode.LIVE),
        request_id="request-voyager",
    )

    assert response.stop_reason == "accepted"
    assert "自动课程" in response.answer
    assert len(response.citations) == 1
    assert response.facets[0].status is FacetStatus.COVERED
    assert response.facets[0].reason == "threshold_met"
    assert response.actions == []
    assert response.budget.token_usage_source is TokenUsageSource.ESTIMATED
    assert len(llm.calls) == 1
    assert paper_context.calls == [("paper-voyager", 5)]


@pytest.mark.asyncio
async def test_complex_live_question_uses_one_budgeted_facet_call_without_workflow_repeat() -> None:
    question = "比较 Voyager 与 ReAct 的智能体架构"
    evidence = Evidence(
        evidence_id="architecture-comparison",
        content="Voyager 与 ReAct 使用不同的智能体架构。",
        source_type=EvidenceSourceType.CHUNK,
        source_id="chunk-architecture-comparison",
        paper_id="paper-comparison",
    )
    candidate = RequirementCandidate(
        kind=FacetKind.COMPARISON,
        description="Voyager 与 ReAct 的智能体架构比较",
        expected_evidence_types=[EvidenceSourceType.CHUNK],
        condition=SatisfactionCondition(
            kind=ConditionKind.TERMS,
            terms=["Voyager", "ReAct", "智能体架构"],
            min_term_matches=2,
        ),
        confidence=0.9,
    )
    requirement_provider = MockRequirementProvider(
        RequirementProviderResponse(
            batch=RequirementBatch(candidates=[candidate]),
            usage=BudgetUsage(llm_calls=1, input_tokens=100, output_tokens=40),
        )
    )
    corrective = CorrectiveWorkflowConfig(facets={"allow_llm": True})
    executor = ActionExecutor({}, max_candidates=20, bounds=corrective.actions)
    runtime = LiveQueryRuntime(
        hybrid=_StaticHybrid([evidence]),  # type: ignore[arg-type]
        correction_config=corrective,
        correction_dependencies=WorkflowDependencies(config=corrective, executor=executor),
        answer_dependencies=AnswerWorkflowDependencies(
            config=GroundedAnswerConfig(
                enabled=True,
                critic={"enabled": False},
                web={"enabled": False, "provider": "disabled"},
            ),
            llm=_FacetAwareAnswerLLM(),  # type: ignore[arg-type]
            answer_prompt=load_structured_prompt(
                Path("prompts/grounded-answer-v1.txt"),
                placeholders={"question", "facets", "evidence_context", "constraints"},
            ),
            correction_config=corrective,
            action_executor=executor,
            model_revision="mock-facet-live-v1",
        ),
        identity=RuntimeIdentitySummary(
            corpus_snapshot="c" * 64,
            dense_version="d" * 64,
            sparse_version="e" * 64,
        ),
        embedding=_UnusedEmbedding(),  # type: ignore[arg-type]
        reranker=None,
        requirement_provider=requirement_provider,
        requirement_prompt="只返回严格 JSON。",
    )

    execution = await runtime.run(QueryRequest(question=question, mode=ApplicationMode.LIVE))

    assert requirement_provider.calls == 1
    assert execution.correction.state.budget.used.llm_calls == 1
    assert execution.correction.state.budget.used.input_tokens == 100
    requirement_events = [
        item for item in execution.correction.state.trace if item.event == "requirements_generated"
    ]
    assert len(requirement_events) == 1
    assert requirement_events[0].details["source"] == "llm_precomputed"
    assert execution.answer.stop_reason.value == "accepted"
