"""任意问题实时链路的惰性装配；应用层只连接既有领域能力。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

from kg_crag.answering import SemanticCritic
from kg_crag.answering.config import load_grounded_answer_config
from kg_crag.answering.prompt import load_structured_prompt
from kg_crag.application.services import (
    ApplicationServiceError,
    GroundedAnswerQueryAdapter,
    LiveQueryExecution,
)
from kg_crag.correction.config import (
    CorrectiveWorkflowConfig,
    corrective_config_hash,
    load_corrective_workflow_config,
)
from kg_crag.correction.coverage import entity_anchor_terms
from kg_crag.correction.executor import (
    ActionExecutor,
    CompoundTool,
    CorrectionTool,
    GraphTool,
    HybridTool,
    RetrieverTool,
)
from kg_crag.correction.identity import build_run_identity
from kg_crag.correction.requirements import (
    LLMRequirementProvider,
    RequirementCache,
    RequirementProvider,
    generate_budgeted_requirements,
)
from kg_crag.errors import KGCRAGError
from kg_crag.models import (
    ApplicationErrorCode,
    BudgetLedger,
    ComponentReadiness,
    ComponentStatus,
    CorrectionAction,
    CorrectionState,
    Evidence,
    EvidenceRequirement,
    QueryRequest,
    QueryResponse,
    RuntimeIdentitySummary,
    SparseIndexIdentity,
    TraceSummary,
)
from kg_crag.providers import (
    EmbeddingService,
    LLMProvider,
    MockLLMProvider,
    MockSearchProvider,
    OpenAICompatibleLLMProvider,
    RecordedSearchProvider,
    SearchProvider,
)
from kg_crag.retrieval import (
    CrossEncoderReranker,
    GraphRetriever,
    HybridRetrievalService,
    ProcessedSourceResolver,
    SparseRetriever,
    load_hybrid_retrieval_config,
)
from kg_crag.retrieval.dense import DenseRetriever
from kg_crag.runtime import build_embedding_service, build_vector_store
from kg_crag.settings import Settings
from kg_crag.sparse_store import SparseStore, SQLiteSparseStore, load_sqlite_index_identity
from kg_crag.web import TavilySearchProvider
from kg_crag.workflow.answer import run_grounded_answer_workflow
from kg_crag.workflow.corrective import run_corrective_workflow
from kg_crag.workflow.nodes.answer import AnswerWorkflowDependencies
from kg_crag.workflow.nodes.corrective import WorkflowDependencies


class LiveQueryRuntime:
    """单进程复用已校验的索引客户端，并按单请求执行现有工作流。"""

    def __init__(
        self,
        *,
        hybrid: HybridRetrievalService,
        correction_config: CorrectiveWorkflowConfig,
        correction_dependencies: WorkflowDependencies,
        answer_dependencies: AnswerWorkflowDependencies,
        identity: RuntimeIdentitySummary,
        embedding: EmbeddingService,
        reranker: CrossEncoderReranker | None,
        paper_context_store: SparseStore | None = None,
        paper_context_limit: int = 5,
        requirement_provider: RequirementProvider | None = None,
        requirement_prompt: str = "",
        requirement_cache: RequirementCache | None = None,
    ) -> None:
        self.hybrid = hybrid
        self.correction_config = correction_config
        self.correction_dependencies = correction_dependencies
        self.answer_dependencies = answer_dependencies
        self.identity = identity
        self.embedding = embedding
        self.reranker = reranker
        self.paper_context_store = paper_context_store
        self.paper_context_limit = paper_context_limit
        self.requirement_provider = requirement_provider
        self.requirement_prompt = requirement_prompt
        self.requirement_cache = requirement_cache

    async def warmup_models(self) -> None:
        """提前完成本地模型加载和最小推理，不访问 Qdrant、LLM 或 Web。"""

        await self.embedding.embed(
            ["scientific literature retrieval"],
            input_type="query",
            use_cache=False,
        )
        if self.reranker is not None:
            await self.reranker.warmup()

    async def run(self, request: QueryRequest) -> LiveQueryExecution:
        question = " ".join(request.question.split())
        question_id = f"live-{__import__('hashlib').sha256(question.encode()).hexdigest()[:24]}"
        config = self.correction_dependencies.config
        facets, facet_budget, _source = await generate_budgeted_requirements(
            question_id,
            question,
            config.facets,
            BudgetLedger(limit=config.budget),
            provider=self.requirement_provider,
            system_prompt=self.requirement_prompt,
            cache=self.requirement_cache,
        )
        retrieval = await self.hybrid.retrieve(question)
        initial_evidence = await self._expand_paper_context(
            facets,
            retrieval.evidence,
        )
        identity = build_run_identity(
            question,
            facets,
            initial_evidence,
            config_hash=corrective_config_hash(config),
            prompt_version=config.facets.prompt_version,
            model_revision=self.answer_dependencies.model_revision,
            corpus_snapshot=self.identity.corpus_snapshot,
            dense_version=self.identity.dense_version,
            sparse_version=self.identity.sparse_version,
            graph_version=self.identity.graph_version or "disabled",
            rules_version=config.facets.version,
            coverage_version=config.coverage.version,
            policy_version=config.policy_version,
        )
        initial = CorrectionState(
            identity=identity,
            question_id=question_id,
            question=question,
            facets=facets,
            budget=facet_budget,
        )
        dependencies = replace(
            self.correction_dependencies,
            initial_evidence=tuple(initial_evidence),
        )
        correction = await run_corrective_workflow(initial, dependencies)
        answer = await run_grounded_answer_workflow(
            correction,
            self.answer_dependencies,
            allow_web=request.allow_web,
        )
        return LiveQueryExecution(
            answer=answer,
            correction=correction,
            runtime_identity=self.identity,
        )

    async def _expand_paper_context(
        self,
        facets: list[EvidenceRequirement],
        evidence: list[Evidence],
    ) -> list[Evidence]:
        """从同一冻结 Sparse 索引补入首个已锚定论文的摘要优先上下文。"""

        if self.paper_context_store is None:
            return [item.model_copy(deep=True) for item in evidence]
        anchors = {anchor for facet in facets for anchor in entity_anchor_terms(facet)}
        if not anchors:
            return [item.model_copy(deep=True) for item in evidence]
        paper_id = next(
            (
                item.paper_id
                for item in evidence
                if item.paper_id and any(anchor in item.content.casefold() for anchor in anchors)
            ),
            None,
        )
        if paper_id is None:
            return [item.model_copy(deep=True) for item in evidence]
        try:
            expanded = await self.paper_context_store.paper_context(
                paper_id,
                limit=self.paper_context_limit,
            )
        except KGCRAGError:
            return [item.model_copy(deep=True) for item in evidence]
        prioritized = [
            item.model_copy(
                update={
                    "metadata": {
                        **item.metadata,
                        "paper_context_expansion": True,
                    }
                },
                deep=True,
            )
            for item in expanded
        ]
        by_source: dict[str, Evidence] = {}
        for item in [*prioritized, *evidence]:
            by_source.setdefault(item.source_id, item.model_copy(deep=True))
        return list(by_source.values())


class LiveQueryApplicationService:
    """按配置后台预热或在首次请求装配依赖的并发安全门面。"""

    def __init__(
        self,
        factory: Callable[[], Awaitable[LiveQueryRuntime]],
        *,
        preload_on_start: bool = False,
    ) -> None:
        self._factory = factory
        self._runtime: LiveQueryRuntime | None = None
        self._adapter: GroundedAnswerQueryAdapter | None = None
        self._lock = asyncio.Lock()
        self._failure: str | None = None
        self._preload_on_start = preload_on_start
        self._preload_task: asyncio.Task[None] | None = None
        self._prewarmed = False

    def start_preload(self) -> None:
        """在 API 事件循环中启动一次后台预热；重复调用保持幂等。"""

        if not self._preload_on_start or self._preload_task is not None:
            return
        self._preload_task = asyncio.create_task(self._preload_models())

    async def shutdown(self) -> None:
        """停止尚未完成的预热协程；线程中的当前本地加载由进程退出回收。"""

        if self._preload_task is None or self._preload_task.done():
            return
        self._preload_task.cancel()
        with suppress(asyncio.CancelledError):
            await self._preload_task

    async def _preload_models(self) -> None:
        try:
            await self._get_adapter()
            if self._runtime is None:  # pragma: no cover - 防御不完整工厂
                raise RuntimeError("live runtime was not initialized")
            await self._runtime.warmup_models()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # 就绪端点只暴露安全类别，不泄漏模型仓库地址或本地路径。
            self._failure = type(exc).__name__
        else:
            self._failure = None
            self._prewarmed = True

    async def _get_adapter(self) -> GroundedAnswerQueryAdapter:
        if self._adapter is not None:
            return self._adapter
        async with self._lock:
            if self._adapter is not None:
                return self._adapter
            try:
                self._runtime = await self._factory()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._failure = type(exc).__name__
                raise ApplicationServiceError(
                    ApplicationErrorCode.DEPENDENCY_UNAVAILABLE,
                    "实时查询依赖未就绪，请检查索引、Qdrant 与模型配置。",
                    status_code=503,
                    retryable=True,
                ) from exc
            self._adapter = GroundedAnswerQueryAdapter(self._runtime.run)
            self._failure = None
            return self._adapter

    async def query(self, request: QueryRequest, *, request_id: str) -> QueryResponse:
        if self._preload_task is not None and not self._preload_task.done():
            raise ApplicationServiceError(
                ApplicationErrorCode.DEPENDENCY_UNAVAILABLE,
                "实时模型正在预热，请稍后重试。",
                status_code=503,
                retryable=True,
            )
        if self._preload_task is not None and self._failure is not None:
            raise ApplicationServiceError(
                ApplicationErrorCode.DEPENDENCY_UNAVAILABLE,
                "实时工作流预热失败，请检查索引、存储与本地模型配置。",
                status_code=503,
                retryable=True,
            )
        adapter = await self._get_adapter()
        try:
            return await adapter.query(request, request_id=request_id)
        except asyncio.CancelledError:
            raise
        except ApplicationServiceError:
            raise
        except KGCRAGError as exc:
            raise ApplicationServiceError(
                ApplicationErrorCode.DEPENDENCY_UNAVAILABLE,
                "实时查询依赖调用失败，未使用回放结果替代。",
                status_code=503,
                retryable=exc.detail.retryable,
            ) from exc

    def trace(self, trace_id: str, *, max_events: int) -> TraceSummary | None:
        return self._adapter.trace(trace_id, max_events=max_events) if self._adapter else None

    def readiness(self) -> ComponentReadiness:
        if self._preload_task is not None and not self._preload_task.done():
            status, message = ComponentStatus.WARMING, "实时模型正在后台预热。"
        elif self._failure is not None:
            status = ComponentStatus.DEGRADED
            message = "实时工作流预热失败；请检查索引、存储与本地模型配置。"
        elif self._prewarmed:
            status, message = ComponentStatus.READY, "实时工作流与本地模型已预热。"
        elif self._adapter is not None:
            status, message = ComponentStatus.READY, "实时工作流已惰性装配。"
        else:
            status, message = ComponentStatus.READY, "实时工作流已配置，将在首次请求时加载。"
        return ComponentReadiness(
            component="live_query",
            status=status,
            required=False,
            message=message,
        )


def _select_sparse_identity(
    workspace_root: Path, settings: Settings, index_root: str
) -> tuple[Path, SparseIndexIdentity]:
    root = workspace_root / index_root
    if settings.live_sparse_index_version:
        versions = [settings.live_sparse_index_version]
    else:
        versions = (
            sorted(path.name for path in root.iterdir() if path.is_dir()) if root.is_dir() else []
        )
        if len(versions) != 1:
            raise ValueError(
                "live_sparse_index_version is required when zero or multiple indices exist"
            )
    db_path = root / versions[0] / "index.sqlite3"
    return db_path, load_sqlite_index_identity(db_path)


def _dense_manifest_identity(
    workspace_root: Path,
    dense_version: str,
    corpus_snapshot: str,
) -> None:
    manifests = sorted((workspace_root / "data/processed/index-runs").glob("*/manifest.json"))
    for path in reversed(manifests):
        payload = json.loads(path.read_text(encoding="utf-8"))
        collection = payload.get("collection", {})
        if (
            collection.get("collection_version") == dense_version
            and payload.get("corpus_snapshot_hash") == corpus_snapshot
            and payload.get("dry_run") is False
        ):
            return
    raise ValueError("no completed Dense manifest matches the Sparse corpus identity")


async def build_live_query_runtime(
    settings: Settings,
    *,
    workspace_root: Path,
) -> LiveQueryRuntime:
    """校验冻结身份后组装既有 Retriever、纠错和回答工作流。"""

    retrieval_config = load_hybrid_retrieval_config(
        workspace_root / settings.live_retrieval_config_path
    )
    if (
        not settings.live_enable_reranker
        or retrieval_config.reranker.cache_root != settings.model_cache_root
        or retrieval_config.reranker.device != settings.reranker_device
    ):
        payload = retrieval_config.model_dump(mode="python")
        payload["reranker"].update(
            {
                "enabled": settings.live_enable_reranker,
                "cache_root": settings.model_cache_root,
                "device": settings.reranker_device,
            }
        )
        retrieval_config = type(retrieval_config).model_validate(payload)
    db_path, sparse_identity = _select_sparse_identity(
        workspace_root,
        settings,
        retrieval_config.sparse.index_root,
    )
    corpus_snapshot = sparse_identity.corpus_snapshot_hash
    if settings.live_corpus_snapshot and settings.live_corpus_snapshot != corpus_snapshot:
        raise ValueError("configured corpus snapshot does not match the Sparse index")

    sparse_store = SQLiteSparseStore(
        db_path,
        sparse_identity,
        max_top_k=retrieval_config.sparse.max_top_k,
        max_candidates=retrieval_config.sparse.max_candidates,
        max_query_chars=retrieval_config.sparse.max_query_chars,
        max_query_tokens=retrieval_config.sparse.max_query_tokens,
    )
    await sparse_store.ensure_index(sparse_identity)
    vector_store = build_vector_store(retrieval_config.dense, settings)
    await vector_store.ensure_existing_collection()
    _dense_manifest_identity(workspace_root, vector_store.collection_version, corpus_snapshot)
    embedding = build_embedding_service(
        retrieval_config.dense,
        workspace_root=workspace_root,
        settings=settings,
    )
    dense = DenseRetriever(embedding, vector_store, retrieval_config.dense)
    sparse = SparseRetriever(sparse_store, retrieval_config.sparse)
    reranker = (
        CrossEncoderReranker(
            retrieval_config.reranker,
            workspace_root=workspace_root,
            local_files_only=settings.model_local_files_only,
        )
        if retrieval_config.reranker.enabled
        else None
    )
    hybrid = HybridRetrievalService(
        dense,
        sparse,
        retrieval_config,
        collection_version=vector_store.collection_version,
        sparse_index_version=sparse_identity.index_version,
        corpus_snapshot_hash=corpus_snapshot,
        reranker=reranker,
        workspace_root=workspace_root,
    )

    correction_config = load_corrective_workflow_config(
        workspace_root / settings.live_workflow_config_path
    )
    llm = _build_llm(settings)
    requirement_prompt = load_structured_prompt(
        workspace_root / correction_config.facets.prompt_path,
        placeholders=set(),
    ).template
    requirement_provider = (
        LLMRequirementProvider(
            llm,
            revision=f"{settings.llm_model}@{correction_config.facets.prompt_version}",
        )
        if correction_config.facets.allow_llm
        else None
    )
    requirement_cache = RequirementCache(
        workspace_root / correction_config.artifacts.cache_root / "requirements"
    )
    enabled_actions: list[CorrectionAction] = [
        item for item in correction_config.actions.enabled if item is not CorrectionAction.GRAPH
    ]
    graph_tool = None
    graph_version: str | None = None
    if settings.live_enable_graph:
        graph_tool, graph_version = await _build_graph_tool(workspace_root, settings)
        enabled_actions.append(CorrectionAction.GRAPH)
    correction_payload = correction_config.model_dump(mode="python")
    correction_payload["actions"]["enabled"] = enabled_actions
    correction_config = type(correction_config).model_validate(correction_payload)
    tools: dict[CorrectionAction, CorrectionTool] = {
        CorrectionAction.DENSE: RetrieverTool(dense),
        CorrectionAction.SPARSE: RetrieverTool(sparse),
        CorrectionAction.HYBRID: HybridTool(hybrid),
        CorrectionAction.REWRITE: CompoundTool(dense),
        CorrectionAction.DECOMPOSE: CompoundTool(dense),
        CorrectionAction.ADJUST: CompoundTool(sparse),
    }
    if graph_tool is not None:
        tools[CorrectionAction.GRAPH] = graph_tool
    executor = ActionExecutor(
        tools,
        max_candidates=correction_config.actions.max_candidates_per_action,
        bounds=correction_config.actions,
        max_evidence_chars=correction_config.coverage.max_evidence_chars,
    )
    correction_dependencies = WorkflowDependencies(
        config=correction_config,
        executor=executor,
    )

    answer_config = load_grounded_answer_config(workspace_root / settings.live_workflow_config_path)
    answer_payload = answer_config.model_dump(mode="python")
    answer_payload["enabled"] = True
    answer_payload["critic"]["enabled"] = settings.live_enable_answer_critic
    web_enabled = settings.enable_web_fallback and settings.web_search_provider != "disabled"
    answer_payload["web"].update(
        {
            "enabled": web_enabled,
            "provider": settings.web_search_provider if web_enabled else "disabled",
            "provider_version": (
                f"{settings.web_search_provider}-v1" if web_enabled else "disabled-v1"
            ),
        }
    )
    answer_config = type(answer_config).model_validate(answer_payload)
    answer_prompt = load_structured_prompt(
        workspace_root / answer_config.prompts.answer_path,
        placeholders={"question", "facets", "evidence_context", "constraints"},
    )
    critic_prompt = load_structured_prompt(
        workspace_root / answer_config.prompts.critic_path,
        placeholders={"question", "claims", "evidence_context", "facets", "conflicts"},
    )
    critic = (
        SemanticCritic(
            llm,
            critic_prompt,
            max_prompt_chars=answer_config.critic.max_context_chars,
            max_response_chars=answer_config.critic.max_response_chars,
        )
        if answer_config.critic.enabled
        else None
    )
    answer_dependencies = AnswerWorkflowDependencies(
        config=answer_config,
        llm=llm,
        answer_prompt=answer_prompt,
        critic=critic,
        search_provider=_build_search(
            settings, answer_config.web.recorded_fixture_path, workspace_root
        ),
        correction_config=correction_config,
        action_executor=executor,
        model_revision=settings.llm_model,
    )
    public_identity = RuntimeIdentitySummary(
        corpus_snapshot=corpus_snapshot,
        dense_version=vector_store.collection_version,
        sparse_version=sparse_identity.index_version,
        graph_version=graph_version,
    )
    return LiveQueryRuntime(
        hybrid=hybrid,
        correction_config=correction_config,
        correction_dependencies=correction_dependencies,
        answer_dependencies=answer_dependencies,
        identity=public_identity,
        embedding=embedding,
        reranker=reranker,
        paper_context_store=sparse_store,
        paper_context_limit=settings.live_paper_context_limit,
        requirement_provider=requirement_provider,
        requirement_prompt=requirement_prompt,
        requirement_cache=requirement_cache,
    )


def _build_llm(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "mock":
        return MockLLMProvider(
            '{"claims":[{"text":"离线 Mock 回答","claim_type":"uncertain",'
            '"citation_ids":[],"facet_ids":[]}],"confidence":0.1}'
        )
    if settings.llm_provider == "openai-compatible":
        return OpenAICompatibleLLMProvider(
            settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout_seconds=settings.llm_timeout_seconds,
            max_output_tokens=settings.llm_max_output_tokens,
            reasoning_effort=settings.llm_reasoning_effort,
        )
    raise ValueError("unsupported LLM provider")


def _build_search(
    settings: Settings, fixture_path: str, workspace_root: Path
) -> SearchProvider | None:
    if not settings.enable_web_fallback or settings.web_search_provider == "disabled":
        return None
    if settings.web_search_provider == "tavily":
        return TavilySearchProvider(
            settings.web_search_api_key,
            timeout_seconds=settings.web_search_timeout_seconds,
        )
    if settings.web_search_provider == "recorded":
        return RecordedSearchProvider(workspace_root / fixture_path)
    if settings.web_search_provider == "mock":
        return MockSearchProvider()
    raise ValueError("unsupported Web search provider")


async def _build_graph_tool(
    workspace_root: Path,
    settings: Settings,
) -> tuple[GraphTool, str]:
    """可选图在首次实时请求时从已发布小语料构建，不影响默认 Hybrid。"""

    del settings
    from kg_crag.graph.config import load_graph_config
    from kg_crag.graph.metadata import plan_metadata_graph
    from kg_crag.graph_store import InMemoryGraphStore
    from kg_crag.indexing.processed import discover_processed, select_processed

    config = load_graph_config(workspace_root / "configs/retrieval.yaml")
    papers = select_processed(
        discover_processed(workspace_root / "data/processed"),
        pilot_manifest=workspace_root / "configs/pilot_corpus.json",
        max_papers=config.selection.max_papers,
    )
    plan = plan_metadata_graph(papers, config, model="offline-metadata")
    store = InMemoryGraphStore()
    await store.ensure_graph(plan.identity)
    for bundle in plan.bundles:
        await store.sync_paper(bundle)
    retriever = GraphRetriever(store, ProcessedSourceResolver(papers))
    return (
        GraphTool(retriever, graph_version=plan.identity.graph_version),
        plan.identity.graph_version,
    )


def build_live_query_service(
    settings: Settings,
    *,
    workspace_root: Path,
) -> LiveQueryApplicationService:
    return LiveQueryApplicationService(
        lambda: build_live_query_runtime(settings, workspace_root=workspace_root),
        preload_on_start=settings.live_preload_models,
    )
