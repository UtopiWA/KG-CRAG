"""公共传输模型与既有领域工作流之间的应用适配层。"""

from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Protocol

import anyio

from kg_crag import __version__
from kg_crag.ingestion.config import load_ingestion_config
from kg_crag.ingestion.pdf import PyMuPDFParser
from kg_crag.ingestion.pipeline import IngestionPipeline
from kg_crag.ingestion.raw import RawPaperInput, discover_raw_inputs
from kg_crag.ingestion.storage import ArtifactStore, paper_key
from kg_crag.models import (
    ActionSummary,
    ApplicationBudgetSummary,
    ApplicationErrorCode,
    ApplicationMode,
    CitationSummary,
    ComponentReadiness,
    ComponentStatus,
    DocumentSummary,
    FacetStatus,
    FacetSummary,
    GroundedAnswerResult,
    IngestionItemSummary,
    IngestionRunRequest,
    IngestionRunResponse,
    Paper,
    PublicSourceType,
    QueryRequest,
    QueryResponse,
    ReadinessResponse,
    TraceEventSummary,
    TraceSummary,
)
from kg_crag.models.domain import Evidence, EvidenceSourceType
from kg_crag.settings import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPLAY_NOTICE = "固定脱敏回放: 非实时结果，不得作为正式实验结论。"


class ApplicationServiceError(Exception):
    """应用边界可安全映射为 HTTP 错误的失败。"""

    def __init__(
        self,
        code: ApplicationErrorCode,
        message: str,
        *,
        status_code: int,
        retryable: bool = False,
        fields: dict[str, str] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
        self.fields = fields or {}
        super().__init__(f"{code.value}: {message}")


class ApplicationService(Protocol):
    """API 唯一依赖的窄应用能力集合。"""

    async def query(self, request: QueryRequest, *, request_id: str) -> QueryResponse: ...

    async def get_document(self, paper_id: str) -> DocumentSummary: ...

    async def run_ingestion(
        self,
        request: IngestionRunRequest,
        *,
        request_id: str,
    ) -> IngestionRunResponse: ...

    async def get_trace(self, trace_id: str, *, max_events: int) -> TraceSummary: ...

    async def readiness(self) -> ReadinessResponse: ...


def _source_channels(evidence: Evidence | None, *, external: bool) -> list[PublicSourceType]:
    """依据已有 Evidence 排名字段投影来源，不重新判断检索结果。"""

    if external or (evidence is not None and evidence.source_type is EvidenceSourceType.WEB):
        return [PublicSourceType.WEB]
    if evidence is not None and evidence.source_type is EvidenceSourceType.GRAPH:
        return [PublicSourceType.GRAPH]
    channels: list[PublicSourceType] = []
    if evidence is not None and evidence.ranks.dense is not None:
        channels.append(PublicSourceType.DENSE)
    if evidence is not None and evidence.ranks.sparse is not None:
        channels.append(PublicSourceType.SPARSE)
    return channels or [PublicSourceType.INTERNAL]


class GroundedAnswerQueryAdapter:
    """把现有 GroundedAnswerResult 工作流投影为公共查询响应。"""

    def __init__(
        self,
        run: Callable[[str], Awaitable[GroundedAnswerResult]],
    ) -> None:
        self._run = run
        self._traces: dict[str, TraceSummary] = {}

    async def query(self, question: str, *, request_id: str) -> QueryResponse:
        result = await self._run(question)
        evidence_by_id = {
            item.evidence_id: item
            for item in [*result.internal_evidence, *result.external_evidence]
        }
        citations = [
            CitationSummary(
                citation_id=item.citation_id,
                evidence_id=item.evidence_id,
                title=(
                    str(evidence_by_id[item.evidence_id].metadata.get("title"))[:300]
                    if item.evidence_id in evidence_by_id
                    and evidence_by_id[item.evidence_id].metadata.get("title")
                    else item.paper_id or item.source_id
                ),
                source_channels=_source_channels(
                    evidence_by_id.get(item.evidence_id),
                    external=item.external,
                ),
                paper_id=item.paper_id,
                section=item.section,
                page=item.page,
                url=item.url,
                external=item.external,
            )
            for item in result.citations
        ]
        missing = set(result.missing_required_facet_ids)
        covered = {
            facet_id
            for claim in result.claims
            for facet_id in claim.facet_ids
            if facet_id not in missing
        }
        facets = [
            FacetSummary(
                facet_id=facet_id,
                label="工作流返回的证据 facet",
                status=FacetStatus.COVERED,
                evidence_ids=[],
            )
            for facet_id in sorted(covered)
        ] + [
            FacetSummary(
                facet_id=facet_id,
                label="工作流报告的缺失 facet",
                status=FacetStatus.MISSING,
                evidence_ids=[],
            )
            for facet_id in sorted(missing)
        ]
        # 纠错动作位于内部结果中；GroundedAnswerResult 只保留最终预算与 Trace，
        # 因此这里不从 Trace 文本反推动作，缺失时保持空列表。
        actions: list[ActionSummary] = []
        internal = result.budget.internal.used
        answer = result.budget.answer.used
        budget = ApplicationBudgetSummary(
            tool_calls=internal.retrieval_rounds,
            model_calls=internal.llm_calls + answer.answer_calls + answer.critic_calls,
            web_calls=answer.web_calls,
            input_tokens=internal.input_tokens + answer.input_tokens,
            output_tokens=internal.output_tokens + answer.output_tokens,
            elapsed_ms=internal.latency_ms + answer.latency_ms,
            stopped=result.stop_reason.value != "accepted",
            stop_reason=result.stop_reason.value,
        )
        response = QueryResponse(
            mode=ApplicationMode.LIVE,
            request_id=request_id,
            trace_id=result.trace_id,
            answer=result.answer,
            citations=citations,
            facets=facets,
            actions=actions,
            retrieval_path=[item.value for item in result.retrieval_path],
            budget=budget,
            stop_reason=result.stop_reason.value,
        )
        self._traces[result.trace_id] = TraceSummary(
            trace_id=result.trace_id,
            mode=ApplicationMode.LIVE,
            events=[
                TraceEventSummary(
                    sequence=item.sequence,
                    phase=item.node,
                    event=item.event,
                    occurred_at=item.occurred_at,
                    details=dict(item.details),
                )
                for item in result.trace
            ],
        )
        return response

    def trace(self, trace_id: str, *, max_events: int) -> TraceSummary | None:
        trace = self._traces.get(trace_id)
        if trace is None:
            return None
        events = trace.events[:max_events]
        return trace.model_copy(
            update={"events": events, "truncated": len(events) < len(trace.events)}
        )


class ReplayCatalog:
    """一次加载并严格校验版本化回放文件。"""

    def __init__(self, path: Path) -> None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("replay fixture cannot be loaded") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != "v1":
            raise ValueError("unsupported replay fixture schema")
        version = payload.get("fixture_version")
        cases = payload.get("cases")
        if not isinstance(version, str) or not version or not isinstance(cases, list):
            raise ValueError("replay fixture is incomplete")
        parsed: dict[str, dict[str, Any]] = {}
        traces: dict[str, TraceSummary] = {}
        for case in cases:
            if not isinstance(case, dict) or not isinstance(case.get("case_id"), str):
                raise ValueError("replay case is invalid")
            case_id = case["case_id"]
            if case_id in parsed:
                raise ValueError("replay case IDs must be unique")
            response = dict(case.get("response", {}))
            response.update(
                {
                    "mode": ApplicationMode.REPLAY,
                    "request_id": "fixture-validation",
                    "replay_fixture_version": version,
                    "replay_notice": REPLAY_NOTICE,
                }
            )
            validated = QueryResponse.model_validate(response)
            trace = TraceSummary.model_validate(
                {
                    **dict(case.get("trace", {})),
                    "mode": ApplicationMode.REPLAY,
                    "replay_fixture_version": version,
                }
            )
            if trace.trace_id != validated.trace_id or trace.trace_id in traces:
                raise ValueError("replay trace identity is invalid")
            parsed[case_id] = {"question": str(case.get("question", "")), "response": validated}
            traces[trace.trace_id] = trace
        if set(parsed) != {"sufficient", "corrected-gap", "conservative-stop"}:
            raise ValueError("replay fixture must contain the three required cases")
        self.version = version
        self.cases = parsed
        self.traces = traces

    def query(self, request: QueryRequest, *, request_id: str) -> QueryResponse:
        case_id = request.replay_case_id or "sufficient"
        case = self.cases.get(case_id)
        if case is None:
            raise ApplicationServiceError(
                ApplicationErrorCode.NOT_FOUND,
                "未找到指定回放案例。",
                status_code=404,
                fields={"replay_case_id": case_id},
            )
        response: QueryResponse = case["response"]
        return response.model_copy(update={"request_id": request_id})

    def trace(self, trace_id: str, *, max_events: int) -> TraceSummary:
        trace = self.traces.get(trace_id)
        if trace is None:
            raise ApplicationServiceError(
                ApplicationErrorCode.NOT_FOUND,
                "未找到指定 Trace。",
                status_code=404,
            )
        events = trace.events[:max_events]
        return trace.model_copy(
            update={"events": events, "truncated": len(events) < len(trace.events)}
        )


class LocalDocumentService:
    """只读取已发布论文元数据和 Chunk ID，不返回 Chunk 正文。"""

    def __init__(self, processed_root: Path) -> None:
        self.processed_root = processed_root

    async def get(self, paper_id: str) -> DocumentSummary:
        return await anyio.to_thread.run_sync(self._get_sync, paper_id)

    def _get_sync(self, paper_id: str) -> DocumentSummary:
        root = self.processed_root / paper_key(paper_id)
        versions = (
            sorted(
                (path for path in root.iterdir() if path.is_dir() and not path.is_symlink()),
                key=lambda path: path.name,
                reverse=True,
            )
            if root.is_dir()
            else []
        )
        for version in versions:
            try:
                paper = Paper.model_validate_json(
                    (version / "paper.json").read_text(encoding="utf-8")
                )
                quality = json.loads((version / "quality_report.json").read_text(encoding="utf-8"))
                chunk_ids: list[str] = []
                with (version / "chunks.jsonl").open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if len(chunk_ids) >= 100:
                            break
                        payload = json.loads(line)
                        chunk_id = payload.get("chunk_id")
                        if isinstance(chunk_id, str):
                            chunk_ids.append(chunk_id)
                if paper.paper_id != paper_id or quality.get("paper_id") != paper_id:
                    continue
                return DocumentSummary(
                    paper_id=paper.paper_id,
                    title=paper.title,
                    abstract=paper.abstract[:2000],
                    year=paper.year,
                    authors=[item.name for item in paper.authors],
                    processing_version=version.name,
                    chunk_ids=chunk_ids,
                    quality_passed=bool(quality.get("passed")),
                )
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        raise ApplicationServiceError(
            ApplicationErrorCode.NOT_FOUND,
            "未找到已发布且可校验的论文产物。",
            status_code=404,
            fields={"paper_id": paper_id},
        )


class LocalIngestionService:
    """复用既有摄取管线的 API 适配器，硬限制为最多五篇。"""

    def __init__(self, workspace_root: Path, settings: Settings) -> None:
        self.workspace_root = workspace_root
        self.settings = settings

    def _select(self, inputs: list[RawPaperInput], ids: list[str]) -> list[RawPaperInput]:
        by_id = {item.paper_id: item for item in inputs}
        unknown = [paper_id for paper_id in ids if paper_id not in by_id]
        if unknown:
            raise ApplicationServiceError(
                ApplicationErrorCode.NOT_FOUND,
                "部分论文 ID 不存在于已校验 raw 语料。",
                status_code=404,
                fields={"paper_ids": ",".join(unknown)[:200]},
            )
        return [by_id[paper_id] for paper_id in ids]

    def _run_sync(
        self,
        request: IngestionRunRequest,
        request_id: str,
    ) -> IngestionRunResponse:
        config = load_ingestion_config(self.workspace_root / "configs" / "default.yaml")
        raw_root = self.workspace_root / config.paths.raw_root
        inputs = discover_raw_inputs(
            raw_root,
            max_pdf_bytes=min(config.limits.max_pdf_bytes, self.settings.api_max_pdf_bytes),
            max_papers=config.limits.max_papers,
        )
        selected = self._select(inputs, request.paper_ids)
        store = ArtifactStore(
            self.workspace_root / config.paths.interim_root,
            self.workspace_root / config.paths.processed_root,
        )
        pipeline = IngestionPipeline(config, PyMuPDFParser(config.parsing), store)
        if request.dry_run:
            items = [
                IngestionItemSummary(
                    paper_id=item.paper_id,
                    status="planned",
                    processing_version=pipeline.version_for(item),
                    reason="dry-run: 仅完成输入与版本校验",
                )
                for item in selected
            ]
            return IngestionRunResponse(
                request_id=request_id,
                run_id=f"dry-run-{uuid.uuid4().hex[:16]}",
                dry_run=True,
                complete=True,
                items=items,
            )
        manifest = pipeline.run_batch(selected, mode="selected", force=request.force)
        items = [
            IngestionItemSummary(
                paper_id=item.paper_id,
                status=item.status.value,
                processing_version=item.processing_version,
                artifact_names=sorted(item.artifacts),
                warning_count=item.warning_count,
                reason=item.reason or (item.error.message if item.error else None),
            )
            for item in manifest.items
        ]
        return IngestionRunResponse(
            request_id=request_id,
            run_id=manifest.run_id,
            dry_run=False,
            complete=all(item.status in {"succeeded", "skipped"} for item in items),
            items=items,
        )

    async def run(
        self,
        request: IngestionRunRequest,
        *,
        request_id: str,
    ) -> IngestionRunResponse:
        return await anyio.to_thread.run_sync(self._run_sync, request, request_id)


class DefaultApplicationService:
    """组合回放、本地产物和可选真实回答适配器。"""

    def __init__(
        self,
        *,
        replay: ReplayCatalog,
        documents: LocalDocumentService,
        ingestion: LocalIngestionService,
        live_query: GroundedAnswerQueryAdapter | None = None,
    ) -> None:
        self.replay = replay
        self.documents = documents
        self.ingestion = ingestion
        self.live_query = live_query

    async def query(self, request: QueryRequest, *, request_id: str) -> QueryResponse:
        if request.mode is ApplicationMode.REPLAY:
            return self.replay.query(request, request_id=request_id)
        if self.live_query is None:
            raise ApplicationServiceError(
                ApplicationErrorCode.DEPENDENCY_UNAVAILABLE,
                "实时查询工作流尚未在当前进程装配；可显式选择回放模式。",
                status_code=503,
                retryable=True,
            )
        return await self.live_query.query(request.question, request_id=request_id)

    async def get_document(self, paper_id: str) -> DocumentSummary:
        return await self.documents.get(paper_id)

    async def run_ingestion(
        self,
        request: IngestionRunRequest,
        *,
        request_id: str,
    ) -> IngestionRunResponse:
        return await self.ingestion.run(request, request_id=request_id)

    async def get_trace(self, trace_id: str, *, max_events: int) -> TraceSummary:
        if trace_id in self.replay.traces:
            return self.replay.trace(trace_id, max_events=max_events)
        if self.live_query is not None:
            trace = self.live_query.trace(trace_id, max_events=max_events)
            if trace is not None:
                return trace
        raise ApplicationServiceError(
            ApplicationErrorCode.NOT_FOUND,
            "未找到指定 Trace。",
            status_code=404,
        )

    async def readiness(self) -> ReadinessResponse:
        live = self.live_query is not None
        components = [
            ComponentReadiness(
                component="api",
                status=ComponentStatus.READY,
                required=True,
                message="API 进程可响应。",
            ),
            ComponentReadiness(
                component="replay",
                status=ComponentStatus.READY,
                required=True,
                message=f"固定回放已加载: {self.replay.version}",
            ),
            ComponentReadiness(
                component="live_query",
                status=ComponentStatus.READY if live else ComponentStatus.DEGRADED,
                required=False,
                message="实时工作流已装配。" if live else "当前仅提供回放和本地产物能力。",
            ),
            ComponentReadiness(
                component="graph",
                status=ComponentStatus.DISABLED,
                required=False,
                message="Graph 是可选工具，由实时工作流 profile 决定是否启用。",
            ),
        ]
        return ReadinessResponse(
            status=ComponentStatus.READY if live else ComponentStatus.DEGRADED,
            version=__version__,
            available_modes=(
                [ApplicationMode.REPLAY, ApplicationMode.LIVE] if live else [ApplicationMode.REPLAY]
            ),
            components=components,
        )


def build_default_application_service(
    settings: Settings,
    *,
    workspace_root: Path = PROJECT_ROOT,
    live_query: GroundedAnswerQueryAdapter | None = None,
) -> DefaultApplicationService:
    """只装配轻量本地服务；真实模型和数据库保持惰性、显式注入。"""

    replay_path = workspace_root / settings.replay_fixture_path
    return DefaultApplicationService(
        replay=ReplayCatalog(replay_path),
        documents=LocalDocumentService(workspace_root / "data" / "processed"),
        ingestion=LocalIngestionService(workspace_root, settings),
        live_query=live_query,
    )
