"""版本化 FastAPI 应用入口与安全错误边界。"""

from __future__ import annotations

import asyncio
import re
import uuid
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from typing import Annotated, TypeVar

from fastapi import FastAPI, Path, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint

from kg_crag import __version__
from kg_crag.application import (
    ApplicationService,
    ApplicationServiceError,
    build_default_application_service,
)
from kg_crag.models import (
    ApplicationErrorCode,
    ApplicationErrorDetail,
    ApplicationErrorResponse,
    DocumentSummary,
    HealthResponse,
    IngestionRunRequest,
    IngestionRunResponse,
    QueryRequest,
    QueryResponse,
    ReadinessResponse,
    TraceSummary,
)
from kg_crag.settings import Settings, get_settings

_REQUEST_ID = re.compile(r"^[A-Za-z0-9-]{1,64}$")
_ResultT = TypeVar("_ResultT")
PaperId = Annotated[str, Path(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9:._-]+$")]
TraceId = Annotated[str, Path(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9:._-]+$")]


def _request_id(request: Request) -> str:
    value = getattr(request.state, "request_id", None)
    return value if isinstance(value, str) else uuid.uuid4().hex


def _error_response(
    request: Request,
    *,
    code: ApplicationErrorCode,
    message: str,
    status_code: int,
    retryable: bool = False,
    fields: dict[str, str] | None = None,
) -> JSONResponse:
    payload = ApplicationErrorResponse(
        request_id=_request_id(request),
        error=ApplicationErrorDetail(
            code=code,
            message=message,
            retryable=retryable,
            fields=fields or {},
        ),
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


def create_app(
    *,
    settings: Settings | None = None,
    service: ApplicationService | None = None,
) -> FastAPI:
    """构造可注入替身的应用；默认装配不会创建外部客户端。"""

    runtime_settings = settings or get_settings()
    runtime_service = service or build_default_application_service(runtime_settings)

    @asynccontextmanager
    async def lifespan(_application: FastAPI) -> AsyncIterator[None]:
        startup = getattr(runtime_service, "startup", None)
        if callable(startup):
            startup()
        try:
            yield
        finally:
            shutdown = getattr(runtime_service, "shutdown", None)
            if callable(shutdown):
                await shutdown()

    application = FastAPI(title="KG-CRAG API", version=__version__, lifespan=lifespan)
    application.state.application_service = runtime_service
    application.state.settings = runtime_settings

    @application.middleware("http")
    async def attach_request_id(
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        candidate = request.headers.get("X-Request-ID", "")
        request.state.request_id = (
            candidate if _REQUEST_ID.fullmatch(candidate) else uuid.uuid4().hex
        )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @application.exception_handler(ApplicationServiceError)
    async def handle_application_error(
        request: Request,
        error: ApplicationServiceError,
    ) -> JSONResponse:
        return _error_response(
            request,
            code=error.code,
            message=error.message,
            status_code=error.status_code,
            retryable=error.retryable,
            fields=error.fields,
        )

    @application.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        # 字段名也可能是 prompt、body 等敏感载体标记；用序号键并只保留
        # schema 位置和错误类型，不回显用户提交的值。
        fields = {
            f"field_{index}": (
                f"{'.'.join(str(part) for part in item.get('loc', ()))}:"
                f"{item.get('type', 'invalid')}"
            )[:200]
            for index, item in enumerate(error.errors()[:16])
        }
        return _error_response(
            request,
            code=ApplicationErrorCode.VALIDATION,
            message="请求不符合公共 API 契约。",
            status_code=422,
            fields=fields,
        )

    @application.exception_handler(Exception)
    async def handle_unknown_error(request: Request, error: Exception) -> JSONResponse:
        del error
        return _error_response(
            request,
            code=ApplicationErrorCode.INTERNAL,
            message="服务发生未分类错误，请使用 request_id 查询本地日志。",
            status_code=500,
        )

    async def with_timeout(awaitable: Awaitable[_ResultT]) -> _ResultT:
        try:
            async with asyncio.timeout(runtime_settings.api_request_timeout_seconds):
                return await awaitable
        except TimeoutError as exc:
            raise ApplicationServiceError(
                ApplicationErrorCode.TIMEOUT,
                "请求超过应用层时间上限，后续外部调用已停止。",
                status_code=504,
                retryable=True,
            ) from exc
        except asyncio.CancelledError as exc:
            raise ApplicationServiceError(
                ApplicationErrorCode.CANCELLED,
                "请求已取消。",
                status_code=499,
            ) from exc

    @application.get("/health", tags=["system"], response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            environment=runtime_settings.env,
        )

    @application.get("/ready", tags=["system"], response_model=ReadinessResponse)
    async def ready() -> ReadinessResponse:
        return await runtime_service.readiness()

    @application.post(
        "/v1/queries",
        tags=["query"],
        response_model=QueryResponse,
        responses={
            422: {"model": ApplicationErrorResponse},
            503: {"model": ApplicationErrorResponse},
            504: {"model": ApplicationErrorResponse},
        },
    )
    async def query(payload: QueryRequest, request: Request) -> QueryResponse:
        return await with_timeout(runtime_service.query(payload, request_id=_request_id(request)))

    @application.get(
        "/v1/documents/{paper_id}",
        tags=["documents"],
        response_model=DocumentSummary,
        responses={404: {"model": ApplicationErrorResponse}},
    )
    async def document(paper_id: PaperId) -> DocumentSummary:
        return await with_timeout(runtime_service.get_document(paper_id))

    @application.post(
        "/v1/ingestion/runs",
        tags=["ingestion"],
        response_model=IngestionRunResponse,
        responses={
            422: {"model": ApplicationErrorResponse},
            504: {"model": ApplicationErrorResponse},
        },
    )
    async def ingestion(payload: IngestionRunRequest, request: Request) -> IngestionRunResponse:
        return await with_timeout(
            runtime_service.run_ingestion(payload, request_id=_request_id(request))
        )

    @application.get(
        "/v1/traces/{trace_id}",
        tags=["traces"],
        response_model=TraceSummary,
        responses={404: {"model": ApplicationErrorResponse}},
    )
    async def trace(trace_id: TraceId) -> TraceSummary:
        return await with_timeout(
            runtime_service.get_trace(
                trace_id,
                max_events=runtime_settings.api_max_trace_events,
            )
        )

    return application


app = create_app()


def run() -> None:
    """通过安装后的命令行入口启动开发服务器。"""

    import uvicorn

    settings = get_settings()
    uvicorn.run("kg_crag.api.app:app", host=settings.api_host, port=settings.api_port)
