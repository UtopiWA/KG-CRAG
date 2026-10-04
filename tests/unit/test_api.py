"""查询应用公共 HTTP 契约测试。"""

from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from kg_crag.api.app import app, create_app
from kg_crag.application import build_default_application_service
from kg_crag.models import (
    ApplicationMode,
    DocumentSummary,
    IngestionRunRequest,
    IngestionRunResponse,
    QueryRequest,
    QueryResponse,
    ReadinessResponse,
    TraceSummary,
)
from kg_crag.settings import Settings


def test_health() -> None:
    # 显式关闭 TestClient，避免其 AnyIO socket 在后续测试中才被垃圾回收。
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": "0.1.0",
        "environment": "development",
    }
    assert not {"api_key", "password", "token", "secret"} & set(response.json())


def test_readiness_reports_replay_without_constructing_live_clients() -> None:
    application = create_app(settings=Settings(_env_file=None))
    with TestClient(application) as client:
        response = client.get("/ready")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["available_modes"] == ["replay"]
    assert {item["component"] for item in payload["components"]} >= {"api", "replay", "graph"}
    assert "password" not in response.text.casefold()


def test_three_replay_queries_and_trace_are_explicitly_labeled() -> None:
    application = create_app(settings=Settings(_env_file=None))
    with TestClient(application) as client:
        for case_id in ("sufficient", "corrected-gap", "conservative-stop"):
            response = client.post(
                "/v1/queries",
                json={
                    "question": "请展示固定案例。",
                    "mode": "replay",
                    "replay_case_id": case_id,
                },
                headers={"X-Request-ID": f"request-{case_id}"},
            )
            assert response.status_code == 200
            payload = response.json()
            assert payload["mode"] == "replay"
            assert payload["request_id"] == f"request-{case_id}"
            assert payload["replay_fixture_version"] == "demo-replay-v1"
            assert "正式实验结论" in payload["replay_notice"]
            trace = client.get(f"/v1/traces/{payload['trace_id']}")
            assert trace.status_code == 200
            assert trace.json()["mode"] == "replay"
            assert [item["sequence"] for item in trace.json()["events"]] == [0, 1, 2]


def test_live_mode_is_not_silently_replaced_by_replay() -> None:
    application = create_app(settings=Settings(_env_file=None))
    with TestClient(application) as client:
        response = client.post(
            "/v1/queries",
            json={"question": "实时问题", "mode": "live"},
        )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "dependency_unavailable"
    assert "回放" in response.json()["error"]["message"]
    assert "trace_id" not in response.json()


def test_validation_error_uses_safe_envelope_and_rejects_unknown_fields() -> None:
    application = create_app(settings=Settings(_env_file=None))
    with TestClient(application) as client:
        response = client.post(
            "/v1/queries",
            json={"question": "问题", "mode": "replay", "prompt": "不要接受"},
        )
    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "validation_error"
    assert "不要接受" not in response.text
    assert payload["request_id"] == response.headers["X-Request-ID"]


def test_actual_ingestion_requires_explicit_confirmation_before_service_call() -> None:
    application = create_app(settings=Settings(_env_file=None))
    with TestClient(application) as client:
        response = client.post(
            "/v1/ingestion/runs",
            json={"paper_ids": ["arxiv:example"], "dry_run": False},
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_unknown_document_and_trace_return_sanitized_not_found() -> None:
    application = create_app(settings=Settings(_env_file=None))
    with TestClient(application) as client:
        document = client.get("/v1/documents/arxiv:does-not-exist")
        trace = client.get("/v1/traces/does-not-exist")
    assert document.status_code == 404
    assert trace.status_code == 404
    assert document.json()["error"]["code"] == "not_found"
    assert trace.json()["error"]["code"] == "not_found"
    assert "D:\\" not in document.text


class _SlowService:
    async def query(self, request: QueryRequest, *, request_id: str) -> QueryResponse:
        del request, request_id
        await asyncio.sleep(0.05)
        raise AssertionError("timeout should cancel the service coroutine")

    async def get_document(self, paper_id: str) -> DocumentSummary:
        raise AssertionError(paper_id)

    async def run_ingestion(
        self,
        request: IngestionRunRequest,
        *,
        request_id: str,
    ) -> IngestionRunResponse:
        raise AssertionError((request, request_id))

    async def get_trace(self, trace_id: str, *, max_events: int) -> TraceSummary:
        raise AssertionError((trace_id, max_events))

    async def readiness(self) -> ReadinessResponse:
        raise AssertionError("not used")


def test_request_timeout_returns_stable_error() -> None:
    settings = Settings(_env_file=None, api_request_timeout_seconds=0.01)
    application = create_app(settings=settings, service=_SlowService())
    with TestClient(application) as client:
        response = client.post(
            "/v1/queries",
            json={"question": "会超时的问题", "mode": "replay"},
        )
    assert response.status_code == 504
    assert response.json()["error"] == {
        "code": "timeout",
        "message": "请求超过应用层时间上限，后续外部调用已停止。",
        "retryable": True,
        "fields": {},
    }


def test_default_service_factory_exposes_only_replay_without_live_adapter() -> None:
    service = build_default_application_service(Settings(_env_file=None))
    assert service.live_query is None
    assert set(service.replay.cases) == {"sufficient", "corrected-gap", "conservative-stop"}
    assert ApplicationMode.REPLAY.value == "replay"
