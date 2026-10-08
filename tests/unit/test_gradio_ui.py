"""Gradio 客户端的载荷、历史和安全错误测试，不导入 Gradio 本体。"""

from __future__ import annotations

import json

import httpx
from pytest import MonkeyPatch

from kg_crag.ui import app as ui_app
from kg_crag.ui.app import _ensure_localhost_proxy_bypass, _post_query, clear_chat


def test_localhost_proxy_bypass_preserves_existing_entries() -> None:
    environment = {
        "NO_PROXY": "internal.example,localhost",
        "no_proxy": "metadata.example,INTERNAL.example",
    }

    _ensure_localhost_proxy_bypass(environment)

    expected = "internal.example,localhost,metadata.example,127.0.0.1,::1"
    assert environment["NO_PROXY"] == expected
    assert environment["no_proxy"] == expected


def test_run_keeps_ui_local_and_disables_public_share(monkeypatch: MonkeyPatch) -> None:
    launch_arguments: dict[str, object] = {}
    bypass_calls: list[bool] = []

    class Demo:
        def launch(self, **kwargs: object) -> None:
            launch_arguments.update(kwargs)

    class Settings:
        ui_host = "127.0.0.1"
        ui_port = 7860

    monkeypatch.setattr(ui_app, "_ensure_localhost_proxy_bypass", lambda: bypass_calls.append(True))
    monkeypatch.setattr(ui_app, "get_settings", Settings)
    monkeypatch.setattr(ui_app, "build_demo", Demo)

    ui_app.run()

    assert bypass_calls == [True]
    assert launch_arguments == {
        "server_name": "127.0.0.1",
        "server_port": 7860,
        "share": False,
        "show_error": False,
    }


def test_client_sends_only_current_question_and_public_controls() -> None:
    observed: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/queries":
            observed.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "mode": "live",
                    "request_id": "request-demo",
                    "trace_id": "trace-demo",
                    "answer": "有依据回答",
                    "citations": [],
                    "facets": [],
                    "actions": [],
                    "retrieval_path": ["hybrid"],
                    "budget": {},
                    "stop_reason": "accepted",
                },
            )
        return httpx.Response(
            200,
            json={"trace_id": "trace-demo", "mode": "live", "events": []},
        )

    result = _post_query(
        "http://test",
        question="第二个独立问题",
        mode="live",
        replay_case_id=None,
        allow_web=True,
        include_trace=True,
        transport=httpx.MockTransport(handler),
    )

    assert observed == {
        "question": "第二个独立问题",
        "mode": "live",
        "allow_web": True,
        "include_trace": True,
    }
    assert result["trace"]["trace_id"] == "trace-demo"
    assert not {"history", "prompt", "threshold", "tools"} & set(observed)


def test_replay_never_sends_web_permission_and_clear_resets_history() -> None:
    observed: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "mode": "replay",
                "request_id": "request-demo",
                "trace_id": "trace-demo",
                "answer": "固定回答",
                "citations": [],
                "facets": [],
                "actions": [],
                "retrieval_path": [],
                "budget": {},
                "stop_reason": "accepted",
                "replay_fixture_version": "v1",
                "replay_notice": "非实时结果",
            },
        )

    _post_query(
        "http://test",
        question="固定案例",
        mode="replay",
        replay_case_id="sufficient",
        allow_web=True,
        include_trace=False,
        transport=httpx.MockTransport(handler),
    )

    assert observed["allow_web"] is False
    assert observed["replay_case_id"] == "sufficient"
    assert clear_chat() == ("", [])


def test_client_error_uses_safe_public_message() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            json={
                "request_id": "request-safe",
                "error": {
                    "code": "dependency_unavailable",
                    "message": "实时依赖未就绪。",
                    "retryable": True,
                    "fields": {},
                },
            },
        )

    try:
        _post_query(
            "http://test",
            question="任意问题",
            mode="live",
            replay_case_id=None,
            allow_web=False,
            include_trace=False,
            transport=httpx.MockTransport(handler),
        )
    except RuntimeError as exc:
        message = str(exc)
    else:  # pragma: no cover - 防止错误信封被误当作成功
        raise AssertionError("error response must raise")
    assert "request-safe" in message
    assert "可重试=True" in message
    assert "Traceback" not in message
