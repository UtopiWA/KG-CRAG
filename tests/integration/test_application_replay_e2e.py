"""API 到固定回放与 Trace 下钻的零网络闭环。"""

from fastapi.testclient import TestClient

from kg_crag.api.app import create_app
from kg_crag.settings import Settings


def test_all_demo_cases_round_trip_without_external_services() -> None:
    application = create_app(settings=Settings(_env_file=None))
    observed_stop_reasons: set[str] = set()
    with TestClient(application) as client:
        for case_id in ("sufficient", "corrected-gap", "conservative-stop"):
            response = client.post(
                "/v1/queries",
                json={
                    "question": "固定演示问题",
                    "mode": "replay",
                    "replay_case_id": case_id,
                },
            )
            assert response.status_code == 200
            result = response.json()
            observed_stop_reasons.add(result["stop_reason"])
            assert result["mode"] == "replay"
            assert result["budget"]["model_calls"] == 0
            assert result["budget"]["web_calls"] == 0

            trace = client.get(f"/v1/traces/{result['trace_id']}")
            assert trace.status_code == 200
            assert trace.json()["replay_fixture_version"] == "demo-replay-v1"
            assert len(trace.json()["events"]) == 3

    assert observed_stop_reasons == {
        "accepted",
        "accepted_after_correction",
        "conservative_stop",
    }


def test_demo_api_rejects_formal_test_controls() -> None:
    """应用验收没有读取 split 或写正式锁的入口。"""

    application = create_app(settings=Settings(_env_file=None))
    with TestClient(application) as client:
        response = client.post(
            "/v1/queries",
            json={
                "question": "不得消费正式测试",
                "mode": "replay",
                "split": "test",
                "confirm_test_run": True,
            },
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
