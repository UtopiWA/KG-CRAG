"""最小服务入口测试。"""

from fastapi.testclient import TestClient

from kg_crag.api.app import app


def test_health() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": "0.1.0",
        "environment": "development",
    }
    assert not {"api_key", "password", "token", "secret"} & set(response.json())
