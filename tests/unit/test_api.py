"""最小服务入口测试。"""

from fastapi.testclient import TestClient

from kg_crag.api.app import app


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
