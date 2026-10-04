"""运行配置的提前失败边界测试。"""

import pytest
from pydantic import ValidationError

from kg_crag.settings import Settings


def test_settings_defaults_are_valid() -> None:
    settings = Settings(_env_file=None)

    assert settings.api_port == 8000
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.neo4j_uri == "bolt://localhost:7687"
    assert settings.application_memory_target_mb == 10_240
    assert settings.application_memory_hard_limit_mb == 12_288


@pytest.mark.parametrize("port", [0, 65536])
def test_settings_rejects_out_of_range_port(port: int) -> None:
    with pytest.raises(ValidationError, match="api_port"):
        Settings(_env_file=None, api_port=port)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("qdrant_url", "localhost:6333"),
        ("qdrant_url", "ftp://localhost:6333"),
        ("neo4j_uri", "http://localhost:7687"),
        ("neo4j_uri", "bolt:///missing-host"),
    ],
)
def test_settings_rejects_invalid_service_addresses(field: str, value: str) -> None:
    with pytest.raises(ValidationError, match=field):
        Settings.model_validate({field: value})


def test_settings_rejects_blank_required_strings() -> None:
    with pytest.raises(ValidationError, match="llm_model"):
        Settings(_env_file=None, llm_model="   ")


def test_settings_rejects_unsafe_application_paths_and_resource_limits() -> None:
    with pytest.raises(ValidationError, match="replay_fixture_path"):
        Settings(_env_file=None, replay_fixture_path="../private.json")
    with pytest.raises(ValidationError, match="memory target"):
        Settings(
            _env_file=None,
            application_memory_target_mb=10_240,
            application_memory_hard_limit_mb=8_192,
        )
