"""由环境变量驱动的运行时配置。"""

from functools import lru_cache
from pathlib import PurePath
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """从 KG_CRAG_* 环境变量和本地 .env 文件加载配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="KG_CRAG_",
        extra="ignore",
        case_sensitive=False,
        str_strip_whitespace=True,
    )

    env: str = Field(default="development", min_length=1)
    log_level: str = Field(default="INFO", min_length=1)
    api_host: str = Field(default="127.0.0.1", min_length=1)
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_request_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    api_max_trace_events: int = Field(default=100, ge=1, le=200)
    api_max_pdf_bytes: int = Field(default=52_428_800, ge=1024, le=104_857_600)
    replay_fixture_path: str = Field(
        default="src/kg_crag/application/replay_cases.json",
        min_length=1,
    )
    ui_api_base_url: str = Field(default="http://127.0.0.1:8000", min_length=1)
    application_memory_target_mb: int = Field(default=10_240, ge=1024, le=10_240)
    application_memory_hard_limit_mb: int = Field(default=12_288, ge=1024, le=12_288)
    qdrant_url: str = Field(default="http://localhost:6333", min_length=1)
    qdrant_api_key: str | None = None
    neo4j_uri: str = Field(default="bolt://localhost:7687", min_length=1)
    neo4j_user: str = Field(default="neo4j", min_length=1)
    neo4j_password: str = ""
    llm_provider: str = Field(default="mock", min_length=1)
    llm_model: str = Field(default="mock-llm-v1", min_length=1)
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_timeout_seconds: float = Field(default=60.0, gt=0, le=300)
    embedding_provider: str = Field(default="sentence-transformers", min_length=1)
    embedding_model: str = Field(default="BAAI/bge-m3", min_length=1)
    embedding_revision: str = Field(
        default="5617a9f61b028005a4858fdac845db406aefb181",
        min_length=1,
    )
    reranker_model: str = Field(default="BAAI/bge-reranker-base", min_length=1)
    reranker_revision: str = Field(default="2cfc18c9415c912f9d8155881c133215df768a70", min_length=1)
    reranker_device: str = Field(default="cpu", pattern=r"^(cpu|cuda(?::[0-9]+)?)$")
    model_cache_root: str = Field(default="data/processed/model-cache", min_length=1)
    web_search_provider: Literal["disabled", "mock", "recorded", "tavily"] = "disabled"
    web_search_api_key: str | None = None
    web_search_timeout_seconds: float = Field(default=15.0, gt=0, le=60)
    enable_web_fallback: bool = False
    max_internal_retrieval_rounds: int = Field(default=2, ge=0, le=10)
    max_web_retrieval_rounds: int = Field(default=1, ge=0, le=3)
    max_reflection_rounds: int = Field(default=1, ge=0, le=5)

    @field_validator("qdrant_url")
    @classmethod
    def qdrant_url_is_http(cls, value: str) -> str:
        """仅接受带主机的 HTTP(S) Qdrant 地址。"""

        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("qdrant_url must be an HTTP(S) URL with a host")
        return value

    @field_validator("neo4j_uri")
    @classmethod
    def neo4j_uri_uses_supported_scheme(cls, value: str) -> str:
        """仅接受带主机的 bolt:// 或 neo4j:// 地址。"""

        parsed = urlparse(value)
        if parsed.scheme not in {"bolt", "neo4j"} or not parsed.netloc:
            raise ValueError("neo4j_uri must be a bolt:// or neo4j:// URI with a host")
        return value

    @field_validator("ui_api_base_url")
    @classmethod
    def ui_api_base_url_is_http(cls, value: str) -> str:
        """界面只允许调用明确的 HTTP(S) API 基址。"""

        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("ui_api_base_url must be an HTTP(S) URL with a host")
        return value.rstrip("/")

    @field_validator("replay_fixture_path")
    @classmethod
    def replay_fixture_path_is_workspace_relative(cls, value: str) -> str:
        """回放文件必须由工作区内的稳定相对路径定位。"""

        path = PurePath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("replay_fixture_path must be workspace-relative and contained")
        return path.as_posix()

    @model_validator(mode="after")
    def application_memory_limits_are_ordered(self) -> "Settings":
        """目标值不能高于阻止继续加载可选组件的硬边界。"""

        if self.application_memory_target_mb > self.application_memory_hard_limit_mb:
            raise ValueError("application memory target must not exceed hard limit")
        return self


# 配置在进程内只解析一次，避免同一请求链中重复读取环境变量和 .env。
@lru_cache
def get_settings() -> Settings:
    """为每个进程返回一个通过校验的配置对象。"""

    return Settings()
