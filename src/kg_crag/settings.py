"""由环境变量驱动的运行时配置。"""

from functools import lru_cache
from urllib.parse import urlparse

from pydantic import Field, field_validator
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
    qdrant_url: str = Field(default="http://localhost:6333", min_length=1)
    qdrant_api_key: str | None = None
    neo4j_uri: str = Field(default="bolt://localhost:7687", min_length=1)
    neo4j_user: str = Field(default="neo4j", min_length=1)
    neo4j_password: str = ""
    llm_provider: str = Field(default="mock", min_length=1)
    llm_model: str = Field(default="mock-llm-v1", min_length=1)
    llm_api_key: str | None = None
    embedding_model: str = Field(default="BAAI/bge-m3", min_length=1)
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3", min_length=1)
    web_search_provider: str = Field(default="disabled", min_length=1)
    web_search_api_key: str | None = None
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


@lru_cache
def get_settings() -> Settings:
    """为每个进程返回一个通过校验的配置对象。"""

    return Settings()
