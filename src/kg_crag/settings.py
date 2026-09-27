"""由环境变量驱动的运行时配置。"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """从 KG_CRAG_* 环境变量和本地 .env 文件加载配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="KG_CRAG_",
        extra="ignore",
        case_sensitive=False,
    )

    env: str = "development"
    log_level: str = "INFO"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    llm_provider: str = "mock"
    llm_model: str = "mock-llm-v1"
    llm_api_key: str | None = None
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    web_search_provider: str = "disabled"
    web_search_api_key: str | None = None
    enable_web_fallback: bool = False
    max_internal_retrieval_rounds: int = Field(default=2, ge=0, le=10)
    max_web_retrieval_rounds: int = Field(default=1, ge=0, le=3)
    max_reflection_rounds: int = Field(default=1, ge=0, le=5)


@lru_cache
def get_settings() -> Settings:
    """为每个进程返回一个通过校验的配置对象。"""

    return Settings()
