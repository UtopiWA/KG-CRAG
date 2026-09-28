"""论文摄取的严格配置加载、哈希和处理版本计算。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePath
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import Field, field_validator, model_validator

from kg_crag.models.domain import StrictModel


class IngestionLimits(StrictModel):
    max_papers: int = Field(default=200, ge=1, le=10_000)
    max_pdf_bytes: int = Field(default=104_857_600, ge=1024, le=1_073_741_824)
    max_pages: int = Field(default=500, ge=1, le=10_000)
    min_page_chars: int = Field(default=20, ge=0, le=100_000)
    min_document_chars: int = Field(default=200, ge=1, le=10_000_000)


class ParsingConfig(StrictModel):
    full_width_ratio: float = Field(default=0.72, gt=0.5, le=1.0)
    column_gap_ratio: float = Field(default=0.08, gt=0, lt=0.5)
    heading_font_ratio: float = Field(default=1.15, gt=1.0, le=3.0)
    heading_max_chars: int = Field(default=160, ge=10, le=1000)


class CleaningConfig(StrictModel):
    version: str = Field(default="clean-v1", min_length=1)
    margin_ratio: float = Field(default=0.10, gt=0, le=0.25)
    repeated_margin_min_pages: int = Field(default=3, ge=2, le=100)
    repeated_margin_coverage: float = Field(default=0.60, gt=0.5, le=1.0)
    remove_adjacent_duplicates: bool = True


class ChunkingConfig(StrictModel):
    tokenizer: Literal["regex-v1"] = "regex-v1"
    target_tokens: int = Field(default=500, ge=1)
    min_tokens: int = Field(default=300, ge=1)
    max_tokens: int = Field(default=700, ge=1)
    overlap_tokens: int = Field(default=75, ge=0)

    @model_validator(mode="after")
    def validate_window(self) -> ChunkingConfig:
        if not (self.overlap_tokens < self.min_tokens <= self.target_tokens <= self.max_tokens):
            raise ValueError("chunking requires overlap < min <= target <= max")
        return self


class IngestionPaths(StrictModel):
    raw_root: str = "data/raw"
    pilot_manifest: str = "configs/pilot_corpus.json"
    pilot_review: str = "configs/pilot_review.json"
    interim_root: str = "data/interim"
    processed_root: str = "data/processed"

    @field_validator("*", mode="after")
    @classmethod
    def paths_are_relative(cls, value: str) -> str:
        path = PurePath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("ingestion paths must be workspace-relative and contained")
        return path.as_posix()


class IngestionConfig(StrictModel):
    limits: IngestionLimits = Field(default_factory=IngestionLimits)
    parsing: ParsingConfig = Field(default_factory=ParsingConfig)
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    paths: IngestionPaths = Field(default_factory=IngestionPaths)


def load_ingestion_config(path: Path) -> IngestionConfig:
    """加载 YAML 中与摄取相关的段，并拒绝未知字段。"""

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a mapping")
    ingestion = payload.get("ingestion", {})
    if not isinstance(ingestion, dict):
        raise ValueError("ingestion configuration must be a mapping")
    selected: dict[str, Any] = {
        "limits": ingestion.get("limits", {}),
        "paths": ingestion.get("paths", {}),
        "parsing": payload.get("parsing", {}),
        "cleaning": payload.get("cleaning", {}),
        "chunking": payload.get("chunking", {}),
    }
    unknown = set(ingestion) - {"limits", "paths"}
    if unknown:
        raise ValueError(f"unknown ingestion configuration fields: {sorted(unknown)}")
    return IngestionConfig.model_validate(selected)


def canonical_config_bytes(config: IngestionConfig) -> bytes:
    """排除机器路径，生成影响内容的规范化配置字节。"""

    payload = config.model_dump(mode="json", exclude={"paths": True})
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def config_hash(config: IngestionConfig) -> str:
    return hashlib.sha256(canonical_config_bytes(config)).hexdigest()


def processing_version(
    input_sha256: str,
    parser_name: str,
    parser_version: str,
    configuration_hash: str,
) -> str:
    """由全部内容影响因素计算稳定的处理版本。"""

    components = [input_sha256, parser_name, parser_version, configuration_hash]
    if any(not component for component in components):
        raise ValueError("processing-version components must not be empty")
    return hashlib.sha256("\n".join(components).encode()).hexdigest()
