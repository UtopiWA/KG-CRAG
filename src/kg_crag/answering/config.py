"""回答反思与 Web 兜底的严格配置和稳定身份。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePath
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import Field, field_validator, model_validator

from kg_crag.models.answer import AnswerBudgetLimit, WebSourceType
from kg_crag.models.domain import StrictModel


def _default_domains() -> list[str]:
    return [
        "arxiv.org",
        "openreview.net",
        "aclanthology.org",
        "proceedings.mlr.press",
        "semanticscholar.org",
        "openalex.org",
        "doi.org",
        "github.com",
    ]


def _default_source_types() -> list[WebSourceType]:
    return list(WebSourceType)


class AnswerPromptConfig(StrictModel):
    answer_path: str = "prompts/grounded-answer-v1.txt"
    critic_path: str = "prompts/answer-critic-v1.txt"
    checker_version: str = "answer-checker-v1"
    policy_version: str = "answer-policy-v1"

    @field_validator("answer_path", "critic_path")
    @classmethod
    def prompt_paths_are_relative(cls, value: str) -> str:
        return _relative_path(value)


class AnswerGenerationConfig(StrictModel):
    max_selected_evidence: int = Field(default=8, ge=1, le=20)
    max_chars_per_evidence: int = Field(default=2000, ge=100, le=4000)
    max_context_chars: int = Field(default=12_000, ge=1000, le=16_000)
    max_response_chars: int = Field(default=3000, ge=500, le=12_000)
    max_claims: int = Field(default=30, ge=1, le=100)


class CriticConfig(StrictModel):
    enabled: bool = True
    max_context_chars: int = Field(default=4000, ge=500, le=12_000)
    max_response_chars: int = Field(default=2000, ge=500, le=6000)


class WebSearchConfig(StrictModel):
    enabled: bool = False
    provider: Literal["disabled", "mock", "recorded", "tavily"] = "disabled"
    provider_version: str = "disabled-v1"
    allowed_domains: list[str] = Field(
        default_factory=_default_domains, min_length=1, max_length=20
    )
    allowed_source_types: list[WebSourceType] = Field(
        default_factory=_default_source_types, min_length=1, max_length=4
    )
    max_results: int = Field(default=5, ge=1, le=5)
    max_excerpt_chars: int = Field(default=2000, ge=100, le=2000)
    max_context_chars: int = Field(default=8000, ge=100, le=8000)
    timeout_seconds: float = Field(default=15.0, gt=0.0, le=60.0)
    recorded_fixture_path: str = "data/evaluation/fixtures/grounded-web-recorded.json"

    @field_validator("allowed_domains")
    @classmethod
    def domains_are_normalized_and_unique(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for domain in value:
            candidate = domain.strip().rstrip(".").casefold()
            if not candidate or ":" in candidate or "/" in candidate:
                raise ValueError("trusted domains must be host names without scheme or port")
            candidate = candidate.encode("idna").decode("ascii")
            normalized.append(candidate)
        if len(normalized) != len(set(normalized)):
            raise ValueError("trusted domains must be unique")
        return normalized

    @field_validator("allowed_source_types")
    @classmethod
    def source_types_are_unique(cls, value: list[WebSourceType]) -> list[WebSourceType]:
        if len(value) != len(set(value)):
            raise ValueError("web source types must be unique")
        return value

    @field_validator("recorded_fixture_path")
    @classmethod
    def fixture_path_is_relative(cls, value: str) -> str:
        return _relative_path(value)

    @model_validator(mode="after")
    def enabled_provider_is_consistent(self) -> WebSearchConfig:
        if self.enabled == (self.provider == "disabled"):
            raise ValueError("enabled Web search requires a non-disabled provider")
        return self


class AnswerArtifactConfig(StrictModel):
    cache_root: str = "data/processed/grounded-answer-cache"
    run_root: str = "data/processed/grounded-answer-runs"
    evaluation_root: str = "data/evaluation/results/grounded-answer"

    @field_validator("cache_root", "run_root", "evaluation_root")
    @classmethod
    def paths_are_relative(cls, value: str) -> str:
        return _relative_path(value)


class GroundedAnswerConfig(StrictModel):
    schema_version: Literal["v1"] = "v1"
    enabled: bool = False
    prompts: AnswerPromptConfig = Field(default_factory=AnswerPromptConfig)
    generation: AnswerGenerationConfig = Field(default_factory=AnswerGenerationConfig)
    critic: CriticConfig = Field(default_factory=CriticConfig)
    web: WebSearchConfig = Field(default_factory=WebSearchConfig)
    budget: AnswerBudgetLimit = Field(default_factory=AnswerBudgetLimit)
    artifacts: AnswerArtifactConfig = Field(default_factory=AnswerArtifactConfig)
    max_trace_events: int = Field(default=100, ge=1, le=200)


class GroundedEvaluationConfig(StrictModel):
    questions_path: str = "data/evaluation/grounded_answer_dev_questions.json"
    results_root: str = "data/evaluation/results/grounded-answer"
    fixture_path: str = "data/evaluation/fixtures/grounded-web-recorded.json"
    max_questions: int = Field(default=20, ge=12, le=20)
    smoke_questions: int = Field(default=5, ge=1, le=5)
    online_min_questions: int = Field(default=5, ge=5, le=10)
    online_max_questions: int = Field(default=10, ge=5, le=10)

    @field_validator("questions_path", "results_root", "fixture_path")
    @classmethod
    def paths_are_relative(cls, value: str) -> str:
        return _relative_path(value)

    @model_validator(mode="after")
    def online_range_is_ordered(self) -> GroundedEvaluationConfig:
        if self.online_min_questions > self.online_max_questions:
            raise ValueError("online question range is reversed")
        return self


def _relative_path(value: str) -> str:
    path = PurePath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("grounded answer paths must stay inside the workspace")
    return path.as_posix()


def _load_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a mapping")
    return payload


def load_grounded_answer_config(path: Path) -> GroundedAnswerConfig:
    section = _load_mapping(path).get("grounded_answer", {})
    if not isinstance(section, dict):
        raise ValueError("grounded_answer configuration must be a mapping")
    return GroundedAnswerConfig.model_validate(section)


def load_grounded_evaluation_config(path: Path) -> GroundedEvaluationConfig:
    section = _load_mapping(path).get("grounded_answer", {})
    if not isinstance(section, dict):
        raise ValueError("grounded_answer evaluation configuration must be a mapping")
    return GroundedEvaluationConfig.model_validate(section)


def grounded_answer_config_hash(config: GroundedAnswerConfig) -> str:
    encoded = json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
