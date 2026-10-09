"""纠错工作流的严格配置、硬资源边界与稳定哈希。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePath
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import Field, field_validator, model_validator

from kg_crag.models.correction import BudgetLimit, CorrectionAction
from kg_crag.models.domain import StrictModel


class FacetRulesConfig(StrictModel):
    version: str = "facet-rules-v3"
    prompt_version: str = "facet-prompt-v2"
    prompt_path: str = "prompts/facet-requirements-v2.txt"
    confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    max_facets: int = Field(default=12, ge=1, le=20)
    max_description_chars: int = Field(default=300, ge=20, le=500)
    allow_llm: bool = False

    @field_validator("prompt_path")
    @classmethod
    def prompt_is_workspace_relative(cls, value: str) -> str:
        return _relative_path(value)


class CoverageConfig(StrictModel):
    version: str = "coverage-v6"
    min_support: float = Field(default=0.45, ge=0.0, le=1.0)
    min_sources: int = Field(default=1, ge=1, le=5)
    max_selected_evidence: int = Field(default=8, ge=1, le=20)
    max_evidence_chars: int = Field(default=4000, ge=100, le=10_000)


class CostWeightsConfig(StrictModel):
    version: str = "cost-v1"
    retrieval: float = Field(default=0.08, ge=0.0, le=10.0)
    model: float = Field(default=0.15, ge=0.0, le=10.0)
    token: float = Field(default=0.00002, ge=0.0, le=1.0)
    latency: float = Field(default=0.00001, ge=0.0, le=1.0)
    repeat: float = Field(default=1.0, ge=0.0, le=10.0)
    minimum_utility: float = Field(default=0.01, ge=0.0, le=100.0)


def _default_actions() -> list[CorrectionAction]:
    return list(CorrectionAction)


class ActionBoundsConfig(StrictModel):
    catalog_version: str = "actions-v1"
    enabled: list[CorrectionAction] = Field(default_factory=_default_actions, min_length=1)
    default_top_k: int = Field(default=8, ge=1, le=100)
    max_top_k: int = Field(default=20, ge=1, le=100)
    max_candidates_per_action: int = Field(default=20, ge=1, le=100)
    max_subquestions: int = Field(default=3, ge=1, le=3)
    max_graph_hops: int = Field(default=3, ge=1, le=3)

    @field_validator("enabled")
    @classmethod
    def enabled_actions_are_unique(cls, value: list[CorrectionAction]) -> list[CorrectionAction]:
        if len(value) != len(set(value)):
            raise ValueError("enabled correction actions must be unique")
        return value

    @model_validator(mode="after")
    def top_k_is_consistent(self) -> ActionBoundsConfig:
        if self.default_top_k > self.max_top_k:
            raise ValueError("default_top_k must not exceed max_top_k")
        return self


class ArtifactPathsConfig(StrictModel):
    cache_root: str = "data/processed/corrective-cache"
    run_root: str = "data/processed/corrective-runs"
    evaluation_root: str = "data/evaluation/results/corrective-workflow"

    @field_validator("cache_root", "run_root", "evaluation_root")
    @classmethod
    def artifacts_are_workspace_relative(cls, value: str) -> str:
        return _relative_path(value)


class CorrectiveWorkflowConfig(StrictModel):
    schema_version: Literal["v1"] = "v1"
    policy_version: str = "policy-v1"
    initial_strategy: Literal["fixed_hybrid", "type_router", "facet_corrective"] = (
        "facet_corrective"
    )
    facets: FacetRulesConfig = Field(default_factory=FacetRulesConfig)
    coverage: CoverageConfig = Field(default_factory=CoverageConfig)
    actions: ActionBoundsConfig = Field(default_factory=ActionBoundsConfig)
    costs: CostWeightsConfig = Field(default_factory=CostWeightsConfig)
    budget: BudgetLimit = Field(default_factory=BudgetLimit)
    artifacts: ArtifactPathsConfig = Field(default_factory=ArtifactPathsConfig)
    max_total_evidence: int = Field(default=100, ge=1, le=100)
    max_trace_events: int = Field(default=100, ge=1, le=200)


EvaluationStrategy = Literal["fixed_hybrid", "type_router", "facet_corrective"]


def _default_evaluation_strategies() -> list[EvaluationStrategy]:
    return ["fixed_hybrid", "type_router", "facet_corrective"]


class CorrectiveEvaluationConfig(StrictModel):
    questions_path: str = "data/evaluation/corrective_dev_questions.json"
    results_root: str = "data/evaluation/results/corrective-workflow"
    strategies: list[EvaluationStrategy] = Field(default_factory=_default_evaluation_strategies)
    max_questions: int = Field(default=40, ge=20, le=40)
    smoke_questions: int = Field(default=5, ge=3, le=5)
    online_smoke_questions: int = Field(default=3, ge=1, le=3)
    online_max_questions: int = Field(default=20, ge=1, le=20)

    @field_validator("questions_path", "results_root")
    @classmethod
    def paths_are_workspace_relative(cls, value: str) -> str:
        return _relative_path(value)

    @field_validator("strategies")
    @classmethod
    def strategies_are_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("corrective evaluation strategies must be unique")
        return value


def _relative_path(value: str) -> str:
    path = PurePath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("corrective workflow paths must stay inside the workspace")
    return path.as_posix()


def _load_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a mapping")
    return payload


def load_corrective_workflow_config(path: Path) -> CorrectiveWorkflowConfig:
    section = _load_mapping(path).get("corrective_workflow")
    if not isinstance(section, dict):
        raise ValueError("corrective_workflow configuration must be a mapping")
    return CorrectiveWorkflowConfig.model_validate(section)


def load_corrective_evaluation_config(path: Path) -> CorrectiveEvaluationConfig:
    section = _load_mapping(path).get("corrective_workflow")
    if not isinstance(section, dict):
        raise ValueError("corrective_workflow evaluation configuration must be a mapping")
    return CorrectiveEvaluationConfig.model_validate(section)


def corrective_config_hash(config: CorrectiveWorkflowConfig) -> str:
    payload = config.model_dump(mode="json")
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
