"""统一评测的严格配置与不可放宽资源边界。"""

from __future__ import annotations

from pathlib import Path, PurePath
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import Field, field_validator, model_validator

from kg_crag.models import UnifiedStrategy
from kg_crag.models.domain import StrictModel

ABSOLUTE_MAX_QUESTIONS = 100
ABSOLUTE_MAX_JUDGE_ANSWERS = 50


def _relative_path(value: str) -> str:
    path = PurePath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("evaluation paths must be workspace-relative and contained")
    return path.as_posix()


class UnifiedEvaluationConfig(StrictModel):
    manifest_path: str = "data/evaluation/unified/manifest.json"
    results_root: str = "data/evaluation/results/unified"
    selection_path: str = "data/evaluation/unified/development-selection.json"
    test_lock_root: str = "data/evaluation/results/unified/test-locks"
    strategies: list[UnifiedStrategy] = Field(
        default_factory=lambda: [strategy for strategy in UnifiedStrategy],
        min_length=3,
        max_length=3,
    )
    top_k: int = Field(default=10, ge=1, le=100)
    near_duplicate_threshold: float = Field(default=0.92, ge=0.8, le=1.0)
    max_questions: int = Field(default=80, ge=1, le=ABSOLUTE_MAX_QUESTIONS)
    max_candidates_per_question: int = Field(default=50, ge=1, le=100)
    max_loops_per_question: int = Field(default=2, ge=0, le=3)
    max_model_calls: int = Field(default=50, ge=0, le=100)
    max_input_tokens: int = Field(default=200_000, ge=0, le=1_000_000)
    max_output_tokens: int = Field(default=50_000, ge=0, le=250_000)
    max_latency_ms: int = Field(default=3_600_000, ge=1, le=86_400_000)
    max_estimated_cost: float = Field(default=10.0, ge=0.0, le=1000.0)
    max_trace_events: int = Field(default=20_000, ge=1, le=20_000)
    online: bool = False
    with_judge: bool = False
    judge_max_answers: int = Field(default=20, ge=1, le=ABSOLUTE_MAX_JUDGE_ANSWERS)
    judge_rounds: int = Field(default=1, ge=1, le=1)
    judge_models: int = Field(default=1, ge=1, le=1)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)

    @field_validator("manifest_path", "results_root", "selection_path", "test_lock_root")
    @classmethod
    def paths_are_contained(cls, value: str) -> str:
        return _relative_path(value)

    @field_validator("strategies")
    @classmethod
    def strategies_are_complete(cls, value: list[UnifiedStrategy]) -> list[UnifiedStrategy]:
        if len(value) != len(set(value)) or set(value) != set(UnifiedStrategy):
            raise ValueError(
                "unified evaluation requires each of the three strategies exactly once"
            )
        return value

    @model_validator(mode="after")
    def online_flags_are_consistent(self) -> UnifiedEvaluationConfig:
        if self.with_judge and not self.online:
            raise ValueError("Judge requires explicit online evaluation")
        return self


def load_unified_evaluation_config(path: Path) -> UnifiedEvaluationConfig:
    """只加载 evaluation.yaml 的统一评测段。"""

    with path.open("r", encoding="utf-8") as handle:
        payload: Any = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict) or "unified_evaluation" not in payload:
        raise ValueError("missing unified_evaluation configuration section")
    return UnifiedEvaluationConfig.model_validate(payload["unified_evaluation"])
