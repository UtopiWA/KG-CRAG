"""检索、回答与 Agent 评测模块。"""

from kg_crag.evaluation.corrective import (
    aggregate_metrics,
    evaluate_offline_matrix,
    load_corrective_questions,
)
from kg_crag.evaluation.dataset import load_unified_dataset, validate_unified_dataset
from kg_crag.evaluation.dense import (
    DenseEvaluationRunner,
    compute_metrics,
    load_question_set,
)
from kg_crag.evaluation.grounded import (
    compute_grounded_metrics,
    evaluate_online_grounded_probe,
    evaluate_recorded_grounded_questions,
    load_grounded_questions,
    validate_recorded_web_fixture,
)
from kg_crag.evaluation.hybrid import (
    HybridEvaluationRunner,
    HybridStrategyOutput,
    compute_hybrid_metrics,
)
from kg_crag.evaluation.runner import (
    CallableStrategyAdapter,
    FixtureStrategyAdapter,
    RecordedStrategyAdapter,
    UnifiedEvaluationRunner,
    build_run_identity,
    fixture_adapters,
    recorded_adapters,
)

__all__ = [
    "CallableStrategyAdapter",
    "DenseEvaluationRunner",
    "FixtureStrategyAdapter",
    "HybridEvaluationRunner",
    "HybridStrategyOutput",
    "RecordedStrategyAdapter",
    "UnifiedEvaluationRunner",
    "aggregate_metrics",
    "build_run_identity",
    "compute_grounded_metrics",
    "compute_hybrid_metrics",
    "compute_metrics",
    "evaluate_offline_matrix",
    "evaluate_online_grounded_probe",
    "evaluate_recorded_grounded_questions",
    "fixture_adapters",
    "load_corrective_questions",
    "load_grounded_questions",
    "load_question_set",
    "load_unified_dataset",
    "recorded_adapters",
    "validate_recorded_web_fixture",
    "validate_unified_dataset",
]
