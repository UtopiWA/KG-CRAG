"""检索、回答与 Agent 评测模块。"""

from kg_crag.evaluation.corrective import (
    aggregate_metrics,
    evaluate_offline_matrix,
    load_corrective_questions,
)
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

__all__ = [
    "DenseEvaluationRunner",
    "HybridEvaluationRunner",
    "HybridStrategyOutput",
    "aggregate_metrics",
    "compute_grounded_metrics",
    "compute_hybrid_metrics",
    "compute_metrics",
    "evaluate_offline_matrix",
    "evaluate_online_grounded_probe",
    "evaluate_recorded_grounded_questions",
    "load_corrective_questions",
    "load_grounded_questions",
    "load_question_set",
    "validate_recorded_web_fixture",
]
