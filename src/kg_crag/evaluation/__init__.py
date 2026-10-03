"""检索、回答与 Agent 评测模块。"""

from kg_crag.evaluation.dense import (
    DenseEvaluationRunner,
    compute_metrics,
    load_question_set,
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
    "compute_hybrid_metrics",
    "compute_metrics",
    "load_question_set",
]
from kg_crag.evaluation.corrective import (
    aggregate_metrics,
    evaluate_offline_matrix,
    load_corrective_questions,
)

__all__ = ["aggregate_metrics", "evaluate_offline_matrix", "load_corrective_questions"]
