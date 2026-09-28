"""检索、回答与 Agent 评测模块。"""

from kg_crag.evaluation.dense import (
    DenseEvaluationRunner,
    compute_metrics,
    load_question_set,
)

__all__ = ["DenseEvaluationRunner", "compute_metrics", "load_question_set"]
