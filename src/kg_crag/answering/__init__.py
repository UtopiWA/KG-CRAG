"""可追溯回答与单轮反思能力。"""

from kg_crag.answering.checker import deterministic_check, merge_critic_findings
from kg_crag.answering.config import (
    GroundedAnswerConfig,
    GroundedEvaluationConfig,
    grounded_answer_config_hash,
    load_grounded_answer_config,
    load_grounded_evaluation_config,
)
from kg_crag.answering.conservative import build_conservative_candidate
from kg_crag.answering.context import (
    BoundEvidence,
    build_answer_context,
    parse_grounded_answer,
    supported_binding_sets,
)
from kg_crag.answering.critic import SemanticCritic
from kg_crag.answering.identity import answer_state_fingerprint, build_answer_identity
from kg_crag.answering.policy import decide_reflection

__all__ = [
    "BoundEvidence",
    "GroundedAnswerConfig",
    "GroundedEvaluationConfig",
    "SemanticCritic",
    "answer_state_fingerprint",
    "build_answer_context",
    "build_answer_identity",
    "build_conservative_candidate",
    "decide_reflection",
    "deterministic_check",
    "grounded_answer_config_hash",
    "load_grounded_answer_config",
    "load_grounded_evaluation_config",
    "merge_critic_findings",
    "parse_grounded_answer",
    "supported_binding_sets",
]
