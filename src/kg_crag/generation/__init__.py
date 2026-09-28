"""基于证据的回答生成模块。"""

from kg_crag.generation.dense import (
    ContextEvidence,
    DenseRAGService,
    build_evidence_context,
    parse_generated_answer,
)
from kg_crag.generation.prompt import VersionedPrompt, load_prompt

__all__ = [
    "ContextEvidence",
    "DenseRAGService",
    "VersionedPrompt",
    "build_evidence_context",
    "load_prompt",
    "parse_generated_answer",
]
