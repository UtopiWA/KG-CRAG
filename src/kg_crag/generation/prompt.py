"""版本化 Dense RAG Prompt 加载。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VersionedPrompt:
    template: str
    version: str

    def render(self, *, question: str, evidence_context: str) -> str:
        if not question.strip():
            raise ValueError("question must not be empty")
        return self.template.replace("{question}", question.strip()).replace(
            "{evidence_context}", evidence_context
        )


def load_prompt(path: Path) -> VersionedPrompt:
    template = path.read_text(encoding="utf-8")
    if template.count("{question}") != 1 or template.count("{evidence_context}") != 1:
        raise ValueError("Dense RAG prompt must contain question and evidence_context placeholders")
    return VersionedPrompt(template=template, version=hashlib.sha256(template.encode()).hexdigest())
