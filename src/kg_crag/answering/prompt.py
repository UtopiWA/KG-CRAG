"""回答与 Critic Prompt 的版本化加载和有界渲染。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StructuredPrompt:
    template: str
    version: str
    placeholders: frozenset[str]

    def render(self, *, max_chars: int, **values: str) -> str:
        if set(values) != set(self.placeholders):
            raise ValueError("prompt render values do not match declared placeholders")
        rendered = self.template
        for name in sorted(self.placeholders):
            rendered = rendered.replace("{" + name + "}", values[name])
        if len(rendered) > max_chars:
            raise ValueError("rendered prompt exceeds the configured size limit")
        return rendered


def load_structured_prompt(path: Path, *, placeholders: set[str]) -> StructuredPrompt:
    template = path.read_text(encoding="utf-8")
    for name in placeholders:
        if template.count("{" + name + "}") != 1:
            raise ValueError(f"prompt must contain exactly one {{{name}}} placeholder")
    return StructuredPrompt(
        template=template,
        version=hashlib.sha256(template.encode()).hexdigest(),
        placeholders=frozenset(placeholders),
    )
