"""受限语义 Critic 的 Prompt 构造和严格输出解析。"""

from __future__ import annotations

import json

from pydantic import Field, ValidationError

from kg_crag.answering.prompt import StructuredPrompt
from kg_crag.models import AnswerFinding, GroundedAnswerCandidate
from kg_crag.models.domain import StrictModel
from kg_crag.providers import LLMProvider


class _CriticPayload(StrictModel):
    findings: list[AnswerFinding] = Field(default_factory=list, max_length=100)


class SemanticCritic:
    def __init__(
        self,
        provider: LLMProvider,
        prompt: StructuredPrompt,
        *,
        max_prompt_chars: int,
        max_response_chars: int,
    ) -> None:
        self.provider = provider
        self.prompt = prompt
        self.max_prompt_chars = max_prompt_chars
        self.max_response_chars = max_response_chars

    async def evaluate(
        self,
        *,
        question: str,
        candidate: GroundedAnswerCandidate,
        evidence_context: str,
        facets: str,
        conflicts: str,
    ) -> list[AnswerFinding]:
        values = {
            "question": question,
            "claims": json.dumps(
                [item.model_dump(mode="json") for item in candidate.claims],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "facets": facets,
            "conflicts": conflicts,
        }
        # max_prompt_chars 是整个 Prompt 的硬上限。先为模板、问题和结论预留
        # 空间，再截取 Evidence，避免较长证据包在调用 Critic 前必然失败。
        fixed = self.prompt.render(
            max_chars=self.max_prompt_chars,
            evidence_context="",
            **values,
        )
        remaining = self.max_prompt_chars - len(fixed)
        rendered = self.prompt.render(
            max_chars=self.max_prompt_chars,
            evidence_context=evidence_context[:remaining],
            **values,
        )
        raw = await self.provider.generate(rendered, temperature=0.0)
        if len(raw) > self.max_response_chars:
            raise ValueError("critic response exceeds the configured size limit")
        normalized = raw.strip()
        # 与回答生成保持相同的单层 JSON 围栏兼容；围栏外文本和多对象仍拒绝。
        if normalized.startswith("```") and normalized.endswith("```"):
            first_newline = normalized.find("\n")
            if first_newline > 0:
                language = normalized[3:first_newline].strip().casefold()
                if language in {"", "json"}:
                    normalized = normalized[first_newline + 1 : -3].strip()
        try:
            payload = _CriticPayload.model_validate(json.loads(normalized))
        except (json.JSONDecodeError, ValidationError) as error:
            raise ValueError("critic response is not valid structured output") from error
        claim_ids = {item.claim_id for item in candidate.claims}
        citation_ids = {item.citation_id for item in candidate.citations}
        facet_ids = {value for item in candidate.claims for value in item.facet_ids}
        for finding in payload.findings:
            if set(finding.claim_ids) - claim_ids:
                raise ValueError("critic finding references an unknown claim")
            if set(finding.citation_ids) - citation_ids:
                raise ValueError("critic finding references an unknown citation")
            if set(finding.facet_ids) - facet_ids:
                raise ValueError("critic finding references an unknown facet")
        return payload.findings
