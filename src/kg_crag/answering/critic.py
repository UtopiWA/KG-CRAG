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
        rendered = self.prompt.render(
            max_chars=self.max_prompt_chars,
            question=question,
            claims=json.dumps(
                [item.model_dump(mode="json") for item in candidate.claims],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            evidence_context=evidence_context,
            facets=facets,
            conflicts=conflicts,
        )
        raw = await self.provider.generate(rendered, temperature=0.0)
        if len(raw) > self.max_response_chars:
            raise ValueError("critic response exceeds the configured size limit")
        try:
            payload = _CriticPayload.model_validate(json.loads(raw))
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
