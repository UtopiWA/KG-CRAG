"""回答上下文、稳定引用编号与结构化候选解析。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import Field, ValidationError

from kg_crag.answering.config import AnswerGenerationConfig
from kg_crag.correction.identity import stable_digest
from kg_crag.models import (
    ClaimType,
    CoverageMatrix,
    Evidence,
    EvidenceRequirement,
    ExternalFacetCoverage,
    GroundedAnswerCandidate,
    GroundedCitation,
    GroundedClaim,
)
from kg_crag.models.domain import StrictModel


class _GeneratedClaim(StrictModel):
    text: str = Field(min_length=1, max_length=3000)
    claim_type: ClaimType
    citation_ids: list[str] = Field(default_factory=list, max_length=8)
    facet_ids: list[str] = Field(default_factory=list, max_length=20)


class _GeneratedAnswer(StrictModel):
    claims: list[_GeneratedClaim] = Field(min_length=1, max_length=100)
    confidence: float = Field(ge=0.0, le=1.0)


@dataclass(frozen=True)
class BoundEvidence:
    citation_id: str
    evidence: Evidence
    facet_ids: tuple[str, ...]
    rendered_content: str


def _facet_bindings(
    evidence_id: str,
    internal_matrix: CoverageMatrix | None,
    external_coverage: list[ExternalFacetCoverage],
) -> tuple[str, ...]:
    internal = (
        {
            item.facet_id
            for item in internal_matrix.entries
            if item.evidence_id == evidence_id and item.matched and not item.conflict
        }
        if internal_matrix is not None
        else set()
    )
    external = {
        item.facet_id
        for item in external_coverage
        if item.evidence_id == evidence_id and item.matched
    }
    return tuple(sorted(internal | external))


def build_answer_context(
    evidence: list[Evidence],
    *,
    internal_matrix: CoverageMatrix | None,
    external_coverage: list[ExternalFacetCoverage],
    config: AnswerGenerationConfig,
) -> tuple[str, list[BoundEvidence]]:
    """在总字符上限内选择 Evidence，并附带可声明的 facet 集合。"""

    blocks: list[str] = []
    selected: list[BoundEvidence] = []
    used = 0
    for index, item in enumerate(evidence[: config.max_selected_evidence], start=1):
        citation_id = f"E{index}"
        facets = _facet_bindings(item.evidence_id, internal_matrix, external_coverage)
        location = str(item.location.url) if item.external else item.paper_id or item.source_id
        header = (
            f"[{citation_id}] source={location} external={str(item.external).lower()} "
            f"facets={','.join(facets) or 'none'}\n"
        )
        remaining = config.max_context_chars - used - len(header)
        if remaining <= 0:
            break
        content = item.content[: min(config.max_chars_per_evidence, remaining)]
        if not content:
            continue
        block = header + content
        blocks.append(block)
        selected.append(BoundEvidence(citation_id, item.model_copy(deep=True), facets, content))
        used += len(block) + 2
        if used >= config.max_context_chars:
            break
    return "\n\n".join(blocks), selected


def _citation(binding: BoundEvidence) -> GroundedCitation:
    item = binding.evidence
    return GroundedCitation(
        citation_id=binding.citation_id,
        evidence_id=item.evidence_id,
        source_id=item.source_id,
        source_type=item.source_type,
        external=item.external,
        paper_id=item.paper_id,
        section=item.location.section,
        page=item.location.page,
        url=item.location.url,
    )


def _claim_id(payload: dict[str, Any]) -> str:
    return "claim-" + stable_digest(payload)[:16]


def parse_grounded_answer(
    raw: str,
    bindings: list[BoundEvidence],
    facets: list[EvidenceRequirement],
    *,
    config: AnswerGenerationConfig,
    allowed_binding_sets: set[tuple[tuple[str, ...], tuple[str, ...]]] | None = None,
) -> GroundedAnswerCandidate:
    """拒绝上下文外引用、facet 错配和重生成新增事实绑定。"""

    if len(raw) > config.max_response_chars:
        raise ValueError("answer response exceeds the configured size limit")
    try:
        payload = _GeneratedAnswer.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as error:
        raise ValueError("answer response is not valid structured output") from error
    if len(payload.claims) > config.max_claims:
        raise ValueError("answer contains too many claims")
    by_id = {item.citation_id: item for item in bindings}
    known_facets = {item.facet_id for item in facets}
    claims: list[GroundedClaim] = []
    used_citations: list[str] = []
    for item in payload.claims:
        citation_ids = list(dict.fromkeys(item.citation_ids))
        facet_ids = list(dict.fromkeys(item.facet_ids))
        if citation_ids != item.citation_ids or facet_ids != item.facet_ids:
            raise ValueError("answer identifiers must be unique")
        if set(citation_ids) - set(by_id):
            raise ValueError("answer contains an unknown citation")
        if set(facet_ids) - known_facets:
            raise ValueError("answer contains an unknown facet")
        supported_facets = {
            facet_id for citation_id in citation_ids for facet_id in by_id[citation_id].facet_ids
        }
        if item.claim_type is not ClaimType.UNCERTAIN and not set(facet_ids) <= supported_facets:
            raise ValueError("claim facet is not supported by its cited Evidence")
        binding_key = (tuple(sorted(citation_ids)), tuple(sorted(facet_ids)))
        if allowed_binding_sets is not None and binding_key not in allowed_binding_sets:
            raise ValueError("regenerated answer introduces a new fact binding")
        claim_payload = {
            "text": " ".join(item.text.split()),
            "claim_type": item.claim_type.value,
            "citation_ids": citation_ids,
            "facet_ids": facet_ids,
        }
        claims.append(GroundedClaim(claim_id=_claim_id(claim_payload), **claim_payload))
        used_citations.extend(value for value in citation_ids if value not in used_citations)
    citations = [_citation(by_id[item]) for item in used_citations]
    answer = "\n".join(
        f"{item.text} {' '.join(f'[{value}]' for value in item.citation_ids)}".rstrip()
        for item in claims
    )
    return GroundedAnswerCandidate(
        answer=answer,
        claims=claims,
        citations=citations,
        confidence=payload.confidence,
    )


def supported_binding_sets(
    candidate: GroundedAnswerCandidate,
) -> set[tuple[tuple[str, ...], tuple[str, ...]]]:
    return {
        (tuple(sorted(item.citation_ids)), tuple(sorted(item.facet_ids)))
        for item in candidate.claims
    }
