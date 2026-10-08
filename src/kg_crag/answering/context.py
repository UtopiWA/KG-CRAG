"""回答上下文、稳定引用编号与结构化候选解析。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
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


class AnswerResponseFailure(StrEnum):
    """可安全写入 Trace 的回答响应失败分类。"""

    RESPONSE_TOO_LARGE = "response_too_large"
    INVALID_JSON = "invalid_json"
    INVALID_SCHEMA = "invalid_schema"
    TOO_MANY_CLAIMS = "too_many_claims"
    DUPLICATE_IDENTIFIER = "duplicate_identifier"
    UNKNOWN_CITATION = "unknown_citation"
    UNKNOWN_FACET = "unknown_facet"
    UNSUPPORTED_FACET_BINDING = "unsupported_facet_binding"
    NEW_FACT_BINDING = "new_fact_binding"


class AnswerResponseError(ValueError):
    """保留有限失败类别，但不携带模型原文。"""

    def __init__(self, category: AnswerResponseFailure) -> None:
        self.category = category
        messages = {
            AnswerResponseFailure.RESPONSE_TOO_LARGE: "answer response exceeds size limit",
            AnswerResponseFailure.INVALID_JSON: "answer response is not valid JSON",
            AnswerResponseFailure.INVALID_SCHEMA: "answer response has invalid schema",
            AnswerResponseFailure.TOO_MANY_CLAIMS: "answer contains too many claims",
            AnswerResponseFailure.DUPLICATE_IDENTIFIER: "answer identifiers must be unique",
            AnswerResponseFailure.UNKNOWN_CITATION: "answer contains an unknown citation",
            AnswerResponseFailure.UNKNOWN_FACET: "answer contains an unknown facet",
            AnswerResponseFailure.UNSUPPORTED_FACET_BINDING: (
                "claim facet is not supported by its cited Evidence"
            ),
            AnswerResponseFailure.NEW_FACT_BINDING: (
                "regenerated answer introduces a new fact binding"
            ),
        }
        super().__init__(messages[category])


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
        raise AnswerResponseError(AnswerResponseFailure.RESPONSE_TOO_LARGE)
    normalized = raw.strip()
    # 只容忍单层 Markdown JSON 围栏，围栏内仍执行完整 Schema 与绑定校验。
    if normalized.startswith("```") and normalized.endswith("```"):
        first_newline = normalized.find("\n")
        if first_newline > 0:
            language = normalized[3:first_newline].strip().casefold()
            if language in {"", "json"}:
                normalized = normalized[first_newline + 1 : -3].strip()
    try:
        decoded = json.loads(normalized)
    except json.JSONDecodeError as error:
        raise AnswerResponseError(AnswerResponseFailure.INVALID_JSON) from error
    try:
        payload = _GeneratedAnswer.model_validate(decoded)
    except ValidationError as error:
        raise AnswerResponseError(AnswerResponseFailure.INVALID_SCHEMA) from error
    if len(payload.claims) > config.max_claims:
        raise AnswerResponseError(AnswerResponseFailure.TOO_MANY_CLAIMS)
    by_id = {item.citation_id: item for item in bindings}
    known_facets = {item.facet_id for item in facets}
    claims: list[GroundedClaim] = []
    used_citations: list[str] = []
    for item in payload.claims:
        citation_ids = list(dict.fromkeys(item.citation_ids))
        facet_ids = list(dict.fromkeys(item.facet_ids))
        if citation_ids != item.citation_ids or facet_ids != item.facet_ids:
            raise AnswerResponseError(AnswerResponseFailure.DUPLICATE_IDENTIFIER)
        if set(citation_ids) - set(by_id):
            raise AnswerResponseError(AnswerResponseFailure.UNKNOWN_CITATION)
        if set(facet_ids) - known_facets:
            raise AnswerResponseError(AnswerResponseFailure.UNKNOWN_FACET)
        supported_facets = {
            facet_id for citation_id in citation_ids for facet_id in by_id[citation_id].facet_ids
        }
        if item.claim_type is not ClaimType.UNCERTAIN and not set(facet_ids) <= supported_facets:
            raise AnswerResponseError(AnswerResponseFailure.UNSUPPORTED_FACET_BINDING)
        binding_key = (tuple(sorted(citation_ids)), tuple(sorted(facet_ids)))
        if allowed_binding_sets is not None and binding_key not in allowed_binding_sets:
            raise AnswerResponseError(AnswerResponseFailure.NEW_FACT_BINDING)
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
