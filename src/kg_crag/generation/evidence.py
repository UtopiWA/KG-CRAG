"""回答生成共用的 Evidence 上下文和引用约束。"""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import Field, ValidationError

from kg_crag.errors import KGCRAGError
from kg_crag.models import AnswerClaim, Citation, ErrorCode, ErrorDetail, Evidence
from kg_crag.models.domain import StrictModel
from kg_crag.retrieval.config import DenseRAGConfig


class _GeneratedPayload(StrictModel):
    claims: list[AnswerClaim] = Field(min_length=1, max_length=100)
    confidence: float = Field(ge=0.0, le=1.0)


@dataclass(frozen=True)
class ContextEvidence:
    citation_id: str
    evidence: Evidence
    rendered_content: str


def build_evidence_context(
    evidence: list[Evidence],
    config: DenseRAGConfig,
) -> tuple[str, list[ContextEvidence]]:
    """在字符和条目上限内给证据分配稳定的局部引用编号。"""

    selected: list[ContextEvidence] = []
    blocks: list[str] = []
    used = 0
    for index, item in enumerate(evidence[: config.generation.max_evidence], start=1):
        citation_id = f"E{index}"
        content = item.content[: config.generation.max_chars_per_evidence]
        header = (
            f"[{citation_id}] paper={item.paper_id or 'unknown'} "
            f"section={item.location.section or 'unknown'} page={item.location.page or 'unknown'}\n"
        )
        remaining = config.generation.max_context_chars - used - len(header)
        if remaining <= 0:
            break
        # 先为引用头保留空间，再截断正文，保证每个进入 Prompt 的片段仍可定位。
        content = content[:remaining]
        block = header + content
        blocks.append(block)
        selected.append(ContextEvidence(citation_id, item.model_copy(deep=True), content))
        used += len(block) + 2
        if used >= config.generation.max_context_chars:
            break
    return "\n\n".join(blocks), selected


def parse_generated_answer(
    raw: str,
    context: list[ContextEvidence],
    *,
    max_chars: int,
) -> tuple[list[AnswerClaim], list[Citation], float, str]:
    """解析结构化回答，并拒绝不在当前上下文内的引用。"""

    if len(raw) > max_chars:
        raise generation_error("LLM response exceeds the configured size limit")
    try:
        payload = _GeneratedPayload.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError, ValueError) as error:
        raise generation_error("LLM response is not valid structured output") from error
    by_id = {item.citation_id: item.evidence for item in context}
    used_ids: list[str] = []
    for claim in payload.claims:
        for citation_id in claim.citation_ids:
            if citation_id not in by_id:
                raise generation_error("LLM response contains an unknown citation")
            if citation_id not in used_ids:
                used_ids.append(citation_id)
    citations: list[Citation] = []
    for citation_id in used_ids:
        evidence = by_id[citation_id]
        if evidence.paper_id is None:
            raise generation_error("cited Evidence is missing a paper ID")
        citations.append(
            Citation(
                citation_id=citation_id,
                evidence_id=evidence.evidence_id,
                source_id=evidence.source_id,
                paper_id=evidence.paper_id,
                section=evidence.location.section,
                page=evidence.location.page,
            )
        )
    answer = "\n".join(
        f"{claim.text} {' '.join(f'[{item}]' for item in claim.citation_ids)}".rstrip()
        for claim in payload.claims
    )
    return payload.claims, citations, payload.confidence, answer


def generation_error(message: str) -> KGCRAGError:
    """构造不泄漏 Prompt、正文或本地路径的生成错误。"""

    return KGCRAGError(
        ErrorDetail(
            code=ErrorCode.DATA,
            message=message,
            retryable=False,
        )
    )
