"""Dense Evidence 上下文、结构化回答与最小 Trace。"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import Field, ValidationError

from kg_crag.errors import KGCRAGError
from kg_crag.generation.prompt import VersionedPrompt
from kg_crag.ingestion.storage import stable_json_bytes
from kg_crag.models import (
    AnswerClaim,
    Citation,
    DenseRAGResult,
    ErrorCode,
    ErrorDetail,
    Evidence,
    TraceEvent,
)
from kg_crag.models.domain import StrictModel
from kg_crag.providers import LLMProvider
from kg_crag.retrieval.config import DenseRAGConfig, dense_config_hash
from kg_crag.retrieval.dense import DenseRetriever


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
    if len(raw) > max_chars:
        raise _generation_error("LLM response exceeds the configured size limit")
    try:
        payload = _GeneratedPayload.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError, ValueError) as error:
        raise _generation_error("LLM response is not valid structured output") from error
    by_id = {item.citation_id: item.evidence for item in context}
    used_ids: list[str] = []
    for claim in payload.claims:
        for citation_id in claim.citation_ids:
            if citation_id not in by_id:
                raise _generation_error("LLM response contains an unknown citation")
            if citation_id not in used_ids:
                used_ids.append(citation_id)
    citations: list[Citation] = []
    for citation_id in used_ids:
        evidence = by_id[citation_id]
        if evidence.paper_id is None:
            raise _generation_error("cited Evidence is missing a paper ID")
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


def _generation_error(message: str) -> KGCRAGError:
    return KGCRAGError(
        ErrorDetail(
            code=ErrorCode.DATA,
            message=message,
            retryable=False,
        )
    )


class DenseRAGService:
    """一次检索、至多一次生成，不进行隐藏重试。"""

    def __init__(
        self,
        retriever: DenseRetriever,
        llm: LLMProvider,
        config: DenseRAGConfig,
        prompt: VersionedPrompt,
        *,
        collection_version: str,
        workspace_root: Path,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.config = config
        self.prompt = prompt
        self.collection_version = collection_version
        self.workspace_root = workspace_root.resolve()

    async def ask(
        self,
        question: str,
        *,
        top_k: int,
        filters: dict[str, str | int | bool] | None = None,
        deduplicate_by_paper: bool | None = None,
        persist: bool = True,
    ) -> DenseRAGResult:
        normalized = question.strip()
        if not normalized:
            raise _generation_error("question must not be empty")
        run_id = uuid.uuid4().hex
        trace_id = f"trace-{run_id}"
        trace: list[TraceEvent] = []
        self._trace(trace, trace_id, "query", "accepted", {"top_k": top_k})
        evidence = await self.retriever.retrieve(
            normalized,
            top_k=top_k,
            filters=filters,
            deduplicate_by_paper=deduplicate_by_paper,
        )
        self._trace(
            trace,
            trace_id,
            "retrieval",
            "completed",
            {"candidate_count": len(evidence), "strategy": "dense"},
        )
        selected = [
            item
            for item in evidence
            if item.scores.dense is not None
            and item.scores.dense >= self.config.generation.min_score
        ][: self.config.generation.max_evidence]
        if len(selected) < self.config.generation.min_evidence:
            self._trace(trace, trace_id, "generation", "skipped", {"reason": "insufficient"})
            result = DenseRAGResult(
                run_id=run_id,
                trace_id=trace_id,
                question=normalized,
                answer="内部 Dense 证据不足，无法可靠回答该问题。",
                claims=[],
                citations=[],
                evidence=selected,
                confidence=0.0,
                insufficient_evidence=True,
                config_hash=dense_config_hash(self.config),
                prompt_version=self.prompt.version,
                collection_version=self.collection_version,
                trace=trace,
            )
        else:
            context_text, context = build_evidence_context(selected, self.config)
            if len(context) < self.config.generation.min_evidence:
                raise _generation_error("bounded context cannot contain the minimum Evidence")
            self._trace(trace, trace_id, "evidence", "selected", {"count": len(context)})
            raw = await self.llm.generate(
                self.prompt.render(question=normalized, evidence_context=context_text),
                temperature=0.0,
            )
            claims, citations, confidence, answer = parse_generated_answer(
                raw,
                context,
                max_chars=self.config.generation.max_response_chars,
            )
            self._trace(trace, trace_id, "generation", "completed", {"claim_count": len(claims)})
            result = DenseRAGResult(
                run_id=run_id,
                trace_id=trace_id,
                question=normalized,
                answer=answer,
                claims=claims,
                citations=citations,
                evidence=[item.evidence for item in context],
                confidence=confidence,
                insufficient_evidence=False,
                config_hash=dense_config_hash(self.config),
                prompt_version=self.prompt.version,
                collection_version=self.collection_version,
                trace=trace,
            )
        if persist:
            self._persist(result)
        return result

    @staticmethod
    def _trace(
        target: list[TraceEvent],
        trace_id: str,
        node: str,
        event: str,
        details: dict[str, str | int | float | bool | None],
    ) -> None:
        target.append(
            TraceEvent(
                trace_id=trace_id,
                sequence=len(target),
                node=node,
                event=event,
                occurred_at=datetime.now(UTC),
                details=details,
            )
        )

    def _persist(self, result: DenseRAGResult) -> Path:
        root = (self.workspace_root / self.config.generation.output_root).resolve()
        if not root.is_relative_to(self.workspace_root):
            raise ValueError("query output root escapes the workspace")
        destination = root / result.run_id
        temporary = root / f".{result.run_id}.tmp-{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        try:
            temporary.mkdir(exist_ok=False)
            data = stable_json_bytes(result)
            path = temporary / "result.json"
            path.write_bytes(data)
            DenseRAGResult.model_validate_json(path.read_text(encoding="utf-8"))
            os.replace(temporary, destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return destination / "result.json"
