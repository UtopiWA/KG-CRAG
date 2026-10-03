"""Dense Evidence 上下文、结构化回答与最小 Trace。"""

from __future__ import annotations

import os
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

from kg_crag.generation.evidence import (
    ContextEvidence,
    build_evidence_context,
    generation_error,
    parse_generated_answer,
)
from kg_crag.generation.prompt import VersionedPrompt
from kg_crag.ingestion.storage import stable_json_bytes
from kg_crag.models import (
    DenseRAGResult,
    TraceEvent,
)
from kg_crag.providers import LLMProvider
from kg_crag.retrieval.config import DenseRAGConfig, dense_config_hash
from kg_crag.retrieval.dense import DenseRetriever

__all__ = [
    "ContextEvidence",
    "DenseRAGService",
    "build_evidence_context",
    "parse_generated_answer",
]


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
            raise generation_error("question must not be empty")
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
            # 证据门槛在调用 LLM 前判断，证据不足时返回固定结果且不消耗 Token。
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
                raise generation_error("bounded context cannot contain the minimum Evidence")
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
            # 在临时目录写入并用公共模型回读，验证成功后才原子发布整个运行目录。
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
