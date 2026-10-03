"""显式启用回答、默认零 LLM 的 Hybrid 查询服务。"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Literal

from kg_crag.generation.evidence import (
    build_evidence_context,
    generation_error,
    parse_generated_answer,
)
from kg_crag.generation.prompt import VersionedPrompt
from kg_crag.ingestion.storage import stable_json_bytes
from kg_crag.models import Evidence, HybridQueryResult, HybridRetrievalResult, RetrievalStageStatus
from kg_crag.providers import LLMProvider
from kg_crag.retrieval import HybridRetrievalConfig, HybridRetrievalService


class HybridQueryService:
    """固定一次检索；只有显式请求且证据充足时才生成一次。"""

    def __init__(
        self,
        retrieval: HybridRetrievalService,
        llm: LLMProvider,
        config: HybridRetrievalConfig,
        prompt: VersionedPrompt,
        *,
        workspace_root: Path,
    ) -> None:
        self.retrieval = retrieval
        self.llm = llm
        self.config = config
        self.prompt = prompt
        self.workspace_root = workspace_root.resolve()

    async def ask(
        self,
        question: str,
        *,
        with_answer: bool = False,
        filters: dict[str, str | int | bool] | None = None,
        persist: bool = False,
    ) -> HybridQueryResult:
        retrieval = await self.retrieval.retrieve(question, filters=filters, persist=False)
        path: list[Literal["dense", "sparse"]] = []
        for stage in retrieval.stages:
            if stage.stage == "dense" and stage.status is not RetrievalStageStatus.FAILED:
                path.append("dense")
            elif stage.stage == "sparse" and stage.status is not RetrievalStageStatus.FAILED:
                path.append("sparse")
        if not path:
            raise generation_error("hybrid retrieval produced no usable path")
        if not with_answer:
            result = HybridQueryResult(
                run_id=retrieval.run_id,
                question=retrieval.query,
                retrieval=retrieval,
                retrieval_path=path,
            )
        else:
            result = await self._answer(retrieval, path)
        if persist:
            self._persist(result)
        return result

    async def _answer(
        self,
        checked: HybridRetrievalResult,
        path: list[Literal["dense", "sparse"]],
    ) -> HybridQueryResult:
        generation = self.config.dense.generation
        selected: list[Evidence] = []
        for item in checked.evidence:
            score = _final_score(item)
            if score is not None and score >= generation.min_score:
                selected.append(item)
            if len(selected) >= generation.max_evidence:
                break
        if len(selected) < generation.min_evidence:
            return HybridQueryResult(
                run_id=checked.run_id,
                question=checked.query,
                retrieval=checked,
                retrieval_path=path,
                with_answer=True,
                answer="内部 Hybrid 证据不足，无法可靠回答该问题。",
                confidence=0.0,
                insufficient_evidence=True,
                prompt_version=self.prompt.version,
            )
        context_text, context = build_evidence_context(selected, self.config.dense)
        if len(context) < generation.min_evidence:
            raise generation_error("bounded context cannot contain the minimum Evidence")
        raw = await self.llm.generate(
            self.prompt.render(question=checked.query, evidence_context=context_text),
            temperature=0.0,
        )
        claims, citations, confidence, answer = parse_generated_answer(
            raw,
            context,
            max_chars=generation.max_response_chars,
        )
        return HybridQueryResult(
            run_id=checked.run_id,
            question=checked.query,
            retrieval=checked,
            retrieval_path=path,
            with_answer=True,
            answer=answer,
            claims=claims,
            citations=citations,
            confidence=confidence,
            prompt_version=self.prompt.version,
            llm_calls=1,
        )

    def _persist(self, result: HybridQueryResult) -> Path:
        root = (self.workspace_root / self.config.hybrid.output_root).resolve()
        if not root.is_relative_to(self.workspace_root):
            raise ValueError("hybrid query output root escapes the workspace")
        destination = root / result.run_id
        temporary = root / f".{result.run_id}.tmp-{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        try:
            temporary.mkdir(exist_ok=False)
            path = temporary / "query-result.json"
            path.write_bytes(stable_json_bytes(result))
            HybridQueryResult.model_validate_json(path.read_text(encoding="utf-8"))
            os.replace(temporary, destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return destination / "query-result.json"


def _final_score(item: Evidence) -> float | None:
    for value in (
        item.scores.rerank,
        item.scores.fusion,
        item.scores.dense,
        item.scores.sparse,
    ):
        if value is not None:
            return value
    return None
