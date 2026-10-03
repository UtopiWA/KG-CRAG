"""固定修订、惰性加载且整批校验的本地 Cross-Encoder Reranker。"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

from kg_crag.errors import KGCRAGError
from kg_crag.models import ErrorCode, ErrorDetail, Evidence
from kg_crag.retrieval.config import RerankerConfig


class CrossEncoderModel(Protocol):
    def predict(
        self,
        sentences: list[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> object: ...


class CrossEncoderReranker:
    """只在非空 rerank 调用时加载权重，不返回部分批次结果。"""

    def __init__(
        self,
        config: RerankerConfig,
        *,
        workspace_root: Path,
        model_factory: Callable[..., CrossEncoderModel] | None = None,
    ) -> None:
        self.config = config
        self.workspace_root = workspace_root.resolve()
        self.cache_root = Path(config.cache_root)
        if not self.cache_root.is_absolute():
            self.cache_root = (self.workspace_root / self.cache_root).resolve()
        self._model_factory = model_factory
        self._model: CrossEncoderModel | None = None

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> CrossEncoderModel:
        if self._model is not None:
            return self._model
        try:
            if self._model_factory is None:
                from sentence_transformers import CrossEncoder

                factory: Callable[..., CrossEncoderModel] = CrossEncoder
            else:
                factory = self._model_factory
            self._model = factory(
                self.config.model,
                revision=self.config.revision,
                device=self.config.device,
                cache_dir=str(self.cache_root),
            )
        except Exception as error:
            raise _reranker_error("reranker model could not be loaded", retryable=True) from error
        return self._model

    async def rerank(
        self,
        query: str,
        candidates: list[Evidence],
        *,
        top_k: int,
    ) -> list[Evidence]:
        normalized = query.strip()
        if not normalized:
            raise _reranker_error("reranker query must not be empty")
        if top_k <= 0 or top_k > self.config.top_k:
            raise _reranker_error("reranker top_k is outside the configured boundary")
        if len(candidates) > self.config.max_candidates:
            raise _reranker_error("reranker candidate count exceeds the configured boundary")
        if not candidates:
            return []
        pairs = [(normalized, item.content) for item in candidates]
        model = self._load()
        try:
            raw = await asyncio.to_thread(
                model.predict,
                pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
                convert_to_numpy=False,
            )
            scores = _to_scores(raw)
        except KGCRAGError:
            raise
        except Exception as error:
            raise _reranker_error("reranker inference failed", retryable=True) from error
        if len(scores) != len(candidates):
            raise _reranker_error("reranker returned an unexpected score count")
        if any(not math.isfinite(score) for score in scores):
            raise _reranker_error("reranker returned a non-finite score")

        scored = list(zip(candidates, scores, strict=True))
        scored.sort(
            key=lambda item: (
                -item[1],
                item[0].ranks.fusion or 2**31 - 1,
                item[0].source_id,
            )
        )
        results: list[Evidence] = []
        for rank, (candidate, score) in enumerate(scored[:top_k], start=1):
            copied = candidate.model_copy(deep=True)
            copied.scores.rerank = score
            copied.ranks.rerank = rank
            results.append(copied)
        return results


def _to_scores(raw: object) -> list[float]:
    converted = raw.tolist() if hasattr(raw, "tolist") else raw
    if not isinstance(converted, list | tuple):
        raise _reranker_error("reranker returned an unsupported score container")
    scores: list[float] = []
    for value in converted:
        if isinstance(value, list | tuple):
            if len(value) != 1:
                raise _reranker_error("reranker returned a non-scalar score")
            value = value[0]
        try:
            scores.append(float(cast(float, value)))
        except (TypeError, ValueError) as error:
            raise _reranker_error("reranker returned a non-numeric score") from error
    return scores


def _reranker_error(message: str, *, retryable: bool = False) -> KGCRAGError:
    return KGCRAGError(
        ErrorDetail(
            code=ErrorCode.EXTERNAL_SERVICE if retryable else ErrorCode.VALIDATION,
            message=message,
            retryable=retryable,
            context={"provider": "sentence-transformers-cross-encoder"},
        )
    )
