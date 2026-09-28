"""惰性加载的 Sentence Transformers Embedding 适配器。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Protocol, cast

from kg_crag.errors import KGCRAGError
from kg_crag.models import ErrorCode, ErrorDetail


class SentenceModel(Protocol):
    def encode(
        self,
        sentences: list[str],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> object: ...


class _ArrayLike(Protocol):
    def tolist(self) -> list[list[float]]: ...


class SentenceTransformerEmbeddingProvider:
    """仅在首次显式 embed 时加载模型，导入模块不会下载权重。"""

    def __init__(
        self,
        model_name: str,
        revision: str,
        *,
        dimensions: int,
        normalize: bool,
        model_factory: Callable[..., SentenceModel] | None = None,
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.dimensions = dimensions
        self.normalize = normalize
        self._model_factory = model_factory
        self._model: SentenceModel | None = None

    def _load(self) -> SentenceModel:
        if self._model is not None:
            return self._model
        try:
            if self._model_factory is None:
                from sentence_transformers import SentenceTransformer

                factory: Callable[..., SentenceModel] = SentenceTransformer
            else:
                factory = self._model_factory
            self._model = factory(self.model_name, revision=self.revision)
        except Exception as error:
            raise KGCRAGError(
                ErrorDetail(
                    code=ErrorCode.EXTERNAL_SERVICE,
                    message="embedding model could not be loaded",
                    retryable=True,
                    context={"provider": "sentence-transformers"},
                )
            ) from error
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        try:
            encoded = await asyncio.to_thread(
                model.encode,
                texts,
                convert_to_numpy=False,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
            raw = (
                cast(_ArrayLike, encoded).tolist()
                if hasattr(encoded, "tolist")
                else cast(list[list[float]], encoded)
            )
            vectors = [[float(value) for value in vector] for vector in raw]
        except KGCRAGError:
            raise
        except Exception as error:
            raise KGCRAGError(
                ErrorDetail(
                    code=ErrorCode.EXTERNAL_SERVICE,
                    message="embedding provider failed",
                    retryable=True,
                    context={"provider": "sentence-transformers"},
                )
            ) from error
        if any(len(vector) != self.dimensions for vector in vectors):
            raise KGCRAGError(
                ErrorDetail(
                    code=ErrorCode.DATA,
                    message="embedding provider returned an unexpected dimension",
                    retryable=False,
                    context={"expected_dimensions": self.dimensions},
                )
            )
        return vectors
