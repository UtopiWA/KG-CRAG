"""惰性加载的 Sentence Transformers Embedding 适配器。"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

from kg_crag.errors import KGCRAGError
from kg_crag.models import ErrorCode, ErrorDetail
from kg_crag.providers.local_models import enforce_local_model_resolution


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
        local_files_only: bool = False,
        cache_folder: str | Path | None = None,
        model_factory: Callable[..., SentenceModel] | None = None,
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.dimensions = dimensions
        self.normalize = normalize
        self.local_files_only = local_files_only
        self.cache_folder = str(Path(cache_folder).resolve()) if cache_folder else None
        self._model_factory = model_factory
        self._model: SentenceModel | None = None
        self._load_lock = threading.Lock()

    def _load(self) -> SentenceModel:
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            try:
                enforce_local_model_resolution(self.local_files_only)
                if self._model_factory is None:
                    from sentence_transformers import SentenceTransformer

                    factory: Callable[..., SentenceModel] = SentenceTransformer
                else:
                    factory = self._model_factory
                kwargs: dict[str, str | bool] = {"revision": self.revision}
                if self.local_files_only:
                    # 离线实验必须禁止库在模型已缓存时仍发起远端版本探测。
                    kwargs["local_files_only"] = True
                if self.cache_folder:
                    kwargs["cache_folder"] = self.cache_folder
                self._model = factory(self.model_name, **kwargs)
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
        # 模型构造可能读取数 GB 权重或访问模型仓库，必须移出 API 事件循环。
        model = await asyncio.to_thread(self._load)
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

    async def warmup(self, text: str = "scientific literature retrieval") -> None:
        """提前完成权重加载和一次有界推理，避免首个用户请求承担冷启动。"""

        await self.embed([text])
