"""Embedding 批处理、校验和内容寻址缓存。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from kg_crag.errors import KGCRAGError
from kg_crag.models import ErrorCode, ErrorDetail
from kg_crag.providers.base import EmbeddingProvider
from kg_crag.retrieval.config import DenseEmbeddingConfig

InputType = Literal["query", "document"]


@dataclass(frozen=True)
class EmbeddingBatchResult:
    vectors: list[list[float]]
    cache_hits: int
    cache_misses: int


def normalize_embedding_text(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value).strip()
    if not normalized:
        raise _embedding_error("embedding input must not be empty", retryable=False)
    return normalized


def _embedding_error(message: str, *, retryable: bool) -> KGCRAGError:
    return KGCRAGError(
        ErrorDetail(
            code=ErrorCode.EXTERNAL_SERVICE if retryable else ErrorCode.VALIDATION,
            message=message,
            retryable=retryable,
        )
    )


class EmbeddingService:
    """保持输入顺序，并仅在完整批次通过校验后发布缓存。"""

    def __init__(
        self,
        provider: EmbeddingProvider,
        config: DenseEmbeddingConfig,
        *,
        workspace_root: Path,
    ) -> None:
        self.provider = provider
        self.config = config
        self.workspace_root = workspace_root.resolve()
        self.cache_root = (self.workspace_root / config.cache_root).resolve()
        if not self.cache_root.is_relative_to(self.workspace_root):
            raise ValueError("embedding cache root escapes the workspace")
        namespace_payload = {
            "provider": config.provider,
            "model": config.model,
            "revision": config.revision,
            "dimensions": config.dimensions,
            "normalization": config.normalization,
            "normalize": config.normalize,
            "query_prefix": config.query_prefix,
            "document_prefix": config.document_prefix,
        }
        # 命名空间绑定所有会改变向量语义的配置，防止不同模型或前缀误用同一缓存。
        encoded = json.dumps(namespace_payload, sort_keys=True, separators=(",", ":")).encode()
        self.namespace = hashlib.sha256(encoded).hexdigest()

    async def embed(
        self,
        texts: list[str],
        *,
        input_type: InputType,
        use_cache: bool = True,
    ) -> EmbeddingBatchResult:
        if not texts:
            return EmbeddingBatchResult(vectors=[], cache_hits=0, cache_misses=0)
        normalized = [normalize_embedding_text(value) for value in texts]
        keys = [self._key(value, input_type) for value in normalized]
        # 以原始位置作为缺失队列，批量请求后仍能恢复调用方输入顺序。
        vectors: list[list[float] | None] = [None] * len(texts)
        missing: list[int] = []
        for index, key in enumerate(keys):
            cached = self._read_cache(key) if use_cache else None
            if cached is None:
                missing.append(index)
            else:
                vectors[index] = cached

        prefix = self.config.query_prefix if input_type == "query" else self.config.document_prefix
        for offset in range(0, len(missing), self.config.batch_size):
            positions = missing[offset : offset + self.config.batch_size]
            batch_texts = [prefix + normalized[index] for index in positions]
            produced = await self.provider.embed(batch_texts)
            checked = self._validate_batch(produced, len(positions))
            if use_cache:
                pairs = [
                    (keys[index], vector) for index, vector in zip(positions, checked, strict=True)
                ]
                self._publish_batch(pairs)
            for index, vector in zip(positions, checked, strict=True):
                vectors[index] = vector

        if any(vector is None for vector in vectors):
            raise _embedding_error("embedding batch is incomplete", retryable=False)
        return EmbeddingBatchResult(
            vectors=[vector for vector in vectors if vector is not None],
            cache_hits=len(texts) - len(missing),
            cache_misses=len(missing),
        )

    def _key(self, normalized: str, input_type: InputType) -> str:
        payload = f"{self.namespace}\n{input_type}\n{normalized}".encode()
        return hashlib.sha256(payload).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_root / self.namespace / f"{key}.json"

    def _read_cache(self, key: str) -> list[float] | None:
        path = self._cache_path(key)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("key") != key:
                return None
            vector = [float(value) for value in payload["vector"]]
            return self._validate_vector(vector)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _validate_batch(self, vectors: list[list[float]], expected: int) -> list[list[float]]:
        if len(vectors) != expected:
            raise _embedding_error(
                "embedding provider returned an unexpected count",
                retryable=False,
            )
        return [self._validate_vector([float(value) for value in vector]) for vector in vectors]

    def _validate_vector(self, vector: list[float]) -> list[float]:
        if len(vector) != self.config.dimensions:
            raise _embedding_error(
                "embedding provider returned an unexpected dimension",
                retryable=False,
            )
        if not all(math.isfinite(value) for value in vector):
            raise _embedding_error("embedding provider returned non-finite values", retryable=False)
        if self.config.normalize:
            norm = math.sqrt(sum(value * value for value in vector))
            if norm == 0.0:
                raise _embedding_error("embedding provider returned a zero vector", retryable=False)
            vector = [value / norm for value in vector]
        return vector

    def _publish_batch(self, entries: list[tuple[str, list[float]]]) -> None:
        created: list[Path] = []
        temporaries: list[Path] = []
        try:
            for key, vector in entries:
                destination = self._cache_path(key)
                destination.parent.mkdir(parents=True, exist_ok=True)
                data = (
                    json.dumps(
                        {"key": key, "vector": vector},
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode()
                temporary = destination.parent / f".{key}.tmp-{uuid.uuid4().hex}"
                temporaries.append(temporary)
                temporary.write_bytes(data)
                if temporary.read_bytes() != data:
                    raise OSError("embedding cache staging verification failed")
                if destination.exists():
                    current = self._read_cache(key)
                    if current == vector:
                        continue
                    destination.unlink()
                try:
                    # 硬链接只会原子创建新目标；并发写入同一键时可检测内容冲突。
                    os.link(temporary, destination)
                    created.append(destination)
                except FileExistsError:
                    if self._read_cache(key) != vector:
                        raise OSError("concurrent embedding cache conflict") from None
        except Exception:
            for path in created:
                path.unlink(missing_ok=True)
            raise
        finally:
            for path in temporaries:
                path.unlink(missing_ok=True)
