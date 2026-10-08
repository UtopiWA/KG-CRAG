"""Qdrant 向量存储适配器。"""

from __future__ import annotations

from typing import Any, Protocol, cast

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from kg_crag.errors import KGCRAGError
from kg_crag.models import (
    Chunk,
    ErrorCode,
    ErrorDetail,
    Evidence,
    EvidenceLocation,
    EvidenceRanks,
    EvidenceScores,
    EvidenceSourceType,
    VectorCollectionIdentity,
)
from kg_crag.vector_store.base import VectorRecordState
from kg_crag.vector_store.identity import identity_point_id, point_id


class _ScoredPoint(Protocol):
    payload: dict[str, Any] | None
    score: float


def _store_error(
    message: str,
    *,
    retryable: bool,
    context: dict[str, Any] | None = None,
) -> KGCRAGError:
    safe_context = {
        key: value
        for key, value in (context or {}).items()
        if isinstance(value, str | int | float | bool) or value is None
    }
    return KGCRAGError(
        ErrorDetail(
            code=ErrorCode.EXTERNAL_SERVICE if retryable else ErrorCode.CONFIGURATION,
            message=message,
            retryable=retryable,
            context=safe_context,
        )
    )


class QdrantVectorStore:
    """只暴露公共 Evidence 和安全的集合生命周期。"""

    def __init__(
        self,
        identity: VectorCollectionIdentity,
        *,
        url: str | None = None,
        api_key: str | None = None,
        allowed_filters: set[str] | None = None,
        timeout_seconds: int = 30,
        client: object | None = None,
    ) -> None:
        self.identity = identity
        self.collection_version = identity.collection_version
        self.collection_name = identity.collection_name
        self.allowed_filters = allowed_filters or {"paper_id", "section", "processing_version"}
        self._client = cast(
            AsyncQdrantClient,
            client
            or AsyncQdrantClient(
                url=url,
                api_key=api_key,
                timeout=timeout_seconds,
            ),
        )

    async def ensure_collection(self, *, rebuild: bool = False) -> None:
        try:
            exists = await self._client.collection_exists(self.collection_name)
            if rebuild and exists:
                await self._client.delete_collection(self.collection_name)
                exists = False
            if not exists:
                await self._client.create_collection(
                    self.collection_name,
                    vectors_config=qm.VectorParams(
                        size=self.identity.dimensions,
                        distance=self._distance(),
                    ),
                )
                for field in ("record_type", "paper_id", "section", "processing_version"):
                    await self._client.create_payload_index(
                        self.collection_name,
                        field_name=field,
                        field_schema=qm.PayloadSchemaType.KEYWORD,
                    )
                # 身份哨兵把模型、维度和 payload 契约写进集合，复用前可做严格兼容检查。
                await self._client.upsert(
                    self.collection_name,
                    points=[
                        qm.PointStruct(
                            id=identity_point_id(self.collection_name),
                            vector=[0.0] * self.identity.dimensions,
                            payload={
                                "record_type": "collection_identity",
                                "identity": self.identity.model_dump(mode="json"),
                            },
                        )
                    ],
                    wait=True,
                )
            await self._validate_identity()
        except KGCRAGError:
            raise
        except Exception as error:
            raise self._translate(error, "Qdrant collection could not be prepared") from error

    async def ensure_existing_collection(self) -> None:
        """实时查询只验证既有集合，绝不因一次请求隐式创建空索引。"""

        try:
            if not await self._client.collection_exists(self.collection_name):
                raise _store_error(
                    "Qdrant collection is not initialized",
                    retryable=False,
                    context={"collection": self.collection_name},
                )
            await self._validate_identity()
        except KGCRAGError:
            raise
        except Exception as error:
            raise self._translate(error, "Qdrant collection could not be validated") from error

    async def _validate_identity(self) -> None:
        records = await self._client.retrieve(
            self.collection_name,
            ids=[identity_point_id(self.collection_name)],
            with_payload=True,
            with_vectors=False,
        )
        expected = self.identity.model_dump(mode="json")
        actual = records[0].payload.get("identity") if records and records[0].payload else None
        if actual != expected:
            raise _store_error(
                "Qdrant collection identity is incompatible",
                retryable=False,
                context={"collection": self.collection_name},
            )

    async def upsert(self, records: list[tuple[Chunk, list[float]]]) -> None:
        if not records:
            return
        points: list[qm.PointStruct] = []
        for chunk, vector in records:
            if len(vector) != self.identity.dimensions:
                raise _store_error(
                    "vector dimension does not match collection identity",
                    retryable=False,
                    context={"expected_dimensions": self.identity.dimensions},
                )
            points.append(
                qm.PointStruct(
                    id=point_id(chunk.chunk_id),
                    vector=vector,
                    payload={
                        "record_type": "chunk",
                        "chunk_id": chunk.chunk_id,
                        "paper_id": chunk.paper_id,
                        "section": chunk.section,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                        "ordinal": chunk.ordinal,
                        "text": chunk.text,
                        "content_hash": chunk.content_hash,
                        "processing_version": chunk.processing_version,
                    },
                )
            )
        try:
            await self._client.upsert(self.collection_name, points=points, wait=True)
        except Exception as error:
            raise self._translate(error, "Qdrant upsert failed") from error

    async def search(
        self,
        vector: list[float],
        *,
        top_k: int,
        filters: dict[str, str | int | bool] | None = None,
    ) -> list[Evidence]:
        if top_k <= 0 or not vector:
            raise _store_error("search vector and positive top_k are required", retryable=False)
        if len(vector) != self.identity.dimensions:
            raise _store_error("search vector dimension is incompatible", retryable=False)
        conditions = [qm.FieldCondition(key="record_type", match=qm.MatchValue(value="chunk"))]
        for key, value in sorted((filters or {}).items()):
            if key not in self.allowed_filters:
                raise _store_error(
                    "unsupported Qdrant filter field",
                    retryable=False,
                    context={"field": key},
                )
            conditions.append(qm.FieldCondition(key=key, match=qm.MatchValue(value=value)))
        try:
            response = await self._client.query_points(
                self.collection_name,
                query=vector,
                query_filter=qm.Filter(must=conditions),
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as error:
            raise self._translate(error, "Qdrant search failed") from error
        points = sorted(
            response.points,
            key=lambda item: (-float(item.score), str((item.payload or {}).get("chunk_id", ""))),
        )
        return [self._evidence(point, rank) for rank, point in enumerate(points, start=1)]

    async def delete_paper(self, paper_id: str) -> None:
        await self._delete_filter(self._paper_filter(paper_id))

    async def record_state(self, paper_id: str) -> dict[str, VectorRecordState]:
        state: dict[str, VectorRecordState] = {}
        offset: int | str | None = None
        try:
            if not await self._client.collection_exists(self.collection_name):
                return {}
            while True:
                # 分页读取论文级状态，避免大集合一次性加载全部 payload。
                records, offset = await self._client.scroll(
                    self.collection_name,
                    scroll_filter=self._paper_filter(paper_id),
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for record in records:
                    payload = record.payload or {}
                    chunk_id = str(payload.get("chunk_id", ""))
                    content_hash = str(payload.get("content_hash", ""))
                    processing_version = str(payload.get("processing_version", ""))
                    if not chunk_id or not content_hash or not processing_version:
                        raise _store_error(
                            "Qdrant point payload is incomplete",
                            retryable=False,
                            context={"collection": self.collection_name},
                        )
                    state[chunk_id] = VectorRecordState(content_hash, processing_version)
                if offset is None:
                    break
        except KGCRAGError:
            raise
        except Exception as error:
            raise self._translate(error, "Qdrant point state could not be read") from error
        return state

    async def delete_stale(self, paper_id: str, keep_chunk_ids: set[str]) -> int:
        state = await self.record_state(paper_id)
        stale = sorted(set(state) - keep_chunk_ids)
        if not stale:
            return 0
        try:
            # 只按已计算出的稳定点 ID 删除，避免使用宽泛过滤器误伤其他论文。
            await self._client.delete(
                self.collection_name,
                points_selector=qm.PointIdsList(points=[point_id(item) for item in stale]),
                wait=True,
            )
        except Exception as error:
            raise self._translate(error, "Qdrant stale point deletion failed") from error
        return len(stale)

    async def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            await close()

    async def _delete_filter(self, query_filter: qm.Filter) -> None:
        try:
            await self._client.delete(
                self.collection_name,
                points_selector=qm.FilterSelector(filter=query_filter),
                wait=True,
            )
        except Exception as error:
            raise self._translate(error, "Qdrant delete failed") from error

    @staticmethod
    def _paper_filter(paper_id: str) -> qm.Filter:
        return qm.Filter(
            must=[
                qm.FieldCondition(key="record_type", match=qm.MatchValue(value="chunk")),
                qm.FieldCondition(key="paper_id", match=qm.MatchValue(value=paper_id)),
            ]
        )

    def _distance(self) -> qm.Distance:
        return {
            "cosine": qm.Distance.COSINE,
            "dot": qm.Distance.DOT,
            "euclid": qm.Distance.EUCLID,
        }[self.identity.distance]

    def _evidence(self, point: _ScoredPoint, rank: int) -> Evidence:
        payload = point.payload or {}
        required = {"chunk_id", "paper_id", "text", "content_hash", "processing_version"}
        if not required.issubset(payload):
            raise _store_error("Qdrant search payload is incomplete", retryable=False)
        chunk_id = str(payload["chunk_id"])
        return Evidence(
            evidence_id=f"dense:{self.collection_version}:{chunk_id}",
            content=str(payload["text"]),
            source_type=EvidenceSourceType.CHUNK,
            source_id=chunk_id,
            paper_id=str(payload["paper_id"]),
            location=EvidenceLocation(
                section=str(payload["section"]) if payload.get("section") is not None else None,
                page=int(payload["page_start"]) if payload.get("page_start") is not None else None,
            ),
            scores=EvidenceScores(dense=float(point.score)),
            ranks=EvidenceRanks(dense=rank),
            external=False,
            metadata={
                "collection_version": self.collection_version,
                "content_hash": str(payload["content_hash"]),
                "processing_version": str(payload["processing_version"]),
                "page_end": (
                    int(payload["page_end"]) if payload.get("page_end") is not None else None
                ),
                "ordinal": int(payload.get("ordinal", 0)),
            },
        )

    def _translate(self, error: Exception, message: str) -> KGCRAGError:
        # 只保留异常类型推导出的重试语义和状态码，不传播可能含请求正文的原始消息。
        name = type(error).__name__.casefold()
        status = getattr(error, "status_code", None)
        retryable = bool(
            "timeout" in name
            or "connect" in name
            or status == 429
            or (isinstance(status, int) and status >= 500)
        )
        return _store_error(
            message,
            retryable=retryable,
            context={"collection": self.collection_name, "status": status},
        )
