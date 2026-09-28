"""向量集合身份、点标识和语料快照纯函数。"""

from __future__ import annotations

import hashlib
import json
import uuid

from kg_crag.models import Chunk, VectorCollectionIdentity
from kg_crag.retrieval.config import DenseRAGConfig

PAYLOAD_FIELDS = [
    "record_type",
    "chunk_id",
    "paper_id",
    "section",
    "page_start",
    "page_end",
    "ordinal",
    "text",
    "content_hash",
    "processing_version",
]


def collection_identity(config: DenseRAGConfig) -> VectorCollectionIdentity:
    payload = {
        "schema_version": config.collection.schema_version,
        "embedding_provider": config.embedding.provider,
        "embedding_model": config.embedding.model,
        "embedding_revision": config.embedding.revision,
        "dimensions": config.embedding.dimensions,
        "distance": config.collection.distance,
        "payload_fields": PAYLOAD_FIELDS,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    version = hashlib.sha256(encoded).hexdigest()
    name = f"{config.collection.prefix}_{version[:16]}"
    return VectorCollectionIdentity(
        **payload,
        collection_version=version,
        collection_name=name,
    )


def corpus_snapshot_hash(chunks: list[Chunk]) -> str:
    rows = [
        {
            "paper_id": chunk.paper_id,
            "processing_version": chunk.processing_version,
            "chunk_id": chunk.chunk_id,
            "content_hash": chunk.content_hash,
        }
        for chunk in sorted(chunks, key=lambda item: (item.paper_id, item.ordinal, item.chunk_id))
    ]
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"kg-crag:chunk:{chunk_id}"))


def identity_point_id(collection_name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"kg-crag:collection:{collection_name}"))
