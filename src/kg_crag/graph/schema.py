"""图名称、实体、事实和版本身份的确定性函数。"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable

from kg_crag.models import GraphBundle, GraphNodeType, GraphProvenance, GraphRelationType


def normalize_name(value: str) -> str:
    """使用 NFC、case-fold 和稳定空白/标点规则生成保守规范键。"""

    normalized = unicodedata.normalize("NFC", value).casefold().strip()
    normalized = re.sub(r"[\s\-_\u2013\u2014]+", " ", normalized)
    normalized = re.sub(r"[^\w .:/+]", "", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def _digest(parts: Iterable[str]) -> str:
    serialized = json.dumps(list(parts), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def entity_id(
    node_type: GraphNodeType,
    name: str,
    *,
    external_id: str | None = None,
    paper_scope: str | None = None,
) -> str:
    identity = normalize_name(external_id) if external_id else normalize_name(name)
    scope = "" if external_id else (paper_scope or "")
    return f"{node_type.value.casefold()}:{_digest((node_type.value, identity, scope))}"


def fact_id(
    schema_version: str,
    relation: GraphRelationType,
    source_entity_id: str,
    target_entity_id: str,
    provenance: GraphProvenance,
) -> str:
    return "fact:" + _digest(
        (
            schema_version,
            relation.value,
            source_entity_id,
            target_entity_id,
            provenance.source_kind,
            provenance.source_id,
            provenance.source_hash,
        )
    )


def corpus_snapshot(values: Iterable[tuple[str, str, str]]) -> str:
    """绑定 Paper、处理版本与内容哈希，不受发现顺序影响。"""

    return _digest("\0".join(item) for item in sorted(values))


def bundle_hash(bundle: GraphBundle) -> str:
    payload = bundle.model_dump(mode="json", exclude={"unresolved_references"})
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
