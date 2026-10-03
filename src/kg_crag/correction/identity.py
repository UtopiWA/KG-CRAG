"""构造绑定全部可变版本的内容寻址运行身份。"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping, Sequence
from typing import cast

from pydantic import BaseModel

from kg_crag.models import Evidence
from kg_crag.models.correction import EvidenceRequirement, RunIdentity


def _canonical(value: object) -> object:
    if isinstance(value, BaseModel):
        return _canonical(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {
            str(key): _canonical(item)
            for key, item in sorted(mapping.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        sequence = cast(Sequence[object], value)
        return [_canonical(item) for item in sequence]
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value).strip()
    return value


def stable_digest(value: object) -> str:
    payload = json.dumps(
        _canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def build_run_identity(
    question: str,
    facets: Sequence[EvidenceRequirement],
    evidence: Sequence[Evidence],
    *,
    config_hash: str,
    prompt_version: str,
    model_revision: str,
    corpus_snapshot: str,
    dense_version: str,
    sparse_version: str,
    graph_version: str,
    rules_version: str,
    coverage_version: str,
    policy_version: str,
) -> RunIdentity:
    """排序集合字段，确保输入排列不影响同一运行的身份。"""

    normalized_question = unicodedata.normalize("NFC", question).strip()
    question_hash = stable_digest(normalized_question)
    facets_hash = stable_digest(sorted(facets, key=lambda item: item.facet_id))
    evidence_hash = stable_digest(sorted(evidence, key=lambda item: item.evidence_id))
    parts = {
        "question_hash": question_hash,
        "facets_hash": facets_hash,
        "evidence_hash": evidence_hash,
        "config_hash": config_hash,
        "prompt_version": prompt_version,
        "model_revision": model_revision,
        "corpus_snapshot": corpus_snapshot,
        "dense_version": dense_version,
        "sparse_version": sparse_version,
        "graph_version": graph_version,
        "rules_version": rules_version,
        "coverage_version": coverage_version,
        "policy_version": policy_version,
    }
    return RunIdentity(run_id=f"run-{stable_digest(parts)[:32]}", **parts)
