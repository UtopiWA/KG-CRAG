"""实体保守合并、歧义复核与人工决定重放。"""

from __future__ import annotations

import difflib
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from kg_crag.graph.artifacts import atomic_write_json
from kg_crag.graph.schema import normalize_name
from kg_crag.models import EntityReviewDecision, EntityReviewItem, GraphEntity


def _review_id(left: str, right: str, reason: str) -> str:
    pair = sorted((left, right))
    return "review:" + hashlib.sha256("\0".join((*pair, reason)).encode()).hexdigest()


def normalize_entities(
    entities: list[GraphEntity],
    *,
    auto_merge_min_confidence: float,
    review_threshold: float,
    confidences: dict[str, float] | None = None,
    source_ids: dict[str, list[str]] | None = None,
) -> tuple[dict[str, str], list[EntityReviewItem]]:
    """仅合并确定同一项；近似名称只进入复核队列。"""

    mapping = {item.entity_id: item.entity_id for item in entities}
    confidences = confidences or {}
    source_ids = source_ids or {}
    reviews: list[EntityReviewItem] = []
    external: dict[tuple[object, str], str] = {}
    keys: dict[tuple[object, str], list[GraphEntity]] = defaultdict(list)
    for item in entities:
        keys[(item.node_type, normalize_name(item.name))].append(item)
        if item.external_id:
            key = (item.node_type, normalize_name(item.external_id))
            existing = external.get(key)
            if existing:
                mapping[item.entity_id] = existing
            else:
                external[key] = item.entity_id
    for same_key in keys.values():
        without_external = [item for item in same_key if item.external_id is None]
        can_merge = all(
            confidences.get(item.entity_id, 1.0) >= auto_merge_min_confidence
            for item in without_external
        )
        if len(without_external) > 1 and can_merge:
            canonical = min(item.entity_id for item in without_external)
            for item in without_external:
                mapping[item.entity_id] = canonical

    ordered = sorted(entities, key=lambda item: item.entity_id)
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if mapping[left.entity_id] == mapping[right.entity_id]:
                continue
            same_name = normalize_name(left.name) == normalize_name(right.name)
            similarity = difflib.SequenceMatcher(
                None, normalize_name(left.name), normalize_name(right.name)
            ).ratio()
            left_words = normalize_name(left.name).split()
            right_words = normalize_name(right.name).split()
            left_initials = "".join(word[0] for word in left_words if word)
            right_initials = "".join(word[0] for word in right_words if word)
            abbreviation = (len(left_words) > 1 and left_initials == "".join(right_words)) or (
                len(right_words) > 1 and right_initials == "".join(left_words)
            )
            low_confidence = (
                min(
                    confidences.get(left.entity_id, 1.0),
                    confidences.get(right.entity_id, 1.0),
                )
                < auto_merge_min_confidence
            )
            reason: str | None = None
            if same_name and left.node_type != right.node_type:
                reason = "cross_type_name"
            elif (
                (same_name or similarity >= review_threshold)
                and left.external_id
                and right.external_id
                and left.external_id != right.external_id
            ):
                reason = "external_id_conflict"
            elif abbreviation:
                reason = "abbreviation"
            elif low_confidence and (same_name or similarity >= review_threshold):
                reason = "low_confidence"
            elif similarity >= review_threshold:
                reason = "similar_name"
            if reason:
                sources = sorted(
                    set(
                        (source_ids.get(left.entity_id) or [left.entity_id])
                        + (source_ids.get(right.entity_id) or [right.entity_id])
                    )
                )[:20]
                reviews.append(
                    EntityReviewItem(
                        review_id=_review_id(left.entity_id, right.entity_id, reason),
                        left_entity_id=left.entity_id,
                        right_entity_id=right.entity_id,
                        left_name=left.name,
                        right_name=right.name,
                        node_type=left.node_type if left.node_type == right.node_type else None,
                        score=similarity,
                        reason=reason,
                        source_ids=sources,
                    )
                )
    return mapping, sorted(reviews, key=lambda item: item.review_id)


def apply_review_decisions(
    mapping: dict[str, str],
    entities: list[GraphEntity],
    decisions: list[EntityReviewDecision],
    *,
    version: str,
    available_source_ids: set[str] | None = None,
    known_review_ids: set[str] | None = None,
) -> dict[str, str]:
    known = {item.entity_id: item for item in entities}
    updated = dict(mapping)
    for decision in decisions:
        if decision.normalization_version != version:
            raise ValueError("review decision normalization version is stale")
        if known_review_ids is not None and decision.review_id not in known_review_ids:
            raise ValueError("review decision references an unknown review candidate")
        if available_source_ids is not None and not set(decision.source_ids).issubset(
            available_source_ids
        ):
            raise ValueError("review decision references a stale source")
        left = known.get(decision.left_entity_id)
        right = known.get(decision.right_entity_id)
        if left is None or right is None:
            raise ValueError("review decision references an unknown entity")
        if decision.decision == "merge":
            if left.node_type != right.node_type:
                raise ValueError("review decision cannot merge different entity types")
            if left.external_id and right.external_id and left.external_id != right.external_id:
                raise ValueError("review decision conflicts with stable external IDs")
            canonical = min(updated[left.entity_id], updated[right.entity_id])
            updated[left.entity_id] = canonical
            updated[right.entity_id] = canonical
    return updated


def load_review_decisions(path: Path) -> list[EntityReviewDecision]:
    """加载版本化人工决定，并拒绝重复 review ID。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("decisions"), list):
        raise ValueError("review decision file must contain a decisions list")
    decisions = [EntityReviewDecision.model_validate(item) for item in payload["decisions"]]
    review_ids = [item.review_id for item in decisions]
    if len(review_ids) != len(set(review_ids)):
        raise ValueError("review decision file contains duplicate review IDs")
    return decisions


def write_review_report(
    path: Path, items: list[EntityReviewItem], *, normalization_version: str
) -> None:
    """只发布有限标量与来源标识，不保存完整 Chunk 正文。"""

    atomic_write_json(
        path,
        {
            "schema_version": "v1",
            "normalization_version": normalization_version,
            "item_count": len(items),
            "items": [
                item.model_dump(mode="json")
                for item in sorted(items, key=lambda value: value.review_id)
            ],
        },
    )
