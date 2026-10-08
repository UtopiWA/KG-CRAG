"""可追溯 Evidence—facet 覆盖矩阵与保守冲突检测。"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Protocol

from kg_crag.correction.identity import stable_digest
from kg_crag.models import (
    ConditionKind,
    ConflictKind,
    CoverageMatrix,
    CoverageStatus,
    Evidence,
    EvidenceConflict,
    EvidenceRequirement,
    EvidenceSourceType,
    FacetCoverage,
    FacetKind,
)


class FacetMatcher(Protocol):
    def match(
        self, facet: EvidenceRequirement, evidence: Evidence
    ) -> tuple[bool, list[str], float]: ...


class SourceVerifier(Protocol):
    def verify(self, evidence: Evidence) -> tuple[bool, str | None]: ...


class CatalogSourceVerifier:
    """使用冻结 source_id、paper_id 和可选 content_hash 目录复核来源。"""

    def __init__(self, catalog: Mapping[str, tuple[str | None, str | None]]) -> None:
        self._catalog = catalog

    def verify(self, evidence: Evidence) -> tuple[bool, str | None]:
        if evidence.external or evidence.source_type is EvidenceSourceType.WEB:
            return False, "external_evidence"
        expected = self._catalog.get(evidence.source_id)
        if expected is None:
            return False, "source_missing"
        paper_id, content_hash = expected
        if paper_id is not None and paper_id != evidence.paper_id:
            return False, "paper_mismatch"
        actual_hash = evidence.metadata.get("content_hash")
        if content_hash is not None and actual_hash != content_hash:
            return False, "content_hash_mismatch"
        if evidence.source_type is EvidenceSourceType.GRAPH:
            if not evidence.metadata.get("path_id") or not evidence.metadata.get("graph_version"):
                return False, "graph_provenance_missing"
        return True, None


class PermissiveInternalVerifier:
    """离线 fixture 使用的最小复核器；仍排除 Web 与 external。"""

    def verify(self, evidence: Evidence) -> tuple[bool, str | None]:
        if evidence.external or evidence.source_type is EvidenceSourceType.WEB:
            return False, "external_evidence"
        if evidence.source_type is EvidenceSourceType.GRAPH and (
            not evidence.metadata.get("path_id") or not evidence.metadata.get("graph_version")
        ):
            return False, "graph_provenance_missing"
        return True, None


class RuleFacetMatcher:
    """只读取受控字段，按词项、ID、类型和图路径元数据匹配。"""

    def match(
        self, facet: EvidenceRequirement, evidence: Evidence
    ) -> tuple[bool, list[str], float]:
        condition = facet.condition
        basis: list[str] = []
        folded = evidence.content.casefold()
        matched_terms = [term for term in condition.terms if term.casefold() in folded]
        if len(matched_terms) >= condition.min_term_matches:
            basis.extend(f"term:{term}" for term in matched_terms)
        entity_values = {
            str(evidence.metadata.get("entity_id", "")),
            *str(evidence.metadata.get("entity_ids", "")).split(","),
        }
        entity_matches = sorted(set(condition.entity_ids) & entity_values)
        basis.extend(f"entity:{item}" for item in entity_matches)
        if evidence.paper_id and evidence.paper_id in condition.paper_ids:
            basis.append(f"paper:{evidence.paper_id}")
        if condition.evidence_types and evidence.source_type in condition.evidence_types:
            basis.append(f"type:{evidence.source_type.value}")
        if condition.max_hops is not None:
            hops = evidence.metadata.get("hop_count")
            if isinstance(hops, int) and 1 <= hops <= condition.max_hops:
                basis.append(f"hops:{hops}")
        selector_match = bool(
            len(matched_terms) >= condition.min_term_matches
            or entity_matches
            or (evidence.paper_id in condition.paper_ids)
        )
        if not (condition.terms or condition.entity_ids or condition.paper_ids):
            selector_match = evidence.source_type in condition.evidence_types
        expected_type = evidence.source_type in facet.expected_evidence_types
        matched = selector_match and expected_type
        denominator = max(
            1, len(condition.terms) + len(condition.entity_ids) + len(condition.paper_ids)
        )
        lexical = min(1.0, (len(matched_terms) + len(entity_matches)) / denominator)
        if matched and lexical == 0.0:
            lexical = 0.5
        return matched, basis[:20], lexical


_QUESTION_WORDS = {
    "what",
    "which",
    "who",
    "where",
    "when",
    "why",
    "how",
    "does",
    "do",
    "did",
    "is",
    "are",
    "was",
    "were",
}


def entity_anchor_terms(facet: EvidenceRequirement) -> tuple[str, ...]:
    """提取问题中的显式拉丁实体名，用于把同论文的答案片段纳入候选。"""

    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_.+-]{2,}", facet.description)
    return tuple(
        dict.fromkeys(
            token.casefold()
            for token in tokens
            if token[0].isupper() and token.casefold() not in _QUESTION_WORDS
        )
    )


def _anchored_papers(
    facet: EvidenceRequirement,
    evidence: Sequence[Evidence],
) -> set[str]:
    """先通过显式实体定位论文，再允许该论文中省略实体名的摘要或续接段落参与覆盖。"""

    anchors = entity_anchor_terms(facet)
    if not anchors:
        return set()
    return {
        item.paper_id
        for item in evidence
        if item.paper_id and any(anchor in item.content.casefold() for anchor in anchors)
    }


def _source_quality(evidence: Evidence) -> float:
    available = [
        value for value in evidence.scores.model_dump().values() if isinstance(value, int | float)
    ]
    if not available:
        return 0.5 if evidence.source_type is EvidenceSourceType.GRAPH else 0.4
    normalized = [max(0.0, min(1.0, float(value))) for value in available]
    return round(max(normalized), 6)


def build_coverage_matrix(
    facets: Sequence[EvidenceRequirement],
    evidence: Sequence[Evidence],
    *,
    rule_version: str,
    matcher: FacetMatcher | None = None,
    verifier: SourceVerifier | None = None,
) -> CoverageMatrix:
    selected_matcher = matcher or RuleFacetMatcher()
    selected_verifier = verifier or PermissiveInternalVerifier()
    sorted_facets = sorted(facets, key=lambda item: item.facet_id)
    sorted_evidence = sorted(evidence, key=lambda item: item.evidence_id)
    anchored_papers = {
        facet.facet_id: _anchored_papers(facet, sorted_evidence) for facet in sorted_facets
    }
    entries: list[FacetCoverage] = []
    for facet in sorted_facets:
        for item in sorted_evidence:
            valid, invalid_reason = selected_verifier.verify(item)
            basis: list[str]
            support: float | None
            if not valid:
                status = (
                    CoverageStatus.EXTERNAL_EXCLUDED
                    if invalid_reason == "external_evidence"
                    else CoverageStatus.INVALID_SOURCE
                )
                matched, basis, support = False, [], None
            else:
                matched, basis, lexical = selected_matcher.match(facet, item)
                # 摘要或续接片段可能不重复论文简称。实体已在同一篇论文的其他候选中
                # 明确锚定时，保留该片段给后续检索排名和回答生成共同判断。
                if (
                    not matched
                    and item.source_type is EvidenceSourceType.CHUNK
                    and item.paper_id in anchored_papers[facet.facet_id]
                ):
                    matched = True
                    basis = [f"paper_scope:{item.paper_id}"]
                    lexical = 0.5
                status = CoverageStatus.MATCHED if matched else CoverageStatus.NOT_MATCHED
                quality = _source_quality(item)
                support = round(0.7 * lexical + 0.3 * quality, 6) if matched else None
            coverage_id = (
                "coverage-"
                + stable_digest(
                    {"facet": facet.facet_id, "evidence": item.evidence_id, "rule": rule_version}
                )[:16]
            )
            entries.append(
                FacetCoverage(
                    coverage_id=coverage_id,
                    facet_id=facet.facet_id,
                    evidence_id=item.evidence_id,
                    status=status,
                    matched=matched,
                    support_strength=support,
                    source_quality=_source_quality(item) if valid else 0.0,
                    basis=basis,
                    reason=invalid_reason,
                )
            )
    preliminary_payload = {
        "facets": [item.model_dump(mode="json") for item in sorted_facets],
        "evidence": [item.model_dump(mode="json") for item in sorted_evidence],
        "rule": rule_version,
        "entries": [item.model_dump(mode="json") for item in entries],
    }
    preliminary = CoverageMatrix(
        matrix_id="matrix-" + stable_digest(preliminary_payload)[:16],
        rule_version=rule_version,
        facet_ids=[item.facet_id for item in sorted_facets],
        evidence_ids=[item.evidence_id for item in sorted_evidence],
        entries=entries,
    )
    conflict_pairs = {
        (facet.facet_id, evidence_id)
        for facet in sorted_facets
        for conflict in detect_conflicts(facet, sorted_evidence, preliminary)
        for evidence_id in conflict.evidence_ids
    }
    marked_entries = [
        item.model_copy(update={"conflict": (item.facet_id, item.evidence_id) in conflict_pairs})
        for item in entries
    ]
    identity_payload = {
        **preliminary_payload,
        "entries": [item.model_dump(mode="json") for item in marked_entries],
    }
    return CoverageMatrix(
        matrix_id="matrix-" + stable_digest(identity_payload)[:16],
        rule_version=rule_version,
        facet_ids=preliminary.facet_ids,
        evidence_ids=preliminary.evidence_ids,
        entries=marked_entries,
    )


_NUMBER = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*([%a-zA-Z/]+)?")


def _normalized_conflict(left: Evidence, right: Evidence) -> ConflictKind | None:
    """只比较显式绑定到同一命题的规范化值。"""

    left_key = left.metadata.get("normalized_value_key")
    right_key = right.metadata.get("normalized_value_key")
    if (
        not isinstance(left_key, str)
        or not left_key.strip()
        or not isinstance(right_key, str)
        or left_key.strip().casefold() != right_key.strip().casefold()
    ):
        return None
    left_value = left.metadata.get("normalized_value")
    right_value = right.metadata.get("normalized_value")
    if left_value is None or right_value is None:
        return None
    if isinstance(left_value, bool) and isinstance(right_value, bool):
        return ConflictKind.NEGATION if left_value != right_value else None
    if (
        isinstance(left_value, int | float)
        and not isinstance(left_value, bool)
        and isinstance(right_value, int | float)
        and not isinstance(right_value, bool)
    ):
        return ConflictKind.NUMERIC if float(left_value) != float(right_value) else None
    return (
        ConflictKind.DISCRETE
        if str(left_value).strip().casefold() != str(right_value).strip().casefold()
        else None
    )


_METRIC_UNITS = {
    "%",
    "percent",
    "percentage",
    "x",
    "ms",
    "s",
    "sec",
    "seconds",
    "point",
    "points",
    "pt",
}


def _metric_measurement(content: str, facet: EvidenceRequirement) -> tuple[float, str] | None:
    """只在 facet 词项邻域提取指标值，并优先选择公认单位。"""

    folded = content.casefold()
    windows: list[str] = []
    for term in sorted(facet.condition.terms, key=len, reverse=True):
        normalized = term.casefold().strip()
        if len(normalized) < 2:
            continue
        start = folded.find(normalized)
        if start >= 0:
            windows.append(content[max(0, start - 120) : start + len(normalized) + 160])
    if not windows:
        return None
    values = [
        (float(number), unit.casefold()) for number, unit in _NUMBER.findall(" ".join(windows))
    ]
    if not values:
        return None
    with_units = [item for item in values if item[1] in _METRIC_UNITS]
    return (with_units or values)[-1]


def detect_conflicts(
    facet: EvidenceRequirement, evidence: Sequence[Evidence], matrix: CoverageMatrix
) -> list[EvidenceConflict]:
    matched_ids = {
        item.evidence_id
        for item in matrix.entries
        if item.facet_id == facet.facet_id and item.matched
    }
    items = [item for item in evidence if item.evidence_id in matched_ids]
    conflicts: list[EvidenceConflict] = []
    for index, left in enumerate(items):
        for right in items[index + 1 :]:
            kind = _normalized_conflict(left, right)
            blocking = True
            review = False
            metric_facet = (
                facet.kind is FacetKind.METRIC or facet.condition.kind is ConditionKind.NUMERIC
            )
            if kind is None and metric_facet:
                left_measurement = _metric_measurement(left.content, facet)
                right_measurement = _metric_measurement(right.content, facet)
                if left_measurement is None or right_measurement is None:
                    continue
                left_number, left_unit = left_measurement
                right_number, right_unit = right_measurement
                if left_unit and right_unit and left_unit != right_unit:
                    kind, blocking, review = ConflictKind.UNKNOWN_UNIT, False, True
                elif left_number != right_number:
                    kind = ConflictKind.NUMERIC
            if kind is None:
                continue
            ids = sorted([left.evidence_id, right.evidence_id])
            conflicts.append(
                EvidenceConflict(
                    conflict_id="conflict-"
                    + stable_digest({"facet": facet.facet_id, "ids": ids, "kind": kind})[:16],
                    facet_id=facet.facet_id,
                    kind=kind,
                    evidence_ids=ids,
                    blocking=blocking,
                    review_required=review,
                    # Evidence ID 已由结构化字段完整保留，原因文本不重复拼接长 ID。
                    reason=f"{kind.value} conflict between two Evidence records",
                )
            )
    return sorted(conflicts, key=lambda item: item.conflict_id)
