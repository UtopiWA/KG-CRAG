"""逐 facet 门槛聚合和有界集合覆盖式 Evidence 选择。"""

from __future__ import annotations

from collections.abc import Sequence

from kg_crag.correction.coverage import detect_conflicts
from kg_crag.correction.identity import stable_digest
from kg_crag.models import (
    CoverageMatrix,
    Evidence,
    EvidenceRequirement,
    FacetAssessment,
    SufficiencyAssessment,
)


def select_evidence(
    facets: Sequence[EvidenceRequirement],
    evidence: Sequence[Evidence],
    matrix: CoverageMatrix,
    *,
    min_support: float,
    max_selected: int,
) -> list[str]:
    """以必需 facet 新增覆盖数为主键，确定性地执行贪心集合覆盖。"""

    required = {item.facet_id for item in facets if item.required}
    coverage_by_evidence: dict[str, set[str]] = {}
    strength_by_evidence: dict[str, float] = {}
    quality_by_evidence: dict[str, float] = {}
    for item in matrix.entries:
        if item.matched and (item.support_strength or 0.0) >= min_support:
            coverage_by_evidence.setdefault(item.evidence_id, set()).add(item.facet_id)
            strength_by_evidence[item.evidence_id] = max(
                strength_by_evidence.get(item.evidence_id, 0.0), item.support_strength or 0.0
            )
            quality_by_evidence[item.evidence_id] = max(
                quality_by_evidence.get(item.evidence_id, 0.0), item.source_quality
            )
    remaining = set(required)
    selected: list[str] = []
    available = {item.evidence_id for item in evidence}
    while remaining and available and len(selected) < max_selected:
        ranked = sorted(
            available,
            key=lambda evidence_id: (
                -len(coverage_by_evidence.get(evidence_id, set()) & remaining),
                -strength_by_evidence.get(evidence_id, 0.0),
                -quality_by_evidence.get(evidence_id, 0.0),
                evidence_id,
            ),
        )
        best = ranked[0]
        gained = coverage_by_evidence.get(best, set()) & remaining
        if not gained:
            break
        selected.append(best)
        remaining -= gained
        available.remove(best)
    return selected


def assess_sufficiency(
    facets: Sequence[EvidenceRequirement],
    evidence: Sequence[Evidence],
    matrix: CoverageMatrix,
    *,
    min_support: float,
    min_sources: int,
    max_selected: int,
) -> SufficiencyAssessment:
    facet_assessments: list[FacetAssessment] = []
    evidence_by_id = {item.evidence_id: item for item in evidence}
    conflicts = []
    covered: list[str] = []
    missing: list[str] = []
    optional: list[str] = []
    for facet in sorted(facets, key=lambda item: item.facet_id):
        matching = [
            item
            for item in matrix.entries
            if item.facet_id == facet.facet_id
            and item.matched
            and (item.support_strength or 0.0) >= min_support
        ]
        facet_conflicts = detect_conflicts(facet, evidence, matrix)
        conflicts.extend(facet_conflicts)
        blocking = any(item.blocking for item in facet_conflicts)
        sources = sorted({item.evidence_id for item in matching})
        independent_sources = {
            evidence_by_id[item].paper_id or evidence_by_id[item].source_id for item in sources
        }
        satisfied = len(independent_sources) >= min_sources and not blocking
        strength = min((item.support_strength or 0.0 for item in matching), default=0.0)
        if satisfied:
            covered.append(facet.facet_id)
        elif facet.required:
            missing.append(facet.facet_id)
        else:
            optional.append(facet.facet_id)
        reason = (
            "blocking_conflict"
            if blocking
            else "threshold_met"
            if satisfied
            else "insufficient_internal_sources"
        )
        facet_assessments.append(
            FacetAssessment(
                facet_id=facet.facet_id,
                required=facet.required,
                satisfied=satisfied,
                evidence_ids=sources,
                support_strength=strength,
                reason=reason,
            )
        )
    selected = select_evidence(
        facets, evidence, matrix, min_support=min_support, max_selected=max_selected
    )
    required_count = sum(1 for item in facets if item.required)
    satisfied_required = required_count - len(missing)
    confidence = satisfied_required / required_count if required_count else 1.0
    payload = {
        "matrix": matrix.matrix_id,
        "facets": [item.model_dump(mode="json") for item in facet_assessments],
        "conflicts": [item.model_dump(mode="json") for item in conflicts],
    }
    return SufficiencyAssessment(
        assessment_id="assessment-" + stable_digest(payload)[:16],
        sufficient=not missing and not any(item.blocking for item in conflicts),
        covered_facet_ids=covered,
        missing_required_facet_ids=missing,
        optional_facet_ids=optional,
        facets=facet_assessments,
        conflicts=conflicts,
        confidence=confidence,
        selected_evidence_ids=selected,
    )
