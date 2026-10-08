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
    """先覆盖全部必需 facet，再按检索排名补充少量同 facet 上下文。"""

    required = {item.facet_id for item in facets if item.required}
    coverage_by_evidence: dict[str, set[str]] = {}
    strength_by_evidence: dict[str, float] = {}
    quality_by_evidence: dict[str, float] = {}
    evidence_by_id = {item.evidence_id: item for item in evidence}
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

    def retrieval_priority(evidence_id: str) -> tuple[int, int, int, int, str]:
        item = evidence_by_id[evidence_id]
        ranks = item.ranks
        return (
            0 if item.metadata.get("paper_context_expansion") is True else 1,
            ranks.rerank or 2**30,
            ranks.fusion or 2**30,
            min(
                (value for value in (ranks.dense, ranks.sparse) if value is not None),
                default=2**30,
            ),
            evidence_id,
        )

    while remaining and available and len(selected) < max_selected:
        ranked = sorted(
            available,
            key=lambda evidence_id: (
                -len(coverage_by_evidence.get(evidence_id, set()) & remaining),
                retrieval_priority(evidence_id),
                -strength_by_evidence.get(evidence_id, 0.0),
                -quality_by_evidence.get(evidence_id, 0.0),
            ),
        )
        best = ranked[0]
        gained = coverage_by_evidence.get(best, set()) & remaining
        if not gained:
            break
        selected.append(best)
        remaining -= gained
        available.remove(best)

    # 单条实体命中只能证明论文相关，往往不足以承载定义、列表或机制答案。
    # 在既有上限内补入其余高排名匹配片段，由生成阶段在同一证据包中选择引用。
    supplemental = sorted(
        (
            evidence_id
            for evidence_id, covered_facets in coverage_by_evidence.items()
            if evidence_id not in selected and covered_facets & required
        ),
        key=retrieval_priority,
    )
    selected_coverage = {
        facet_id: sum(
            facet_id in coverage_by_evidence.get(evidence_id, set()) for evidence_id in selected
        )
        for facet_id in required
    }
    for evidence_id in supplemental:
        if len(selected) >= max_selected:
            break
        covered = coverage_by_evidence[evidence_id] & required
        if not any(selected_coverage[facet_id] < 3 for facet_id in covered):
            continue
        selected.append(evidence_id)
        for facet_id in covered:
            selected_coverage[facet_id] += 1
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
