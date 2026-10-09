"""可信搜索结果到外部 Evidence 与独立 facet 覆盖的转换。"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from kg_crag.answering.config import WebSearchConfig
from kg_crag.correction.coverage import (
    entity_anchor_match_sufficient,
    entity_anchor_terms,
    matching_entity_anchor_terms,
)
from kg_crag.correction.identity import stable_digest
from kg_crag.models import (
    Evidence,
    EvidenceLocation,
    EvidenceRequirement,
    EvidenceSourceType,
    ExternalFacetCoverage,
    SearchResult,
    WebSourceType,
)

_PRIMARY_ACADEMIC_SOURCE_TYPES = {
    WebSourceType.ARXIV.value,
    WebSourceType.PAPER_SITE.value,
    WebSourceType.ACADEMIC_DATABASE.value,
}


def _content_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def convert_web_results(
    results: Sequence[SearchResult],
    *,
    provider: str,
    converter_version: str,
    config: WebSearchConfig,
) -> list[Evidence]:
    """按条目和总上下文上限转换，绝不伪造 Paper/Chunk 身份。"""

    converted: list[Evidence] = []
    used = 0
    for search_rank, result in enumerate(results[: config.max_results], start=1):
        excerpt = result.excerpt[: config.max_excerpt_chars].strip()
        remaining = config.max_context_chars - used
        if remaining <= 0:
            break
        excerpt = excerpt[:remaining].rstrip()
        if not excerpt:
            continue
        url = str(result.final_url or result.url)
        digest = _content_hash(excerpt)
        identity = stable_digest(
            {
                "url": url,
                "excerpt_hash": digest,
                "provider_result_id": result.provider_result_id,
                "converter_version": converter_version,
            }
        )
        converted.append(
            Evidence(
                evidence_id="web-" + identity[:32],
                content=excerpt,
                source_type=EvidenceSourceType.WEB,
                source_id=result.provider_result_id,
                paper_id=None,
                location=EvidenceLocation(url=url),
                external=True,
                metadata={
                    "title": result.title,
                    "accessed_at": result.accessed_at.isoformat(),
                    "web_source_type": result.source_type.value,
                    "provider": provider,
                    "provider_result_id": result.provider_result_id,
                    "content_hash": digest,
                    "converter_version": converter_version,
                    "search_rank": search_rank,
                    "search_score": result.score,
                },
            )
        )
        used += len(excerpt)
    return converted


def build_external_coverage(
    facets: Sequence[EvidenceRequirement],
    evidence: Sequence[Evidence],
    *,
    rule_version: str,
) -> list[ExternalFacetCoverage]:
    coverage: list[ExternalFacetCoverage] = []
    for facet in sorted(facets, key=lambda item: item.facet_id):
        anchors = entity_anchor_terms(facet)
        facet_rows: list[tuple[Evidence, bool, float, list[str]]] = []
        for item in sorted(evidence, key=lambda value: value.evidence_id):
            if not item.external or item.source_type is not EvidenceSourceType.WEB:
                raise ValueError("external coverage only accepts Web Evidence")
            title = str(item.metadata.get("title", ""))
            folded = f"{title}\n{item.content}".casefold()
            matched_terms = [term for term in facet.condition.terms if term.casefold() in folded]
            matched_anchors = matching_entity_anchor_terms(facet, folded)
            direct_match = len(matched_terms) >= facet.condition.min_term_matches
            anchor_gate = not anchors or entity_anchor_match_sufficient(facet, folded)
            # 中英文问题常只共享论文简称。可信学术结果若标题命中目标实体且摘要
            # 非空，可作为待生成与 Critic 复核的外部证据，而不伪造翻译词表。
            title_folded = title.casefold()
            anchored_abstract = bool(
                entity_anchor_match_sufficient(facet, folded)
                and len(item.content.strip()) >= 80
                and (
                    any(anchor in title_folded for anchor in matched_anchors)
                    or len(matched_anchors) >= 2
                )
            )
            matched = (direct_match and anchor_gate) or anchored_abstract
            lexical_strength = len(matched_terms) / max(1, len(facet.condition.terms))
            anchor_strength = 0.55 + min(0.2, 0.1 * max(0, len(matched_anchors) - 1))
            strength = min(
                1.0,
                max(lexical_strength, anchor_strength if anchored_abstract else 0.0),
            )
            facet_rows.append(
                (
                    item,
                    matched,
                    strength,
                    (
                        [f"term:{value}" for value in matched_terms]
                        + [f"anchor:{value}" for value in matched_anchors]
                    )[:20],
                )
            )

        has_primary_match = any(
            matched
            and str(item.metadata.get("web_source_type", "")) in _PRIMARY_ACADEMIC_SOURCE_TYPES
            for item, matched, _strength, _basis in facet_rows
        )
        for item, raw_matched, strength, basis in facet_rows:
            source_type = str(item.metadata.get("web_source_type", ""))
            # 项目页和代码仓库可在找不到论文正文时兜底；只要同轮已有
            # 学术主来源覆盖该 facet，就不允许次要页面竞争答案绑定。
            matched = raw_matched and not (
                has_primary_match and source_type == WebSourceType.PROJECT_PAGE.value
            )
            payload = {
                "facet": facet.facet_id,
                "evidence": item.evidence_id,
                "rule": rule_version,
            }
            coverage.append(
                ExternalFacetCoverage(
                    coverage_id="external-coverage-" + stable_digest(payload)[:16],
                    facet_id=facet.facet_id,
                    evidence_id=item.evidence_id,
                    matched=matched,
                    support_strength=round(strength, 6) if matched else 0.0,
                    basis=basis,
                )
            )
    return coverage
