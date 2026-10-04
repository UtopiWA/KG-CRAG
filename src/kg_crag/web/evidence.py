"""可信搜索结果到外部 Evidence 与独立 facet 覆盖的转换。"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from kg_crag.answering.config import WebSearchConfig
from kg_crag.correction.identity import stable_digest
from kg_crag.models import (
    Evidence,
    EvidenceLocation,
    EvidenceRequirement,
    EvidenceSourceType,
    ExternalFacetCoverage,
    SearchResult,
)


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
    for result in results[: config.max_results]:
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
        for item in sorted(evidence, key=lambda value: value.evidence_id):
            if not item.external or item.source_type is not EvidenceSourceType.WEB:
                raise ValueError("external coverage only accepts Web Evidence")
            folded = item.content.casefold()
            matched_terms = [term for term in facet.condition.terms if term.casefold() in folded]
            matched = len(matched_terms) >= facet.condition.min_term_matches
            strength = min(1.0, len(matched_terms) / max(1, len(facet.condition.terms)))
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
                    basis=[f"term:{value}" for value in matched_terms[:20]],
                )
            )
    return coverage
