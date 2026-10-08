"""确定性回答检查和 Critic 发现项合并。"""

from __future__ import annotations

import re

from kg_crag.models import (
    AnswerCheckCode,
    AnswerEvaluation,
    AnswerFinding,
    ClaimType,
    EvidenceRequirement,
    FindingSeverity,
    GroundedAnswerCandidate,
    SufficiencyAssessment,
)

_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:\s*%)?")
_ABSTENTION = re.compile(
    r"(?:暂无|没有|缺少|不足).{0,12}(?:可信|可靠|充分|足够|安全).{0,12}(?:结论|证据|信息)"
    r"|无法.{0,16}(?:回答|判断|确定|得出)"
    r"|cannot.{0,16}(?:answer|determine|conclude)"
    r"|insufficient\s+(?:evidence|information)"
    r"|not\s+enough\s+(?:evidence|information)",
    re.IGNORECASE,
)


def _is_abstention(text: str) -> bool:
    """识别把系统证据边界误写成事实答案的常见拒答表述。"""

    return bool(_ABSTENTION.search(" ".join(text.split())))


def deterministic_check(
    candidate: GroundedAnswerCandidate,
    facets: list[EvidenceRequirement],
    sufficiency: SufficiencyAssessment,
    *,
    externally_covered_facet_ids: set[str] | None = None,
) -> AnswerEvaluation:
    """只依赖结构化字段完成可复现的硬性检查。"""

    findings: list[AnswerFinding] = []
    external = externally_covered_facet_ids or set()
    required = {item.facet_id for item in facets if item.required}
    abstaining_claim_ids = {item.claim_id for item in candidate.claims if _is_abstention(item.text)}
    claimed_facets = {
        facet_id
        for item in candidate.claims
        if item.claim_type is not ClaimType.UNCERTAIN and item.claim_id not in abstaining_claim_ids
        for facet_id in item.facet_ids
    }
    missing = required - claimed_facets - external
    if missing:
        findings.append(
            AnswerFinding(
                code=AnswerCheckCode.INCOMPLETE,
                reason="required facets are absent from the answer",
                facet_ids=sorted(missing),
            )
        )
    for claim in candidate.claims:
        if claim.claim_id in abstaining_claim_ids and claim.facet_ids:
            findings.append(
                AnswerFinding(
                    code=AnswerCheckCode.ABSTENTION,
                    reason="an abstention cannot satisfy a required answer facet",
                    claim_ids=[claim.claim_id],
                    facet_ids=claim.facet_ids,
                )
            )
    for claim in candidate.claims:
        if _NUMBER.search(claim.text) and not claim.citation_ids:
            findings.append(
                AnswerFinding(
                    code=AnswerCheckCode.UNCITED_NUMBER,
                    reason="numeric claim has no citation",
                    claim_ids=[claim.claim_id],
                    facet_ids=claim.facet_ids,
                )
            )
    blocking_facets = {item.facet_id for item in sufficiency.conflicts if item.blocking}
    for claim in candidate.claims:
        blocked = blocking_facets & set(claim.facet_ids)
        if blocked and claim.claim_type not in {ClaimType.CONFLICT, ClaimType.UNCERTAIN}:
            findings.append(
                AnswerFinding(
                    code=AnswerCheckCode.BLOCKING_CONFLICT,
                    reason="claim treats a blocking conflict as resolved",
                    claim_ids=[claim.claim_id],
                    facet_ids=sorted(blocked),
                )
            )
    complete = not any(item.code is AnswerCheckCode.INCOMPLETE for item in findings)
    conflicts_handled = not any(item.code is AnswerCheckCode.BLOCKING_CONFLICT for item in findings)
    citations_supported = not any(
        item.code in {AnswerCheckCode.UNKNOWN_CITATION, AnswerCheckCode.UNCITED_NUMBER}
        for item in findings
    )
    faithful = not abstaining_claim_ids
    acceptable = complete and faithful and conflicts_handled and citations_supported
    return AnswerEvaluation(
        complete=complete,
        faithful=faithful,
        citations_supported=citations_supported,
        conflicts_handled=conflicts_handled,
        attribution_correct=True,
        required_facets_covered=complete,
        findings=findings,
        acceptable=acceptable,
        critic_used=False,
    )


def merge_critic_findings(
    base: AnswerEvaluation,
    findings: list[AnswerFinding],
    *,
    critic_failed: bool = False,
) -> AnswerEvaluation:
    combined = [*base.findings, *findings]
    unsupported = any(item.code is AnswerCheckCode.UNSUPPORTED for item in combined)
    wrong_attribution = any(item.code is AnswerCheckCode.WRONG_ATTRIBUTION for item in combined)
    failed = critic_failed or any(item.code is AnswerCheckCode.CRITIC_FAILED for item in combined)
    faithful = base.faithful and not unsupported and not failed
    attribution = base.attribution_correct and not wrong_attribution and not failed
    acceptable = (
        base.complete
        and faithful
        and base.citations_supported
        and base.conflicts_handled
        and attribution
        and base.required_facets_covered
        and not any(item.severity is FindingSeverity.ERROR for item in combined)
    )
    return AnswerEvaluation(
        complete=base.complete,
        faithful=faithful,
        citations_supported=base.citations_supported,
        conflicts_handled=base.conflicts_handled,
        attribution_correct=attribution,
        required_facets_covered=base.required_facets_covered,
        findings=combined,
        acceptable=acceptable,
        critic_used=True,
    )
