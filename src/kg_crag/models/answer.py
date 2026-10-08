"""可追溯回答、受控反思与 Web 兜底的严格公共契约。"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field, HttpUrl, field_validator, model_validator

from kg_crag.models.correction import (
    ActionResult,
    BudgetLedger,
    CorrectionRunResult,
    EvidenceConflict,
    EvidenceRequirement,
    FacetAssessment,
    StopReason,
)
from kg_crag.models.domain import Evidence, EvidenceSourceType, RetrievalStrategy, StrictModel
from kg_crag.models.foundation import ErrorDetail, TraceEvent
from kg_crag.models.rag import ClaimType


class WebSourceType(StrEnum):
    PAPER_SITE = "paper_site"
    ARXIV = "arxiv"
    ACADEMIC_DATABASE = "academic_database"
    PROJECT_PAGE = "project_page"


class FindingSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class AnswerCheckCode(StrEnum):
    INVALID_STRUCTURE = "invalid_structure"
    UNKNOWN_CITATION = "unknown_citation"
    SOURCE_MISMATCH = "source_mismatch"
    FACET_MISMATCH = "facet_mismatch"
    UNCITED_NUMBER = "uncited_number"
    BLOCKING_CONFLICT = "blocking_conflict"
    INCOMPLETE = "incomplete"
    ABSTENTION = "abstention"
    UNSUPPORTED = "unsupported"
    WRONG_ATTRIBUTION = "wrong_attribution"
    CRITIC_FAILED = "critic_failed"


class ReflectionAction(StrEnum):
    ACCEPT = "accept"
    REGENERATE = "regenerate"
    RERETRIEVE = "reretrieve"
    WEB_SEARCH = "web_search"
    CONSERVATIVE_STOP = "conservative_stop"


class AnswerStopReason(StrEnum):
    ACCEPTED = "accepted"
    CONSERVATIVE = "conservative"
    GENERATION_FAILED = "generation_failed"
    CRITIC_FAILED = "critic_failed"
    SEARCH_FAILED = "search_failed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    EXECUTION_FAILED = "execution_failed"


class AnswerWorkflowStage(StrEnum):
    PREPARE = "prepare"
    GENERATE = "generate"
    CHECK = "check"
    DECIDE = "decide"
    REMEDIATE = "remediate"
    FINALIZE = "finalize"


class GroundedEvaluationScenario(StrEnum):
    SUFFICIENT = "sufficient"
    UNCITED_NUMBER = "uncited_number"
    WRONG_ATTRIBUTION = "wrong_attribution"
    UNSUPPORTED = "unsupported"
    CONFLICT = "conflict"
    INTERNAL_MISSING = "internal_missing"
    WEB_DISABLED = "web_disabled"
    WEB_FAILED = "web_failed"
    WEB_UNTRUSTED = "web_untrusted"
    BUDGET_EXHAUSTED = "budget_exhausted"


class SearchResult(StrictModel):
    provider_result_id: str = Field(min_length=1, max_length=300)
    title: str = Field(min_length=1, max_length=500)
    url: HttpUrl
    final_url: HttpUrl | None = None
    accessed_at: datetime
    excerpt: str = Field(min_length=1, max_length=2000)
    source_type: WebSourceType
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    requires_auth: bool = False

    @field_validator("accessed_at")
    @classmethod
    def accessed_at_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("search result access time must include timezone")
        return value


class GroundedCitation(StrictModel):
    citation_id: str = Field(pattern=r"^E[1-9][0-9]*$")
    evidence_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_type: EvidenceSourceType
    external: bool
    paper_id: str | None = None
    section: str | None = None
    page: int | None = Field(default=None, ge=1)
    url: HttpUrl | None = None

    @model_validator(mode="after")
    def provenance_matches_source_type(self) -> GroundedCitation:
        if self.source_type is EvidenceSourceType.WEB:
            if not self.external or self.url is None or self.paper_id is not None:
                raise ValueError("web citations require external URL provenance without paper_id")
        elif self.external or not self.paper_id:
            raise ValueError("internal citations require paper_id and external=false")
        return self


class GroundedClaim(StrictModel):
    claim_id: str = Field(pattern=r"^claim-[a-f0-9]{16}$")
    text: str = Field(min_length=1, max_length=3000)
    claim_type: ClaimType
    citation_ids: list[str] = Field(default_factory=list, max_length=8)
    facet_ids: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("citation_ids", "facet_ids")
    @classmethod
    def identifiers_are_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("claim identifiers must be unique")
        return value

    @model_validator(mode="after")
    def supported_claims_have_provenance(self) -> GroundedClaim:
        if self.claim_type is not ClaimType.UNCERTAIN and not self.citation_ids:
            raise ValueError("non-uncertain grounded claims require citations")
        if self.claim_type is not ClaimType.UNCERTAIN and not self.facet_ids:
            raise ValueError("non-uncertain grounded claims require facet bindings")
        return self


class GroundedAnswerCandidate(StrictModel):
    answer: str = Field(min_length=1, max_length=12_000)
    claims: list[GroundedClaim] = Field(default_factory=list, max_length=100)
    citations: list[GroundedCitation] = Field(default_factory=list, max_length=20)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def references_are_unique_and_resolvable(self) -> GroundedAnswerCandidate:
        claim_ids = [item.claim_id for item in self.claims]
        citation_ids = [item.citation_id for item in self.citations]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("claim IDs must be unique")
        if len(citation_ids) != len(set(citation_ids)):
            raise ValueError("citation IDs must be unique")
        known = set(citation_ids)
        if any(set(item.citation_ids) - known for item in self.claims):
            raise ValueError("claim references an unknown citation")
        return self


class AnswerFinding(StrictModel):
    code: AnswerCheckCode
    severity: FindingSeverity = FindingSeverity.ERROR
    reason: str = Field(min_length=1, max_length=300)
    claim_ids: list[str] = Field(default_factory=list, max_length=20)
    citation_ids: list[str] = Field(default_factory=list, max_length=20)
    facet_ids: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("claim_ids", "citation_ids", "facet_ids")
    @classmethod
    def finding_ids_are_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("finding identifiers must be unique")
        return value


class AnswerEvaluation(StrictModel):
    complete: bool
    faithful: bool
    citations_supported: bool
    conflicts_handled: bool
    attribution_correct: bool
    required_facets_covered: bool
    findings: list[AnswerFinding] = Field(default_factory=list, max_length=100)
    acceptable: bool
    critic_used: bool = False

    @model_validator(mode="after")
    def acceptable_matches_dimensions(self) -> AnswerEvaluation:
        dimensions = (
            self.complete,
            self.faithful,
            self.citations_supported,
            self.conflicts_handled,
            self.attribution_correct,
            self.required_facets_covered,
        )
        expected = all(dimensions) and not any(
            item.severity is FindingSeverity.ERROR for item in self.findings
        )
        if self.acceptable != expected:
            raise ValueError("acceptable must agree with evaluation dimensions and findings")
        return self


class ExternalFacetCoverage(StrictModel):
    coverage_id: str = Field(pattern=r"^external-coverage-[a-f0-9]{16}$")
    facet_id: str = Field(pattern=r"^facet-[a-f0-9]{16}$")
    evidence_id: str = Field(min_length=1)
    matched: bool
    support_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    basis: list[str] = Field(default_factory=list, max_length=20)


class AnswerBudgetUsage(StrictModel):
    answer_calls: int = Field(default=0, ge=0, le=2)
    critic_calls: int = Field(default=0, ge=0, le=1)
    web_calls: int = Field(default=0, ge=0, le=1)
    reflection_rounds: int = Field(default=0, ge=0, le=1)
    input_tokens: int = Field(default=0, ge=0, le=12_000)
    output_tokens: int = Field(default=0, ge=0, le=12_000)
    web_results: int = Field(default=0, ge=0, le=5)
    context_chars: int = Field(default=0, ge=0, le=16_000)
    external_context_chars: int = Field(default=0, ge=0, le=8_000)
    latency_ms: int = Field(default=0, ge=0, le=3_600_000)

    @model_validator(mode="after")
    def combined_limits_are_respected(self) -> AnswerBudgetUsage:
        if self.answer_calls + self.critic_calls > 3:
            raise ValueError("answer-stage LLM calls exceed the hard limit")
        if self.input_tokens + self.output_tokens > 12_000:
            raise ValueError("answer-stage combined token usage exceeds the hard limit")
        return self


class AnswerBudgetLimit(AnswerBudgetUsage):
    answer_calls: int = Field(default=2, ge=0, le=2)
    critic_calls: int = Field(default=1, ge=0, le=1)
    web_calls: int = Field(default=1, ge=0, le=1)
    reflection_rounds: int = Field(default=1, ge=0, le=1)
    # 为两次最多 1600 Token 的生成预留完整输出空间，同时保持总上限不变。
    input_tokens: int = Field(default=8800, ge=0, le=12_000)
    output_tokens: int = Field(default=3200, ge=0, le=12_000)
    web_results: int = Field(default=5, ge=0, le=5)
    context_chars: int = Field(default=16_000, ge=1, le=16_000)
    external_context_chars: int = Field(default=8000, ge=0, le=8000)
    latency_ms: int = Field(default=30_000, ge=1, le=3_600_000)


class AnswerBudgetLedger(StrictModel):
    limit: AnswerBudgetLimit = Field(default_factory=AnswerBudgetLimit)
    used: AnswerBudgetUsage = Field(default_factory=AnswerBudgetUsage)
    reserved: AnswerBudgetUsage = Field(default_factory=AnswerBudgetUsage)

    @model_validator(mode="after")
    def usage_and_reservations_fit(self) -> AnswerBudgetLedger:
        for name in AnswerBudgetUsage.model_fields:
            if getattr(self.used, name) + getattr(self.reserved, name) > getattr(self.limit, name):
                raise ValueError(f"answer budget dimension exceeds limit: {name}")
        total_llm = (
            self.used.answer_calls
            + self.used.critic_calls
            + self.reserved.answer_calls
            + self.reserved.critic_calls
        )
        if total_llm > 3:
            raise ValueError("answer-stage LLM reservations exceed the hard limit")
        total_tokens = (
            self.used.input_tokens
            + self.used.output_tokens
            + self.reserved.input_tokens
            + self.reserved.output_tokens
        )
        if total_tokens > 12_000:
            raise ValueError("answer-stage token reservations exceed the hard limit")
        return self


class CombinedBudgetSummary(StrictModel):
    internal: BudgetLedger
    answer: AnswerBudgetLedger


class ReflectionDecision(StrictModel):
    decision_id: str = Field(pattern=r"^answer-decision-[a-f0-9]{16}$")
    action: ReflectionAction
    reason_code: str = Field(min_length=1, max_length=100)
    target_facet_ids: list[str] = Field(default_factory=list, max_length=20)
    allowed_claim_ids: list[str] = Field(default_factory=list, max_length=100)
    estimated_usage: AnswerBudgetUsage = Field(default_factory=AnswerBudgetUsage)


class AnswerRunIdentity(StrictModel):
    run_id: str = Field(pattern=r"^answer-run-[a-f0-9]{32}$")
    internal_run_id: str = Field(pattern=r"^run-[a-f0-9]{32}$")
    input_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    config_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    answer_prompt_version: str = Field(pattern=r"^[a-f0-9]{64}$")
    critic_prompt_version: str = Field(pattern=r"^[a-f0-9]{64}$")
    checker_version: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    search_provider: str = Field(min_length=1)
    search_provider_version: str = Field(min_length=1)


class GroundedAnswerResult(StrictModel):
    identity: AnswerRunIdentity
    trace_id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=4000)
    answer: str = Field(min_length=1, max_length=12_000)
    claims: list[GroundedClaim] = Field(default_factory=list, max_length=100)
    citations: list[GroundedCitation] = Field(default_factory=list, max_length=20)
    internal_evidence: list[Evidence] = Field(default_factory=list, max_length=20)
    external_evidence: list[Evidence] = Field(default_factory=list, max_length=5)
    external_coverage: list[ExternalFacetCoverage] = Field(default_factory=list, max_length=100)
    evaluation: AnswerEvaluation | None = None
    requirements: list[EvidenceRequirement] = Field(default_factory=list, max_length=20)
    facet_assessments: list[FacetAssessment] = Field(default_factory=list, max_length=20)
    conflicts: list[EvidenceConflict] = Field(default_factory=list, max_length=100)
    internal_actions: list[ActionResult] = Field(default_factory=list, max_length=2)
    missing_required_facet_ids: list[str] = Field(default_factory=list, max_length=20)
    retrieval_path: list[RetrievalStrategy] = Field(default_factory=list, max_length=10)
    internal_stop_reason: StopReason
    stop_reason: AnswerStopReason
    used_external: bool
    confidence: float = Field(ge=0.0, le=1.0)
    budget: CombinedBudgetSummary
    trace: list[TraceEvent] = Field(default_factory=list, max_length=200)
    errors: list[ErrorDetail] = Field(default_factory=list, max_length=20)
    started_at: datetime
    finished_at: datetime

    @model_validator(mode="after")
    def result_is_consistent(self) -> GroundedAnswerResult:
        if self.finished_at < self.started_at:
            raise ValueError("answer result time range is reversed")
        if self.used_external != bool(self.external_evidence):
            raise ValueError("used_external must agree with external evidence")
        if any(not item.external for item in self.external_evidence):
            raise ValueError("external evidence must be explicitly marked")
        if any(item.external for item in self.internal_evidence):
            raise ValueError("internal evidence cannot be marked external")
        return self


class AnswerWorkflowState(StrictModel):
    identity: AnswerRunIdentity
    correction: CorrectionRunResult
    stage: AnswerWorkflowStage = AnswerWorkflowStage.PREPARE
    candidate: GroundedAnswerCandidate | None = None
    evaluation: AnswerEvaluation | None = None
    decision: ReflectionDecision | None = None
    external_evidence: list[Evidence] = Field(default_factory=list, max_length=5)
    external_coverage: list[ExternalFacetCoverage] = Field(default_factory=list, max_length=100)
    budget: AnswerBudgetLedger = Field(default_factory=AnswerBudgetLedger)
    remediation_used: bool = False
    trace: list[TraceEvent] = Field(default_factory=list, max_length=200)
    errors: list[ErrorDetail] = Field(default_factory=list, max_length=20)
    result: GroundedAnswerResult | None = None


class AnswerCheckpoint(StrictModel):
    identity: AnswerRunIdentity
    stage: AnswerWorkflowStage
    sequence: int = Field(ge=0)
    state: AnswerWorkflowState
    created_at: datetime

    @model_validator(mode="after")
    def checkpoint_is_consistent(self) -> AnswerCheckpoint:
        if self.identity != self.state.identity or self.stage is not self.state.stage:
            raise ValueError("answer checkpoint identity or stage mismatch")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("answer checkpoint creation time must include timezone")
        return self


class GroundedEvaluationItem(StrictModel):
    question_id: str = Field(min_length=1)
    succeeded: bool
    expected_claim_ids: list[str] = Field(default_factory=list)
    supported_claim_ids: list[str] = Field(default_factory=list)
    expected_citation_ids: list[str] = Field(default_factory=list)
    predicted_citation_ids: list[str] = Field(default_factory=list)
    valid_citation_ids: list[str] = Field(default_factory=list)
    required_facets: int = Field(default=0, ge=0)
    covered_required_facets: int = Field(default=0, ge=0)
    unsupported_claims: int = Field(default=0, ge=0)
    web_eligible: bool = False
    web_triggered: bool = False
    web_used: bool = False
    model_calls: int = Field(default=0, ge=0, le=3)
    search_calls: int = Field(default=0, ge=0, le=1)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)
    stop_reason: AnswerStopReason
    error: ErrorDetail | None = None


class GroundedEvaluationQuestion(StrictModel):
    question_id: str = Field(pattern=r"^gaq-[0-9]{3}$")
    question: str = Field(min_length=1, max_length=4000)
    scenario: GroundedEvaluationScenario
    expected_stop_reason: AnswerStopReason
    web_eligible: bool = False
    expected_web_trigger: bool = False

    @model_validator(mode="after")
    def web_expectation_is_consistent(self) -> GroundedEvaluationQuestion:
        if self.expected_web_trigger and not self.web_eligible:
            raise ValueError("web trigger expectation requires an eligible question")
        return self


class GroundedEvaluationQuestionSet(StrictModel):
    dataset_version: str = Field(min_length=1)
    questions: list[GroundedEvaluationQuestion] = Field(min_length=12, max_length=20)

    @model_validator(mode="after")
    def questions_are_unique_and_cover_scenarios(self) -> GroundedEvaluationQuestionSet:
        identifiers = [item.question_id for item in self.questions]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("grounded evaluation question IDs must be unique")
        covered = {item.scenario for item in self.questions}
        missing = set(GroundedEvaluationScenario) - covered
        if missing:
            raise ValueError(f"grounded evaluation scenarios are missing: {sorted(missing)}")
        return self


class GroundedEvaluationMetrics(StrictModel):
    total: int = Field(ge=1)
    failures: int = Field(ge=0)
    task_score: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    faithfulness: float = Field(ge=0.0, le=1.0)
    citation_precision: float = Field(ge=0.0, le=1.0)
    citation_recall: float = Field(ge=0.0, le=1.0)
    unsupported_claim_rate: float = Field(ge=0.0, le=1.0)
    web_trigger_rate: float = Field(ge=0.0, le=1.0)
    web_usage_rate: float = Field(ge=0.0, le=1.0)
    model_calls: int = Field(ge=0)
    search_calls: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: int = Field(ge=0)
    stop_reasons: dict[AnswerStopReason, int]


class GroundedEvaluationReport(StrictModel):
    evaluation_id: str = Field(pattern=r"^answer-eval-[a-f0-9]{32}$")
    offline: bool
    question_set_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    items: list[GroundedEvaluationItem] = Field(min_length=1)
    metrics: GroundedEvaluationMetrics
    created_at: datetime
