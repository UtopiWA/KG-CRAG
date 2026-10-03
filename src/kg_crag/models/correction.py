"""证据充分性诊断与有界纠错工作流的公共契约。"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from kg_crag.models.domain import Evidence, EvidenceSourceType, StrictModel
from kg_crag.models.foundation import ErrorDetail, TraceEvent


class FacetKind(StrEnum):
    CONTENT = "content"
    ENTITY = "entity"
    RELATIONSHIP = "relationship"
    COMPARISON = "comparison"
    METRIC = "metric"
    MULTI_HOP = "multi_hop"


class RequirementSource(StrEnum):
    RULE = "rule"
    LLM = "llm"


class ConditionKind(StrEnum):
    TERMS = "terms"
    ENTITY = "entity"
    PAPER = "paper"
    EVIDENCE_TYPE = "evidence_type"
    GRAPH_PATH = "graph_path"
    NUMERIC = "numeric"


class SatisfactionCondition(StrictModel):
    """仅包含可由 Evidence 字段验证的满足条件。"""

    kind: ConditionKind
    terms: list[str] = Field(default_factory=list, max_length=20)
    entity_ids: list[str] = Field(default_factory=list, max_length=20)
    paper_ids: list[str] = Field(default_factory=list, max_length=20)
    evidence_types: list[EvidenceSourceType] = Field(default_factory=list, max_length=3)
    min_term_matches: int = Field(default=1, ge=1, le=20)
    max_hops: int | None = Field(default=None, ge=1, le=3)
    unit: str | None = Field(default=None, max_length=32)

    @field_validator("terms", "entity_ids", "paper_ids")
    @classmethod
    def values_are_normalized_and_unique(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value]
        if any(not item for item in normalized) or len(normalized) != len(set(normalized)):
            raise ValueError("condition values must be non-empty and unique")
        return normalized

    @model_validator(mode="after")
    def condition_has_verifiable_content(self) -> SatisfactionCondition:
        populated = bool(self.terms or self.entity_ids or self.paper_ids or self.evidence_types)
        if not populated:
            raise ValueError("condition requires at least one verifiable selector")
        if self.min_term_matches > max(1, len(self.terms)):
            raise ValueError("min_term_matches exceeds the available terms")
        if self.kind is ConditionKind.GRAPH_PATH and self.max_hops is None:
            raise ValueError("graph path conditions require max_hops")
        return self


class EvidenceRequirement(StrictModel):
    facet_id: str = Field(pattern=r"^facet-[a-f0-9]{16}$")
    question_id: str = Field(min_length=1, max_length=128)
    kind: FacetKind
    description: str = Field(min_length=1, max_length=500)
    required: bool = True
    expected_evidence_types: list[EvidenceSourceType] = Field(min_length=1, max_length=3)
    condition: SatisfactionCondition
    source: RequirementSource = RequirementSource.RULE
    target_entity: str | None = Field(default=None, max_length=200)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("expected_evidence_types")
    @classmethod
    def evidence_types_are_unique(cls, value: list[EvidenceSourceType]) -> list[EvidenceSourceType]:
        if len(value) != len(set(value)) or EvidenceSourceType.WEB in value:
            raise ValueError("expected evidence types must be unique internal sources")
        return value


class CoverageStatus(StrEnum):
    MATCHED = "matched"
    NOT_MATCHED = "not_matched"
    INVALID_SOURCE = "invalid_source"
    EXTERNAL_EXCLUDED = "external_excluded"
    REVIEW_REQUIRED = "review_required"


class FacetCoverage(StrictModel):
    coverage_id: str = Field(pattern=r"^coverage-[a-f0-9]{16}$")
    facet_id: str = Field(pattern=r"^facet-[a-f0-9]{16}$")
    evidence_id: str = Field(min_length=1)
    status: CoverageStatus
    matched: bool
    support_strength: float | None = Field(default=None, ge=0.0, le=1.0)
    source_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    basis: list[str] = Field(default_factory=list, max_length=20)
    conflict: bool = False
    reason: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def status_matches_flags(self) -> FacetCoverage:
        if self.matched != (self.status is CoverageStatus.MATCHED):
            raise ValueError("coverage matched flag must agree with status")
        if self.matched and self.support_strength is None:
            raise ValueError("matched coverage requires support strength")
        return self


class CoverageMatrix(StrictModel):
    matrix_id: str = Field(pattern=r"^matrix-[a-f0-9]{16}$")
    rule_version: str = Field(min_length=1)
    facet_ids: list[str]
    evidence_ids: list[str]
    entries: list[FacetCoverage]

    @model_validator(mode="after")
    def identities_are_unique_and_complete(self) -> CoverageMatrix:
        if len(self.facet_ids) != len(set(self.facet_ids)):
            raise ValueError("coverage matrix facet IDs must be unique")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("coverage matrix evidence IDs must be unique")
        pairs = [(item.facet_id, item.evidence_id) for item in self.entries]
        if len(pairs) != len(set(pairs)):
            raise ValueError("coverage matrix entries must be unique")
        if any(item.facet_id not in self.facet_ids for item in self.entries):
            raise ValueError("coverage entry references an unknown facet")
        if any(item.evidence_id not in self.evidence_ids for item in self.entries):
            raise ValueError("coverage entry references unknown evidence")
        return self


class ConflictKind(StrEnum):
    DISCRETE = "discrete"
    NUMERIC = "numeric"
    NEGATION = "negation"
    UNKNOWN_UNIT = "unknown_unit"


class EvidenceConflict(StrictModel):
    conflict_id: str = Field(pattern=r"^conflict-[a-f0-9]{16}$")
    facet_id: str = Field(pattern=r"^facet-[a-f0-9]{16}$")
    kind: ConflictKind
    evidence_ids: list[str] = Field(min_length=2, max_length=20)
    blocking: bool = True
    review_required: bool = False
    reason: str = Field(min_length=1, max_length=300)

    @field_validator("evidence_ids")
    @classmethod
    def conflict_sources_are_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("conflict evidence IDs must be unique")
        return value


class FacetAssessment(StrictModel):
    facet_id: str = Field(pattern=r"^facet-[a-f0-9]{16}$")
    required: bool
    satisfied: bool
    evidence_ids: list[str] = Field(default_factory=list)
    support_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=300)


class SufficiencyAssessment(StrictModel):
    assessment_id: str = Field(pattern=r"^assessment-[a-f0-9]{16}$")
    sufficient: bool
    covered_facet_ids: list[str] = Field(default_factory=list)
    missing_required_facet_ids: list[str] = Field(default_factory=list)
    optional_facet_ids: list[str] = Field(default_factory=list)
    facets: list[FacetAssessment]
    conflicts: list[EvidenceConflict] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    selected_evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def sufficient_state_is_consistent(self) -> SufficiencyAssessment:
        facet_ids = [item.facet_id for item in self.facets]
        if len(facet_ids) != len(set(facet_ids)):
            raise ValueError("facet assessments must be unique")
        blocked = any(item.blocking for item in self.conflicts)
        expected = not self.missing_required_facet_ids and not blocked
        if self.sufficient != expected:
            raise ValueError("sufficient flag conflicts with missing facets or blocking conflicts")
        return self


class CorrectionAction(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    GRAPH = "graph"
    REWRITE = "rewrite"
    DECOMPOSE = "decompose"
    ADJUST = "adjust"


class GapType(StrEnum):
    TERMINOLOGY_MISMATCH = "terminology_mismatch"
    ENTITY_ALIAS = "entity_alias"
    COMPARISON_SIDE = "comparison_side"
    MULTI_HOP_GAP = "multi_hop_gap"
    METRIC_MISSING = "metric_missing"
    CONFLICT = "conflict"
    INTERNAL_MISSING = "internal_missing"


class BudgetUsage(StrictModel):
    retrieval_rounds: int = Field(default=0, ge=0, le=2)
    subquestions: int = Field(default=0, ge=0, le=3)
    llm_calls: int = Field(default=0, ge=0, le=4)
    input_tokens: int = Field(default=0, ge=0, le=20_000)
    output_tokens: int = Field(default=0, ge=0, le=20_000)
    candidates: int = Field(default=0, ge=0, le=10_000)
    context_chars: int = Field(default=0, ge=0, le=1_000_000)
    latency_ms: int = Field(default=0, ge=0, le=3_600_000)

    @model_validator(mode="after")
    def total_tokens_are_hard_bounded(self) -> BudgetUsage:
        if self.input_tokens + self.output_tokens > 20_000:
            raise ValueError("combined token usage exceeds the hard limit")
        return self


class BudgetLimit(BudgetUsage):
    retrieval_rounds: int = Field(default=2, ge=1, le=2)
    subquestions: int = Field(default=3, ge=0, le=3)
    llm_calls: int = Field(default=4, ge=0, le=4)
    input_tokens: int = Field(default=16_000, ge=0, le=20_000)
    output_tokens: int = Field(default=4_000, ge=0, le=20_000)
    candidates: int = Field(default=100, ge=1, le=10_000)
    context_chars: int = Field(default=20_000, ge=1, le=1_000_000)
    latency_ms: int = Field(default=30_000, ge=1, le=3_600_000)


class BudgetLedger(StrictModel):
    limit: BudgetLimit = Field(default_factory=BudgetLimit)
    used: BudgetUsage = Field(default_factory=BudgetUsage)
    reserved: BudgetUsage = Field(default_factory=BudgetUsage)

    @model_validator(mode="after")
    def usage_and_reservations_fit_limits(self) -> BudgetLedger:
        for name in BudgetUsage.model_fields:
            if getattr(self.used, name) + getattr(self.reserved, name) > getattr(self.limit, name):
                raise ValueError(f"budget dimension exceeds limit: {name}")
        return self


class ActionEstimate(BudgetUsage):
    expected_required_facet_gain: float = Field(default=0.0, ge=0.0, le=100.0)


class ActionRequest(StrictModel):
    action_id: str = Field(pattern=r"^action-[a-f0-9]{16}$")
    action: CorrectionAction
    target_facet_ids: list[str] = Field(min_length=1, max_length=20)
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=100)
    max_candidates: int = Field(default=20, ge=1, le=100)
    max_hops: int = Field(default=1, ge=1, le=3)
    subqueries: list[str] = Field(default_factory=list, max_length=3)
    filters: dict[str, str | int | bool] = Field(default_factory=dict, max_length=10)

    @field_validator("target_facet_ids", "subqueries")
    @classmethod
    def request_lists_are_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("action request values must be unique")
        return value

    @model_validator(mode="after")
    def request_parameters_are_consistent(self) -> ActionRequest:
        if self.top_k > self.max_candidates:
            raise ValueError("action top_k must not exceed max_candidates")
        if self.action is not CorrectionAction.GRAPH and self.max_hops != 1:
            raise ValueError("only graph action may request multiple hops")
        if self.action is not CorrectionAction.DECOMPOSE and self.subqueries:
            raise ValueError("only decompose action may carry subqueries")
        return self


class ActionStatus(StrEnum):
    SUCCEEDED = "succeeded"
    EMPTY = "empty"
    FAILED = "failed"
    CACHE_HIT = "cache_hit"


class ActionResult(StrictModel):
    request: ActionRequest
    status: ActionStatus
    evidence: list[Evidence] = Field(default_factory=list, max_length=100)
    estimated_usage: ActionEstimate
    actual_usage: BudgetUsage
    error: ErrorDetail | None = None
    diagnostics: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def result_status_is_consistent(self) -> ActionResult:
        if self.status is ActionStatus.FAILED and self.error is None:
            raise ValueError("failed action requires an error")
        if self.status is not ActionStatus.FAILED and self.error is not None:
            raise ValueError("successful action cannot carry an error")
        if self.status in {ActionStatus.EMPTY, ActionStatus.FAILED} and self.evidence:
            raise ValueError("empty or failed action cannot carry evidence")
        if self.status is ActionStatus.CACHE_HIT and any(
            getattr(self.actual_usage, name) for name in BudgetUsage.model_fields
        ):
            raise ValueError("cache hits must not consume external-call budget")
        return self


class StopReason(StrEnum):
    SUFFICIENT = "sufficient"
    NO_POSITIVE_GAIN = "no_positive_gain"
    BUDGET_EXHAUSTED = "budget_exhausted"
    INTERNAL_KNOWLEDGE_MISSING = "internal_knowledge_missing"
    EXECUTION_FAILED = "execution_failed"
    INVALID_INPUT = "invalid_input"


class CorrectionDecision(StrictModel):
    decision_id: str = Field(pattern=r"^decision-[a-f0-9]{16}$")
    selected: ActionRequest | None = None
    estimate: ActionEstimate | None = None
    utility: float
    reason_code: str = Field(min_length=1, max_length=100)
    candidate_action_ids: list[str] = Field(default_factory=list, max_length=20)
    rejected_reasons: dict[str, str] = Field(default_factory=dict, max_length=20)
    stop_reason: StopReason | None = None

    @model_validator(mode="after")
    def decision_selects_action_or_stop(self) -> CorrectionDecision:
        if (self.selected is None) == (self.stop_reason is None):
            raise ValueError("decision must select exactly one action or stop reason")
        if self.selected is not None and self.estimate is None:
            raise ValueError("selected action requires an estimate")
        if self.selected is not None and self.utility <= 0:
            raise ValueError("selected action requires positive utility")
        return self


class StopResult(StrictModel):
    reason: StopReason
    message: str = Field(min_length=1, max_length=500)
    missing_facet_ids: list[str] = Field(default_factory=list)
    selected_evidence_ids: list[str] = Field(default_factory=list)
    budget: BudgetLedger


class RunIdentity(StrictModel):
    run_id: str = Field(pattern=r"^run-[a-f0-9]{32}$")
    question_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    facets_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    evidence_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    config_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    prompt_version: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    corpus_snapshot: str = Field(min_length=1)
    dense_version: str = Field(min_length=1)
    sparse_version: str = Field(min_length=1)
    graph_version: str = Field(min_length=1)
    rules_version: str = Field(min_length=1)
    coverage_version: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)


class CorrectionState(StrictModel):
    identity: RunIdentity
    question_id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=4000)
    facets: list[EvidenceRequirement] = Field(default_factory=list, max_length=20)
    candidates: list[Evidence] = Field(default_factory=list, max_length=100)
    selected_evidence: list[Evidence] = Field(default_factory=list, max_length=20)
    coverage_matrix: CoverageMatrix | None = None
    sufficiency: SufficiencyAssessment | None = None
    candidate_actions: list[ActionRequest] = Field(default_factory=list, max_length=20)
    decision: CorrectionDecision | None = None
    action_history: list[ActionResult] = Field(default_factory=list, max_length=2)
    retrieval_round: int = Field(default=0, ge=0, le=2)
    budget: BudgetLedger = Field(default_factory=BudgetLedger)
    errors: list[ErrorDetail] = Field(default_factory=list, max_length=20)
    trace: list[TraceEvent] = Field(default_factory=list, max_length=200)
    state_fingerprint: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    stop: StopResult | None = None

    @model_validator(mode="after")
    def state_identities_are_unique(self) -> CorrectionState:
        facet_ids = [item.facet_id for item in self.facets]
        evidence_ids = [item.evidence_id for item in self.candidates]
        if len(facet_ids) != len(set(facet_ids)):
            raise ValueError("state facet IDs must be unique")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("state evidence IDs must be unique")
        return self


class WorkflowStage(StrEnum):
    REQUIREMENTS = "requirements"
    INITIAL_RETRIEVE = "initial_retrieve"
    ASSESS = "assess"
    DECIDE = "decide"
    EXECUTE = "execute"
    REASSESS = "reassess"
    FINALIZE = "finalize"


class CorrectionCheckpoint(StrictModel):
    identity: RunIdentity
    stage: WorkflowStage
    sequence: int = Field(ge=0)
    state: CorrectionState
    created_at: datetime

    @model_validator(mode="after")
    def checkpoint_identity_matches_state(self) -> CorrectionCheckpoint:
        if self.identity != self.state.identity:
            raise ValueError("checkpoint identity must match state identity")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("checkpoint creation time must include timezone")
        return self


class CorrectionRunResult(StrictModel):
    identity: RunIdentity
    state: CorrectionState
    stop: StopResult
    started_at: datetime
    finished_at: datetime

    @model_validator(mode="after")
    def result_is_consistent(self) -> CorrectionRunResult:
        if self.state.identity != self.identity or self.state.stop != self.stop:
            raise ValueError("run result identity or stop result is inconsistent")
        if self.finished_at < self.started_at:
            raise ValueError("run result time range is reversed")
        return self


class CorrectiveStrategy(StrEnum):
    FIXED_HYBRID = "fixed_hybrid"
    TYPE_ROUTER = "type_router"
    FACET_CORRECTIVE = "facet_corrective"


class CorrectiveEvaluationQuestion(StrictModel):
    question_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    category: GapType
    corpus_snapshot: str = Field(min_length=1)
    dense_version: str = Field(min_length=1)
    sparse_version: str = Field(min_length=1)
    graph_version: str = Field(min_length=1)
    expected_facet_ids: list[str] = Field(min_length=1, max_length=20)
    initial_evidence: list[Evidence] = Field(default_factory=list, max_length=100)
    action_evidence: dict[CorrectionAction, list[Evidence]] = Field(default_factory=dict)
    target_evidence_ids: list[str] = Field(default_factory=list, max_length=100)
    allowed_actions: list[CorrectionAction] = Field(default_factory=list)
    expected_stop: StopReason

    @model_validator(mode="after")
    def targets_and_actions_are_frozen(self) -> CorrectiveEvaluationQuestion:
        if len(self.expected_facet_ids) != len(set(self.expected_facet_ids)):
            raise ValueError("expected facet IDs must be unique")
        if len(self.target_evidence_ids) != len(set(self.target_evidence_ids)):
            raise ValueError("target evidence IDs must be unique")
        if len(self.allowed_actions) != len(set(self.allowed_actions)):
            raise ValueError("allowed corrective actions must be unique")
        available = {item.evidence_id for item in self.initial_evidence} | {
            item.evidence_id for items in self.action_evidence.values() for item in items
        }
        if not set(self.target_evidence_ids) <= available:
            raise ValueError("target evidence is absent from frozen fixtures")
        if self.category is not GapType.INTERNAL_MISSING and not self.target_evidence_ids:
            raise ValueError("non-missing questions require target evidence")
        if set(self.action_evidence) - set(self.allowed_actions):
            raise ValueError("action fixtures must use allowed actions")
        return self


class CorrectiveEvaluationQuestionSet(StrictModel):
    schema_version: Literal["v1"] = "v1"
    questions: list[CorrectiveEvaluationQuestion] = Field(min_length=20, max_length=40)

    @model_validator(mode="after")
    def questions_share_identity(self) -> CorrectiveEvaluationQuestionSet:
        ids = [item.question_id for item in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("corrective evaluation question IDs must be unique")
        versions = {
            (q.corpus_snapshot, q.dense_version, q.sparse_version, q.graph_version)
            for q in self.questions
        }
        if len(versions) != 1:
            raise ValueError("corrective evaluation questions must share frozen versions")
        return self


class CorrectiveEvaluationItem(StrictModel):
    question_id: str
    strategy: CorrectiveStrategy
    first_sufficient: bool = False
    diagnosis_correct: bool = False
    required_facets: int = Field(default=0, ge=0)
    covered_required_facets: int = Field(default=0, ge=0)
    recovered: bool = False
    correction_triggered: bool = False
    meaningless_actions: int = Field(default=0, ge=0)
    retrieval_rounds: int = Field(default=0, ge=0, le=2)
    tool_calls: int = Field(default=0, ge=0)
    llm_calls: int = Field(default=0, ge=0, le=4)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)
    stop_reason: StopReason
    error: ErrorDetail | None = None


class CorrectiveEvaluationMetrics(StrictModel):
    total: int = Field(ge=1)
    failures: int = Field(ge=0)
    first_sufficiency_rate: float = Field(ge=0.0, le=1.0)
    diagnosis_accuracy: float = Field(ge=0.0, le=1.0)
    required_coverage_rate: float = Field(ge=0.0, le=1.0)
    recovery_rate: float = Field(ge=0.0, le=1.0)
    correction_trigger_rate: float = Field(ge=0.0, le=1.0)
    meaningless_action_rate: float = Field(ge=0.0, le=1.0)
    average_retrieval_rounds: float = Field(ge=0.0, le=2.0)
    tool_calls: int = Field(ge=0)
    llm_calls: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: int = Field(ge=0)
    stop_reasons: dict[StopReason, int]


class CorrectiveEvaluationReport(StrictModel):
    evaluation_id: str = Field(pattern=r"^eval-[a-f0-9]{32}$")
    offline: bool
    question_set_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    items: list[CorrectiveEvaluationItem] = Field(min_length=1)
    metrics: dict[CorrectiveStrategy, CorrectiveEvaluationMetrics]
    created_at: datetime
