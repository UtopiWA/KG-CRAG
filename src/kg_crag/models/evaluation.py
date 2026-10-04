"""统一评测、运行身份、逐题观察与报告契约。"""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from kg_crag.models.domain import StrictModel
from kg_crag.models.foundation import ErrorDetail, EvaluationTraceEvent


class EvaluationSplit(StrEnum):
    DEV = "dev"
    TEST = "test"


class UnifiedQuestionType(StrEnum):
    FACTOID = "factoid"
    COMPARISON = "comparison"
    RELATIONSHIP = "relationship"
    MULTI_HOP = "multi_hop"
    METRIC = "metric"
    SYNTHESIS = "synthesis"


class StressCategory(StrEnum):
    CONTROL = "control"
    TERMINOLOGY_MISMATCH = "terminology_mismatch"
    ENTITY_ALIAS = "entity_alias"
    COMPARISON_SIDE = "comparison_side"
    MULTI_HOP_GAP = "multi_hop_gap"
    METRIC_MISSING = "metric_missing"
    CONFLICT = "conflict"
    INTERNAL_MISSING = "internal_missing"


class KnowledgeSufficiency(StrEnum):
    SUFFICIENT = "sufficient"
    CONFLICTING = "conflicting"
    INSUFFICIENT = "insufficient"


class EvidenceMatchMode(StrEnum):
    ANY = "any"
    ALL = "all"


class EvaluationQuestionOrigin(StrEnum):
    FIXTURE_MIGRATION = "fixture_migration"
    CHUNK_DERIVED = "chunk_derived"


class UnifiedStrategy(StrEnum):
    FIXED_HYBRID = "fixed_hybrid"
    TYPE_ROUTER = "type_router"
    FACET_CORRECTIVE = "facet_corrective"


class EvaluationStage(StrEnum):
    VALIDATE = "validate"
    RETRIEVE = "retrieve"
    ASSESS = "assess"
    ANSWER = "answer"
    JUDGE = "judge"
    AGGREGATE = "aggregate"
    PUBLISH = "publish"


class EvaluationStageStatus(StrEnum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    BUDGET_STOPPED = "budget_stopped"


class EvaluationAnswerPoint(StrictModel):
    point_id: str = Field(pattern=r"^point-[a-f0-9]{16}$")
    description: str = Field(min_length=1, max_length=500)
    evidence_ids: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("evidence_ids")
    @classmethod
    def evidence_ids_are_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("answer point evidence IDs must be unique")
        return value


class EvaluationFacetTarget(StrictModel):
    facet_id: str = Field(pattern=r"^facet-[a-f0-9]{16}$")
    description: str = Field(min_length=1, max_length=500)
    required: bool = True
    target_evidence_ids: list[str] = Field(default_factory=list, max_length=20)
    evidence_match: EvidenceMatchMode = EvidenceMatchMode.ANY

    @field_validator("target_evidence_ids")
    @classmethod
    def targets_are_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("facet target evidence IDs must be unique")
        return value

    def is_covered_by(self, evidence_ids: set[str]) -> bool:
        """按 facet 声明的任一/全部语义判断 Evidence 覆盖。"""

        targets = set(self.target_evidence_ids)
        if not targets:
            return False
        if self.evidence_match is EvidenceMatchMode.ALL:
            return targets <= evidence_ids
        return bool(targets & evidence_ids)


class UnifiedEvaluationQuestion(StrictModel):
    """冻结问题及其最小可验证真值。"""

    question_id: str = Field(pattern=r"^ueq-[a-z0-9-]{3,120}$")
    split: EvaluationSplit
    question: str = Field(min_length=1, max_length=4000)
    normalized_question: str = Field(min_length=1, max_length=4000)
    question_types: list[UnifiedQuestionType] = Field(min_length=1, max_length=3)
    stress_category: StressCategory
    knowledge_sufficiency: KnowledgeSufficiency
    answer_points: list[EvaluationAnswerPoint] = Field(default_factory=list, max_length=20)
    facets: list[EvaluationFacetTarget] = Field(min_length=1, max_length=20)
    relevant_evidence: dict[str, int] = Field(default_factory=dict, max_length=100)
    minimum_sufficient_evidence_sets: list[list[str]] = Field(default_factory=list, max_length=10)
    leakage_group_id: str = Field(min_length=1, max_length=128)
    source_group_ids: list[str] = Field(min_length=1, max_length=20)
    source_fixture: str = Field(min_length=1, max_length=300)

    @field_validator("question_types", "source_group_ids")
    @classmethod
    def values_are_unique(cls, value: list[object]) -> list[object]:
        if len(value) != len(set(value)):
            raise ValueError("question annotation values must be unique")
        return value

    @field_validator("relevant_evidence")
    @classmethod
    def relevance_is_graded(cls, value: dict[str, int]) -> dict[str, int]:
        if any(not evidence_id or grade < 1 or grade > 3 for evidence_id, grade in value.items()):
            raise ValueError("relevance grades must be between 1 and 3")
        return value

    @model_validator(mode="after")
    def annotations_are_consistent(self) -> UnifiedEvaluationQuestion:
        facet_ids = [item.facet_id for item in self.facets]
        point_ids = [item.point_id for item in self.answer_points]
        if len(facet_ids) != len(set(facet_ids)):
            raise ValueError("facet IDs must be unique within a question")
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("answer point IDs must be unique within a question")
        if not any(item.required for item in self.facets):
            raise ValueError("each question requires at least one required facet")
        known = set(self.relevant_evidence)
        referenced = {
            evidence_id for item in self.facets for evidence_id in item.target_evidence_ids
        }
        referenced.update(
            evidence_id for item in self.answer_points for evidence_id in item.evidence_ids
        )
        if referenced - known:
            raise ValueError("annotation references unknown relevant evidence")
        normalized_sets: list[tuple[str, ...]] = []
        for evidence_set in self.minimum_sufficient_evidence_sets:
            if not evidence_set or len(evidence_set) != len(set(evidence_set)):
                raise ValueError("minimum sufficient evidence sets must be non-empty and unique")
            if set(evidence_set) - known:
                raise ValueError("minimum sufficient set references unknown evidence")
            normalized_sets.append(tuple(sorted(evidence_set)))
        if len(normalized_sets) != len(set(normalized_sets)):
            raise ValueError("minimum sufficient evidence sets must be unique")
        if self.knowledge_sufficiency is KnowledgeSufficiency.INSUFFICIENT:
            if known or self.minimum_sufficient_evidence_sets:
                raise ValueError("knowledge-missing questions cannot declare sufficient evidence")
        elif not known:
            raise ValueError("available or conflicting knowledge requires relevant evidence")
        if (
            self.stress_category is StressCategory.INTERNAL_MISSING
            and self.knowledge_sufficiency is not KnowledgeSufficiency.INSUFFICIENT
        ):
            raise ValueError("internal-missing stress requires insufficient knowledge")
        if (
            self.stress_category is StressCategory.CONFLICT
            and self.knowledge_sufficiency is not KnowledgeSufficiency.CONFLICTING
        ):
            raise ValueError("conflict stress requires conflicting knowledge")
        if self.knowledge_sufficiency is KnowledgeSufficiency.SUFFICIENT:
            if not self.minimum_sufficient_evidence_sets:
                raise ValueError("sufficient knowledge requires a minimum sufficient evidence set")
            required = [item for item in self.facets if item.required]
            if any(not item.target_evidence_ids for item in required):
                raise ValueError("required facets need target evidence")
            for evidence_set in self.minimum_sufficient_evidence_sets:
                if any(not item.is_covered_by(set(evidence_set)) for item in required):
                    raise ValueError("minimum sufficient set does not cover every required facet")
        return self


class UnifiedQuestionSet(StrictModel):
    schema_version: Literal["v1"] = "v1"
    dataset_version: str = Field(min_length=1, max_length=100)
    split: EvaluationSplit
    questions: list[UnifiedEvaluationQuestion] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def questions_are_unique_and_match_split(self) -> UnifiedQuestionSet:
        identifiers = [item.question_id for item in self.questions]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("unified evaluation question IDs must be unique")
        if any(item.split is not self.split for item in self.questions):
            raise ValueError("question split differs from its containing question set")
        return self


class EvaluationEvidenceRecord(StrictModel):
    """供统一评测标注复核的冻结 Evidence 记录。"""

    evidence_id: str = Field(min_length=1, max_length=200)
    source_group_id: str = Field(min_length=1, max_length=128)
    source_kind: Literal["chunk", "graph", "fixture"]
    source_locator: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1, max_length=20_000)
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def content_matches_hash(self) -> EvaluationEvidenceRecord:
        actual = hashlib.sha256(self.content.encode()).hexdigest()
        if actual != self.content_hash:
            raise ValueError("evaluation Evidence content hash drifted")
        return self


class EvaluationEvidenceCatalog(StrictModel):
    """统一题目引用的有界、内容寻址 Evidence 目录。"""

    schema_version: Literal["v1"] = "v1"
    dataset_version: str = Field(min_length=1, max_length=100)
    evidence_version: str = Field(min_length=1, max_length=200)
    records: list[EvaluationEvidenceRecord] = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def evidence_ids_are_unique(self) -> EvaluationEvidenceCatalog:
        identifiers = [item.evidence_id for item in self.records]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("evaluation Evidence IDs must be unique")
        return self


class EvaluationQuestionSourceRecord(StrictModel):
    """当前题目文本的直接来源及其上游派生依据。"""

    question_id: str = Field(pattern=r"^ueq-[a-z0-9-]{3,120}$")
    question: str = Field(min_length=1, max_length=4000)
    origin: EvaluationQuestionOrigin
    origin_locator: str = Field(min_length=1, max_length=500)
    evidence_ids: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("evidence_ids")
    @classmethod
    def evidence_ids_are_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("question source Evidence IDs must be unique")
        return value


class EvaluationQuestionSourceCatalog(StrictModel):
    """统一题目的冻结直接来源目录。"""

    schema_version: Literal["v1"] = "v1"
    dataset_version: str = Field(min_length=1, max_length=100)
    records: list[EvaluationQuestionSourceRecord] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def question_ids_are_unique(self) -> EvaluationQuestionSourceCatalog:
        identifiers = [item.question_id for item in self.records]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("evaluation question source IDs must be unique")
        return self


class EvaluationDatasetManifest(StrictModel):
    schema_version: Literal["v1"] = "v1"
    dataset_version: str = Field(min_length=1, max_length=100)
    corpus_snapshot: str = Field(min_length=1, max_length=200)
    evidence_version: str = Field(min_length=1, max_length=200)
    evidence_path: str = Field(min_length=1, max_length=300)
    evidence_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    question_sources_path: str = Field(min_length=1, max_length=300)
    question_sources_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    dev_path: str = Field(min_length=1, max_length=300)
    test_path: str = Field(min_length=1, max_length=300)
    dev_count: int = Field(ge=40, le=50)
    test_count: int = Field(ge=20, le=30)
    dev_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    test_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    dataset_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    annotation_guide_version: str = Field(min_length=1, max_length=100)
    reviewed: bool
    review_note: str = Field(min_length=1, max_length=1000)


class EvaluationResourceUsage(StrictModel):
    tool_calls: int = Field(default=0, ge=0, le=10_000)
    model_calls: int = Field(default=0, ge=0, le=1_000)
    judge_calls: int = Field(default=0, ge=0, le=50)
    web_calls: int = Field(default=0, ge=0, le=100)
    input_tokens: int = Field(default=0, ge=0, le=10_000_000)
    output_tokens: int = Field(default=0, ge=0, le=10_000_000)
    latency_ms: int = Field(default=0, ge=0, le=86_400_000)
    estimated_cost: float | None = Field(default=None, ge=0.0)


class StrategyObservation(StrictModel):
    question_id: str = Field(pattern=r"^ueq-[a-z0-9-]{3,120}$")
    strategy: UnifiedStrategy
    status: EvaluationStageStatus
    ranked_evidence_ids: list[str] = Field(default_factory=list, max_length=100)
    initial_covered_facet_ids: list[str] = Field(default_factory=list, max_length=20)
    final_covered_facet_ids: list[str] = Field(default_factory=list, max_length=20)
    predicted_sufficient: bool = False
    action_ids: list[str] = Field(default_factory=list, max_length=20)
    route: str | None = Field(default=None, max_length=100)
    loop_count: int = Field(default=0, ge=0, le=10)
    web_used: bool = False
    matched_answer_point_ids: list[str] = Field(default_factory=list, max_length=20)
    cited_evidence_ids: list[str] = Field(default_factory=list, max_length=20)
    usage: EvaluationResourceUsage = Field(default_factory=EvaluationResourceUsage)
    error: ErrorDetail | None = None

    @field_validator(
        "initial_covered_facet_ids",
        "final_covered_facet_ids",
        "action_ids",
        "matched_answer_point_ids",
        "cited_evidence_ids",
    )
    @classmethod
    def observation_lists_are_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("strategy observation lists must be unique")
        return value

    @model_validator(mode="after")
    def status_matches_error(self) -> StrategyObservation:
        failed = self.status in {
            EvaluationStageStatus.FAILED,
            EvaluationStageStatus.BUDGET_STOPPED,
        }
        if failed != (self.error is not None):
            raise ValueError("failed or budget-stopped observation must carry exactly one error")
        return self


class StrategyObservationSet(StrictModel):
    schema_version: Literal["v1"] = "v1"
    dataset_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    split_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_versions: dict[str, str] = Field(min_length=1, max_length=50)
    observations: list[StrategyObservation] = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def observation_keys_are_unique(self) -> StrategyObservationSet:
        keys = [(item.question_id, item.strategy) for item in self.observations]
        if len(keys) != len(set(keys)):
            raise ValueError("strategy observation keys must be unique")
        return self


class EvaluationMetricValue(StrictModel):
    value: float | None = None
    numerator: float | None = None
    denominator: float | None = None
    question_ids: list[str] = Field(default_factory=list, max_length=100)
    not_applicable_reason: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def applicability_is_explicit(self) -> EvaluationMetricValue:
        if self.value is None and self.not_applicable_reason is None:
            raise ValueError("non-numeric metric requires a not-applicable reason")
        if self.value is not None and self.not_applicable_reason is not None:
            raise ValueError("numeric metric cannot carry a not-applicable reason")
        if self.denominator == 0 and self.value is not None:
            raise ValueError("zero-denominator metric cannot have a numeric value")
        return self


class EvaluationSliceMetrics(StrictModel):
    slice_name: str = Field(min_length=1, max_length=200)
    question_ids: list[str] = Field(min_length=1, max_length=100)
    metrics: dict[str, EvaluationMetricValue] = Field(min_length=1, max_length=100)


class EvaluationRunIdentity(StrictModel):
    run_id: str = Field(pattern=r"^unified-run-[a-f0-9]{32}$")
    dataset_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    split_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    corpus_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    evidence_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    index_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    config_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    strategy_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    prompt_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    metric_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    judge_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    seed: int = Field(ge=0, le=2**32 - 1)


class EvaluationItemResult(StrictModel):
    question_id: str = Field(pattern=r"^ueq-[a-z0-9-]{3,120}$")
    strategy: UnifiedStrategy
    observation: StrategyObservation
    metrics: dict[str, EvaluationMetricValue] = Field(default_factory=dict, max_length=100)
    trace_sequences: list[int] = Field(default_factory=list, max_length=200)
    source_artifact_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def observation_matches_item(self) -> EvaluationItemResult:
        if (
            self.question_id != self.observation.question_id
            or self.strategy is not self.observation.strategy
        ):
            raise ValueError("evaluation item does not match its strategy observation")
        return self


class EvaluationRunManifest(StrictModel):
    identity: EvaluationRunIdentity
    split: EvaluationSplit
    status: EvaluationStageStatus
    selected_answer_strategy: UnifiedStrategy | None = None
    completed_item_keys: list[str] = Field(default_factory=list, max_length=300)
    checkpoint_hashes: dict[str, str] = Field(default_factory=dict, max_length=300)
    budget: EvaluationResourceUsage = Field(default_factory=EvaluationResourceUsage)
    failures: dict[str, int] = Field(default_factory=dict, max_length=50)
    started_at: datetime
    finished_at: datetime | None = None


class EvaluationFailureRecord(StrictModel):
    question_id: str = Field(pattern=r"^ueq-[a-z0-9-]{3,120}$")
    strategy: UnifiedStrategy
    status: EvaluationStageStatus
    error: ErrorDetail
    usage: EvaluationResourceUsage


class EvaluationFailureReport(StrictModel):
    run_id: str = Field(pattern=r"^unified-run-[a-f0-9]{32}$")
    failures: list[EvaluationFailureRecord] = Field(default_factory=list, max_length=300)


class EvaluationCheckpoint(StrictModel):
    identity: EvaluationRunIdentity
    stage: EvaluationStage
    item: EvaluationItemResult
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def created_at_is_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("evaluation checkpoint time must include a timezone")
        return value


class UnifiedEvaluationReport(StrictModel):
    schema_version: Literal["v1"] = "v1"
    identity: EvaluationRunIdentity
    split: EvaluationSplit
    offline: bool
    fixture_mode: bool
    selected_answer_strategy: UnifiedStrategy | None = None
    items: list[EvaluationItemResult] = Field(min_length=1, max_length=300)
    slices: dict[str, list[EvaluationSliceMetrics]] = Field(min_length=1)
    trace: list[EvaluationTraceEvent] = Field(default_factory=list, max_length=20_000)
    judge_status: Literal["not_requested", "completed", "failed"] = "not_requested"
    judge_records: list[JudgeRecord] = Field(default_factory=list, max_length=50)
    created_at: datetime


class DevelopmentSelection(StrictModel):
    dataset_version: str = Field(min_length=1)
    dataset_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    selected_strategy: UnifiedStrategy
    config_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    prompt_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    report_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    version_declarations: dict[str, str] = Field(min_length=1, max_length=20)
    thresholds: dict[str, str | int | float | bool] = Field(min_length=1, max_length=20)
    frozen_at: datetime


class FormalTestLock(StrictModel):
    dataset_version: str = Field(min_length=1)
    dataset_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    selection_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    run_id: str = Field(pattern=r"^unified-run-[a-f0-9]{32}$")
    report_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    created_at: datetime


class JudgeVerdict(StrictModel):
    question_id: str = Field(pattern=r"^ueq-[a-z0-9-]{3,120}$")
    strategy: UnifiedStrategy
    judge_version: str = Field(min_length=1, max_length=200)
    prompt_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    input_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    correct: bool
    complete: bool
    faithful: bool
    citation_correct: bool
    reason_labels: list[str] = Field(default_factory=list, max_length=20)
    usage: EvaluationResourceUsage


class JudgeRecord(StrictModel):
    question_id: str = Field(pattern=r"^ueq-[a-z0-9-]{3,120}$")
    strategy: UnifiedStrategy
    status: EvaluationStageStatus
    verdict: JudgeVerdict | None = None
    usage: EvaluationResourceUsage = Field(default_factory=EvaluationResourceUsage)
    error: ErrorDetail | None = None

    @model_validator(mode="after")
    def result_is_consistent(self) -> JudgeRecord:
        if self.status is EvaluationStageStatus.SUCCEEDED:
            if self.verdict is None or self.error is not None:
                raise ValueError("successful Judge record requires only a verdict")
        elif self.verdict is not None or self.error is None:
            raise ValueError("failed Judge record requires only an error")
        return self


class HumanReview(StrictModel):
    question_id: str = Field(pattern=r"^ueq-[a-z0-9-]{3,120}$")
    strategy: UnifiedStrategy
    review_version: str = Field(min_length=1, max_length=100)
    correct: bool
    complete: bool
    faithful: bool
    citation_correct: bool
    note: str = Field(default="", max_length=1000)


UnifiedEvaluationReport.model_rebuild()
