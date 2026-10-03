"""知识图谱构建、查询与评测使用的严格公共契约。"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from kg_crag.models.domain import StrictModel

Sha256 = str


class GraphNodeType(StrEnum):
    PAPER = "Paper"
    AUTHOR = "Author"
    INSTITUTION = "Institution"
    METHOD = "Method"
    MODEL = "Model"
    DATASET = "Dataset"
    TASK = "Task"
    METRIC = "Metric"
    RESULT = "Result"
    CHUNK = "Chunk"


class GraphRelationType(StrEnum):
    AUTHORED_BY = "AUTHORED_BY"
    AFFILIATED_WITH = "AFFILIATED_WITH"
    CITES = "CITES"
    CONTAINS = "CONTAINS"
    HAS_METHOD = "HAS_METHOD"
    USES_MODEL = "USES_MODEL"
    EVALUATED_ON = "EVALUATED_ON"
    ADDRESSES_TASK = "ADDRESSES_TASK"
    REPORTS_RESULT = "REPORTS_RESULT"
    USES_METRIC = "USES_METRIC"
    ON_DATASET = "ON_DATASET"
    FOR_TASK = "FOR_TASK"


RELATION_ENDPOINTS: dict[GraphRelationType, tuple[GraphNodeType, GraphNodeType]] = {
    GraphRelationType.AUTHORED_BY: (GraphNodeType.PAPER, GraphNodeType.AUTHOR),
    GraphRelationType.AFFILIATED_WITH: (GraphNodeType.AUTHOR, GraphNodeType.INSTITUTION),
    GraphRelationType.CITES: (GraphNodeType.PAPER, GraphNodeType.PAPER),
    GraphRelationType.CONTAINS: (GraphNodeType.PAPER, GraphNodeType.CHUNK),
    GraphRelationType.HAS_METHOD: (GraphNodeType.PAPER, GraphNodeType.METHOD),
    GraphRelationType.USES_MODEL: (GraphNodeType.METHOD, GraphNodeType.MODEL),
    GraphRelationType.EVALUATED_ON: (GraphNodeType.METHOD, GraphNodeType.DATASET),
    GraphRelationType.ADDRESSES_TASK: (GraphNodeType.METHOD, GraphNodeType.TASK),
    GraphRelationType.REPORTS_RESULT: (GraphNodeType.METHOD, GraphNodeType.RESULT),
    GraphRelationType.USES_METRIC: (GraphNodeType.RESULT, GraphNodeType.METRIC),
    GraphRelationType.ON_DATASET: (GraphNodeType.RESULT, GraphNodeType.DATASET),
    GraphRelationType.FOR_TASK: (GraphNodeType.RESULT, GraphNodeType.TASK),
}


class GraphIdentity(StrictModel):
    schema_version: str = Field(min_length=1)
    graph_version: str = Field(min_length=1)
    corpus_snapshot: Sha256 = Field(pattern=r"^[a-f0-9]{64}$")
    metadata_builder_version: str = Field(min_length=1)
    selector_version: str = Field(min_length=1)
    extractor_version: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    model: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    normalization_version: str = Field(min_length=1)
    query_template_version: str = Field(min_length=1)


class GraphEntity(StrictModel):
    entity_id: str = Field(min_length=1)
    node_type: GraphNodeType
    name: str = Field(min_length=1, max_length=500)
    normalized_name: str = Field(min_length=1, max_length=500)
    aliases: list[str] = Field(default_factory=list, max_length=50)
    external_id: str | None = Field(default=None, max_length=500)
    paper_scope: str | None = Field(default=None, max_length=200)
    properties: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict, max_length=32
    )

    @field_validator("aliases")
    @classmethod
    def aliases_are_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("graph entity aliases must be unique")
        return value


class GraphProvenance(StrictModel):
    source_kind: Literal["metadata", "chunk"]
    paper_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_hash: Sha256 = Field(pattern=r"^[a-f0-9]{64}$")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    extractor_version: str | None = None
    prompt_version: str | None = None
    created_at: datetime | None = None
    evidence_quote: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def provenance_fields_match_kind(self) -> GraphProvenance:
        """正文来源必须完整，元数据来源不能伪装成模型抽取。"""

        if self.source_kind == "chunk":
            required = (self.extractor_version, self.prompt_version, self.created_at)
            if any(value is None for value in required):
                raise ValueError("chunk provenance requires extractor, prompt and creation time")
            if self.created_at and self.created_at.tzinfo is None:
                raise ValueError("chunk provenance creation time must include timezone")
        elif any((self.extractor_version, self.prompt_version, self.created_at)):
            raise ValueError("metadata provenance must not carry extraction fields")
        return self


class GraphFact(StrictModel):
    fact_id: str = Field(min_length=1)
    relation: GraphRelationType
    source_entity_id: str = Field(min_length=1)
    source_type: GraphNodeType
    target_entity_id: str = Field(min_length=1)
    target_type: GraphNodeType
    provenance: GraphProvenance
    graph_schema_version: str = Field(min_length=1)
    corpus_snapshot: Sha256 = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def endpoints_are_whitelisted(self) -> GraphFact:
        if RELATION_ENDPOINTS[self.relation] != (self.source_type, self.target_type):
            raise ValueError("graph relation endpoints are not allowed by the schema")
        if (
            self.source_entity_id == self.target_entity_id
            and self.relation != GraphRelationType.CITES
        ):
            raise ValueError("non-citation graph facts cannot be self-referential")
        return self


class GraphBundle(StrictModel):
    identity: GraphIdentity
    paper_id: str = Field(min_length=1)
    entities: list[GraphEntity] = Field(min_length=1)
    facts: list[GraphFact] = Field(default_factory=list)
    unresolved_references: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def bundle_is_complete_and_consistent(self) -> GraphBundle:
        ids = [item.entity_id for item in self.entities]
        if len(ids) != len(set(ids)):
            raise ValueError("graph bundle contains duplicate entity IDs")
        known = set(ids)
        fact_ids = [item.fact_id for item in self.facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("graph bundle contains duplicate fact IDs")
        for fact in self.facts:
            if fact.source_entity_id not in known or fact.target_entity_id not in known:
                raise ValueError("graph fact endpoint is absent from the bundle")
            if fact.provenance.paper_id != self.paper_id:
                raise ValueError("graph fact provenance belongs to another paper")
            if (
                fact.graph_schema_version != self.identity.schema_version
                or fact.corpus_snapshot != self.identity.corpus_snapshot
            ):
                raise ValueError("graph fact identity does not match its bundle")
        return self


class GraphQueryTemplate(StrEnum):
    ENTITY_NEIGHBORS = "entity_neighbors"
    RELATION_LOOKUP = "relation_lookup"
    BOUNDED_PATH = "bounded_path"
    METHOD_EVIDENCE = "method_evidence"


class GraphQueryRequest(StrictModel):
    graph_version: str = Field(min_length=1)
    template: GraphQueryTemplate
    entity_ids: list[str] = Field(default_factory=list, max_length=20)
    entity_names: list[str] = Field(default_factory=list, max_length=20)
    node_types: list[GraphNodeType] = Field(default_factory=list, max_length=10)
    relation_types: list[GraphRelationType] = Field(default_factory=list, max_length=12)
    paper_ids: list[str] = Field(default_factory=list, max_length=100)
    top_k: int = Field(default=20, ge=1, le=100)
    max_candidates: int = Field(default=100, ge=1, le=500)
    max_hops: int = Field(default=1, ge=1, le=3)
    timeout_seconds: float = Field(default=10.0, gt=0.0, le=30.0)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def query_is_bounded(self) -> GraphQueryRequest:
        if not self.entity_ids and not self.entity_names:
            raise ValueError("graph query requires an entity ID or name")
        if self.top_k > self.max_candidates:
            raise ValueError("graph top_k must not exceed max_candidates")
        if self.template != GraphQueryTemplate.BOUNDED_PATH and self.max_hops != 1:
            raise ValueError("only bounded_path may request multiple hops")
        return self


class GraphPathHit(StrictModel):
    path_id: str = Field(min_length=1)
    graph_version: str = Field(min_length=1)
    template: GraphQueryTemplate
    entity_ids: list[str] = Field(min_length=2)
    fact_ids: list[str] = Field(min_length=1)
    provenances: list[GraphProvenance] = Field(min_length=1)
    summary: str = Field(min_length=1, max_length=2000)
    score: float
    hop_count: int = Field(ge=1, le=3)

    @model_validator(mode="after")
    def path_lengths_are_consistent(self) -> GraphPathHit:
        if len(self.fact_ids) != self.hop_count or len(self.entity_ids) != self.hop_count + 1:
            raise ValueError("graph path IDs do not match hop_count")
        if len(self.provenances) != len(self.fact_ids):
            raise ValueError("every graph fact requires provenance")
        return self


class GraphRecordState(StrictModel):
    paper_id: str
    bundle_hash: Sha256 = Field(pattern=r"^[a-f0-9]{64}$")
    entity_count: int = Field(ge=0)
    fact_count: int = Field(ge=0)


class GraphSyncResult(StrictModel):
    paper_id: str
    status: Literal["created", "updated", "unchanged", "failed"]
    entity_count: int = Field(ge=0)
    fact_count: int = Field(ge=0)
    error: str | None = Field(default=None, max_length=500)


class ExtractionUsage(StrictModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    estimated: bool = False


class GraphExtractionCandidate(StrictModel):
    source_type: GraphNodeType
    source_name: str = Field(min_length=1, max_length=500)
    relation: GraphRelationType
    target_type: GraphNodeType
    target_name: str = Field(min_length=1, max_length=500)
    evidence_quote: str = Field(min_length=1, max_length=1000)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def endpoints_are_whitelisted(self) -> GraphExtractionCandidate:
        if RELATION_ENDPOINTS[self.relation] != (self.source_type, self.target_type):
            raise ValueError("extracted relation endpoints are not allowed")
        return self


class GraphExtractionResponse(StrictModel):
    candidates: list[GraphExtractionCandidate] = Field(default_factory=list, max_length=100)
    usage: ExtractionUsage | None = None


class EntityReviewItem(StrictModel):
    review_id: str = Field(min_length=1)
    left_entity_id: str = Field(min_length=1)
    right_entity_id: str = Field(min_length=1)
    left_name: str = Field(min_length=1, max_length=500)
    right_name: str = Field(min_length=1, max_length=500)
    node_type: GraphNodeType | None = None
    score: float = Field(ge=0.0, le=1.0)
    reason: Literal[
        "cross_type_name",
        "external_id_conflict",
        "ambiguous_key",
        "abbreviation",
        "similar_name",
        "low_confidence",
    ]
    source_ids: list[str] = Field(min_length=1, max_length=20)


class EntityReviewDecision(StrictModel):
    review_id: str = Field(min_length=1)
    left_entity_id: str = Field(min_length=1)
    right_entity_id: str = Field(min_length=1)
    decision: Literal["merge", "keep_separate"]
    reason: str = Field(min_length=1, max_length=500)
    normalization_version: str = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1, max_length=20)


class GraphEvaluationQuestion(StrictModel):
    question_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    category: Literal[
        "relationship", "ambiguity", "single_hop", "multi_hop", "no_path", "bad_source"
    ]
    corpus_snapshot: Sha256 = Field(pattern=r"^[a-f0-9]{64}$")
    graph_version: str = Field(min_length=1)
    request: GraphQueryRequest
    relevant_fact_ids: list[str] = Field(default_factory=list)
    relevant_path_ids: list[str] = Field(default_factory=list)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    k: int = Field(default=10, ge=1, le=100)

    @model_validator(mode="after")
    def target_matches_request_identity(self) -> GraphEvaluationQuestion:
        if self.request.graph_version != self.graph_version:
            raise ValueError("graph evaluation request uses a different graph version")
        if self.k > self.request.top_k:
            raise ValueError("graph evaluation K exceeds the request top_k")
        targets = self.relevant_fact_ids or self.relevant_path_ids or self.relevant_chunk_ids
        if self.category != "no_path" and not targets:
            raise ValueError("non-empty graph evaluation questions require a target")
        return self


class GraphEvaluationQuestionSet(StrictModel):
    schema_version: Literal["v1"] = "v1"
    questions: list[GraphEvaluationQuestion] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def questions_share_frozen_identity(self) -> GraphEvaluationQuestionSet:
        ids = [item.question_id for item in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("graph evaluation question IDs must be unique")
        identities = {(item.corpus_snapshot, item.graph_version) for item in self.questions}
        if len(identities) != 1:
            raise ValueError("graph evaluation questions must share one frozen identity")
        return self
