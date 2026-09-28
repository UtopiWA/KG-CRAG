"""摄取、检索与生成模块共享的校验数据契约。"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class StrictModel(BaseModel):
    """拒绝未知字段，使流水线的模式漂移尽早失败。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Author(StrictModel):
    author_id: str | None = None
    name: str = Field(min_length=1)


class Paper(StrictModel):
    paper_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    abstract: str = ""
    year: int | None = Field(default=None, ge=1900, le=2100)
    authors: list[Author] = Field(default_factory=list)
    doi: str | None = None
    arxiv_id: str | None = None
    semantic_scholar_id: str | None = None
    venue: str | None = None
    source_url: HttpUrl | None = None
    pdf_path: str | None = None
    references: list[str] = Field(default_factory=list)
    ingestion_version: str = "v1"


class Chunk(StrictModel):
    chunk_id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    section: str | None = None
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    text: str = Field(min_length=1)
    token_count: int = Field(ge=1)
    content_hash: str = Field(min_length=8)
    ordinal: int = Field(default=0, ge=0)
    processing_version: str = Field(default="v1", min_length=1)

    @model_validator(mode="after")
    def page_range_is_ordered(self) -> Chunk:
        """确保页码映射不会反向。"""

        if self.page_start and self.page_end and self.page_end < self.page_start:
            raise ValueError("page_end must be greater than or equal to page_start")
        return self


class EvidenceSourceType(StrEnum):
    CHUNK = "chunk"
    GRAPH = "graph"
    WEB = "web"


class EvidenceLocation(StrictModel):
    section: str | None = None
    page: int | None = Field(default=None, ge=1)
    url: HttpUrl | None = None


class EvidenceScores(StrictModel):
    dense: float | None = None
    sparse: float | None = None
    graph: float | None = None
    fusion: float | None = None
    rerank: float | None = None


class Evidence(StrictModel):
    evidence_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    source_type: EvidenceSourceType
    source_id: str = Field(min_length=1)
    paper_id: str | None = None
    location: EvidenceLocation = Field(default_factory=EvidenceLocation)
    scores: EvidenceScores = Field(default_factory=EvidenceScores)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class QueryType(StrEnum):
    FACT = "fact"
    SEMANTIC = "semantic"
    ENTITY = "entity"
    RELATIONSHIP = "relationship"
    COMPARATIVE = "comparative"
    MULTI_HOP = "multi_hop"
    KNOWLEDGE_MISSING = "knowledge_missing"


class RetrievalStrategy(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"
    GRAPH = "graph"
    HYBRID = "hybrid"
    WEB = "web"


class RetrievalWeights(StrictModel):
    dense: float = Field(default=0.0, ge=0.0, le=1.0)
    sparse: float = Field(default=0.0, ge=0.0, le=1.0)
    graph: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> RetrievalWeights:
        """保持融合权重归一化，确保实验可比。"""

        if abs(self.dense + self.sparse + self.graph - 1.0) > 1e-6:
            raise ValueError("retrieval weights must sum to 1.0")
        return self


class RouteDecision(StrictModel):
    query_types: list[QueryType] = Field(min_length=1)
    strategy: RetrievalStrategy
    subqueries: list[str] = Field(default_factory=list)
    weights: RetrievalWeights
    reason: str = Field(min_length=1)


class RetrievalLabel(StrEnum):
    RELEVANT = "relevant"
    AMBIGUOUS = "ambiguous"
    INSUFFICIENT = "insufficient"
    IRRELEVANT = "irrelevant"


class RecommendedAction(StrEnum):
    ANSWER = "answer"
    REWRITE = "rewrite"
    DECOMPOSE = "decompose"
    SWITCH_RETRIEVER = "switch_retriever"
    WEB_SEARCH = "web_search"


class RetrievalEvaluationScores(StrictModel):
    relevance: float = Field(ge=0.0, le=1.0)
    coverage: float = Field(ge=0.0, le=1.0)
    consistency: float = Field(ge=0.0, le=1.0)
    groundability: float = Field(ge=0.0, le=1.0)
    source_quality: float = Field(default=0.0, ge=0.0, le=1.0)


class RetrievalEvaluation(StrictModel):
    label: RetrievalLabel
    scores: RetrievalEvaluationScores
    missing_aspects: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    recommended_action: RecommendedAction
