"""论文摄取阶段使用的严格、可序列化数据契约。"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from kg_crag.models.domain import StrictModel
from kg_crag.models.foundation import ErrorDetail, ScalarValue, _reject_sensitive_keys

Sha256 = str


class LayoutLabel(StrEnum):
    """试点语料人工确认的版面特征。"""

    PLAIN_TEXT = "plain_text"
    TWO_COLUMN = "two_column"
    TABLE = "table"
    FORMULA = "formula"
    FULL_WIDTH_ELEMENT = "full_width_element"
    ANOMALOUS_LAYOUT = "anomalous_layout"


class ReviewStatus(StrEnum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"


class PilotPaper(StrictModel):
    paper_id: str = Field(min_length=1)
    arxiv_id: str = Field(min_length=1)
    input_sha256: Sha256 = Field(pattern=r"^[a-f0-9]{64}$")
    layout_labels: list[LayoutLabel] = Field(min_length=1)
    selection_reason: str = Field(min_length=1)

    @field_validator("layout_labels")
    @classmethod
    def labels_are_unique(cls, value: list[LayoutLabel]) -> list[LayoutLabel]:
        if len(value) != len(set(value)):
            raise ValueError("layout_labels must be unique")
        return value


class PilotManifest(StrictModel):
    schema_version: Literal["v1"] = "v1"
    papers: list[PilotPaper] = Field(min_length=10, max_length=20)

    @model_validator(mode="after")
    def validate_selection(self) -> PilotManifest:
        """保证试点唯一，且覆盖规范要求的全部版面。"""

        paper_ids = [paper.paper_id for paper in self.papers]
        arxiv_ids = [paper.arxiv_id for paper in self.papers]
        if len(paper_ids) != len(set(paper_ids)) or len(arxiv_ids) != len(set(arxiv_ids)):
            raise ValueError("pilot paper identifiers must be unique")
        covered = {label for paper in self.papers for label in paper.layout_labels}
        missing = set(LayoutLabel) - covered
        if missing:
            raise ValueError(f"pilot layout coverage is incomplete: {sorted(missing)}")
        return self


class PilotReview(StrictModel):
    schema_version: Literal["v1"] = "v1"
    manifest_hash: Sha256 = Field(pattern=r"^[a-f0-9]{64}$")
    status: ReviewStatus = ReviewStatus.PENDING
    processing_versions: dict[str, str] = Field(default_factory=dict)
    reviewed_labels: list[LayoutLabel] = Field(default_factory=list)
    reviewed_at: datetime | None = None
    notes: str = ""

    @field_validator("reviewed_at")
    @classmethod
    def reviewed_at_is_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("reviewed_at must include a timezone")
        return value


class BoundingBox(StrictModel):
    x0: float = Field(ge=0)
    y0: float = Field(ge=0)
    x1: float = Field(ge=0)
    y1: float = Field(ge=0)

    @model_validator(mode="after")
    def coordinates_are_ordered(self) -> BoundingBox:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("bounding-box coordinates must be ordered")
        return self


class BlockKind(StrEnum):
    TEXT = "text"
    TITLE = "title"
    TABLE = "table"
    FORMULA = "formula"
    IMAGE = "image"
    UNKNOWN = "unknown"


class DocumentBlock(StrictModel):
    block_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    order: int = Field(ge=0)
    bbox: BoundingBox
    text: str = Field(min_length=1)
    kind: BlockKind = BlockKind.TEXT
    font_size: float | None = Field(default=None, ge=0)
    bold: bool = False
    source_block_ids: list[str] = Field(default_factory=list)


class DocumentPage(StrictModel):
    page_number: int = Field(ge=1)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    blocks: list[DocumentBlock] = Field(default_factory=list)

    @model_validator(mode="after")
    def block_order_is_valid(self) -> DocumentPage:
        if any(block.page_number != self.page_number for block in self.blocks):
            raise ValueError("block page_number must match its parent page")
        orders = [block.order for block in self.blocks]
        if orders != list(range(len(orders))):
            raise ValueError("block order must be contiguous and start at zero")
        return self


class DocumentSection(StrictModel):
    section_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    block_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def page_range_is_ordered(self) -> DocumentSection:
        if self.page_end < self.page_start:
            raise ValueError("page_end must be greater than or equal to page_start")
        return self


class WarningSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ParseWarningCode(StrEnum):
    MULTI_COLUMN = "multi_column"
    COMPLEX_LAYOUT = "complex_layout"
    TABLE_DETECTED = "table_detected"
    FORMULA_DETECTED = "formula_detected"
    IMAGE_SKIPPED = "image_skipped"
    LOW_TEXT = "low_text"
    OVERSIZED_BLOCK = "oversized_block"
    HEADING_FALLBACK = "heading_fallback"
    HEADER_FOOTER_REMOVED = "header_footer_removed"
    ADJACENT_DUPLICATE_REMOVED = "adjacent_duplicate_removed"


class ParseWarning(StrictModel):
    code: ParseWarningCode
    severity: WarningSeverity
    message: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)
    block_id: str | None = None
    details: dict[str, ScalarValue] = Field(default_factory=dict, max_length=32)

    _validate_details = field_validator("details")(_reject_sensitive_keys)


class ParseRequest(StrictModel):
    paper_id: str = Field(min_length=1)
    input_sha256: Sha256 = Field(pattern=r"^[a-f0-9]{64}$")
    max_pdf_bytes: int = Field(gt=0)
    max_pages: int = Field(gt=0)
    min_document_chars: int = Field(ge=0)
    full_width_ratio: float = Field(gt=0, le=1)


class ParsedDocument(StrictModel):
    schema_version: Literal["v1"] = "v1"
    paper_id: str = Field(min_length=1)
    input_sha256: Sha256 = Field(pattern=r"^[a-f0-9]{64}$")
    parser_name: str = Field(min_length=1)
    parser_version: str = Field(min_length=1)
    page_count: int = Field(ge=1)
    pages: list[DocumentPage] = Field(min_length=1)
    sections: list[DocumentSection] = Field(min_length=1)
    warnings: list[ParseWarning] = Field(default_factory=list)

    @model_validator(mode="after")
    def page_count_matches(self) -> ParsedDocument:
        if self.page_count != len(self.pages):
            raise ValueError("page_count must match pages")
        return self


class CleanedDocument(StrictModel):
    schema_version: Literal["v1"] = "v1"
    paper_id: str = Field(min_length=1)
    input_sha256: Sha256 = Field(pattern=r"^[a-f0-9]{64}$")
    cleaning_version: str = Field(min_length=1)
    pages: list[DocumentPage] = Field(min_length=1)
    sections: list[DocumentSection] = Field(min_length=1)
    warnings: list[ParseWarning] = Field(default_factory=list)
    statistics: dict[str, int] = Field(default_factory=dict)


class IngestionStatus(StrEnum):
    PLANNED = "planned"
    SUCCEEDED = "succeeded"
    SKIPPED = "skipped"
    FAILED = "failed"


class IngestionItemResult(StrictModel):
    paper_id: str = Field(min_length=1)
    status: IngestionStatus
    processing_version: str | None = None
    artifacts: dict[str, Sha256] = Field(default_factory=dict)
    warning_count: int = Field(default=0, ge=0)
    error: ErrorDetail | None = None
    reason: str | None = None

    @field_validator("artifacts")
    @classmethod
    def artifact_hashes_are_valid(cls, value: dict[str, Sha256]) -> dict[str, Sha256]:
        if any(
            len(item) != 64 or any(char not in "0123456789abcdef" for char in item)
            for item in value.values()
        ):
            raise ValueError("artifact hashes must be lowercase SHA-256 values")
        return value


class IngestionRunManifest(StrictModel):
    schema_version: Literal["v1"] = "v1"
    run_id: str = Field(min_length=1)
    mode: Literal["pilot", "selected", "all", "dry-run"]
    config_hash: Sha256 = Field(pattern=r"^[a-f0-9]{64}$")
    pilot_manifest_hash: Sha256 | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    started_at: datetime
    finished_at: datetime
    items: list[IngestionItemResult]

    @model_validator(mode="after")
    def timestamps_are_valid(self) -> IngestionRunManifest:
        for value in (self.started_at, self.finished_at):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("run timestamps must include a timezone")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        return self


class QualityReport(StrictModel):
    schema_version: Literal["v1"] = "v1"
    paper_id: str = Field(min_length=1)
    processing_version: str = Field(min_length=1)
    page_count: int = Field(ge=1)
    block_count: int = Field(ge=0)
    section_count: int = Field(ge=1)
    chunk_count: int = Field(ge=0)
    character_count: int = Field(ge=0)
    empty_text_pages: list[int] = Field(default_factory=list)
    warning_counts: dict[str, int] = Field(default_factory=dict)
    min_chunk_tokens: int = Field(default=0, ge=0)
    max_chunk_tokens: int = Field(default=0, ge=0)
    average_chunk_tokens: float = Field(default=0, ge=0)
    pages_with_chunks: list[int] = Field(default_factory=list)
    page_coverage_ratio: float = Field(default=0, ge=0, le=1)
    passed: bool = False
    failures: list[str] = Field(default_factory=list)
    artifact_hashes: dict[str, Sha256] = Field(default_factory=dict)
