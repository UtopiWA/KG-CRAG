"""单篇与有界批次共用的论文摄取编排。"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal

from pydantic import ValidationError

from kg_crag.errors import KGCRAGError
from kg_crag.ingestion.base import DocumentParser
from kg_crag.ingestion.chunking import chunk_document
from kg_crag.ingestion.cleaning import clean_document
from kg_crag.ingestion.config import IngestionConfig, config_hash, processing_version
from kg_crag.ingestion.metadata import normalize_paper
from kg_crag.ingestion.quality import build_quality_report
from kg_crag.ingestion.raw import RawPaperInput
from kg_crag.ingestion.storage import ArtifactStore
from kg_crag.models import (
    ErrorCode,
    ErrorDetail,
    IngestionItemResult,
    IngestionRunManifest,
    IngestionStatus,
    ParseRequest,
)


class IngestionPipeline:
    """把校验后的单篇输入转换为版本化 Paper/Chunk 产物。"""

    def __init__(
        self,
        config: IngestionConfig,
        parser: DocumentParser,
        store: ArtifactStore,
    ) -> None:
        self.config = config
        self.parser = parser
        self.store = store
        self.configuration_hash = config_hash(config)

    def version_for(self, raw: RawPaperInput) -> str:
        return processing_version(
            raw.input_sha256,
            self.parser.name,
            self.parser.version,
            self.configuration_hash,
        )

    def process(self, raw: RawPaperInput, *, force: bool = False) -> IngestionItemResult:
        version = self.version_for(raw)
        # 处理版本同时绑定原始文件、解析器和配置；只有哈希复验通过才可复用产物。
        if (
            not force
            and (hashes := self.store.validate_existing(raw.paper_id, version)) is not None
        ):
            return IngestionItemResult(
                paper_id=raw.paper_id,
                status=IngestionStatus.SKIPPED,
                processing_version=version,
                artifacts=hashes,
                reason="matching verified artifacts already exist",
            )
        # 在各阶段重复核对论文和输入哈希，避免元数据、PDF 与派生产物串篇。
        paper = normalize_paper(raw.metadata, processing_version=version)
        if paper.paper_id != raw.paper_id:
            raise ValueError("normalized paper ID conflicts with the validated raw input")
        parsed = self.parser.parse(
            raw.pdf_path,
            ParseRequest(
                paper_id=paper.paper_id,
                input_sha256=raw.input_sha256,
                max_pdf_bytes=self.config.limits.max_pdf_bytes,
                max_pages=self.config.limits.max_pages,
                min_document_chars=self.config.limits.min_document_chars,
                full_width_ratio=self.config.parsing.full_width_ratio,
            ),
        )
        if parsed.paper_id != paper.paper_id or parsed.input_sha256 != raw.input_sha256:
            raise ValueError("parser result identity conflicts with the validated raw input")
        cleaned = clean_document(parsed, self.config.cleaning)
        chunking = chunk_document(
            cleaned,
            self.config.chunking,
            processing_version=version,
        )
        quality = build_quality_report(
            parsed,
            cleaned,
            chunking,
            processing_version=version,
        )
        # 质量门禁位于发布之前，未通过的中间结果不会成为可复用的正式产物。
        if not quality.passed:
            raise KGCRAGError(
                ErrorDetail(
                    code=ErrorCode.DATA,
                    message="automatic ingestion quality gate failed",
                    context={"paper_id": paper.paper_id},
                )
            )
        hashes = self.store.publish(paper, parsed, cleaned, chunking.chunks, quality)
        return IngestionItemResult(
            paper_id=paper.paper_id,
            status=IngestionStatus.SUCCEEDED,
            processing_version=version,
            artifacts=hashes,
            warning_count=len(cleaned.warnings) + len(chunking.warnings),
        )

    @staticmethod
    def _safe_failure(raw: RawPaperInput, error: Exception) -> IngestionItemResult:
        if isinstance(error, KGCRAGError):
            detail = error.detail
        elif isinstance(error, ValueError | ValidationError):
            detail = ErrorDetail(
                code=ErrorCode.DATA,
                message="input or derived data failed validation",
                context={"paper_id": raw.paper_id},
            )
        elif isinstance(error, OSError):
            detail = ErrorDetail(
                code=ErrorCode.INTERNAL,
                message="artifact I/O failed",
                retryable=True,
                context={"paper_id": raw.paper_id},
            )
        else:
            detail = ErrorDetail(
                code=ErrorCode.INTERNAL,
                message="unexpected ingestion failure",
                context={"paper_id": raw.paper_id},
            )
        return IngestionItemResult(
            paper_id=raw.paper_id,
            status=IngestionStatus.FAILED,
            error=detail,
        )

    def run_batch(
        self,
        inputs: list[RawPaperInput],
        *,
        mode: Literal["pilot", "selected", "all", "dry-run"],
        force: bool = False,
        pilot_manifest_hash: str | None = None,
    ) -> IngestionRunManifest:
        started = datetime.now(UTC)
        results: list[IngestionItemResult] = []
        for raw in inputs:
            try:
                results.append(self.process(raw, force=force))
            except Exception as error:  # 批次边界必须隔离单篇未知故障。
                results.append(self._safe_failure(raw, error))
        finished = datetime.now(UTC)
        run_id = f"{started:%Y%m%dT%H%M%SZ}-{self.configuration_hash[:8]}-{uuid.uuid4().hex[:8]}"
        manifest = IngestionRunManifest(
            run_id=run_id,
            mode=mode,
            config_hash=self.configuration_hash,
            pilot_manifest_hash=pilot_manifest_hash,
            started_at=started,
            finished_at=finished,
            items=results,
        )
        self.store.write_run_manifest(manifest)
        return manifest
