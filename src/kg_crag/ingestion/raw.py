"""只读发现并预校验原始论文 PDF 与元数据。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kg_crag.errors import KGCRAGError
from kg_crag.models import ErrorCode, ErrorDetail


@dataclass(frozen=True)
class RawPaperInput:
    """已通过输入边界校验的本地论文对。"""

    paper_id: str
    arxiv_id: str
    metadata_path: Path
    pdf_path: Path
    metadata: dict[str, Any]
    input_sha256: str
    size_bytes: int


def _data_error(message: str, paper_id: str | None = None) -> KGCRAGError:
    context = {"paper_id": paper_id} if paper_id else {}
    return KGCRAGError(
        ErrorDetail(code=ErrorCode.DATA, message=message, retryable=False, context=context)
    )


def _contained_regular_file(root: Path, value: str, *, expected_suffix: str) -> Path:
    """在读取内容前验证路径位于可信根目录且不是符号链接。"""

    root_resolved = root.resolve(strict=True)
    candidate = Path(value)
    if candidate.name != value and candidate.parent.name not in {"papers", "."}:
        raise _data_error("PDF path must identify a file in the configured raw directory")
    target = root / candidate.name
    try:
        resolved = target.resolve(strict=True)
        resolved.relative_to(root_resolved)
    except (OSError, ValueError) as exc:
        raise _data_error("PDF path is outside the configured raw directory") from exc
    if (
        target.is_symlink()
        or not resolved.is_file()
        or resolved.suffix.casefold() != expected_suffix
    ):
        raise _data_error("PDF path must point to a regular PDF file")
    return resolved


def validate_raw_record(
    metadata_path: Path,
    pdf_root: Path,
    *,
    max_pdf_bytes: int,
) -> RawPaperInput:
    """完整校验一条元数据和 PDF，不修改任何原始文件。"""

    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise _data_error("metadata path must point to a regular file")
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _data_error("metadata is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise _data_error("metadata root must be an object")
    paper_id = str(payload.get("paper_id", "")).strip()
    arxiv_id = str(payload.get("arxiv_id", "")).strip()
    if not paper_id or not arxiv_id or not str(payload.get("title", "")).strip():
        raise _data_error("metadata is missing paper_id, arxiv_id, or title", paper_id or None)
    pdf_value = payload.get("pdf_path")
    if not isinstance(pdf_value, str) or not pdf_value.strip():
        raise _data_error("metadata is missing pdf_path", paper_id)
    pdf_path = _contained_regular_file(pdf_root, pdf_value, expected_suffix=".pdf")
    size_bytes = pdf_path.stat().st_size
    if size_bytes > max_pdf_bytes:
        raise _data_error("PDF exceeds the configured size limit", paper_id)
    with pdf_path.open("rb") as handle:
        signature = handle.read(5)
        handle.seek(0)
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if signature != b"%PDF-":
        raise _data_error("file does not have a valid PDF signature", paper_id)
    acquisition = payload.get("acquisition")
    if not isinstance(acquisition, dict):
        raise _data_error("metadata is missing acquisition information", paper_id)
    if acquisition.get("size_bytes") != size_bytes:
        raise _data_error("PDF size does not match metadata", paper_id)
    if acquisition.get("sha256") != digest:
        raise _data_error("PDF SHA-256 does not match metadata", paper_id)
    return RawPaperInput(
        paper_id=paper_id,
        arxiv_id=arxiv_id,
        metadata_path=metadata_path.resolve(),
        pdf_path=pdf_path,
        metadata=payload,
        input_sha256=digest,
        size_bytes=size_bytes,
    )


def discover_raw_inputs(
    raw_root: Path,
    *,
    max_pdf_bytes: int,
    max_papers: int,
) -> list[RawPaperInput]:
    """按元数据文件名稳定排序，并在返回前校验全部输入。"""

    metadata_root = raw_root / "metadata"
    pdf_root = raw_root / "papers"
    if not metadata_root.is_dir() or not pdf_root.is_dir():
        raise _data_error("raw root must contain metadata and papers directories")
    paths = sorted(metadata_root.glob("*.json"))
    if not paths:
        raise _data_error("no raw metadata records were found")
    if len(paths) > max_papers:
        raise _data_error("raw corpus exceeds the configured paper limit")
    records = [validate_raw_record(path, pdf_root, max_pdf_bytes=max_pdf_bytes) for path in paths]
    ids = [record.paper_id for record in records]
    hashes = [record.input_sha256 for record in records]
    if len(ids) != len(set(ids)):
        raise _data_error("raw corpus contains duplicate paper identifiers")
    if len(hashes) != len(set(hashes)):
        raise _data_error("raw corpus contains duplicate PDF content")
    return records
