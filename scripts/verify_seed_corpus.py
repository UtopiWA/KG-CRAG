#!/usr/bin/env python3
"""校验种子语料的元数据、PDF 签名、大小与校验和。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """解析输入位置，使校验逻辑也能用于 CI fixture。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=PROJECT_ROOT / "data/raw/metadata")
    parser.add_argument("--pdf-dir", type=Path, default=PROJECT_ROOT / "data/raw/papers")
    parser.add_argument("--allow-missing-pdf", action="store_true")
    return parser.parse_args()


def validate_record(metadata_path: Path, pdf_dir: Path, allow_missing_pdf: bool) -> list[str]:
    """返回单条元数据记录中所有可处理的校验错误。"""

    errors: list[str] = []
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read metadata: {exc}"]

    required = ("paper_id", "title", "arxiv_id", "source_url", "ingestion_version")
    for field in required:
        if not payload.get(field):
            errors.append(f"missing {field}")

    pdf_path_value = payload.get("pdf_path")
    if not pdf_path_value:
        if not allow_missing_pdf:
            errors.append("missing pdf_path")
        return errors

    pdf_path = pdf_dir / Path(str(pdf_path_value)).name
    if not pdf_path.is_file():
        errors.append(f"PDF does not exist: {pdf_path.name}")
        return errors
    data = pdf_path.read_bytes()
    if not data.startswith(b"%PDF-"):
        errors.append("invalid PDF signature")

    acquisition = payload.get("acquisition")
    if not isinstance(acquisition, dict):
        errors.append("missing acquisition record")
        return errors
    actual_sha256 = hashlib.sha256(data).hexdigest()
    if acquisition.get("sha256") != actual_sha256:
        errors.append("SHA-256 mismatch")
    if acquisition.get("size_bytes") != len(data):
        errors.append("size mismatch")
    return errors


def main() -> int:
    """检查每个元数据文件，遇到不完整或损坏数据时返回非零状态。"""

    args = parse_args()
    metadata_files = sorted(args.metadata_dir.glob("*.json"))
    if not metadata_files:
        print(f"No metadata records found in {args.metadata_dir}")
        return 1

    failure_count = 0
    for metadata_path in metadata_files:
        errors = validate_record(metadata_path, args.pdf_dir, args.allow_missing_pdf)
        if errors:
            failure_count += 1
            print(f"FAIL {metadata_path.name}: {'; '.join(errors)}")
        else:
            print(f"PASS {metadata_path.name}")
    print(f"Checked {len(metadata_files)} records; failures={failure_count}")
    return 1 if failure_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
