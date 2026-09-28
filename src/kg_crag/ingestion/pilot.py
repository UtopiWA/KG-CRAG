"""试点清单加载、原始输入绑定与完整语料门禁。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from kg_crag.ingestion.raw import RawPaperInput
from kg_crag.models import LayoutLabel, PilotManifest, PilotReview, ReviewStatus


def manifest_hash(manifest: PilotManifest) -> str:
    payload = json.dumps(
        manifest.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def load_pilot_manifest(path: Path) -> PilotManifest:
    return PilotManifest.model_validate_json(path.read_text(encoding="utf-8"))


def resolve_pilot_inputs(
    manifest: PilotManifest,
    raw_inputs: list[RawPaperInput],
) -> list[RawPaperInput]:
    """按清单顺序解析输入，并验证 ID 与输入哈希全部一致。"""

    by_id = {item.paper_id: item for item in raw_inputs}
    selected: list[RawPaperInput] = []
    for paper in manifest.papers:
        raw = by_id.get(paper.paper_id)
        if raw is None:
            raise ValueError(f"pilot paper is not present in raw corpus: {paper.paper_id}")
        if raw.arxiv_id != paper.arxiv_id or raw.input_sha256 != paper.input_sha256:
            raise ValueError(f"pilot input identity or hash mismatch: {paper.paper_id}")
        selected.append(raw)
    return selected


def load_pilot_review(path: Path) -> PilotReview:
    return PilotReview.model_validate_json(path.read_text(encoding="utf-8"))


def validate_pilot_gate(
    review: PilotReview,
    manifest: PilotManifest,
    processing_versions: dict[str, str],
) -> None:
    """完整语料运行前要求人工结论、版本和标签同时匹配。"""

    if review.status is not ReviewStatus.PASSED:
        raise ValueError("pilot review has not passed")
    if review.manifest_hash != manifest_hash(manifest):
        raise ValueError("pilot review manifest hash does not match")
    expected_ids = {paper.paper_id for paper in manifest.papers}
    if (
        review.processing_versions
        != {
            key: processing_versions[key]
            for key in sorted(expected_ids)
            if key in processing_versions
        }
        or set(review.processing_versions) != expected_ids
    ):
        raise ValueError("pilot review processing versions do not match")
    if set(review.reviewed_labels) != set(LayoutLabel):
        raise ValueError("pilot review does not cover every required layout label")
