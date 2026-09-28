"""试点清单和完整语料门禁测试。"""

from pathlib import Path

import pytest

from kg_crag.ingestion.pilot import (
    load_pilot_manifest,
    manifest_hash,
    validate_pilot_gate,
)
from kg_crag.models import LayoutLabel, PilotReview, ReviewStatus


def test_checked_in_pilot_has_exactly_fifteen_unique_valid_records() -> None:
    manifest = load_pilot_manifest(Path("configs/pilot_corpus.json"))
    assert len(manifest.papers) == 15
    assert len({paper.paper_id for paper in manifest.papers}) == 15
    assert set(LayoutLabel) == {label for paper in manifest.papers for label in paper.layout_labels}


def test_gate_rejects_status_hash_version_and_label_mismatch() -> None:
    manifest = load_pilot_manifest(Path("configs/pilot_corpus.json"))
    versions = {paper.paper_id: "version" for paper in manifest.papers}
    valid = {
        "manifest_hash": manifest_hash(manifest),
        "status": ReviewStatus.PASSED,
        "processing_versions": versions,
        "reviewed_labels": list(LayoutLabel),
    }
    validate_pilot_gate(PilotReview.model_validate(valid), manifest, versions)
    for mutation in (
        {"status": ReviewStatus.PENDING},
        {"manifest_hash": "0" * 64},
        {"processing_versions": {}},
        {"reviewed_labels": [LayoutLabel.PLAIN_TEXT]},
    ):
        with pytest.raises(ValueError):
            validate_pilot_gate(
                PilotReview.model_validate({**valid, **mutation}), manifest, versions
            )
