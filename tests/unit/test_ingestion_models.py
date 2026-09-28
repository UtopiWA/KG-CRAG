"""论文摄取公共模型的边界测试。"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from kg_crag.models import (
    BlockKind,
    BoundingBox,
    DocumentBlock,
    DocumentPage,
    IngestionRunManifest,
    LayoutLabel,
    ParseWarning,
    ParseWarningCode,
    PilotManifest,
    PilotPaper,
    WarningSeverity,
)

SHA = "a" * 64


def _pilot_paper(index: int, labels: list[LayoutLabel]) -> PilotPaper:
    return PilotPaper(
        paper_id=f"paper-{index}",
        arxiv_id=f"2401.{index:05d}",
        input_sha256=SHA,
        layout_labels=labels,
        selection_reason="人工检查版面后纳入试点。",
    )


def test_pilot_manifest_requires_unique_ids_and_all_layout_labels() -> None:
    papers = [_pilot_paper(index, [LayoutLabel.PLAIN_TEXT]) for index in range(10)]
    papers[0] = _pilot_paper(0, list(LayoutLabel))
    assert len(PilotManifest(papers=papers).papers) == 10

    papers[0] = _pilot_paper(0, [LayoutLabel.PLAIN_TEXT])
    with pytest.raises(ValidationError, match="coverage"):
        PilotManifest(papers=papers)


@pytest.mark.parametrize(
    ("bbox", "message"),
    [
        ({"x0": 4, "y0": 0, "x1": 3, "y1": 1}, "coordinates"),
        ({"x0": -1, "y0": 0, "x1": 3, "y1": 1}, "greater than or equal"),
    ],
)
def test_bounding_box_rejects_invalid_coordinates(bbox: dict[str, int], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        BoundingBox.model_validate(bbox)


def test_page_rejects_non_contiguous_order_and_mismatched_page() -> None:
    block = DocumentBlock(
        block_id="b1",
        page_number=2,
        order=1,
        bbox=BoundingBox(x0=0, y0=0, x1=10, y1=10),
        text="content",
        kind=BlockKind.TEXT,
    )
    with pytest.raises(ValidationError, match="page_number"):
        DocumentPage(page_number=1, width=100, height=100, blocks=[block])


def test_warning_rejects_unknown_fields_and_sensitive_details() -> None:
    payload = {
        "code": ParseWarningCode.LOW_TEXT,
        "severity": WarningSeverity.WARNING,
        "message": "文本过少",
    }
    with pytest.raises(ValidationError, match="Extra inputs"):
        ParseWarning.model_validate({**payload, "unknown": True})
    with pytest.raises(ValidationError, match="sensitive"):
        ParseWarning.model_validate({**payload, "details": {"api_token": "secret"}})


def test_run_manifest_requires_hash_and_aware_ordered_timestamps() -> None:
    with pytest.raises(ValidationError):
        IngestionRunManifest(
            run_id="run",
            mode="pilot",
            config_hash="short",
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
            items=[],
        )
