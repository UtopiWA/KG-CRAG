"""应用服务投影、回放和本地产物读取测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from kg_crag.application.services import LocalDocumentService, ReplayCatalog
from kg_crag.ingestion.storage import paper_key
from kg_crag.models import (
    ApplicationMode,
    Author,
    IngestionRunRequest,
    Paper,
    QueryRequest,
    QueryResponse,
)


def _replay_path() -> Path:
    return Path(__file__).parents[2] / "src" / "kg_crag" / "application" / "replay_cases.json"


def test_replay_catalog_is_deterministic_and_request_scoped() -> None:
    catalog = ReplayCatalog(_replay_path())
    request = QueryRequest(
        question="展示纠错案例",
        mode=ApplicationMode.REPLAY,
        replay_case_id="corrected-gap",
    )
    first = catalog.query(request, request_id="request-a")
    second = catalog.query(request, request_id="request-b")
    assert first.trace_id == second.trace_id == "demo-trace-corrected-v1"
    assert first.request_id == "request-a"
    assert second.request_id == "request-b"
    assert first.actions[0].target_facet_ids == ["demo-facet-b"]
    assert catalog.cases["corrected-gap"]["response"].request_id == "fixture-validation"


def test_replay_response_requires_explicit_fixture_metadata() -> None:
    with pytest.raises(ValidationError, match="replay responses require"):
        QueryResponse(
            mode=ApplicationMode.REPLAY,
            request_id="request",
            trace_id="trace",
            answer="answer",
            stop_reason="accepted",
        )


def test_actual_ingestion_model_requires_confirmation_and_unique_ids() -> None:
    with pytest.raises(ValidationError, match="confirm=true"):
        IngestionRunRequest(paper_ids=["paper-a"], dry_run=False)
    with pytest.raises(ValidationError, match="unique"):
        IngestionRunRequest(paper_ids=["paper-a", "paper-a"])


@pytest.mark.asyncio
async def test_document_service_returns_metadata_and_ids_without_chunk_text(tmp_path: Path) -> None:
    paper_id = "arxiv:demo"
    version = "a" * 64
    destination = tmp_path / paper_key(paper_id) / version
    destination.mkdir(parents=True)
    paper = Paper(
        paper_id=paper_id,
        title="Demo Paper",
        abstract="bounded abstract",
        year=2026,
        authors=[Author(name="Researcher")],
        ingestion_version=version,
    )
    (destination / "paper.json").write_text(
        paper.model_dump_json(),
        encoding="utf-8",
    )
    (destination / "quality_report.json").write_text(
        json.dumps({"paper_id": paper_id, "passed": True}),
        encoding="utf-8",
    )
    (destination / "chunks.jsonl").write_text(
        json.dumps({"chunk_id": "chunk-demo", "text": "不得进入公共响应"}) + "\n",
        encoding="utf-8",
    )

    result = await LocalDocumentService(tmp_path).get(paper_id)

    assert result.chunk_ids == ["chunk-demo"]
    assert "不得进入" not in result.model_dump_json()
    assert result.authors == ["Researcher"]
