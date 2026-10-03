"""Hybrid CLI 覆盖、边界、dry-run 和默认零 LLM 测试。"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from kg_crag.retrieval import load_hybrid_retrieval_config
from kg_crag.retrieval.hybrid_cli import _run, apply_overrides, parse_args, parse_filters

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_hybrid_cli_parses_filters_and_rejects_invalid_overrides() -> None:
    assert parse_filters(["paper_id=p1", "section=Methods"]) == {
        "paper_id": "p1",
        "section": "Methods",
    }
    with pytest.raises(ValueError, match="KEY=VALUE"):
        parse_filters(["invalid"])
    with pytest.raises(ValueError, match="unique"):
        parse_filters(["paper_id=p1", "paper_id=p2"])

    config = load_hybrid_retrieval_config(PROJECT_ROOT / "configs/retrieval.yaml")
    args = parse_args(["query", "--dense-weight", "0.9", "--sparse-weight", "0.9"])
    with pytest.raises(ValidationError, match="sum to 1.0"):
        apply_overrides(config, args)


async def test_hybrid_cli_dry_run_needs_no_index_model_or_service(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = await _run(
        [
            "What is GPT-4?",
            "--config",
            str(PROJECT_ROOT / "configs/retrieval.yaml"),
            "--method",
            "weighted",
            "--dense-weight",
            "0.4",
            "--sparse-weight",
            "0.6",
            "--no-reranker",
            "--dry-run",
        ]
    )
    assert result == 0
    output = capsys.readouterr().out
    assert '"llm_calls": 0' in output
    assert '"reranker": false' in output
