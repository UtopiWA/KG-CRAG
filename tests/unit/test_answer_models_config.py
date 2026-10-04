"""回答反思公共契约、预算、配置和 Prompt 测试。"""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from kg_crag.answering.budget import AnswerBudgetManager
from kg_crag.answering.config import GroundedAnswerConfig, load_grounded_answer_config
from kg_crag.answering.prompt import load_structured_prompt
from kg_crag.models import (
    AnswerBudgetLedger,
    AnswerBudgetUsage,
    GroundedAnswerCandidate,
    GroundedCitation,
    GroundedClaim,
)


def test_grounded_contract_rejects_unknown_duplicate_and_wrong_provenance() -> None:
    with pytest.raises(ValidationError, match="extra"):
        GroundedClaim.model_validate(
            {
                "claim_id": "claim-" + "a" * 16,
                "text": "supported",
                "claim_type": "fact",
                "citation_ids": ["E1"],
                "facet_ids": ["facet-" + "b" * 16],
                "unknown": True,
            }
        )
    with pytest.raises(ValidationError, match="unique"):
        GroundedClaim(
            claim_id="claim-" + "a" * 16,
            text="supported",
            claim_type="fact",
            citation_ids=["E1", "E1"],
            facet_ids=["facet-" + "b" * 16],
        )
    with pytest.raises(ValidationError, match="external URL"):
        GroundedCitation(
            citation_id="E1",
            evidence_id="web-1",
            source_id="result-1",
            source_type="web",
            external=False,
            url="https://arxiv.org/abs/1234.5678",
        )


def test_candidate_requires_resolvable_citations() -> None:
    claim = GroundedClaim(
        claim_id="claim-" + "a" * 16,
        text="supported",
        claim_type="fact",
        citation_ids=["E1"],
        facet_ids=["facet-" + "b" * 16],
    )
    with pytest.raises(ValidationError, match="unknown citation"):
        GroundedAnswerCandidate(answer="supported", claims=[claim], confidence=0.8)


def test_answer_budget_reserves_settles_and_charges_failures() -> None:
    manager = AnswerBudgetManager(AnswerBudgetLedger())
    generation = AnswerBudgetUsage(answer_calls=1, input_tokens=3000, output_tokens=1000)
    first = manager.reserve(generation)
    assert first is not None
    manager.settle(first, AnswerBudgetUsage(answer_calls=1, input_tokens=100, output_tokens=50))
    second = manager.reserve(generation)
    assert second is not None
    manager.settle(second, failed=True)
    assert manager.ledger.used.answer_calls == 2
    assert manager.reserve(AnswerBudgetUsage(answer_calls=1)) is None
    critic = manager.reserve(AnswerBudgetUsage(critic_calls=1, input_tokens=1000))
    assert critic is not None
    manager.settle(critic, failed=True)
    assert manager.ledger.used.answer_calls + manager.ledger.used.critic_calls == 3


def test_answer_config_defaults_and_hard_bounds() -> None:
    config = load_grounded_answer_config(Path("configs/default.yaml"))
    assert config.enabled is False
    assert config.web.enabled is False
    assert config.web.provider == "disabled"
    assert config.budget.answer_calls == 2
    assert config.budget.input_tokens + config.budget.output_tokens == 12_000
    with pytest.raises(ValidationError, match="non-disabled"):
        GroundedAnswerConfig.model_validate(
            {**config.model_dump(), "web": {**config.web.model_dump(), "enabled": True}}
        )
    with pytest.raises(ValidationError, match="workspace"):
        GroundedAnswerConfig.model_validate(
            {
                **config.model_dump(),
                "artifacts": {**config.artifacts.model_dump(), "run_root": "../outside"},
            }
        )
    default_yaml = Path("configs/default.yaml").read_text(encoding="utf-8")
    assert "api_key" not in default_yaml.casefold()


def test_structured_prompts_require_placeholders_and_are_versioned(tmp_path: Path) -> None:
    answer = load_structured_prompt(
        Path("prompts/grounded-answer-v1.txt"),
        placeholders={"question", "facets", "evidence_context", "constraints"},
    )
    rendered = answer.render(
        max_chars=5000,
        question="q",
        facets="f",
        evidence_context="e",
        constraints="c",
    )
    assert "q" in rendered and len(answer.version) == 64
    changed = tmp_path / "changed.txt"
    changed.write_text(
        Path("prompts/grounded-answer-v1.txt").read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    assert (
        load_structured_prompt(
            changed,
            placeholders={"question", "facets", "evidence_context", "constraints"},
        ).version
        != answer.version
    )
    with pytest.raises(ValueError, match="exceeds"):
        answer.render(
            max_chars=1,
            question="q",
            facets="f",
            evidence_context="e",
            constraints="c",
        )


def test_search_result_requires_timezone() -> None:
    from kg_crag.models import SearchResult

    payload = {
        "provider_result_id": "r1",
        "title": "Paper",
        "url": "https://arxiv.org/abs/1234.5678",
        "excerpt": "Abstract",
        "source_type": "arxiv",
        "accessed_at": datetime.now(),
    }
    with pytest.raises(ValidationError, match="timezone"):
        SearchResult.model_validate(payload)
    assert SearchResult.model_validate({**payload, "accessed_at": datetime.now(UTC)})
