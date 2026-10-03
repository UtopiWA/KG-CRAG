"""Hybrid 评测命令行的零 LLM 默认值与预算边界测试。"""

import pytest

from kg_crag.evaluation.hybrid_cli import parse_args, validate_cli_args


def test_cli_defaults_to_zero_llm_smoke() -> None:
    args = parse_args(["--smoke", "--dry-run"])
    assert validate_cli_args(args, max_questions=40, smoke=5) == 5
    assert args.with_answer is False
    assert args.answer_budget == 0


def test_cli_requires_unique_selected_answer_configuration_and_exact_budget() -> None:
    missing = parse_args(["--with-answer", "--answer-budget", "5", "--smoke"])
    with pytest.raises(ValueError, match="selected"):
        validate_cli_args(missing, max_questions=40, smoke=5)

    wrong_budget = parse_args(
        [
            "--with-answer",
            "--selected-strategy",
            "fusion_rerank",
            "--answer-budget",
            "4",
            "--smoke",
        ]
    )
    with pytest.raises(ValueError, match="exactly match"):
        validate_cli_args(wrong_budget, max_questions=40, smoke=5)

    valid = parse_args(
        [
            "--with-answer",
            "--selected-strategy",
            "fusion_rerank",
            "--answer-budget",
            "5",
            "--smoke",
        ]
    )
    assert validate_cli_args(valid, max_questions=40, smoke=5) == 5
