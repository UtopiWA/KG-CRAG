"""回答反思离线评测与显式在线 Provider 验收入口。"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from kg_crag.evaluation.grounded import (
    evaluate_online_grounded_probe,
    evaluate_recorded_grounded_questions,
    load_grounded_questions,
    validate_recorded_web_fixture,
)
from kg_crag.providers import OpenAICompatibleLLMProvider
from kg_crag.settings import Settings
from kg_crag.web import TavilySearchProvider

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "data/evaluation/grounded_answer_dev_questions.json",
    )
    parser.add_argument(
        "--web-fixture",
        type=Path,
        default=PROJECT_ROOT / "data/evaluation/fixtures/grounded-web-recorded.json",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "data/processed/evaluation/grounded-answer",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def validate_online_gate(args: argparse.Namespace, settings: Settings) -> int:
    """在构造任何网络客户端前验证双重门禁、题数和凭据。"""

    if not args.online:
        return int(args.limit or 0)
    if not args.confirm:
        raise ValueError("online evaluation requires both --online and --confirm")
    if args.limit is None or not 5 <= args.limit <= 10:
        raise ValueError("online evaluation requires an explicit --limit between 5 and 10")
    if settings.llm_provider != "openai-compatible" or not settings.llm_api_key:
        raise ValueError("online evaluation requires an OpenAI-compatible LLM credential")
    if settings.web_search_provider != "tavily" or not settings.web_search_api_key:
        raise ValueError("online evaluation requires a Tavily credential")
    return int(args.limit)


async def _run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    questions = load_grounded_questions(args.questions)
    fixture_queries = validate_recorded_web_fixture(args.web_fixture)
    settings = Settings()
    limit = validate_online_gate(args, settings)
    selected = limit or min(len(questions.questions), 20)
    budget = {
        "questions": selected,
        "max_model_calls": selected,
        "max_search_calls": min(selected, sum(item.web_eligible for item in questions.questions)),
        "max_input_output_tokens": selected * 12000,
        "max_web_results": selected * 5,
        "recorded_fixture_queries": fixture_queries,
    }
    print(json.dumps({"budget_preview": budget}, ensure_ascii=False, sort_keys=True))
    if args.dry_run:
        return 0
    if args.online:
        llm = OpenAICompatibleLLMProvider(
            settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout_seconds=settings.llm_timeout_seconds,
        )
        search = TavilySearchProvider(
            settings.web_search_api_key,
            timeout_seconds=settings.web_search_timeout_seconds,
        )
        report = await evaluate_online_grounded_probe(
            questions,
            llm=llm,
            search=search,
            limit=selected,
            provider_identity=(
                f"{settings.llm_provider}:{settings.llm_model}:tavily:{search.provider_version}"
            ),
            results_root=args.results_root,
            workspace_root=PROJECT_ROOT,
        )
    else:
        report = evaluate_recorded_grounded_questions(
            questions,
            limit=selected,
            results_root=args.results_root,
            workspace_root=PROJECT_ROOT,
        )
    print(report.model_dump_json())
    return int(report.metrics.failures > 0)


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_run(argv))
    except (OSError, ValueError) as error:
        print(f"grounded answer evaluation failed: {error}", file=sys.stderr)
        return 2
