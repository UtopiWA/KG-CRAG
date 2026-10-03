"""运行纠错工作流的冻结离线矩阵或受控在线 smoke。"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from kg_crag.correction.config import (
    load_corrective_evaluation_config,
    load_corrective_workflow_config,
)
from kg_crag.correction.requirements import LLMRequirementProvider, RequirementCache
from kg_crag.evaluation.corrective import evaluate_offline_matrix, load_corrective_questions
from kg_crag.providers import OpenAICompatibleLLMProvider
from kg_crag.settings import Settings
from kg_crag.workflow.artifacts import ContentAddressedCache

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="只展示规模与最坏预算")
    parser.add_argument("--smoke", action="store_true", help="仅运行前 5 题离线 smoke")
    parser.add_argument("--limit", type=int, help="离线调试题数，范围 1-40")
    parser.add_argument("--online", action="store_true", help="启用最多 3 题在线 smoke")
    parser.add_argument("--online-validation", action="store_true", help="启用最多 20 题在线验证")
    parser.add_argument("--confirm-budget", action="store_true", help="确认在线最坏预算")
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    workflow = load_corrective_workflow_config(PROJECT_ROOT / "configs/default.yaml")
    evaluation = load_corrective_evaluation_config(PROJECT_ROOT / "configs/evaluation.yaml")
    questions = load_corrective_questions(PROJECT_ROOT / evaluation.questions_path)
    online = bool(args.online or args.online_validation)
    maximum = (
        evaluation.online_max_questions
        if args.online_validation
        else evaluation.online_smoke_questions
    )
    if args.limit is not None and not 1 <= args.limit <= evaluation.max_questions:
        print("--limit must be in 1..40")
        return 1
    count = args.limit or (evaluation.smoke_questions if args.smoke else len(questions.questions))
    if online:
        count = min(count, maximum)
    worst = {
        "questions": count,
        "strategies": len(evaluation.strategies),
        "online": online,
        "max_llm_calls": count * workflow.budget.llm_calls if online else 0,
        "max_tokens": count * (workflow.budget.input_tokens + workflow.budget.output_tokens)
        if online
        else 0,
        "uses_answer_generation": False,
        "uses_web": False,
    }
    print(json.dumps(worst, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    if online and not args.confirm_budget:
        print("online evaluation requires --confirm-budget")
        return 1
    provider = None
    prompt = ""
    if online:
        settings = Settings()
        if settings.llm_provider != "openai-compatible":
            print("online mode requires KG_CRAG_LLM_PROVIDER=openai-compatible")
            return 1
        provider = LLMRequirementProvider(
            OpenAICompatibleLLMProvider(
                settings.llm_model,
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                timeout_seconds=settings.llm_timeout_seconds,
            ),
            revision=settings.llm_model,
        )
        prompt = (PROJECT_ROOT / workflow.facets.prompt_path).read_text(encoding="utf-8")
        workflow = workflow.model_copy(
            update={"facets": workflow.facets.model_copy(update={"allow_llm": True})}
        )
    report = await evaluate_offline_matrix(
        questions,
        workflow,
        limit=count,
        offline=not online,
        requirement_provider=provider,
        requirement_prompt=prompt,
        requirement_cache=RequirementCache(
            PROJECT_ROOT / workflow.artifacts.cache_root / "requirements"
        ),
        artifact_cache=ContentAddressedCache(
            PROJECT_ROOT / workflow.artifacts.cache_root, workspace_root=PROJECT_ROOT
        ),
        results_root=PROJECT_ROOT / evaluation.results_root,
        workspace_root=PROJECT_ROOT,
    )
    print(report.model_dump_json(indent=2))
    return 0 if all(item.error is None for item in report.items) else 2


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_run(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
