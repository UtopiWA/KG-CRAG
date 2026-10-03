"""运行单题有界证据纠错工作流。"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from kg_crag.correction.config import load_corrective_workflow_config
from kg_crag.correction.executor import ActionExecutor, RetrieverTool
from kg_crag.correction.identity import build_run_identity, stable_digest
from kg_crag.correction.requirements import (
    LLMRequirementProvider,
    RequirementCache,
    analyze_question,
)
from kg_crag.models import (
    BudgetLedger,
    CorrectionAction,
    CorrectionState,
    CorrectiveEvaluationQuestionSet,
    StopReason,
)
from kg_crag.providers import OpenAICompatibleLLMProvider
from kg_crag.retrieval.mock import MockRetriever
from kg_crag.settings import Settings
from kg_crag.workflow.artifacts import CheckpointStore, ContentAddressedCache
from kg_crag.workflow.corrective import run_corrective_workflow
from kg_crag.workflow.nodes import WorkflowDependencies

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question-id", default="q01", help="冻结压力集中的问题 ID")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=PROJECT_ROOT / "data/evaluation/corrective_dev_questions.json",
        help="离线问题与 Evidence fixture",
    )
    parser.add_argument("--dry-run", action="store_true", help="只展示身份与最坏预算")
    parser.add_argument("--online", action="store_true", help="显式启用已配置的真实 LLM")
    parser.add_argument("--confirm-budget", action="store_true", help="确认在线最坏预算")
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    questions = CorrectiveEvaluationQuestionSet.model_validate_json(
        args.fixture.read_text(encoding="utf-8")
    )
    try:
        question = next(
            item for item in questions.questions if item.question_id == args.question_id
        )
    except StopIteration:
        print(f"unknown question_id: {args.question_id}")
        return 1
    config = load_corrective_workflow_config(PROJECT_ROOT / "configs/default.yaml")
    print(
        json.dumps(
            {
                "question_id": question.question_id,
                "online": args.online,
                "max_retrieval_rounds": config.budget.retrieval_rounds,
                "max_llm_calls": config.budget.llm_calls,
                "max_tokens": config.budget.input_tokens + config.budget.output_tokens,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.dry_run:
        return 0
    if args.online and not args.confirm_budget:
        print("online mode requires --confirm-budget")
        return 1

    correction_evidence = [item for values in question.action_evidence.values() for item in values]
    retriever = MockRetriever(correction_evidence)
    executor = ActionExecutor(
        {action: RetrieverTool(retriever) for action in CorrectionAction},
        max_candidates=config.actions.max_candidates_per_action,
        bounds=config.actions,
        max_evidence_chars=config.coverage.max_evidence_chars,
    )
    provider = None
    prompt = ""
    if args.online:
        settings = Settings()
        if settings.llm_provider != "openai-compatible":
            print("online mode requires KG_CRAG_LLM_PROVIDER=openai-compatible")
            return 1
        llm = OpenAICompatibleLLMProvider(
            settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout_seconds=settings.llm_timeout_seconds,
        )
        provider = LLMRequirementProvider(llm, revision=settings.llm_model)
        prompt = (PROJECT_ROOT / config.facets.prompt_path).read_text(encoding="utf-8")
        config = config.model_copy(
            update={"facets": config.facets.model_copy(update={"allow_llm": True})}
        )
    rule_facets = list(analyze_question(question.question_id, question.question).facets)
    identity = build_run_identity(
        question.question,
        rule_facets,
        question.initial_evidence,
        config_hash=stable_digest(config),
        prompt_version=config.facets.prompt_version,
        model_revision=provider.revision if provider else "offline-mock-v1",
        corpus_snapshot=question.corpus_snapshot,
        dense_version=question.dense_version,
        sparse_version=question.sparse_version,
        graph_version=question.graph_version,
        rules_version=config.facets.version,
        coverage_version=config.coverage.version,
        policy_version=config.policy_version,
    )
    state = CorrectionState(
        identity=identity,
        question_id=question.question_id,
        question=question.question,
        facets=rule_facets,
        budget=BudgetLedger(limit=config.budget),
    )
    deps = WorkflowDependencies(
        config=config,
        executor=executor,
        initial_evidence=tuple(question.initial_evidence),
        requirement_provider=provider,
        requirement_prompt=prompt,
        requirement_cache=RequirementCache(
            PROJECT_ROOT / config.artifacts.cache_root / "requirements"
        ),
        artifact_cache=ContentAddressedCache(
            PROJECT_ROOT / config.artifacts.cache_root, workspace_root=PROJECT_ROOT
        ),
    )
    checkpoint_store = CheckpointStore(
        PROJECT_ROOT / config.artifacts.run_root, workspace_root=PROJECT_ROOT
    )
    try:
        result = await run_corrective_workflow(state, deps, checkpoint_store=checkpoint_store)
    except Exception as error:
        print(f"workflow failed: {type(error).__name__}")
        return 1
    print(result.model_dump_json(indent=2))
    if result.stop.reason is StopReason.SUFFICIENT:
        return 0
    if result.stop.reason in {
        StopReason.NO_POSITIVE_GAIN,
        StopReason.BUDGET_EXHAUSTED,
        StopReason.INTERNAL_KNOWLEDGE_MISSING,
    }:
        return 2
    return 1


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_run(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
