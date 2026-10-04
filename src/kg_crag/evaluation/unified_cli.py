"""统一评测的默认离线 CLI 编排。"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping
from pathlib import Path

from kg_crag.evaluation.artifacts import (
    ensure_formal_test_allowed,
    freeze_development_selection,
    load_test_selection,
    publish_formal_test_lock,
)
from kg_crag.evaluation.config import load_unified_evaluation_config
from kg_crag.evaluation.dataset import canonical_digest, load_unified_dataset
from kg_crag.evaluation.observation_collection import prepare_extractive_answer_validation
from kg_crag.evaluation.runner import (
    UnifiedEvaluationRunner,
    UnifiedStrategyAdapter,
    build_run_identity,
    fixture_adapters,
    recorded_adapters,
)
from kg_crag.models import (
    EvaluationDatasetManifest,
    EvaluationSplit,
    StrategyObservationSet,
    UnifiedStrategy,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "evaluation.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", choices=[item.value for item in EvaluationSplit], default="dev")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--fixture-mode",
        action="store_true",
        help="仅以确定性标注回放验收管线，不产生正式研究结果",
    )
    parser.add_argument(
        "--observations",
        type=Path,
        help="三个真实策略预先发布的统一 StrategyObservationSet",
    )
    parser.add_argument(
        "--selected-answer-strategy",
        choices=[item.value for item in UnifiedStrategy],
    )
    parser.add_argument(
        "--extractive-answer-validation",
        action="store_true",
        help="仅为入选策略按 Top-8 Evidence 执行零 LLM 抽取式回答/引用核算",
    )
    parser.add_argument(
        "--freeze-selection",
        choices=[item.value for item in UnifiedStrategy],
        help="以当前开发集报告一次性冻结入选系统",
    )
    parser.add_argument("--confirm-test-run", action="store_true")
    parser.add_argument("--tuning", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--index-version", default="offline-index-bundle-v1")
    parser.add_argument("--strategy-version", default="unified-strategy-adapters-v1")
    parser.add_argument("--model-version", default="no-model")
    parser.add_argument("--prompt-version", default="no-prompt")
    parser.add_argument("--metric-version", default="unified-metrics-v1")
    parser.add_argument("--judge-version", default="judge-disabled")
    return parser.parse_args(argv)


def _versions(args: argparse.Namespace) -> dict[str, str]:
    return {
        "index": args.index_version,
        "strategy": args.strategy_version,
        "model": args.model_version,
        "prompt": args.prompt_version,
        "metric": args.metric_version,
        "judge": args.judge_version,
    }


def _safe_summary(report_path: Path, *, run_id: str, items: int, failures: int) -> None:
    """只输出身份、计数和路径，不回显题目、Prompt 或 Evidence 正文。"""

    print(f"run_id={run_id}")
    print(f"items={items} failures={failures}")
    print(f"report={report_path}")


async def _run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_unified_evaluation_config(args.config)
    manifest_path = PROJECT_ROOT / config.manifest_path
    manifest = EvaluationDatasetManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    split = EvaluationSplit(args.split)
    selection = None
    lock_path = PROJECT_ROOT / config.test_lock_root / f"{manifest.dataset_version}.json"
    if split is EvaluationSplit.TEST:
        if args.selected_answer_strategy or args.freeze_selection:
            raise ValueError("test split cannot select or freeze a strategy")
        selection = load_test_selection(
            PROJECT_ROOT / config.selection_path,
            manifest=manifest,
        )
        ensure_formal_test_allowed(
            lock_path,
            confirmed=args.confirm_test_run,
            tuning_requested=args.tuning,
        )
    loaded_manifest, dev, test = load_unified_dataset(
        manifest_path,
        workspace_root=PROJECT_ROOT,
        near_duplicate_threshold=config.near_duplicate_threshold,
    )
    question_set = dev if split is EvaluationSplit.DEV else test
    print(
        f"validated dataset={loaded_manifest.dataset_version} split={split.value} "
        f"questions={len(question_set.questions)} hash={loaded_manifest.dataset_hash}"
    )
    if args.dry_run:
        print("dry-run: no evaluation artifacts written")
        return 0
    if args.fixture_mode == bool(args.observations):
        raise ValueError("select exactly one input mode: --fixture-mode or --observations")
    if args.fixture_mode and args.freeze_selection:
        raise ValueError("fixture-mode reports cannot freeze the development selection")
    if args.extractive_answer_validation and not args.selected_answer_strategy:
        raise ValueError("extractive answer validation requires --selected-answer-strategy")
    if args.extractive_answer_validation and args.fixture_mode:
        raise ValueError("extractive answer validation requires recorded observations")
    if args.fixture_mode and split is EvaluationSplit.TEST:
        raise ValueError("fixture-mode results cannot consume the one-time formal test run")
    selected = (
        UnifiedStrategy(args.selected_answer_strategy)
        if args.selected_answer_strategy
        else selection.selected_strategy
        if selection
        else None
    )
    adapters: Mapping[UnifiedStrategy, UnifiedStrategyAdapter]
    observation_set: StrategyObservationSet | None = None
    if args.fixture_mode:
        adapters = fixture_adapters()
    else:
        assert args.observations is not None
        observation_set = StrategyObservationSet.model_validate_json(
            args.observations.read_text(encoding="utf-8")
        )
        if args.extractive_answer_validation:
            observation_set = prepare_extractive_answer_validation(
                observation_set,
                question_set,
                selected_strategy=UnifiedStrategy(args.selected_answer_strategy),
            )
        adapters = recorded_adapters(
            observation_set,
            question_set,
            dataset_hash=loaded_manifest.dataset_hash,
        )
    runner = UnifiedEvaluationRunner(
        adapters,
        config,
        results_root=PROJECT_ROOT / config.results_root,
        workspace_root=PROJECT_ROOT,
        fixture_mode=args.fixture_mode,
    )
    versions = _versions(args)
    if observation_set is not None:
        # 观察内容本身会改变结果，必须进入身份，避免复用同名但内容已变化的文件。
        versions["strategy"] = f"{versions['strategy']}:{canonical_digest(observation_set)}"
    planned_identity = build_run_identity(
        loaded_manifest,
        question_set,
        config,
        version_bindings=versions,
        fixture_mode=args.fixture_mode,
        selected_answer_strategy=selected,
    )
    if selection and (
        planned_identity.config_hash != selection.config_hash
        or planned_identity.prompt_hash != selection.prompt_hash
        or planned_identity.model_hash != selection.model_hash
    ):
        raise ValueError("formal test versions differ from the frozen development selection")
    report = await runner.run(
        loaded_manifest,
        question_set,
        version_bindings=versions,
        selected_answer_strategy=selected,
    )
    report_path = PROJECT_ROOT / config.results_root / report.identity.run_id / "report.json"
    if args.freeze_selection:
        assert observation_set is not None
        freeze_development_selection(
            PROJECT_ROOT / config.selection_path,
            manifest=loaded_manifest,
            report=report,
            selected_strategy=UnifiedStrategy(args.freeze_selection),
            version_declarations={
                "index": args.index_version,
                "strategy": args.strategy_version,
                "observation_hash": canonical_digest(observation_set),
                "model": args.model_version,
                "prompt": args.prompt_version,
                "metric": args.metric_version,
                "judge": args.judge_version,
            },
            thresholds={
                "top_k": config.top_k,
                "near_duplicate_threshold": config.near_duplicate_threshold,
                "max_candidates_per_question": config.max_candidates_per_question,
                "max_loops_per_question": config.max_loops_per_question,
                "with_judge": config.with_judge,
            },
            workspace_root=PROJECT_ROOT,
        )
    if split is EvaluationSplit.TEST:
        assert selection is not None
        publish_formal_test_lock(
            lock_path,
            manifest=loaded_manifest,
            selection=selection,
            report=report,
            workspace_root=PROJECT_ROOT,
        )
    failures = sum(item.observation.error is not None for item in report.items)
    _safe_summary(
        report_path,
        run_id=report.identity.run_id,
        items=len(report.items),
        failures=failures,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_run(argv))
    except (OSError, ValueError, RuntimeError) as error:
        print(f"unified evaluation failed: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
