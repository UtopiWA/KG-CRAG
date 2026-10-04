"""从已提交评测 fixture 迁移统一 60 题标注；默认只预览。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from kg_crag.evaluation.dataset import (
    canonical_digest,
    dataset_digest,
    normalize_question,
    validate_unified_dataset,
    write_json_atomic,
)
from kg_crag.models import (
    EvaluationAnswerPoint,
    EvaluationDatasetManifest,
    EvaluationFacetTarget,
    EvaluationSplit,
    KnowledgeSufficiency,
    StressCategory,
    UnifiedEvaluationQuestion,
    UnifiedQuestionSet,
    UnifiedQuestionType,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "data" / "evaluation"
OUTPUT_ROOT = SOURCE_ROOT / "unified"
DATASET_VERSION = "unified-agent-qa-v1"


def _digest_id(prefix: str, value: str) -> str:
    return f"{prefix}-{hashlib.sha256(value.encode()).hexdigest()[:16]}"


def _load(name: str) -> list[dict[str, Any]]:
    payload = json.loads((SOURCE_ROOT / name).read_text(encoding="utf-8"))
    questions = payload.get("questions")
    if not isinstance(questions, list):
        raise ValueError(f"source fixture has no question list: {name}")
    return questions


def _common(
    source: str,
    item: dict[str, Any],
    *,
    split: EvaluationSplit,
) -> dict[str, Any]:
    source_id = str(item["question_id"])
    question = str(item["question"])
    stem = source.removesuffix(".json").replace("_", "-")
    return {
        "question_id": f"ueq-{stem}-{source_id}".lower(),
        "split": split,
        "question": question,
        "normalized_question": normalize_question(question),
        "leakage_group_id": f"leak-{stem}-{source_id}",
        "source_group_ids": [f"source-{stem}-{source_id}"],
        "source_fixture": f"data/evaluation/{source}#{source_id}",
    }


def _target_annotations(
    question_id: str,
    description: str,
    targets: list[str],
) -> tuple[list[EvaluationAnswerPoint], list[EvaluationFacetTarget]]:
    facet_id = _digest_id("facet", f"{question_id}:core")
    if not targets:
        return [], [
            EvaluationFacetTarget(
                facet_id=facet_id,
                description=description,
                target_evidence_ids=[],
            )
        ]
    return [
        EvaluationAnswerPoint(
            point_id=_digest_id("point", f"{question_id}:core"),
            description=description,
            evidence_ids=targets,
        )
    ], [
        EvaluationFacetTarget(
            facet_id=facet_id,
            description=description,
            target_evidence_ids=targets,
        )
    ]


def _corrective_questions() -> list[UnifiedEvaluationQuestion]:
    source = "corrective_dev_questions.json"
    output: list[UnifiedEvaluationQuestion] = []
    type_map = {
        "comparison_side": UnifiedQuestionType.COMPARISON,
        "multi_hop_gap": UnifiedQuestionType.MULTI_HOP,
        "metric_missing": UnifiedQuestionType.METRIC,
        "conflict": UnifiedQuestionType.METRIC,
    }
    for item in _load(source):
        common = _common(source, item, split=EvaluationSplit.DEV)
        targets = [str(value) for value in item.get("target_evidence_ids", [])]
        category = StressCategory(str(item["category"]))
        knowledge = (
            KnowledgeSufficiency.INSUFFICIENT
            if category is StressCategory.INTERNAL_MISSING
            else KnowledgeSufficiency.CONFLICTING
            if category is StressCategory.CONFLICT
            else KnowledgeSufficiency.SUFFICIENT
        )
        answer_points, generated_facets = _target_annotations(
            common["question_id"],
            "回答该压力题的核心证据要求。",
            targets,
        )
        expected_facets = [str(value) for value in item.get("expected_facet_ids", [])]
        facets = [
            EvaluationFacetTarget(
                facet_id=facet_id,
                description="由纠错开发集冻结的必需 facet。",
                target_evidence_ids=targets,
            )
            for facet_id in expected_facets
        ] or generated_facets
        output.append(
            UnifiedEvaluationQuestion(
                **common,
                question_types=[type_map.get(category.value, UnifiedQuestionType.FACTOID)],
                stress_category=category,
                knowledge_sufficiency=knowledge,
                answer_points=answer_points,
                facets=facets,
                relevant_evidence={value: 3 for value in targets},
                minimum_sufficient_evidence_sets=(
                    [targets] if knowledge is KnowledgeSufficiency.SUFFICIENT else []
                ),
            )
        )
    return output


def _hybrid_questions() -> list[UnifiedEvaluationQuestion]:
    source = "hybrid_dev_questions.json"
    output: list[UnifiedEvaluationQuestion] = []
    for item in _load(source)[:20]:
        common = _common(source, item, split=EvaluationSplit.DEV)
        targets = [str(value) for value in item.get("target_chunk_ids", [])]
        answer_points, facets = _target_annotations(
            common["question_id"],
            str(item.get("rationale") or "目标 Chunk 应支持问题答案。"),
            targets,
        )
        output.append(
            UnifiedEvaluationQuestion(
                **common,
                question_types=[UnifiedQuestionType.FACTOID],
                stress_category=StressCategory.CONTROL,
                knowledge_sufficiency=KnowledgeSufficiency.SUFFICIENT,
                answer_points=answer_points,
                facets=facets,
                relevant_evidence={value: 3 for value in targets},
                minimum_sufficient_evidence_sets=[targets],
            )
        )
    return output


def _graph_questions() -> list[UnifiedEvaluationQuestion]:
    source = "graph_dev_questions.json"
    output: list[UnifiedEvaluationQuestion] = []
    for item in _load(source):
        common = _common(source, item, split=EvaluationSplit.TEST)
        targets = [str(value) for value in item.get("relevant_fact_ids", [])]
        category = str(item.get("category", "relationship"))
        multi_hop = "multi" in category
        knowledge = (
            KnowledgeSufficiency.SUFFICIENT if targets else KnowledgeSufficiency.INSUFFICIENT
        )
        answer_points, facets = _target_annotations(
            common["question_id"],
            "冻结图事实应支持关系或多跳答案。",
            targets,
        )
        output.append(
            UnifiedEvaluationQuestion(
                **common,
                question_types=[
                    UnifiedQuestionType.MULTI_HOP if multi_hop else UnifiedQuestionType.RELATIONSHIP
                ],
                stress_category=(
                    StressCategory.INTERNAL_MISSING
                    if not targets
                    else StressCategory.MULTI_HOP_GAP
                    if multi_hop
                    else StressCategory.CONTROL
                ),
                knowledge_sufficiency=knowledge,
                answer_points=answer_points,
                facets=facets,
                relevant_evidence={value: 3 for value in targets},
                minimum_sufficient_evidence_sets=(
                    [targets] if knowledge is KnowledgeSufficiency.SUFFICIENT else []
                ),
            )
        )
    return output


def _missing_questions() -> list[UnifiedEvaluationQuestion]:
    source = "grounded_answer_dev_questions.json"
    output: list[UnifiedEvaluationQuestion] = []
    for item in _load(source)[:8]:
        common = _common(source, item, split=EvaluationSplit.TEST)
        _answer_points, facets = _target_annotations(
            common["question_id"],
            "该迁移子集不提供可作为内部真值的 Evidence。",
            [],
        )
        output.append(
            UnifiedEvaluationQuestion(
                **common,
                question_types=[UnifiedQuestionType.SYNTHESIS],
                stress_category=StressCategory.INTERNAL_MISSING,
                knowledge_sufficiency=KnowledgeSufficiency.INSUFFICIENT,
                answer_points=[],
                facets=facets,
                relevant_evidence={},
                minimum_sufficient_evidence_sets=[],
            )
        )
    return output


def build_dataset() -> tuple[EvaluationDatasetManifest, UnifiedQuestionSet, UnifiedQuestionSet]:
    dev = UnifiedQuestionSet(
        dataset_version=DATASET_VERSION,
        split=EvaluationSplit.DEV,
        questions=[*_corrective_questions(), *_hybrid_questions()],
    )
    test = UnifiedQuestionSet(
        dataset_version=DATASET_VERSION,
        split=EvaluationSplit.TEST,
        questions=[*_graph_questions(), *_missing_questions()],
    )
    dev_hash = canonical_digest(dev)
    test_hash = canonical_digest(test)
    corpus_snapshot = "unified-source-fixtures-v1"
    evidence_version = "unified-evidence-annotations-v1"
    manifest = EvaluationDatasetManifest(
        dataset_version=DATASET_VERSION,
        corpus_snapshot=corpus_snapshot,
        evidence_version=evidence_version,
        dev_path="data/evaluation/unified/dev.json",
        test_path="data/evaluation/unified/test.json",
        dev_count=len(dev.questions),
        test_count=len(test.questions),
        dev_hash=dev_hash,
        test_hash=test_hash,
        dataset_hash=dataset_digest(
            dataset_version=DATASET_VERSION,
            corpus_snapshot=corpus_snapshot,
            evidence_version=evidence_version,
            dev_hash=dev_hash,
            test_hash=test_hash,
        ),
        annotation_guide_version="unified-annotation-guide-v1",
        reviewed=False,
        review_note=(
            "候选集已通过来源、稳定 ID、facet/Evidence 引用、压力类型、规模与跨划分泄漏的"
            "自动校验；仍需人工逐题复核后才能用于真实开发选择或正式测试。"
        ),
    )
    validate_unified_dataset(manifest, dev, test)
    return manifest, dev, test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="写入冻结数据；默认仅 dry-run")
    parser.add_argument("--force", action="store_true", help="允许覆盖已存在的统一数据")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest, dev, test = build_dataset()
    print(
        f"dataset={manifest.dataset_version} dev={len(dev.questions)} "
        f"test={len(test.questions)} hash={manifest.dataset_hash}"
    )
    if not args.write:
        print("dry-run: no files written")
        return 0
    destinations = [
        OUTPUT_ROOT / "manifest.json",
        OUTPUT_ROOT / "dev.json",
        OUTPUT_ROOT / "test.json",
    ]
    if not args.force and any(path.exists() for path in destinations):
        raise FileExistsError("unified dataset exists; use --force for an intentional rebuild")
    write_json_atomic(destinations[1], dev)
    write_json_atomic(destinations[2], test)
    write_json_atomic(destinations[0], manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
