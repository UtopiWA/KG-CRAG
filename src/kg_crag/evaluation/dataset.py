"""统一评测数据加载、内容哈希与泄漏门禁。"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

from pydantic import BaseModel

from kg_crag.ingestion.storage import stable_json_bytes
from kg_crag.models import (
    EvaluationDatasetManifest,
    EvaluationSplit,
    StressCategory,
    UnifiedEvaluationQuestion,
    UnifiedQuestionSet,
)

_REQUIRED_STRESS = set(StressCategory) - {StressCategory.CONTROL}


def canonical_digest(value: BaseModel | dict[str, object] | str) -> str:
    """对模型或普通 JSON 值计算排序稳定的 SHA-256。"""

    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def normalize_question(text: str) -> str:
    """以轻量确定性规则生成泄漏检查用文本。"""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[\w]+", normalize_question(text), flags=re.UNICODE))


def token_jaccard(left: str, right: str) -> float:
    left_tokens = _token_set(left)
    right_tokens = _token_set(right)
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def dataset_digest(
    *,
    dataset_version: str,
    corpus_snapshot: str,
    evidence_version: str,
    dev_hash: str,
    test_hash: str,
) -> str:
    return canonical_digest(
        {
            "dataset_version": dataset_version,
            "corpus_snapshot": corpus_snapshot,
            "evidence_version": evidence_version,
            "dev_hash": dev_hash,
            "test_hash": test_hash,
        }
    )


def _contained(workspace_root: Path, relative: str) -> Path:
    root = workspace_root.resolve()
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError("evaluation dataset path escapes workspace root")
    return path


def validate_cross_split_leakage(
    dev: list[UnifiedEvaluationQuestion],
    test: list[UnifiedEvaluationQuestion],
    *,
    threshold: float,
) -> None:
    """拒绝显式分组、答案来源或高相似文本跨越 dev/test。"""

    dev_groups = {item.leakage_group_id for item in dev}
    test_groups = {item.leakage_group_id for item in test}
    if overlap := dev_groups & test_groups:
        raise ValueError(f"cross-split leakage groups: {sorted(overlap)}")
    dev_sources = {group for item in dev for group in item.source_group_ids}
    test_sources = {group for item in test for group in item.source_group_ids}
    if overlap := dev_sources & test_sources:
        raise ValueError(f"cross-split answer source groups: {sorted(overlap)}")
    for left in dev:
        for right in test:
            similarity = token_jaccard(left.normalized_question, right.normalized_question)
            if similarity >= threshold:
                raise ValueError(
                    "cross-split near duplicate: "
                    f"{left.question_id}/{right.question_id} ({similarity:.3f})"
                )


def validate_unified_dataset(
    manifest: EvaluationDatasetManifest,
    dev: UnifiedQuestionSet,
    test: UnifiedQuestionSet,
    *,
    near_duplicate_threshold: float = 0.92,
) -> None:
    """在任何评测调用前完成规模、版本、哈希、压力类型与泄漏校验。"""

    if dev.split is not EvaluationSplit.DEV or test.split is not EvaluationSplit.TEST:
        raise ValueError("unified dataset files use the wrong split")
    if {manifest.dataset_version, dev.dataset_version, test.dataset_version} != {
        manifest.dataset_version
    }:
        raise ValueError("unified dataset version drifted")
    if len(dev.questions) != manifest.dev_count or len(test.questions) != manifest.test_count:
        raise ValueError("unified dataset question counts drifted")
    if len(dev.questions) + len(test.questions) > 100:
        raise ValueError("unified dataset exceeds the absolute question limit")
    dev_hash = canonical_digest(dev)
    test_hash = canonical_digest(test)
    if dev_hash != manifest.dev_hash or test_hash != manifest.test_hash:
        raise ValueError("unified dataset content hash drifted")
    expected_dataset_hash = dataset_digest(
        dataset_version=manifest.dataset_version,
        corpus_snapshot=manifest.corpus_snapshot,
        evidence_version=manifest.evidence_version,
        dev_hash=dev_hash,
        test_hash=test_hash,
    )
    if expected_dataset_hash != manifest.dataset_hash:
        raise ValueError("unified dataset manifest hash drifted")
    all_questions = [*dev.questions, *test.questions]
    for question in all_questions:
        if question.normalized_question != normalize_question(question.question):
            raise ValueError(f"normalized question drifted: {question.question_id}")
    identifiers = [item.question_id for item in all_questions]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("question IDs must be unique across all splits")
    covered = {item.stress_category for item in all_questions}
    if missing := _REQUIRED_STRESS - covered:
        raise ValueError(f"required stress categories are missing: {sorted(missing)}")
    validate_cross_split_leakage(
        dev.questions,
        test.questions,
        threshold=near_duplicate_threshold,
    )


def load_unified_dataset(
    manifest_path: Path,
    *,
    workspace_root: Path,
    near_duplicate_threshold: float = 0.92,
) -> tuple[EvaluationDatasetManifest, UnifiedQuestionSet, UnifiedQuestionSet]:
    """加载并完整校验冻结清单；失败时不返回部分数据。"""

    manifest = EvaluationDatasetManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    dev_path = _contained(workspace_root, manifest.dev_path)
    test_path = _contained(workspace_root, manifest.test_path)
    dev = UnifiedQuestionSet.model_validate_json(dev_path.read_text(encoding="utf-8"))
    test = UnifiedQuestionSet.model_validate_json(test_path.read_text(encoding="utf-8"))
    validate_unified_dataset(
        manifest,
        dev,
        test,
        near_duplicate_threshold=near_duplicate_threshold,
    )
    return manifest, dev, test


def write_json_atomic(path: Path, value: BaseModel) -> None:
    """供迁移工具写入经过 JSON 回读校验的确定性文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    payload = stable_json_bytes(value)
    try:
        temporary.write_bytes(payload)
        json.loads(temporary.read_text(encoding="utf-8"))
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
