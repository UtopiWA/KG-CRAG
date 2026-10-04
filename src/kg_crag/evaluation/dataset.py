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
    EvaluationEvidenceCatalog,
    EvaluationQuestionOrigin,
    EvaluationQuestionSourceCatalog,
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
    evidence_hash: str,
    question_sources_hash: str,
    dev_hash: str,
    test_hash: str,
) -> str:
    return canonical_digest(
        {
            "dataset_version": dataset_version,
            "corpus_snapshot": corpus_snapshot,
            "evidence_version": evidence_version,
            "evidence_hash": evidence_hash,
            "question_sources_hash": question_sources_hash,
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
    evidence_catalog: EvaluationEvidenceCatalog,
    question_source_catalog: EvaluationQuestionSourceCatalog,
    workspace_root: Path | None = None,
    near_duplicate_threshold: float = 0.92,
) -> None:
    """在任何评测调用前完成规模、版本、哈希、压力类型与泄漏校验。"""

    if dev.split is not EvaluationSplit.DEV or test.split is not EvaluationSplit.TEST:
        raise ValueError("unified dataset files use the wrong split")
    if {manifest.dataset_version, dev.dataset_version, test.dataset_version} != {
        manifest.dataset_version
    }:
        raise ValueError("unified dataset version drifted")
    if (
        evidence_catalog.dataset_version != manifest.dataset_version
        or evidence_catalog.evidence_version != manifest.evidence_version
    ):
        raise ValueError("unified Evidence catalog version drifted")
    if question_source_catalog.dataset_version != manifest.dataset_version:
        raise ValueError("unified question source catalog version drifted")
    if len(dev.questions) != manifest.dev_count or len(test.questions) != manifest.test_count:
        raise ValueError("unified dataset question counts drifted")
    if len(dev.questions) + len(test.questions) > 100:
        raise ValueError("unified dataset exceeds the absolute question limit")
    dev_hash = canonical_digest(dev)
    test_hash = canonical_digest(test)
    evidence_hash = canonical_digest(evidence_catalog)
    question_sources_hash = canonical_digest(question_source_catalog)
    if dev_hash != manifest.dev_hash or test_hash != manifest.test_hash:
        raise ValueError("unified dataset content hash drifted")
    if evidence_hash != manifest.evidence_hash:
        raise ValueError("unified Evidence catalog hash drifted")
    if question_sources_hash != manifest.question_sources_hash:
        raise ValueError("unified question source catalog hash drifted")
    expected_dataset_hash = dataset_digest(
        dataset_version=manifest.dataset_version,
        corpus_snapshot=manifest.corpus_snapshot,
        evidence_version=manifest.evidence_version,
        evidence_hash=evidence_hash,
        question_sources_hash=question_sources_hash,
        dev_hash=dev_hash,
        test_hash=test_hash,
    )
    if expected_dataset_hash != manifest.dataset_hash:
        raise ValueError("unified dataset manifest hash drifted")
    all_questions = [*dev.questions, *test.questions]
    evidence_by_id = {item.evidence_id: item for item in evidence_catalog.records}
    source_by_id = {item.question_id: item for item in question_source_catalog.records}
    referenced_evidence: set[str] = set()
    for question in all_questions:
        if question.normalized_question != normalize_question(question.question):
            raise ValueError(f"normalized question drifted: {question.question_id}")
        question_evidence = set(question.relevant_evidence)
        referenced_evidence.update(question_evidence)
        if missing := question_evidence - evidence_by_id.keys():
            raise ValueError(
                f"question {question.question_id} relevant_evidence is unresolved: "
                f"{sorted(missing)}"
            )
        invalid_sources = {
            evidence_id: evidence_by_id[evidence_id].source_group_id
            for evidence_id in question_evidence
            if evidence_by_id[evidence_id].source_group_id not in question.source_group_ids
        }
        if invalid_sources:
            raise ValueError(
                f"question {question.question_id} source_group_ids do not cover Evidence: "
                f"{invalid_sources}"
            )
        try:
            source = source_by_id[question.question_id]
        except KeyError as error:
            raise ValueError(
                f"question {question.question_id} has no question source record"
            ) from error
        if source.question != question.question:
            raise ValueError(f"question {question.question_id} source text drifted")
        if set(source.evidence_ids) != question_evidence:
            raise ValueError(f"question {question.question_id} source Evidence drifted")
        if source.origin is EvaluationQuestionOrigin.FIXTURE_MIGRATION:
            if question.source_fixture != source.origin_locator:
                raise ValueError(f"question {question.question_id} fixture locator drifted")
            if workspace_root is not None:
                _validate_fixture_question(workspace_root, source.origin_locator, question)
        else:
            expected = f"{manifest.question_sources_path}#{question.question_id}"
            if question.source_fixture != expected:
                raise ValueError(f"question {question.question_id} derived source locator drifted")
            if not source.origin_locator.startswith("evidence:"):
                raise ValueError(f"question {question.question_id} has invalid derived origin")
            derived_evidence = {
                value
                for value in source.origin_locator.removeprefix("evidence:").split(",")
                if value
            }
            if derived_evidence != question_evidence:
                raise ValueError(
                    f"question {question.question_id} derived Evidence locator drifted"
                )
    catalog_ids = set(evidence_by_id)
    if unused := catalog_ids - referenced_evidence:
        raise ValueError(f"unified Evidence catalog contains unused records: {sorted(unused)}")
    identifiers = [item.question_id for item in all_questions]
    if extra_sources := source_by_id.keys() - set(identifiers):
        raise ValueError(
            f"unified question source catalog contains unused records: {sorted(extra_sources)}"
        )
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


def _validate_fixture_question(
    workspace_root: Path,
    locator: str,
    question: UnifiedEvaluationQuestion,
) -> None:
    """确认迁移来源中的题号和原始问题文本确实与当前题一致。"""

    try:
        relative_path, source_id = locator.rsplit("#", maxsplit=1)
    except ValueError as error:
        raise ValueError(f"question {question.question_id} has invalid fixture locator") from error
    source_path = _contained(workspace_root, relative_path)
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        candidates = payload["questions"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"question {question.question_id} fixture cannot be loaded") from error
    matched = [item for item in candidates if str(item.get("question_id")) == source_id]
    if len(matched) != 1 or str(matched[0].get("question")) != question.question:
        raise ValueError(f"question {question.question_id} does not match its source fixture")


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
    evidence_path = _contained(workspace_root, manifest.evidence_path)
    question_sources_path = _contained(workspace_root, manifest.question_sources_path)
    dev_path = _contained(workspace_root, manifest.dev_path)
    test_path = _contained(workspace_root, manifest.test_path)
    evidence_catalog = EvaluationEvidenceCatalog.model_validate_json(
        evidence_path.read_text(encoding="utf-8")
    )
    question_source_catalog = EvaluationQuestionSourceCatalog.model_validate_json(
        question_sources_path.read_text(encoding="utf-8")
    )
    dev = UnifiedQuestionSet.model_validate_json(dev_path.read_text(encoding="utf-8"))
    test = UnifiedQuestionSet.model_validate_json(test_path.read_text(encoding="utf-8"))
    validate_unified_dataset(
        manifest,
        dev,
        test,
        evidence_catalog=evidence_catalog,
        question_source_catalog=question_source_catalog,
        workspace_root=workspace_root,
        near_duplicate_threshold=near_duplicate_threshold,
    )
    return manifest, dev, test


def write_json_atomic(path: Path, value: BaseModel) -> None:
    """供迁移工具写入经过 JSON 回读校验的确定性文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    # Windows 受控目录可能拒绝创建点号前缀临时文件，仍在同目录内完成原子替换。
    temporary = path.with_name(f"{path.name}.tmp")
    payload = stable_json_bytes(value)
    try:
        temporary.write_bytes(payload)
        json.loads(temporary.read_text(encoding="utf-8"))
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
