"""有界 Judge 协议、内容寻址缓存与人工复核对照。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from kg_crag.evaluation.config import ABSOLUTE_MAX_JUDGE_ANSWERS, UnifiedEvaluationConfig
from kg_crag.evaluation.dataset import canonical_digest
from kg_crag.models import (
    ErrorCode,
    ErrorDetail,
    EvaluationItemResult,
    EvaluationResourceUsage,
    EvaluationStageStatus,
    HumanReview,
    JudgeRecord,
    JudgeVerdict,
    UnifiedEvaluationQuestion,
    UnifiedStrategy,
)
from kg_crag.workflow.artifacts import ContentAddressedCache


class JudgeProvider(Protocol):
    revision: str

    async def judge(
        self,
        question: UnifiedEvaluationQuestion,
        item: EvaluationItemResult,
        *,
        prompt_hash: str,
    ) -> JudgeVerdict: ...


def validate_judge_gate(
    config: UnifiedEvaluationConfig,
    items: Sequence[EvaluationItemResult],
    *,
    selected_strategy: UnifiedStrategy | None,
    confirmed_budget: bool,
) -> None:
    """必须在构造或调用在线 Provider 之前执行。"""

    if not config.online or not config.with_judge:
        raise ValueError("Judge requires explicit online and with_judge flags")
    if not confirmed_budget:
        raise ValueError("Judge requires explicit budget confirmation")
    if selected_strategy is None:
        raise ValueError("Judge requires one selected answer strategy")
    if not items:
        raise ValueError("Judge requires a predeclared non-empty review subset")
    if any(item.strategy is not selected_strategy for item in items):
        raise ValueError("Judge items must belong to the selected strategy only")
    limit = min(config.judge_max_answers, ABSOLUTE_MAX_JUDGE_ANSWERS)
    if len(items) > limit:
        raise ValueError(f"Judge answer count exceeds the hard limit: {limit}")
    if config.judge_rounds != 1 or config.judge_models != 1:
        raise ValueError("multi-round or multi-model Judge is not supported")


async def evaluate_with_judge(
    provider: JudgeProvider,
    questions: dict[str, UnifiedEvaluationQuestion],
    items: Sequence[EvaluationItemResult],
    *,
    prompt_hash: str,
    cache: ContentAddressedCache,
) -> list[JudgeRecord]:
    """执行一次 Judge；失败调用仍保留调用计数与结构化错误。"""

    records: list[JudgeRecord] = []
    for item in items:
        question = questions[item.question_id]
        cache_key = canonical_digest(
            {
                "provider": provider.revision,
                "prompt_hash": prompt_hash,
                "question": question.model_dump(mode="json"),
                "item": item.model_dump(mode="json"),
            }
        )
        cached = cache.load("unified-judge", cache_key, JudgeRecord)
        if cached is not None:
            records.append(cached)
            continue
        try:
            verdict = await provider.judge(question, item, prompt_hash=prompt_hash)
            if (
                verdict.question_id != item.question_id
                or verdict.strategy is not item.strategy
                or verdict.judge_version != provider.revision
                or verdict.prompt_hash != prompt_hash
            ):
                raise ValueError("Judge verdict identity mismatch")
            record = JudgeRecord(
                question_id=item.question_id,
                strategy=item.strategy,
                status=EvaluationStageStatus.SUCCEEDED,
                verdict=verdict,
                usage=verdict.usage,
            )
        except Exception as error:
            record = JudgeRecord(
                question_id=item.question_id,
                strategy=item.strategy,
                status=EvaluationStageStatus.FAILED,
                usage=EvaluationResourceUsage(judge_calls=1),
                error=ErrorDetail(
                    code=ErrorCode.EXTERNAL_SERVICE,
                    message=str(error)[:1000] or "Judge call failed",
                    retryable=True,
                ),
            )
        cache.save("unified-judge", cache_key, record)
        records.append(record)
    return records


def human_judge_agreement(
    records: Sequence[JudgeRecord],
    reviews: Sequence[HumanReview],
) -> float | None:
    """Judge 只与人工子集对照，不覆盖任何冻结真值。"""

    review_map = {(item.question_id, item.strategy): item for item in reviews}
    compared = [
        (record.verdict, review_map.get((record.question_id, record.strategy)))
        for record in records
        if record.verdict is not None
    ]
    valid = [(verdict, review) for verdict, review in compared if review is not None]
    if not valid:
        return None
    matches = sum(
        verdict.correct == review.correct
        and verdict.complete == review.complete
        and verdict.faithful == review.faithful
        and verdict.citation_correct == review.citation_correct
        for verdict, review in valid
    )
    return matches / len(valid)
