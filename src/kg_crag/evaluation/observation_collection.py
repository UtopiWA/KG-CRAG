"""为统一开发集生成零 LLM 的检索策略观察。"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping, Sequence

from kg_crag.models import (
    CorrectiveEvaluationQuestion,
    ErrorCode,
    ErrorDetail,
    EvaluationResourceUsage,
    EvaluationStageStatus,
    KnowledgeSufficiency,
    StrategyObservation,
    StrategyObservationSet,
    UnifiedEvaluationQuestion,
    UnifiedQuestionSet,
    UnifiedQuestionType,
    UnifiedStrategy,
)

RetrievalCallback = Callable[[str], Awaitable[Sequence[str]]]


class LocalRetrievalCallbacks:
    """封装本地 Dense、Sparse、Hybrid 与 Graph 检索入口。"""

    def __init__(
        self,
        *,
        dense: RetrievalCallback,
        sparse: RetrievalCallback,
        hybrid: RetrievalCallback,
        graph: RetrievalCallback,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.hybrid = hybrid
        self.graph = graph


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _covered(question: UnifiedEvaluationQuestion, evidence_ids: Sequence[str]) -> list[str]:
    available = set(evidence_ids)
    return [facet.facet_id for facet in question.facets if facet.is_covered_by(available)]


def _is_sufficient(question: UnifiedEvaluationQuestion, covered: Sequence[str]) -> bool:
    required = {facet.facet_id for facet in question.facets if facet.required}
    return question.knowledge_sufficiency is KnowledgeSufficiency.SUFFICIENT and required <= set(
        covered
    )


def _type_route(question: UnifiedEvaluationQuestion) -> str:
    types = set(question.question_types)
    if UnifiedQuestionType.MULTI_HOP in types or UnifiedQuestionType.RELATIONSHIP in types:
        return "graph"
    if UnifiedQuestionType.METRIC in types or UnifiedQuestionType.FACTOID in types:
        return "sparse"
    if UnifiedQuestionType.SYNTHESIS in types and UnifiedQuestionType.COMPARISON not in types:
        return "dense"
    return "hybrid"


async def _retrieve(
    callbacks: LocalRetrievalCallbacks,
    route: str,
    query: str,
) -> list[str]:
    callback = getattr(callbacks, route)
    return _unique(list(await callback(query)))


def _failure(
    question: UnifiedEvaluationQuestion,
    strategy: UnifiedStrategy,
    error: Exception,
    *,
    latency_ms: int,
) -> StrategyObservation:
    return StrategyObservation(
        question_id=question.question_id,
        strategy=strategy,
        status=EvaluationStageStatus.FAILED,
        usage=EvaluationResourceUsage(latency_ms=latency_ms),
        error=ErrorDetail(
            code=ErrorCode.INTERNAL,
            message="local observation collection failed",
            retryable=False,
            context={"error_type": type(error).__name__, "strategy": strategy.value},
        ),
    )


async def _literature_observation(
    question: UnifiedEvaluationQuestion,
    strategy: UnifiedStrategy,
    callbacks: LocalRetrievalCallbacks,
) -> StrategyObservation:
    """执行实际本地检索；facet 纠错只根据缺失 facet 改写一次查询。"""

    started = time.perf_counter()
    try:
        route = "hybrid" if strategy is not UnifiedStrategy.TYPE_ROUTER else _type_route(question)
        ranked = await _retrieve(callbacks, route, question.question)
        initial_covered = _covered(question, ranked)
        final = list(ranked)
        action_ids: list[str] = []
        loop_count = 0
        tool_calls = 1
        if strategy is UnifiedStrategy.FACET_CORRECTIVE and not _is_sufficient(
            question, initial_covered
        ):
            missing = [
                facet.description
                for facet in question.facets
                if facet.required and facet.facet_id not in initial_covered
            ]
            if missing:
                correction_route = _type_route(question)
                corrected_query = f"{question.question}\nRequired evidence: {'; '.join(missing)}"
                corrected = await _retrieve(callbacks, correction_route, corrected_query)
                final = _unique([*ranked, *corrected])
                action_ids = [f"facet-query-{correction_route}"]
                loop_count = 1
                tool_calls += 1
        final_covered = _covered(question, final)
        return StrategyObservation(
            question_id=question.question_id,
            strategy=strategy,
            status=EvaluationStageStatus.SUCCEEDED,
            ranked_evidence_ids=final,
            initial_covered_facet_ids=initial_covered,
            final_covered_facet_ids=final_covered,
            predicted_sufficient=_is_sufficient(question, final_covered),
            action_ids=action_ids,
            route=f"local-index:{route}",
            loop_count=loop_count,
            usage=EvaluationResourceUsage(
                tool_calls=tool_calls,
                latency_ms=max(0, round((time.perf_counter() - started) * 1000)),
            ),
        )
    except Exception as error:  # 单题失败必须固定进入分母，不能中断整个矩阵。
        return _failure(
            question,
            strategy,
            error,
            latency_ms=max(0, round((time.perf_counter() - started) * 1000)),
        )


def _controlled_observation(
    question: UnifiedEvaluationQuestion,
    strategy: UnifiedStrategy,
    source: CorrectiveEvaluationQuestion,
) -> StrategyObservation:
    """回放冻结纠错 fixture，并在 route 中显式保留其合成来源。"""

    initial = [item.evidence_id for item in source.initial_evidence]
    initial_covered = _covered(question, initial)
    final = list(initial)
    action_ids: list[str] = []
    if strategy is UnifiedStrategy.FACET_CORRECTIVE and source.action_evidence:
        relevant = set(question.relevant_evidence)
        ranked_actions = sorted(
            source.action_evidence.items(),
            key=lambda item: (
                -len({evidence.evidence_id for evidence in item[1]} & relevant),
                item[0].value,
            ),
        )
        action, evidence = ranked_actions[0]
        final = _unique([*final, *(item.evidence_id for item in evidence)])
        action_ids = [f"controlled-{action.value}"]
    final_covered = _covered(question, final)
    return StrategyObservation(
        question_id=question.question_id,
        strategy=strategy,
        status=EvaluationStageStatus.SUCCEEDED,
        ranked_evidence_ids=final,
        initial_covered_facet_ids=initial_covered,
        final_covered_facet_ids=final_covered,
        predicted_sufficient=_is_sufficient(question, final_covered),
        action_ids=action_ids,
        route=f"controlled-replay:{strategy.value}",
        loop_count=1 if action_ids else 0,
        usage=EvaluationResourceUsage(tool_calls=1 + int(bool(action_ids))),
    )


async def collect_strategy_observations(
    question_set: UnifiedQuestionSet,
    *,
    dataset_hash: str,
    split_hash: str,
    callbacks: LocalRetrievalCallbacks,
    corrective_questions: Mapping[str, CorrectiveEvaluationQuestion],
    source_versions: Mapping[str, str],
) -> StrategyObservationSet:
    """按固定题序生成完整的三策略观察集。"""

    observations: list[StrategyObservation] = []
    for question in question_set.questions:
        source_id = question.source_fixture.rsplit("#", maxsplit=1)[-1]
        controlled = (
            corrective_questions.get(source_id) if "corrective" in question.question_id else None
        )
        if "corrective" in question.question_id and controlled is None:
            raise ValueError(f"missing corrective fixture for {question.question_id}")
        for strategy in UnifiedStrategy:
            if controlled is not None:
                observation = _controlled_observation(question, strategy, controlled)
            else:
                observation = await _literature_observation(question, strategy, callbacks)
            observations.append(observation)
    return StrategyObservationSet(
        dataset_hash=dataset_hash,
        split_hash=split_hash,
        source_versions=dict(source_versions),
        observations=observations,
    )


def prepare_extractive_answer_validation(
    observation_set: StrategyObservationSet,
    question_set: UnifiedQuestionSet,
    *,
    selected_strategy: UnifiedStrategy,
    max_citations: int = 8,
) -> StrategyObservationSet:
    """仅为入选策略生成有界抽取式引用，并按冻结要点做事后核算。"""

    if max_citations <= 0 or max_citations > 20:
        raise ValueError("extractive answer citation limit must be between 1 and 20")
    questions = {item.question_id: item for item in question_set.questions}
    enriched: list[StrategyObservation] = []
    for observation in observation_set.observations:
        if observation.strategy is not selected_strategy:
            enriched.append(observation)
            continue
        question = questions.get(observation.question_id)
        if question is None:
            raise ValueError("observation references a question outside the selected split")
        citations = observation.ranked_evidence_ids[:max_citations]
        cited = set(citations)
        matched = [
            point.point_id for point in question.answer_points if cited & set(point.evidence_ids)
        ]
        enriched.append(
            observation.model_copy(
                update={
                    "matched_answer_point_ids": matched,
                    "cited_evidence_ids": citations,
                }
            )
        )
    versions = dict(observation_set.source_versions)
    versions["answer_validation"] = f"extractive-top-{max_citations}-v1"
    return observation_set.model_copy(
        update={"source_versions": versions, "observations": enriched}
    )
