"""统一开发集观察生成的本地检索与受控回放测试。"""

import pytest

from kg_crag.evaluation.observation_collection import (
    LocalRetrievalCallbacks,
    collect_strategy_observations,
    prepare_extractive_answer_validation,
)
from kg_crag.models import (
    CorrectionAction,
    CorrectiveEvaluationQuestion,
    EvaluationAnswerPoint,
    EvaluationFacetTarget,
    EvaluationSplit,
    Evidence,
    EvidenceSourceType,
    GapType,
    KnowledgeSufficiency,
    StopReason,
    StressCategory,
    UnifiedEvaluationQuestion,
    UnifiedQuestionSet,
    UnifiedQuestionType,
    UnifiedStrategy,
)


def _question(
    question_id: str,
    *,
    evidence_id: str,
    source_fixture: str,
) -> UnifiedEvaluationQuestion:
    return UnifiedEvaluationQuestion(
        question_id=question_id,
        split=EvaluationSplit.DEV,
        question="Which paper reports the metric?",
        normalized_question="which paper reports the metric?",
        question_types=[UnifiedQuestionType.FACTOID],
        stress_category=StressCategory.CONTROL,
        knowledge_sufficiency=KnowledgeSufficiency.SUFFICIENT,
        answer_points=[
            EvaluationAnswerPoint(
                point_id=f"point-{evidence_id[-1] * 16}",
                description="The requested answer",
                evidence_ids=[evidence_id],
            )
        ],
        facets=[
            EvaluationFacetTarget(
                facet_id=f"facet-{evidence_id[-1] * 16}",
                description="The requested metric",
                target_evidence_ids=[evidence_id],
            )
        ],
        relevant_evidence={evidence_id: 3},
        minimum_sufficient_evidence_sets=[[evidence_id]],
        leakage_group_id=f"leak-{evidence_id}",
        source_group_ids=[f"source-{evidence_id}"],
        source_fixture=source_fixture,
    )


@pytest.mark.asyncio
async def test_collects_local_and_controlled_observations_without_using_gold_for_queries() -> None:
    literature = _question(
        "ueq-literature-fixture-a",
        evidence_id="evidence-a",
        source_fixture="fixture.json#literature-a",
    )
    controlled = _question(
        "ueq-corrective-dev-questions-q01",
        evidence_id="evidence-b",
        source_fixture="corrective.json#q01",
    )
    questions = UnifiedQuestionSet(
        dataset_version="fixture-v1",
        split=EvaluationSplit.DEV,
        questions=[literature, controlled],
    )
    queries: list[tuple[str, str]] = []

    async def empty(route: str, query: str) -> list[str]:
        queries.append((route, query))
        return []

    async def sparse(query: str) -> list[str]:
        queries.append(("sparse", query))
        return ["evidence-a"]

    callbacks = LocalRetrievalCallbacks(
        dense=lambda query: empty("dense", query),
        sparse=sparse,
        hybrid=lambda query: empty("hybrid", query),
        graph=lambda query: empty("graph", query),
    )
    source = CorrectiveEvaluationQuestion(
        question_id="q01",
        question=controlled.question,
        category=GapType.TERMINOLOGY_MISMATCH,
        corpus_snapshot="fixture-corpus",
        dense_version="dense-v1",
        sparse_version="sparse-v1",
        graph_version="graph-v1",
        expected_facet_ids=[controlled.facets[0].facet_id],
        action_evidence={
            CorrectionAction.SPARSE: [
                Evidence(
                    evidence_id="evidence-b",
                    content="Frozen corrective evidence",
                    source_type=EvidenceSourceType.CHUNK,
                    source_id="chunk-b",
                )
            ]
        },
        target_evidence_ids=["evidence-b"],
        allowed_actions=[CorrectionAction.SPARSE],
        expected_stop=StopReason.SUFFICIENT,
    )
    result = await collect_strategy_observations(
        questions,
        dataset_hash="1" * 64,
        split_hash="2" * 64,
        callbacks=callbacks,
        corrective_questions={"q01": source},
        source_versions={"mode": "fixture"},
    )

    assert len(result.observations) == 6
    by_key = {(item.question_id, item.strategy): item for item in result.observations}
    routed = by_key[(literature.question_id, UnifiedStrategy.TYPE_ROUTER)]
    assert routed.route == "local-index:sparse"
    assert routed.predicted_sufficient is True
    corrected = by_key[(literature.question_id, UnifiedStrategy.FACET_CORRECTIVE)]
    assert corrected.action_ids == ["facet-query-sparse"]
    assert corrected.predicted_sufficient is True
    replayed = by_key[(controlled.question_id, UnifiedStrategy.FACET_CORRECTIVE)]
    assert replayed.route == "controlled-replay:facet_corrective"
    assert replayed.ranked_evidence_ids == ["evidence-b"]
    assert replayed.predicted_sufficient is True
    assert all("evidence-a" not in query for _route, query in queries)

    enriched = prepare_extractive_answer_validation(
        result,
        questions,
        selected_strategy=UnifiedStrategy.FACET_CORRECTIVE,
        max_citations=1,
    )
    enriched_by_key = {(item.question_id, item.strategy): item for item in enriched.observations}
    selected = enriched_by_key[(literature.question_id, UnifiedStrategy.FACET_CORRECTIVE)]
    assert selected.cited_evidence_ids == ["evidence-a"]
    assert selected.matched_answer_point_ids == [literature.answer_points[0].point_id]
    assert enriched.source_versions["answer_validation"] == "extractive-top-1-v1"
    baseline = enriched_by_key[(literature.question_id, UnifiedStrategy.FIXED_HYBRID)]
    assert baseline.cited_evidence_ids == []
