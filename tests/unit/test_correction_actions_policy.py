"""动作白名单、执行门禁、效用策略与原子预算测试。"""

from concurrent.futures import ThreadPoolExecutor

import pytest

from kg_crag.correction.actions import action_catalog, build_action_request
from kg_crag.correction.budget import BudgetManager
from kg_crag.correction.config import CorrectiveWorkflowConfig
from kg_crag.correction.coverage import build_coverage_matrix
from kg_crag.correction.executor import ActionExecutor, RetrieverTool
from kg_crag.correction.policy import calculate_utility, candidate_actions, choose_action
from kg_crag.correction.requirements import analyze_question
from kg_crag.correction.sufficiency import assess_sufficiency
from kg_crag.errors import KGCRAGError
from kg_crag.models import (
    ActionEstimate,
    ActionRequest,
    BudgetLedger,
    BudgetLimit,
    BudgetUsage,
    CorrectionAction,
    Evidence,
    EvidenceRequirement,
    EvidenceSourceType,
    StopReason,
    SufficiencyAssessment,
)
from kg_crag.retrieval.mock import MockRetriever


def _evidence(evidence_id: str = "e1") -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        content="Voyager architecture uses a skill library",
        source_type=EvidenceSourceType.CHUNK,
        source_id=f"chunk-{evidence_id}",
        paper_id="paper-1",
    )


def _assessment() -> tuple[list[EvidenceRequirement], SufficiencyAssessment]:
    facets = list(analyze_question("q1", "What architecture does Voyager use?").facets)
    matrix = build_coverage_matrix(facets, [], rule_version="v1")
    return facets, assess_sufficiency(
        facets, [], matrix, min_support=0.45, min_sources=1, max_selected=8
    )


def test_catalog_contains_only_bounded_whitelisted_actions() -> None:
    config = CorrectiveWorkflowConfig()
    catalog = action_catalog(config.actions)
    assert set(catalog) == set(CorrectionAction)
    assert all("web" not in item.tool and "generate" not in item.tool for item in catalog.values())
    assert max(item.max_hops for item in catalog.values()) <= 3
    assert max(item.max_subquestions for item in catalog.values()) <= 3


def test_candidate_generation_is_targeted_deterministic_and_utility_is_manual() -> None:
    config = CorrectiveWorkflowConfig()
    facets, assessment = _assessment()
    first = candidate_actions("question", facets, assessment, BudgetLedger(), set(), config)
    second = candidate_actions("question", facets, assessment, BudgetLedger(), set(), config)
    assert first == second
    assert {item[0].action for item in first} <= {
        CorrectionAction.SPARSE,
        CorrectionAction.HYBRID,
        CorrectionAction.REWRITE,
    }
    estimate = ActionEstimate(
        expected_required_facet_gain=1.0,
        retrieval_rounds=1,
        llm_calls=1,
        input_tokens=100,
        output_tokens=50,
        latency_ms=100,
    )
    expected = 1.0 - 0.08 - 0.15 - 150 * 0.00002 - 100 * 0.00001
    assert calculate_utility(estimate, repeated=False, config=config) == pytest.approx(expected)


def test_policy_stops_on_repeat_or_no_budget() -> None:
    config = CorrectiveWorkflowConfig()
    facets, assessment = _assessment()
    selected = choose_action("question", facets, assessment, BudgetLedger(), set(), config)
    assert selected.selected is not None and selected.utility > 0
    stopped = choose_action(
        "question",
        facets,
        assessment,
        BudgetLedger(),
        {selected.selected.action_id},
        config,
        state_improved=False,
    )
    assert stopped.stop_reason is StopReason.NO_POSITIVE_GAIN
    exhausted = BudgetLedger(
        limit=BudgetLimit(), used=BudgetUsage(retrieval_rounds=2), reserved=BudgetUsage()
    )
    assert choose_action("question", facets, assessment, exhausted, set(), config).stop_reason is (
        StopReason.BUDGET_EXHAUSTED
    )


def test_budget_reservation_settlement_failure_and_concurrency() -> None:
    manager = BudgetManager(BudgetLedger(limit=BudgetLimit(retrieval_rounds=1)))
    estimate = ActionEstimate(retrieval_rounds=1, expected_required_facet_gain=1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        tokens = list(pool.map(lambda _index: manager.reserve(estimate), range(4)))
    reserved = [item for item in tokens if item is not None]
    assert len(reserved) == 1
    manager.settle(reserved[0], failed=True)
    assert manager.ledger.used.retrieval_rounds == 1
    assert manager.reserve(estimate) is None

    cached = BudgetManager(BudgetLedger())
    assert cached.ledger.used == BudgetUsage()


@pytest.mark.asyncio
async def test_executor_calls_only_bound_tool_and_rejects_missing_binding() -> None:
    config = CorrectiveWorkflowConfig()
    facets, _assessment_value = _assessment()
    request = build_action_request(CorrectionAction.SPARSE, "question", facets, config.actions)
    retriever = MockRetriever([_evidence()])
    executor = ActionExecutor(
        {CorrectionAction.SPARSE: RetrieverTool(retriever)}, max_candidates=20
    )
    result = await executor.execute(
        request, ActionEstimate(retrieval_rounds=1, candidates=20, expected_required_facet_gain=1)
    )
    assert result.evidence[0].evidence_id == "e1"
    assert len(retriever.calls) == 1

    over_limit = request.model_copy(update={"top_k": 21, "max_candidates": 21})
    with pytest.raises(ValueError, match="boundary"):
        await executor.execute(
            over_limit,
            ActionEstimate(retrieval_rounds=1, expected_required_facet_gain=1),
        )
    assert len(retriever.calls) == 1

    missing = request.model_copy(update={"action": CorrectionAction.GRAPH})
    with pytest.raises(KGCRAGError, match="no bound tool"):
        await executor.execute(
            ActionRequest.model_validate(missing.model_dump()),
            ActionEstimate(retrieval_rounds=1, expected_required_facet_gain=1),
        )
