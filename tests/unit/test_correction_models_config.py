"""纠错公共契约、配置硬上限与运行身份测试。"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from kg_crag.correction.config import CorrectiveWorkflowConfig, load_corrective_workflow_config
from kg_crag.correction.identity import build_run_identity
from kg_crag.models import (
    BudgetLedger,
    BudgetLimit,
    BudgetUsage,
    ConditionKind,
    CoverageMatrix,
    Evidence,
    EvidenceRequirement,
    EvidenceSourceType,
    FacetKind,
    RequirementSource,
    SatisfactionCondition,
    SufficiencyAssessment,
)

HASH = "a" * 64


def _facet(name: str = "method") -> EvidenceRequirement:
    return EvidenceRequirement(
        facet_id=f"facet-{name.encode().hex()[:16]:0<16}",
        question_id="q1",
        kind=FacetKind.CONTENT,
        description=name,
        expected_evidence_types=[EvidenceSourceType.CHUNK],
        condition=SatisfactionCondition(kind=ConditionKind.TERMS, terms=[name]),
        source=RequirementSource.RULE,
    )


def _evidence(evidence_id: str = "e1") -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        content="method evidence",
        source_type=EvidenceSourceType.CHUNK,
        source_id="chunk-1",
        paper_id="paper-1",
    )


def test_models_reject_unknown_duplicate_and_inconsistent_values() -> None:
    with pytest.raises(ValidationError, match="extra"):
        _facet().__class__.model_validate({**_facet().model_dump(), "unknown": 1})
    with pytest.raises(ValidationError, match="internal sources"):
        _facet().__class__.model_validate(
            {**_facet().model_dump(), "expected_evidence_types": ["web"]}
        )
    with pytest.raises(ValidationError, match="unique"):
        CoverageMatrix(
            matrix_id="matrix-" + "a" * 16,
            rule_version="v1",
            facet_ids=[_facet().facet_id, _facet().facet_id],
            evidence_ids=[],
            entries=[],
        )
    with pytest.raises(ValidationError, match="sufficient flag"):
        SufficiencyAssessment(
            assessment_id="assessment-" + "a" * 16,
            sufficient=True,
            missing_required_facet_ids=[_facet().facet_id],
            facets=[],
            confidence=0.0,
        )


def test_budget_rejects_each_dimension_and_combined_token_overflow() -> None:
    with pytest.raises(ValidationError, match="combined token"):
        BudgetUsage(input_tokens=15_000, output_tokens=6_000)
    with pytest.raises(ValidationError, match="retrieval_rounds"):
        BudgetLedger(
            limit=BudgetLimit(),
            used=BudgetUsage(retrieval_rounds=2),
            reserved=BudgetUsage(retrieval_rounds=1),
        )


def test_config_hard_limits_paths_and_repository_file() -> None:
    config = load_corrective_workflow_config(Path("configs/default.yaml"))
    assert config.budget.retrieval_rounds == 2
    assert config.budget.llm_calls == 4
    with pytest.raises(ValidationError):
        CorrectiveWorkflowConfig.model_validate(
            {**config.model_dump(), "budget": {**config.budget.model_dump(), "llm_calls": 5}}
        )
    with pytest.raises(ValidationError, match="workspace"):
        CorrectiveWorkflowConfig.model_validate(
            {
                **config.model_dump(),
                "artifacts": {
                    **config.artifacts.model_dump(),
                    "run_root": "../outside",
                },
            }
        )


def test_run_identity_is_order_independent_and_binds_every_version() -> None:
    facets = [_facet("method"), _facet("result")]
    evidence = [_evidence("e1"), _evidence("e2")]
    kwargs = {
        "config_hash": HASH,
        "prompt_version": "prompt-v1",
        "model_revision": "model-v1",
        "corpus_snapshot": "corpus-v1",
        "dense_version": "dense-v1",
        "sparse_version": "sparse-v1",
        "graph_version": "graph-v1",
        "rules_version": "rules-v1",
        "coverage_version": "coverage-v1",
        "policy_version": "policy-v1",
    }
    first = build_run_identity(" question ", facets, evidence, **kwargs)
    second = build_run_identity(
        "question", list(reversed(facets)), list(reversed(evidence)), **kwargs
    )
    assert first == second
    changed = build_run_identity("question", facets, evidence, **{**kwargs, "graph_version": "v2"})
    assert changed.run_id != first.run_id
