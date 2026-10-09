"""规则优先的证据需求生成与受控 LLM 候选边界。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import Field, ValidationError

from kg_crag.correction.budget import BudgetManager
from kg_crag.correction.config import FacetRulesConfig
from kg_crag.correction.identity import stable_digest
from kg_crag.ingestion.storage import stable_json_bytes
from kg_crag.models import (
    BudgetLedger,
    BudgetUsage,
    ConditionKind,
    EvidenceRequirement,
    EvidenceSourceType,
    FacetKind,
    RequirementSource,
    SatisfactionCondition,
)
from kg_crag.models.domain import StrictModel
from kg_crag.providers import LLMProvider, UsageAwareLLMProvider


class RequirementCandidate(StrictModel):
    """模型仅能提出领域候选，运行身份和预算由代码注入。"""

    kind: FacetKind
    description: str = Field(min_length=1, max_length=500)
    required: bool = True
    expected_evidence_types: list[EvidenceSourceType] = Field(min_length=1, max_length=3)
    condition: SatisfactionCondition
    target_entity: str | None = Field(default=None, max_length=200)
    confidence: float = Field(ge=0.0, le=1.0)


class RequirementBatch(StrictModel):
    candidates: list[RequirementCandidate] = Field(default_factory=list, max_length=20)


class RequirementProviderResponse(StrictModel):
    batch: RequirementBatch
    usage: BudgetUsage
    usage_estimated: bool = False


class RequirementProvider(Protocol):
    revision: str

    async def propose(
        self, question: str, *, system_prompt: str
    ) -> RequirementProviderResponse: ...


class MockRequirementProvider:
    """确定性测试替身，不执行网络或隐藏重试。"""

    revision = "mock-requirements-v1"

    def __init__(self, response: RequirementProviderResponse) -> None:
        self.response = response
        self.calls = 0

    async def propose(self, question: str, *, system_prompt: str) -> RequirementProviderResponse:
        self.calls += 1
        return self.response.model_copy(deep=True)


class LLMRequirementProvider:
    """把通用 LLM 文本限制为一次严格 JSON 响应。"""

    def __init__(self, llm: LLMProvider, *, revision: str) -> None:
        self._llm = llm
        self.revision = revision

    async def propose(self, question: str, *, system_prompt: str) -> RequirementProviderResponse:
        schema = json.dumps(
            RequirementBatch.model_json_schema(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        bounded_system_prompt = f"{system_prompt.strip()}\nJSON Schema:\n{schema}".strip()
        if isinstance(self._llm, UsageAwareLLMProvider):
            generation = await self._llm.generate_with_usage(
                question,
                system_prompt=bounded_system_prompt,
                temperature=0.0,
            )
            raw = generation.content
            input_tokens = generation.input_tokens
            output_tokens = generation.output_tokens
            usage_estimated = not generation.has_actual_usage
        else:
            raw = await self._llm.generate(
                question,
                system_prompt=bounded_system_prompt,
                temperature=0.0,
            )
            input_tokens = None
            output_tokens = None
            usage_estimated = True
        normalized = raw.strip()
        # 兼容常见的单层 JSON 围栏，围栏外文本仍会被严格拒绝并回退规则。
        if normalized.startswith("```") and normalized.endswith("```"):
            first_newline = normalized.find("\n")
            if first_newline > 0 and normalized[3:first_newline].strip().casefold() in {"", "json"}:
                normalized = normalized[first_newline + 1 : -3].strip()
        batch = RequirementBatch.model_validate_json(normalized)
        input_tokens = input_tokens or max(1, (len(bounded_system_prompt) + len(question) + 2) // 3)
        output_tokens = output_tokens or max(1, (len(raw) + 2) // 3)
        return RequirementProviderResponse(
            batch=batch,
            usage=BudgetUsage(
                llm_calls=1,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                context_chars=len(bounded_system_prompt) + len(question),
            ),
            usage_estimated=usage_estimated,
        )


@dataclass(frozen=True)
class RuleAnalysis:
    facets: tuple[EvidenceRequirement, ...]
    complex: bool
    confidence: float


def _normalized_terms(text: str) -> list[str]:
    terms = re.findall(r"[A-Za-z][A-Za-z0-9_.+-]*|[\u4e00-\u9fff]{2,}", text.casefold())
    return list(dict.fromkeys(terms))[:20]


def _needs_cross_lingual_facets(text: str) -> bool:
    """中文问题包含可区分的拉丁项目名时，需生成文献侧英文证据词。"""

    if re.search(r"[\u4e00-\u9fff]", text) is None:
        return False
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_.+-]{2,}", text)
    return any(
        token.isupper()
        or any(char.isupper() for char in token[1:])
        or any(char.isdigit() or char in "-_.+" for char in token)
        for token in tokens
    )


def _facet(
    question_id: str,
    kind: FacetKind,
    description: str,
    terms: list[str],
    *,
    target: str | None = None,
    evidence_types: list[EvidenceSourceType] | None = None,
    required: bool = True,
    source: RequirementSource = RequirementSource.RULE,
    confidence: float = 1.0,
) -> EvidenceRequirement:
    normalized_description = " ".join(description.split())
    identity = stable_digest(
        {"question_id": question_id, "kind": kind, "description": normalized_description}
    )[:16]
    selector = list(
        dict.fromkeys(
            terms or _normalized_terms(normalized_description) or [normalized_description]
        )
    )
    condition_kind = {
        FacetKind.ENTITY: ConditionKind.ENTITY,
        FacetKind.RELATIONSHIP: ConditionKind.TERMS,
        FacetKind.COMPARISON: ConditionKind.TERMS,
        FacetKind.METRIC: ConditionKind.NUMERIC,
        FacetKind.MULTI_HOP: ConditionKind.GRAPH_PATH,
    }.get(kind, ConditionKind.TERMS)
    types = evidence_types or [EvidenceSourceType.CHUNK, EvidenceSourceType.GRAPH]
    return EvidenceRequirement(
        facet_id=f"facet-{identity}",
        question_id=question_id,
        kind=kind,
        description=normalized_description,
        required=required,
        expected_evidence_types=types,
        condition=SatisfactionCondition(
            kind=condition_kind,
            terms=selector[:20],
            min_term_matches=min(2, len(selector))
            if kind in {FacetKind.COMPARISON, FacetKind.METRIC}
            else 1,
            max_hops=3 if kind is FacetKind.MULTI_HOP else None,
        ),
        source=source,
        target_entity=target,
        confidence=confidence,
    )


def analyze_question(question_id: str, question: str) -> RuleAnalysis:
    """以稳定规则识别比较、关系、多跳、指标和普通内容需求。"""

    normalized = " ".join(question.strip().split())
    if not normalized:
        raise ValueError("question must not be empty")
    folded = normalized.casefold()
    comparison = any(
        token in folded for token in (" compare ", " vs ", "versus", "比较", "对比", "区别")
    )
    multi_hop = any(
        token in folded for token in ("multi-hop", "multihop", "多跳", "关系链", "如何通过")
    )
    relationship = multi_hop or any(
        token in folded for token in ("relationship", "related", "关系", "引用", "依赖")
    )
    metric = bool(
        re.search(r"\b(metric|accuracy|f1|recall|latency)\b|指标|准确率|召回率|延迟", folded)
    )
    terms = _normalized_terms(normalized)
    facets: list[EvidenceRequirement] = []

    if comparison:
        # 比较题按最多两个对象和已识别维度展开；无法可靠分割时仍保留两侧占位需求。
        sides = re.split(r"\b(?:vs\.?|versus|and)\b|与|和|以及", normalized, maxsplit=2)
        objects = [part.strip(" ,，。?") for part in sides if part.strip(" ,，。?")][:2]
        while len(objects) < 2:
            objects.append(f"comparison-side-{len(objects) + 1}")
        dimensions = ["architecture", "results"]
        if metric:
            dimensions.append("metric")
        for target in objects:
            for dimension in dimensions:
                facets.append(
                    _facet(
                        question_id,
                        FacetKind.METRIC if dimension == "metric" else FacetKind.COMPARISON,
                        f"{target}: {dimension}",
                        [*_normalized_terms(target), dimension],
                        target=target,
                    )
                )
    else:
        kind = (
            FacetKind.MULTI_HOP
            if multi_hop
            else FacetKind.RELATIONSHIP
            if relationship
            else FacetKind.METRIC
            if metric
            else FacetKind.CONTENT
        )
        facets.append(_facet(question_id, kind, normalized, terms))

    complex_question = (
        comparison or multi_hop or len(terms) > 12 or _needs_cross_lingual_facets(normalized)
    )
    confidence = 0.9 if not complex_question else 0.7
    return RuleAnalysis(
        tuple(sorted(facets, key=lambda item: item.facet_id)), complex_question, confidence
    )


class RequirementCache:
    """按请求身份存储严格响应；写入使用同目录原子替换。"""

    def __init__(self, root: Path) -> None:
        self.root = root

    def load(self, key: str) -> RequirementProviderResponse | None:
        path = self.root / f"{key}.json"
        try:
            return RequirementProviderResponse.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, ValueError):
            return None

    def save(self, key: str, response: RequirementProviderResponse) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{key}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_bytes(stable_json_bytes(response))
        RequirementProviderResponse.model_validate_json(temporary.read_text(encoding="utf-8"))
        temporary.replace(path)


def _materialize_candidates(
    question_id: str, candidates: list[RequirementCandidate], config: FacetRulesConfig
) -> list[EvidenceRequirement]:
    if len(candidates) > config.max_facets:
        raise ValueError("provider returned too many facets")
    materialized = [
        _facet(
            question_id,
            item.kind,
            item.description,
            item.condition.terms,
            target=item.target_entity,
            evidence_types=item.expected_evidence_types,
            required=item.required,
            source=RequirementSource.LLM,
            confidence=item.confidence,
        ).model_copy(update={"condition": item.condition})
        for item in candidates
    ]
    if any(len(item.description) > config.max_description_chars for item in materialized):
        raise ValueError("provider facet description is too long")
    ids = [item.facet_id for item in materialized]
    if len(ids) != len(set(ids)):
        raise ValueError("provider returned duplicate facets")
    return sorted(materialized, key=lambda item: item.facet_id)


async def generate_requirements(
    question_id: str,
    question: str,
    config: FacetRulesConfig,
    *,
    provider: RequirementProvider | None = None,
    system_prompt: str = "",
    cache: RequirementCache | None = None,
) -> tuple[list[EvidenceRequirement], BudgetUsage, str]:
    """简单题直接返回规则结果，模型失败时原子回退且只调用一次。"""

    analysis = analyze_question(question_id, question)
    should_call = (
        config.allow_llm
        and provider is not None
        and (analysis.complex or analysis.confidence < config.confidence_threshold)
    )
    if not should_call:
        return list(analysis.facets), BudgetUsage(), "rule"
    assert provider is not None
    cache_key = stable_digest(
        {
            "question": question,
            "rules": config.version,
            "prompt": config.prompt_version,
            "provider": provider.revision,
        }
    )
    cached = cache.load(cache_key) if cache else None
    if cached is not None:
        candidates = _materialize_candidates(question_id, cached.batch.candidates, config)
        return candidates, BudgetUsage(), "llm_cache"
    try:
        response = await provider.propose(question, system_prompt=system_prompt)
        candidates = _materialize_candidates(question_id, response.batch.candidates, config)
        if not candidates:
            raise ValueError("provider returned no facets")
        if cache:
            cache.save(cache_key, response)
        return candidates, response.usage, "llm"
    except Exception:
        # Provider 的超时、外部错误和结构错误均只失败一次，不隐藏重试或切换。
        return list(analysis.facets), BudgetUsage(), "rule_fallback"


async def generate_budgeted_requirements(
    question_id: str,
    question: str,
    config: FacetRulesConfig,
    ledger: BudgetLedger,
    *,
    provider: RequirementProvider | None = None,
    system_prompt: str = "",
    cache: RequirementCache | None = None,
) -> tuple[list[EvidenceRequirement], BudgetLedger, str]:
    """在模型调用前预留预算，并返回已经结算的 facet 与账本。"""

    analysis = analyze_question(question_id, question)
    will_call_provider = bool(
        config.allow_llm
        and provider is not None
        and (analysis.complex or analysis.confidence < config.confidence_threshold)
    )
    if not will_call_provider:
        return list(analysis.facets), ledger.model_copy(deep=True), "rule"

    manager = BudgetManager(ledger)
    reservation = manager.reserve(
        BudgetUsage(
            llm_calls=1,
            input_tokens=4000,
            output_tokens=2000,
            context_chars=10_000,
            latency_ms=5000,
        )
    )
    if reservation is None:
        return list(analysis.facets), manager.ledger, "rule_budget_fallback"

    facets, usage, source = await generate_requirements(
        question_id,
        question,
        config,
        provider=provider,
        system_prompt=system_prompt,
        cache=cache,
    )
    if source == "rule_fallback":
        manager.settle(reservation, failed=True)
    else:
        manager.settle(reservation, usage)
    return facets, manager.ledger, source
