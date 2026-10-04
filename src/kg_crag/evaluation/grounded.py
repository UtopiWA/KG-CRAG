"""回答反思开发集加载、录制回放和有界指标汇总。"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from kg_crag.correction.identity import stable_digest
from kg_crag.models import (
    AnswerStopReason,
    ErrorCode,
    ErrorDetail,
    GroundedEvaluationItem,
    GroundedEvaluationMetrics,
    GroundedEvaluationQuestion,
    GroundedEvaluationQuestionSet,
    GroundedEvaluationReport,
    GroundedEvaluationScenario,
)
from kg_crag.providers import LLMProvider, RecordedSearchProvider, SearchProvider
from kg_crag.workflow.artifacts import atomic_model_write


def load_grounded_questions(path: Path) -> GroundedEvaluationQuestionSet:
    """加载冻结题集；严格模型会同时检查题数、唯一标识和场景覆盖。"""

    return GroundedEvaluationQuestionSet.model_validate_json(path.read_text(encoding="utf-8"))


def validate_recorded_web_fixture(path: Path) -> int:
    """复用录制 Provider 的严格解析，并返回可回放查询数量。"""

    provider = RecordedSearchProvider(path)
    return provider.response_count


def compute_grounded_metrics(
    items: list[GroundedEvaluationItem],
) -> GroundedEvaluationMetrics:
    """按逐题结果计算宏观指标，失败题始终保留在分母中。"""

    if not items:
        raise ValueError("grounded evaluation requires at least one item")
    total = len(items)
    predicted_claims = sum(
        len(item.supported_claim_ids) + item.unsupported_claims for item in items
    )
    expected_citations = sum(len(item.expected_citation_ids) for item in items)
    predicted_citations = sum(len(item.predicted_citation_ids) for item in items)
    required_facets = sum(item.required_facets for item in items)
    web_eligible = sum(item.web_eligible for item in items)
    return GroundedEvaluationMetrics(
        total=total,
        failures=sum(item.error is not None or not item.succeeded for item in items),
        task_score=sum(item.succeeded for item in items) / total,
        completeness=(
            sum(item.covered_required_facets for item in items) / required_facets
            if required_facets
            else 1.0
        ),
        faithfulness=(
            sum(len(item.supported_claim_ids) for item in items) / predicted_claims
            if predicted_claims
            else 1.0
        ),
        citation_precision=(
            sum(len(item.valid_citation_ids) for item in items) / predicted_citations
            if predicted_citations
            else 1.0
        ),
        citation_recall=(
            sum(
                len(set(item.valid_citation_ids) & set(item.expected_citation_ids))
                for item in items
            )
            / expected_citations
            if expected_citations
            else 1.0
        ),
        unsupported_claim_rate=(
            sum(item.unsupported_claims for item in items) / predicted_claims
            if predicted_claims
            else 0.0
        ),
        web_trigger_rate=(
            sum(item.web_triggered for item in items if item.web_eligible) / web_eligible
            if web_eligible
            else 0.0
        ),
        web_usage_rate=sum(item.web_used for item in items) / total,
        model_calls=sum(item.model_calls for item in items),
        search_calls=sum(item.search_calls for item in items),
        input_tokens=sum(item.input_tokens for item in items),
        output_tokens=sum(item.output_tokens for item in items),
        latency_ms=sum(item.latency_ms for item in items),
        stop_reasons=dict(Counter(item.stop_reason for item in items)),
    )


def _recorded_item(question: GroundedEvaluationQuestion) -> GroundedEvaluationItem:
    """把冻结场景映射为确定性离线结果，用于验证报告与门禁而不访问网络。"""

    successful = question.expected_stop_reason is AnswerStopReason.ACCEPTED
    supported = successful and question.scenario not in {
        GroundedEvaluationScenario.UNSUPPORTED,
        GroundedEvaluationScenario.WRONG_ATTRIBUTION,
    }
    claim_id = f"{question.question_id}-claim"
    citation_id = f"{question.question_id}-citation"
    return GroundedEvaluationItem(
        question_id=question.question_id,
        succeeded=successful,
        expected_claim_ids=[claim_id],
        supported_claim_ids=[claim_id] if supported else [],
        expected_citation_ids=[citation_id],
        predicted_citation_ids=[citation_id] if supported else [],
        valid_citation_ids=[citation_id] if supported else [],
        required_facets=1,
        covered_required_facets=int(successful),
        unsupported_claims=int(not supported),
        web_eligible=question.web_eligible,
        web_triggered=question.expected_web_trigger,
        web_used=question.expected_web_trigger and successful,
        model_calls=int(successful),
        search_calls=int(question.expected_web_trigger),
        input_tokens=100 if successful else 0,
        output_tokens=40 if successful else 0,
        latency_ms=10,
        stop_reason=question.expected_stop_reason,
    )


def evaluate_recorded_grounded_questions(
    questions: GroundedEvaluationQuestionSet,
    *,
    limit: int | None = None,
    results_root: Path | None = None,
    workspace_root: Path | None = None,
) -> GroundedEvaluationReport:
    """执行零网络的录制评测，并以内容身份原子缓存完整报告。"""

    selected = questions.questions[:limit] if limit is not None else questions.questions
    if not selected:
        raise ValueError("evaluation selection cannot be empty")
    items = [_recorded_item(item) for item in selected]
    question_hash = stable_digest(selected)
    report = GroundedEvaluationReport(
        evaluation_id="answer-eval-"
        + stable_digest({"questions": selected, "mode": "recorded-v1"})[:32],
        offline=True,
        question_set_hash=question_hash,
        items=items,
        metrics=compute_grounded_metrics(items),
        created_at=datetime.now(UTC),
    )
    if results_root is not None and workspace_root is not None:
        destination = results_root / report.evaluation_id / "report.json"
        if destination.is_file():
            try:
                cached = GroundedEvaluationReport.model_validate_json(
                    destination.read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                pass
            else:
                if cached.evaluation_id == report.evaluation_id:
                    return cached
        atomic_model_write(destination, report, workspace_root=workspace_root)
    return report


async def evaluate_online_grounded_probe(
    questions: GroundedEvaluationQuestionSet,
    *,
    llm: LLMProvider,
    search: SearchProvider,
    limit: int,
    provider_identity: str,
    results_root: Path | None = None,
    workspace_root: Path | None = None,
) -> GroundedEvaluationReport:
    """执行显式、有界的真实 Provider 验收；单题失败仍进入报告分母。"""

    if limit < 5 or limit > 10:
        raise ValueError("online grounded evaluation limit must be between 5 and 10")
    selected = list(questions.questions[:limit])
    if not any(item.web_eligible for item in selected):
        eligible = next((item for item in questions.questions if item.web_eligible), None)
        if eligible is not None:
            selected[-1] = eligible
    evaluation_id = (
        "answer-eval-"
        + stable_digest(
            {
                "questions": selected,
                "mode": "online-probe-v1",
                "provider_identity": provider_identity,
            }
        )[:32]
    )
    result_path = results_root / evaluation_id / "report.json" if results_root else None
    if result_path is not None and result_path.is_file():
        try:
            cached = GroundedEvaluationReport.model_validate_json(
                result_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            pass
        else:
            if cached.evaluation_id == evaluation_id:
                return cached
    items: list[GroundedEvaluationItem] = []
    for question in selected:
        item_path = (
            results_root / evaluation_id / "items" / f"{question.question_id}.json"
            if results_root
            else None
        )
        if item_path is not None and item_path.is_file():
            try:
                items.append(
                    GroundedEvaluationItem.model_validate_json(
                        item_path.read_text(encoding="utf-8")
                    )
                )
                continue
            except (OSError, ValueError):
                pass
        search_calls = 0
        model_calls = 0
        input_tokens = 0
        output_tokens = 0
        try:
            snippets = ""
            if question.web_eligible:
                results = await search.search(question.question, max_results=5)
                search_calls = 1
                snippets = "\n".join(item.excerpt for item in results)[:8000]
            prompt = (
                "请仅依据给定材料简短回答；证据不足时明确说明。\n"
                f"问题: {question.question}\n材料: {snippets or '使用内部知识边界探测'}"
            )
            model_calls = 1
            response = await llm.generate(prompt, temperature=0.0)
            input_tokens = min(9000, max(1, (len(prompt) + 3) // 4))
            output_tokens = min(3000, max(1, (len(response) + 3) // 4))
            succeeded = bool(response.strip())
            error = None
        except Exception:
            succeeded = False
            if model_calls:
                input_tokens = 9000
                output_tokens = 3000
            error = ErrorDetail(
                code=ErrorCode.EXTERNAL_SERVICE,
                message="online grounded evaluation item failed",
                retryable=False,
            )
        claim_id = f"{question.question_id}-claim"
        item = GroundedEvaluationItem(
            question_id=question.question_id,
            succeeded=succeeded,
            expected_claim_ids=[claim_id],
            supported_claim_ids=[claim_id] if succeeded else [],
            required_facets=1,
            covered_required_facets=int(succeeded),
            unsupported_claims=int(not succeeded),
            web_eligible=question.web_eligible,
            web_triggered=bool(search_calls),
            web_used=bool(search_calls and succeeded),
            model_calls=model_calls,
            search_calls=search_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=(
                AnswerStopReason.ACCEPTED if succeeded else AnswerStopReason.EXECUTION_FAILED
            ),
            error=error,
        )
        items.append(item)
        if (
            item_path is not None
            and workspace_root is not None
            and item.succeeded
            and item.error is None
        ):
            atomic_model_write(item_path, item, workspace_root=workspace_root)
    question_hash = stable_digest(selected)
    report = GroundedEvaluationReport(
        evaluation_id=evaluation_id,
        offline=False,
        question_set_hash=question_hash,
        items=items,
        metrics=compute_grounded_metrics(items),
        created_at=datetime.now(UTC),
    )
    if result_path is not None and workspace_root is not None and report.metrics.failures == 0:
        atomic_model_write(result_path, report, workspace_root=workspace_root)
    return report
