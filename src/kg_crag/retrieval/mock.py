"""供工作流单元测试使用的确定性检索与重排序替身。"""

from __future__ import annotations

from dataclasses import dataclass

from kg_crag.errors import KGCRAGError
from kg_crag.models import ErrorCode, ErrorDetail, Evidence

FilterValue = str | int | bool


@dataclass(frozen=True)
class RetrievalCall:
    """一次 Retriever 调用的不可变参数快照。"""

    query: str
    top_k: int
    filters: tuple[tuple[str, FilterValue], ...]


@dataclass(frozen=True)
class RerankCall:
    """一次 Reranker 调用的不可变参数快照。"""

    query: str
    candidate_ids: tuple[str, ...]
    top_k: int


def _invalid_parameter(message: str, **context: FilterValue) -> KGCRAGError:
    return KGCRAGError(
        ErrorDetail(
            code=ErrorCode.VALIDATION,
            message=message,
            retryable=False,
            context=context,
        )
    )


def _matches_filters(evidence: Evidence, filters: dict[str, FilterValue]) -> bool:
    """对公共字段和元数据执行精确匹配，未知键不产生命中。"""

    for key, expected in filters.items():
        if key == "paper_id":
            actual: str | int | float | bool | None = evidence.paper_id
        elif key == "section":
            actual = evidence.location.section
        elif key in evidence.metadata:
            actual = evidence.metadata[key]
        else:
            return False
        if actual != expected:
            return False
    return True


class MockRetriever:
    """按预设顺序返回 Evidence 的离线 Retriever。"""

    def __init__(self, evidence: list[Evidence]) -> None:
        self._evidence = tuple(item.model_copy(deep=True) for item in evidence)
        self.calls: list[RetrievalCall] = []

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, FilterValue] | None = None,
    ) -> list[Evidence]:
        """记录参数，精确过滤后稳定截断结果。"""

        if top_k <= 0:
            raise _invalid_parameter("top_k must be positive", top_k=top_k)
        selected_filters = dict(filters or {})
        self.calls.append(
            RetrievalCall(
                query=query,
                top_k=top_k,
                filters=tuple(sorted(selected_filters.items())),
            )
        )
        return [
            item.model_copy(deep=True)
            for item in self._evidence
            if _matches_filters(item, selected_filters)
        ][:top_k]


class MockReranker:
    """依据预设分数执行稳定排序的离线 Reranker。"""

    def __init__(self, scores: dict[str, float] | None = None) -> None:
        self._scores = dict(scores or {})
        self.calls: list[RerankCall] = []

    async def rerank(
        self,
        query: str,
        candidates: list[Evidence],
        *,
        top_k: int,
    ) -> list[Evidence]:
        """按分数降序排序，同分时保留候选输入顺序。"""

        if top_k <= 0:
            raise _invalid_parameter("top_k must be positive", top_k=top_k)
        self.calls.append(
            RerankCall(
                query=query,
                candidate_ids=tuple(item.evidence_id for item in candidates),
                top_k=top_k,
            )
        )
        ranked = sorted(
            enumerate(candidates),
            key=lambda item: (-self._scores.get(item[1].evidence_id, 0.0), item[0]),
        )
        results: list[Evidence] = []
        for _, candidate in ranked[:top_k]:
            copied = candidate.model_copy(deep=True)
            copied.scores.rerank = self._scores.get(candidate.evidence_id, 0.0)
            results.append(copied)
        return results
