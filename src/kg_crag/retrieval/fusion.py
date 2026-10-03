"""无副作用、可重放的 Dense/Sparse 融合与去重。"""

from __future__ import annotations

import hashlib
import json

from kg_crag.models import Evidence
from kg_crag.retrieval.config import FusionConfig


def fusion_version(config: FusionConfig) -> str:
    payload = {
        "implementation": "hybrid-fusion-v1",
        **config.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def fuse_evidence(
    dense: list[Evidence],
    sparse: list[Evidence],
    config: FusionConfig,
) -> list[Evidence]:
    """按配置执行 RRF 或候选池归一化加权融合。"""

    dense_by_id = _best_by_source(dense, "dense")
    sparse_by_id = _best_by_source(sparse, "sparse")
    version = fusion_version(config)
    if config.method == "rrf":
        scores = {
            source_id: _rrf_score(
                dense_by_id.get(source_id), sparse_by_id.get(source_id), config.rrf_k
            )
            for source_id in dense_by_id.keys() | sparse_by_id.keys()
        }
    else:
        dense_normalized = _normalize(dense_by_id, "dense")
        sparse_normalized = _normalize(sparse_by_id, "sparse")
        scores = {
            source_id: config.dense_weight * dense_normalized.get(source_id, 0.0)
            + config.sparse_weight * sparse_normalized.get(source_id, 0.0)
            for source_id in dense_by_id.keys() | sparse_by_id.keys()
        }

    merged = [
        _merge_candidate(
            source_id,
            dense_by_id.get(source_id),
            sparse_by_id.get(source_id),
            score,
            version,
        )
        for source_id, score in scores.items()
    ]
    merged.sort(key=lambda item: (-_fusion_score(item), _best_rank(item), item.source_id))
    results: list[Evidence] = []
    for rank, item in enumerate(merged[: config.final_top_k], start=1):
        copied = item.model_copy(deep=True)
        copied.ranks.fusion = rank
        results.append(copied)
    return results


def deduplicate_evidence_by_paper(evidence: list[Evidence], *, max_results: int) -> list[Evidence]:
    """按当前顺序每篇只保留首条，不循环补取或改写已有阶段排名。"""

    selected: list[Evidence] = []
    seen: set[str] = set()
    for item in evidence:
        key = item.paper_id or f"source:{item.source_id}"
        if key in seen:
            continue
        seen.add(key)
        selected.append(item.model_copy(deep=True))
        if len(selected) >= max_results:
            break
    return selected


def _best_by_source(evidence: list[Evidence], route: str) -> dict[str, Evidence]:
    selected: dict[str, Evidence] = {}
    for fallback_rank, item in enumerate(evidence, start=1):
        copied = item.model_copy(deep=True)
        rank = getattr(copied.ranks, route) or fallback_rank
        setattr(copied.ranks, route, rank)
        current = selected.get(copied.source_id)
        if current is None or rank < (getattr(current.ranks, route) or fallback_rank):
            selected[copied.source_id] = copied
    return selected


def _rrf_score(dense: Evidence | None, sparse: Evidence | None, rrf_k: int) -> float:
    score = 0.0
    if dense is not None and dense.ranks.dense is not None:
        score += 1.0 / (rrf_k + dense.ranks.dense)
    if sparse is not None and sparse.ranks.sparse is not None:
        score += 1.0 / (rrf_k + sparse.ranks.sparse)
    return score


def _normalize(candidates: dict[str, Evidence], route: str) -> dict[str, float]:
    raw = {
        source_id: float(getattr(item.scores, route))
        for source_id, item in candidates.items()
        if getattr(item.scores, route) is not None
    }
    if not raw:
        return {}
    minimum = min(raw.values())
    maximum = max(raw.values())
    if maximum == minimum:
        return {source_id: 1.0 for source_id in raw}
    return {source_id: (score - minimum) / (maximum - minimum) for source_id, score in raw.items()}


def _merge_candidate(
    source_id: str,
    dense: Evidence | None,
    sparse: Evidence | None,
    score: float,
    version: str,
) -> Evidence:
    base = dense or sparse
    if base is None:  # pragma: no cover - 由调用方的 key 并集保证不可达
        raise ValueError("fusion candidate is missing both sources")
    merged = base.model_copy(deep=True)
    if dense is not None:
        merged.scores.dense = dense.scores.dense
        merged.ranks.dense = dense.ranks.dense
        merged.metadata.update(dense.metadata)
        merged.metadata["dense_evidence_id"] = dense.evidence_id
    if sparse is not None:
        merged.scores.sparse = sparse.scores.sparse
        merged.ranks.sparse = sparse.ranks.sparse
        merged.metadata.update(sparse.metadata)
        merged.metadata["sparse_evidence_id"] = sparse.evidence_id
    merged.evidence_id = f"hybrid:{version}:{source_id}"
    merged.scores.fusion = score
    merged.metadata["fusion_version"] = version
    return merged


def _fusion_score(item: Evidence) -> float:
    return item.scores.fusion if item.scores.fusion is not None else float("-inf")


def _best_rank(item: Evidence) -> int:
    ranks = [rank for rank in (item.ranks.dense, item.ranks.sparse) if rank is not None]
    return min(ranks) if ranks else 2**31 - 1
