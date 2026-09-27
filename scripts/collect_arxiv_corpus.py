#!/usr/bin/env python3
"""有界发现并下载 Agent/RAG arXiv 语料，支持断点续跑。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from kg_crag.ingestion.arxiv import (
    ArxivHttpClient,
    ArxivPaper,
    append_jsonl,
    build_metadata,
    index_local_corpus,
    merge_candidates,
    normalize_arxiv_id,
    normalize_title,
    score_relevance,
    write_json,
)
from kg_crag.ingestion.openalex import OpenAlexClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "corpus_collection.json"
DEFAULT_SEEDS = PROJECT_ROOT / "configs" / "seed_papers.json"
DEFAULT_PDF_DIR = PROJECT_ROOT / "data" / "raw" / "papers"
DEFAULT_METADATA_DIR = PROJECT_ROOT / "data" / "raw" / "metadata"
DEFAULT_CACHE = PROJECT_ROOT / "data" / "raw" / "arxiv_candidates.json"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "raw" / "collection_report.json"
DEFAULT_LOG = PROJECT_ROOT / "data" / "raw" / "collection_manifest.jsonl"
USER_AGENT = "kg-crag-course-project/0.2 (contact: local-course-research; bounded-arxiv-client)"


@dataclass(frozen=True)
class SearchQuery:
    name: str
    search_query: str
    openalex_search: str
    tags: list[str]


@dataclass(frozen=True)
class CuratedSeed:
    arxiv_id: str
    title: str
    year: int
    tags: list[str]


@dataclass(frozen=True)
class CollectionConfig:
    target_papers: int
    max_results_per_query: int
    discovery_pages_per_query: int
    min_year: int
    allowed_categories: frozenset[str]
    minimum_relevance_score: float
    relevance_terms: dict[str, float]
    queries: list[SearchQuery]
    config_hash: str


def bounded_int(minimum: int, maximum: int) -> Callable[[str], int]:
    """创建具有明确安全范围的 argparse 类型转换器。"""

    def convert(value: str) -> int:
        parsed = int(value)
        if not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(f"must be between {minimum} and {maximum}")
        return parsed

    return convert


def parse_args() -> argparse.Namespace:
    """解析采集参数，并禁止无界发现或下载。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--target", type=bounded_int(1, 200), help="Override target_papers")
    parser.add_argument(
        "--max-results-per-query",
        type=bounded_int(1, 50),
        help="Override the bounded discovery page size",
    )
    parser.add_argument(
        "--pages-per-query",
        type=bounded_int(1, 5),
        help="Override the bounded number of discovery pages",
    )
    parser.add_argument(
        "--discovery-source",
        choices=("openalex", "arxiv", "both"),
        default="openalex",
        help="Metadata discovery source; PDFs always come from arXiv",
    )
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--retries", type=bounded_int(1, 6), default=3)
    parser.add_argument("--delay", type=float, default=3.0, help="Polite delay between requests")
    parser.add_argument("--refresh-discovery", action="store_true")
    parser.add_argument("--discovery-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_collection_config(path: Path) -> CollectionConfig:
    """在发出任何外部请求前校验采集策略。"""

    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    queries_payload = payload.get("queries")
    terms_payload = payload.get("relevance_terms")
    if not isinstance(queries_payload, list) or not queries_payload:
        raise ValueError("config.queries must be a non-empty list")
    if not isinstance(terms_payload, dict) or not terms_payload:
        raise ValueError("config.relevance_terms must be a non-empty object")

    queries: list[SearchQuery] = []
    names: set[str] = set()
    for item in queries_payload:
        if not isinstance(item, dict):
            raise ValueError("each query must be an object")
        name = str(item.get("name", "")).strip()
        query = str(item.get("search_query", "")).strip()
        openalex_search = str(item.get("openalex_search", "")).strip()
        tags = item.get("tags", [])
        if not name or name in names:
            raise ValueError(f"query name must be unique and non-empty: {name!r}")
        if not query or not openalex_search:
            raise ValueError(f"query {name!r} requires arXiv and OpenAlex search strings")
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ValueError(f"query {name!r} tags must be strings")
        names.add(name)
        queries.append(
            SearchQuery(
                name=name,
                search_query=query,
                openalex_search=openalex_search,
                tags=tags,
            )
        )

    target = int(payload.get("target_papers", 20))
    max_results = int(payload.get("max_results_per_query", 15))
    discovery_pages = int(payload.get("discovery_pages_per_query", 2))
    if not 1 <= target <= 200:
        raise ValueError("target_papers must be between 1 and 200")
    if not 1 <= max_results <= 50:
        raise ValueError("max_results_per_query must be between 1 and 50")
    if not 1 <= discovery_pages <= 5:
        raise ValueError("discovery_pages_per_query must be between 1 and 5")
    categories = payload.get("allowed_categories", [])
    if not isinstance(categories, list) or not all(isinstance(value, str) for value in categories):
        raise ValueError("allowed_categories must contain strings")
    return CollectionConfig(
        target_papers=target,
        max_results_per_query=max_results,
        discovery_pages_per_query=discovery_pages,
        min_year=int(payload.get("min_year", 2020)),
        allowed_categories=frozenset(categories),
        minimum_relevance_score=float(payload.get("minimum_relevance_score", 0.0)),
        relevance_terms={str(key): float(value) for key, value in terms_payload.items()},
        queries=queries,
        config_hash=hashlib.sha256(raw).hexdigest(),
    )


def load_seed_records(path: Path) -> tuple[dict[str, CuratedSeed], str]:
    """加载不依赖 API 的核心元数据，并返回内容哈希。"""

    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    raw_papers = payload.get("papers")
    if not isinstance(raw_papers, list):
        raise ValueError("seed manifest papers must be a list")
    seeds: dict[str, CuratedSeed] = {}
    for item in raw_papers:
        if not isinstance(item, dict):
            raise ValueError("seed paper must be an object")
        arxiv_id = normalize_arxiv_id(str(item.get("arxiv_id", "")))
        title = str(item.get("title", "")).strip()
        year = int(item.get("year", 0))
        tags = item.get("tags", [])
        if not title or not 1900 <= year <= 2100:
            raise ValueError(f"seed {arxiv_id} requires a title and valid year")
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ValueError(f"seed tags for {arxiv_id} must be strings")
        seeds[arxiv_id] = CuratedSeed(
            arxiv_id=arxiv_id,
            title=title,
            year=year,
            tags=tags,
        )
    return seeds, hashlib.sha256(raw).hexdigest()


def load_cached_candidates(path: Path, config_hash: str) -> list[ArxivPaper] | None:
    """只有候选缓存由相同策略配置生成时才复用。"""

    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("config_hash") != config_hash:
        return None
    records = payload.get("candidates")
    if not isinstance(records, list):
        return None
    return [ArxivPaper.from_cache_record(record) for record in records if isinstance(record, dict)]


def candidate_is_allowed(paper: ArxivPaper, config: CollectionConfig) -> bool:
    """应用可解释的年份、分类和最低分数筛选。"""

    if paper.curated:
        return True
    if paper.year is None or paper.year < config.min_year:
        return False
    if (
        config.allowed_categories
        and paper.categories
        and not config.allowed_categories.intersection(paper.categories)
    ):
        return False
    return paper.relevance_score >= config.minimum_relevance_score


def discover_candidates(
    arxiv_client: ArxivHttpClient,
    openalex_client: OpenAlexClient,
    config: CollectionConfig,
    seeds: dict[str, CuratedSeed],
    *,
    max_results: int,
    pages_per_query: int,
    discovery_source: str,
    cache_path: Path,
    cache_hash: str,
) -> tuple[list[ArxivPaper], list[str]]:
    """合并人工维护记录与在线检索结果，生成排序候选缓存。"""

    discovered = [
        ArxivPaper(
            arxiv_id=seed.arxiv_id,
            title=seed.title,
            abstract="",
            published=str(seed.year),
            updated="",
            authors=[],
            categories=[],
            primary_category=None,
            doi=None,
            journal_reference=None,
            source_url=f"https://arxiv.org/abs/{seed.arxiv_id}",
            pdf_url=f"https://arxiv.org/pdf/{seed.arxiv_id}",
            license_url=None,
            tags=list(seed.tags),
            discovered_by=["seed_manifest"],
            curated=True,
        )
        for seed in seeds.values()
    ]
    failures: list[str] = []
    for query in config.queries:
        if discovery_source in {"arxiv", "both"}:
            for page in range(pages_per_query):
                try:
                    papers = arxiv_client.query(
                        {
                            "search_query": query.search_query,
                            "start": page * max_results,
                            "max_results": max_results,
                            "sortBy": "relevance",
                            "sortOrder": "descending",
                        }
                    )
                    for paper in papers:
                        paper.tags = list(query.tags)
                        paper.discovered_by = [f"arxiv:{query.name}"]
                        discovered.append(paper)
                except (OSError, ValueError, RuntimeError, ET.ParseError) as exc:
                    failures.append(f"arxiv query {query.name} page {page + 1}: {exc}")
                    break
                time.sleep(arxiv_client.delay)

        if discovery_source in {"openalex", "both"}:
            for page in range(1, pages_per_query + 1):
                try:
                    papers = openalex_client.search(
                        query.openalex_search,
                        min_year=config.min_year,
                        page=page,
                        per_page=max_results,
                    )
                    for paper in papers:
                        paper.tags = list(query.tags)
                        paper.discovered_by = [f"openalex:{query.name}"]
                        discovered.append(paper)
                except (OSError, ValueError, RuntimeError) as exc:
                    failures.append(f"openalex query {query.name} page {page}: {exc}")
                    break
                time.sleep(openalex_client.delay)

    merged = merge_candidates(discovered)
    for paper in merged:
        paper.relevance_score = score_relevance(paper, config.relevance_terms)
    accepted = [paper for paper in merged if candidate_is_allowed(paper, config)]
    accepted.sort(
        key=lambda paper: (
            paper.curated,
            paper.relevance_score,
            paper.year or 0,
            paper.versionless_id,
        ),
        reverse=True,
    )
    write_json(
        cache_path,
        {
            "schema_version": 1,
            "config_hash": cache_hash,
            "generated_at": datetime.now(UTC).isoformat(),
            "candidate_count": len(accepted),
            "discovery_failures": failures,
            "candidates": [paper.to_cache_record() for paper in accepted],
        },
    )
    return accepted, failures


def collection_event(
    paper: ArxivPaper,
    status: str,
    *,
    error: str | None = None,
    sha256: str | None = None,
) -> dict[str, object]:
    """为成功、跳过和失败路径构造统一的精简审计事件。"""

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "arxiv_id": paper.versionless_id,
        "title": paper.title,
        "status": status,
        "sha256": sha256,
        "error": error,
    }


def collect(
    candidates: list[ArxivPaper],
    client: ArxivHttpClient,
    args: argparse.Namespace,
    target: int,
) -> tuple[int, list[dict[str, object]]]:
    """按排序下载尚未获取的论文，直到达到本地校验目标。"""

    local = index_local_corpus(args.metadata_dir, args.pdf_dir)
    known_ids = set(local.arxiv_ids)
    known_titles = set(local.normalized_titles)
    current_count = local.valid_count
    events: list[dict[str, object]] = []
    for paper in candidates:
        if current_count >= target:
            break
        normalized_title = normalize_title(paper.title)
        if paper.versionless_id in known_ids or normalized_title in known_titles:
            continue

        filename = f"{paper.filename_stem}.pdf"
        pdf_target = args.pdf_dir / filename
        metadata_target = args.metadata_dir / f"{paper.filename_stem}.json"
        try:
            result = client.download_pdf(paper, pdf_target)
            relative_pdf_path = Path("data") / "raw" / "papers" / filename
            write_json(metadata_target, build_metadata(paper, result, relative_pdf_path))
            event = collection_event(paper, "downloaded", sha256=result.sha256)
            known_ids.add(paper.versionless_id)
            known_titles.add(normalized_title)
            current_count += 1
            print(f"[{current_count}/{target}] downloaded {paper.versionless_id}: {paper.title}")
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            event = collection_event(paper, "failed", error=str(exc))
            print(f"failed {paper.versionless_id}: {exc}", file=sys.stderr)
        append_jsonl(args.log, event)
        events.append(event)
        time.sleep(client.delay)
    return current_count, events


def write_report(
    path: Path,
    *,
    config: CollectionConfig,
    target: int,
    initial_count: int,
    final_count: int,
    candidates: list[ArxivPaper],
    discovery_failures: list[str],
    events: list[dict[str, object]],
) -> None:
    """持久化足以诊断和复现采集运行的摘要状态。"""

    write_json(
        path,
        {
            "schema_version": 1,
            "generated_at": datetime.now(UTC).isoformat(),
            "config_hash": config.config_hash,
            "target": target,
            "initial_valid_count": initial_count,
            "final_valid_count": final_count,
            "target_reached": final_count >= target,
            "candidate_count": len(candidates),
            "discovery_failures": discovery_failures,
            "run_events": events,
        },
    )


def main() -> int:
    """校验策略、发现或恢复候选，并达到有界采集目标。"""

    args = parse_args()
    if args.timeout <= 0 or args.delay < 0:
        raise ValueError("timeout must be positive and delay cannot be negative")
    config = load_collection_config(args.config.resolve())
    seeds, seed_hash = load_seed_records(args.seeds.resolve())
    target = args.target or config.target_papers
    max_results = args.max_results_per_query or config.max_results_per_query
    pages_per_query = args.pages_per_query or config.discovery_pages_per_query
    cache_hash = hashlib.sha256(
        (
            f"{config.config_hash}:{seed_hash}:{args.discovery_source}:"
            f"{max_results}:{pages_per_query}"
        ).encode()
    ).hexdigest()
    local = index_local_corpus(args.metadata_dir, args.pdf_dir)
    if local.invalid_metadata_files:
        print("Warning: invalid local records:", file=sys.stderr)
        for error in local.invalid_metadata_files:
            print(f"- {error}", file=sys.stderr)

    print(
        f"Verified local corpus: {local.valid_count}; target: {target}; "
        f"missing: {max(0, target - local.valid_count)}"
    )
    if args.dry_run or local.valid_count >= target:
        return 0

    arxiv_client = ArxivHttpClient(
        timeout=args.timeout,
        retries=args.retries,
        delay=args.delay,
        user_agent=USER_AGENT,
    )
    openalex_client = OpenAlexClient(
        timeout=args.timeout,
        retries=args.retries,
        delay=args.delay,
        user_agent=USER_AGENT,
    )
    candidates = None if args.refresh_discovery else load_cached_candidates(args.cache, cache_hash)
    discovery_failures: list[str] = []
    if candidates is None:
        candidates, discovery_failures = discover_candidates(
            arxiv_client,
            openalex_client,
            config,
            seeds,
            max_results=max_results,
            pages_per_query=pages_per_query,
            discovery_source=args.discovery_source,
            cache_path=args.cache,
            cache_hash=cache_hash,
        )
    else:
        print(f"Using {len(candidates)} cached candidates from {args.cache}")
    print(f"Eligible unique candidates: {len(candidates)}")

    if args.discovery_only:
        final_count = local.valid_count
        events: list[dict[str, object]] = []
    else:
        final_count, events = collect(candidates, arxiv_client, args, target)
    write_report(
        args.report,
        config=config,
        target=target,
        initial_count=local.valid_count,
        final_count=final_count,
        candidates=candidates,
        discovery_failures=discovery_failures,
        events=events,
    )
    print(f"Final verified corpus: {final_count}/{target}")
    return 0 if final_count >= target or args.discovery_only else 2


if __name__ == "__main__":
    raise SystemExit(main())
