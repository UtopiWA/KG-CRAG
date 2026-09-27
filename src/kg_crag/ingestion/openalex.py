"""生成带 arXiv 来源候选论文的 OpenAlex 发现适配器。"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from http.client import HTTPResponse
from typing import Any, cast

from kg_crag.ingestion.arxiv import ArxivPaper, normalize_arxiv_id

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
ARXIV_URL_PATTERN = re.compile(r"arxiv\.org/(?:abs|pdf)/([^?#]+)", re.I)
ARXIV_DOI_PATTERN = re.compile(r"10\.48550/arxiv\.([^?#]+)", re.I)


def reconstruct_abstract(inverted_index: object) -> str:
    """根据 OpenAlex 的词元位置索引重建可读摘要。"""

    if not isinstance(inverted_index, dict):
        return ""
    positions: dict[int, str] = {}
    for token, raw_positions in inverted_index.items():
        if not isinstance(token, str) or not isinstance(raw_positions, list):
            continue
        for position in raw_positions:
            if isinstance(position, int) and position >= 0:
                positions[position] = token
    return " ".join(positions[index] for index in sorted(positions))


def extract_arxiv_id(work: Mapping[str, Any]) -> str | None:
    """从显式 ID、DOI 或开放获取地址中提取 arXiv ID。"""

    ids = work.get("ids")
    if isinstance(ids, dict):
        arxiv_value = ids.get("arxiv")
        if isinstance(arxiv_value, str):
            try:
                return normalize_arxiv_id(arxiv_value)
            except ValueError:
                pass
        doi_value = ids.get("doi")
        if isinstance(doi_value, str) and (match := ARXIV_DOI_PATTERN.search(doi_value)):
            try:
                return normalize_arxiv_id(match.group(1))
            except ValueError:
                pass

    location_values: list[object] = []
    for name in ("primary_location", "best_oa_location"):
        value = work.get(name)
        if value is not None:
            location_values.append(value)
    locations = work.get("locations")
    if isinstance(locations, list):
        location_values.extend(locations)
    for value in location_values:
        if not isinstance(value, dict):
            continue
        for field_name in ("landing_page_url", "pdf_url"):
            url = value.get(field_name)
            if isinstance(url, str) and (match := ARXIV_URL_PATTERN.search(url)):
                try:
                    return normalize_arxiv_id(match.group(1))
                except ValueError:
                    continue
    return None


def _author_names(work: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    authorships = work.get("authorships")
    if not isinstance(authorships, list):
        return names
    for authorship in authorships:
        if not isinstance(authorship, dict):
            continue
        author = authorship.get("author")
        if isinstance(author, dict) and isinstance(author.get("display_name"), str):
            names.append(author["display_name"])
    return names


def parse_openalex_works(payload: Mapping[str, Any]) -> list[ArxivPaper]:
    """把包含 arXiv 地址的 OpenAlex 结果转换为下载候选。"""

    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("OpenAlex response has no results list")
    papers: list[ArxivPaper] = []
    for work in raw_results:
        if not isinstance(work, dict):
            continue
        arxiv_id = extract_arxiv_id(work)
        title = work.get("title") or work.get("display_name")
        year = work.get("publication_year")
        if arxiv_id is None or not isinstance(title, str) or not isinstance(year, int):
            continue
        doi_value = work.get("doi")
        doi = doi_value.removeprefix("https://doi.org/") if isinstance(doi_value, str) else None
        openalex_id = str(work.get("id", ""))
        external_ids = {"openalex": openalex_id} if openalex_id else {}
        papers.append(
            ArxivPaper(
                arxiv_id=arxiv_id,
                title=title,
                abstract=reconstruct_abstract(work.get("abstract_inverted_index")),
                published=str(work.get("publication_date") or year),
                updated="",
                authors=_author_names(work),
                categories=[],
                primary_category=None,
                doi=doi,
                journal_reference=None,
                source_url=f"https://arxiv.org/abs/{arxiv_id}",
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
                license_url=None,
                metadata_provider="OpenAlex",
                external_ids=external_ids,
            )
        )
    return papers


class OpenAlexClient:
    """限制响应规模、重试次数和字段集合的 OpenAlex Works 客户端。"""

    def __init__(self, *, timeout: float, retries: int, delay: float, user_agent: str) -> None:
        self.timeout = timeout
        self.retries = retries
        self.delay = delay
        self.user_agent = user_agent

    def search(self, query: str, *, min_year: int, page: int, per_page: int) -> list[ArxivPaper]:
        """搜索开放获取论文，只保留关联 arXiv 预印本的记录。"""

        parameters = {
            "search": query,
            "filter": f"from_publication_date:{min_year}-01-01,open_access.is_oa:true",
            "page": page,
            "per-page": per_page,
            "select": (
                "id,doi,title,display_name,publication_year,publication_date,ids,"
                "authorships,primary_location,best_oa_location,locations,abstract_inverted_index"
            ),
        }
        url = f"{OPENALEX_WORKS_URL}?{urllib.parse.urlencode(parameters)}"
        request = urllib.request.Request(  # noqa: S310 - 固定的 OpenAlex HTTPS 端点
            url,
            headers={"User-Agent": self.user_agent, "Accept": "application/json"},
        )
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = cast(
                    HTTPResponse,
                    urllib.request.urlopen(request, timeout=self.timeout),  # noqa: S310
                )
                with response:
                    data = response.read(20 * 1024 * 1024 + 1)
                if len(data) > 20 * 1024 * 1024:
                    raise ValueError("OpenAlex response exceeds 20 MiB")
                payload = json.loads(data.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("OpenAlex response must be an object")
                return parse_openalex_works(payload)
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in {408, 429, 500, 502, 503, 504}:
                    break
            except (
                urllib.error.URLError,
                TimeoutError,
                UnicodeDecodeError,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc
            if attempt + 1 < self.retries:
                time.sleep(self.delay * (2**attempt))
        raise RuntimeError(f"OpenAlex request failed after {self.retries} attempts") from last_error
