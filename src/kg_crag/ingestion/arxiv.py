"""可复用的 arXiv 发现、排序、下载与本地语料辅助逻辑。"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from http.client import HTTPResponse
from pathlib import Path
from typing import Any, cast

ARXIV_API_BASE = "https://export.arxiv.org/api/query"
ARXIV_PDF_BASE = "https://arxiv.org/pdf"
ATOM = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
ARXIV_ID_PATTERN = re.compile(r"^(?:[a-z-]+(?:\.[A-Z]{2})?/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?$", re.I)
VERSION_PATTERN = re.compile(r"v\d+$", re.I)
NON_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")
NON_TITLE_PATTERN = re.compile(r"[^a-z0-9]+")
TRUSTED_URL_PREFIXES = ("https://arxiv.org/", "https://export.arxiv.org/")


@dataclass
class ArxivPaper:
    """在发现与下载阶段之间传递的规范化 arXiv 记录。"""

    arxiv_id: str
    title: str
    abstract: str
    published: str
    updated: str
    authors: list[str]
    categories: list[str]
    primary_category: str | None
    doi: str | None
    journal_reference: str | None
    source_url: str
    pdf_url: str
    license_url: str | None
    metadata_provider: str = "arXiv"
    external_ids: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    discovered_by: list[str] = field(default_factory=list)
    relevance_score: float = 0.0
    curated: bool = False

    @property
    def versionless_id(self) -> str:
        """返回用于语料级去重的稳定标识。"""

        return VERSION_PATTERN.sub("", self.arxiv_id)

    @property
    def year(self) -> int | None:
        """在 arXiv 提供日期时提取发表年份。"""

        return int(self.published[:4]) if len(self.published) >= 4 else None

    @property
    def filename_stem(self) -> str:
        """构造可读且不易冲突的文件名主体。"""

        identifier = self.versionless_id.replace("/", "_").replace(".", "_").lower()
        title_slug = NON_SLUG_PATTERN.sub("_", self.title.lower()).strip("_")[:64]
        return f"arxiv_{identifier}_{title_slug or 'paper'}"

    def to_cache_record(self) -> dict[str, object]:
        """序列化记录，同时保留合并后的发现来源。"""

        return asdict(self)

    @classmethod
    def from_cache_record(cls, record: Mapping[str, Any]) -> ArxivPaper:
        """恢复由本模块生成的缓存记录。"""

        return cls(
            arxiv_id=str(record["arxiv_id"]),
            title=str(record["title"]),
            abstract=str(record["abstract"]),
            published=str(record["published"]),
            updated=str(record["updated"]),
            authors=[str(value) for value in record.get("authors", [])],
            categories=[str(value) for value in record.get("categories", [])],
            primary_category=(
                str(record["primary_category"]) if record.get("primary_category") else None
            ),
            doi=str(record["doi"]) if record.get("doi") else None,
            journal_reference=(
                str(record["journal_reference"]) if record.get("journal_reference") else None
            ),
            source_url=str(record["source_url"]),
            pdf_url=str(record["pdf_url"]),
            license_url=str(record["license_url"]) if record.get("license_url") else None,
            metadata_provider=str(record.get("metadata_provider", "arXiv")),
            external_ids={
                str(key): str(value) for key, value in record.get("external_ids", {}).items()
            },
            tags=[str(value) for value in record.get("tags", [])],
            discovered_by=[str(value) for value in record.get("discovered_by", [])],
            relevance_score=float(record.get("relevance_score", 0.0)),
            curated=bool(record.get("curated", False)),
        )


@dataclass(frozen=True)
class DownloadResult:
    sha256: str
    size_bytes: int
    content_type: str


@dataclass(frozen=True)
class LocalCorpusIndex:
    valid_count: int
    arxiv_ids: frozenset[str]
    normalized_titles: frozenset[str]
    invalid_metadata_files: tuple[str, ...]


def normalize_arxiv_id(value: str) -> str:
    """去除 URL 和版本修饰，并拒绝不符合 arXiv ID 格式的值。"""

    normalized = value.strip().removeprefix("arXiv:")
    normalized = normalized.rsplit("/abs/", maxsplit=1)[-1]
    normalized = normalized.rsplit("/pdf/", maxsplit=1)[-1].removesuffix(".pdf")
    if not ARXIV_ID_PATTERN.fullmatch(normalized):
        raise ValueError(f"invalid arXiv ID: {value!r}")
    return VERSION_PATTERN.sub("", normalized)


def normalize_title(value: str) -> str:
    """规范化标题，用于保守的精确重复检测。"""

    return NON_TITLE_PATTERN.sub("", value.lower())


def _entry_text(entry: ET.Element, name: str) -> str:
    element = entry.find(name, ATOM)
    return " ".join((element.text or "").split()) if element is not None else ""


def parse_atom_feed(xml_bytes: bytes) -> list[ArxivPaper]:
    """将限制大小的 arXiv Atom 响应解析为规范化记录。"""

    root = ET.fromstring(xml_bytes)  # noqa: S314 - 响应来自可信 arXiv 且大小受限
    papers: list[ArxivPaper] = []
    for entry in root.findall("atom:entry", ATOM):
        raw_id = _entry_text(entry, "atom:id")
        arxiv_id = normalize_arxiv_id(raw_id)
        links = entry.findall("atom:link", ATOM)
        source_url = next(
            (link.get("href", "") for link in links if link.get("rel") == "alternate"),
            f"https://arxiv.org/abs/{arxiv_id}",
        )
        pdf_url = next(
            (link.get("href", "") for link in links if link.get("type") == "application/pdf"),
            f"{ARXIV_PDF_BASE}/{arxiv_id}",
        )
        categories = [
            term
            for node in entry.findall("atom:category", ATOM)
            if (term := node.get("term")) is not None
        ]
        primary_node = entry.find("arxiv:primary_category", ATOM)
        papers.append(
            ArxivPaper(
                arxiv_id=arxiv_id,
                title=_entry_text(entry, "atom:title"),
                abstract=_entry_text(entry, "atom:summary"),
                published=_entry_text(entry, "atom:published"),
                updated=_entry_text(entry, "atom:updated"),
                authors=[
                    " ".join((node.text or "").split())
                    for node in entry.findall("atom:author/atom:name", ATOM)
                ],
                categories=categories,
                primary_category=primary_node.get("term") if primary_node is not None else None,
                doi=_entry_text(entry, "arxiv:doi") or None,
                journal_reference=_entry_text(entry, "arxiv:journal_ref") or None,
                source_url=source_url,
                pdf_url=pdf_url,
                license_url=_entry_text(entry, "arxiv:license") or None,
            )
        )
    return papers


def score_relevance(paper: ArxivPaper, terms: Mapping[str, float]) -> float:
    """在高成本语料摄取前应用可解释的词法先验。"""

    title = paper.title.lower()
    abstract = paper.abstract.lower()
    score = 100.0 if paper.curated else 0.0
    for term, weight in terms.items():
        normalized_term = term.lower()
        if normalized_term in title:
            score += weight * 2.0
        elif normalized_term in abstract:
            score += weight
    return score


def merge_candidates(candidates: Iterable[ArxivPaper]) -> list[ArxivPaper]:
    """按稳定 arXiv ID 合并重复命中并保留来源。"""

    merged: dict[str, ArxivPaper] = {}
    for candidate in candidates:
        key = candidate.versionless_id
        current = merged.get(key)
        if current is None:
            merged[key] = candidate
            continue
        current.tags = sorted(set(current.tags) | set(candidate.tags))
        current.discovered_by = sorted(set(current.discovered_by) | set(candidate.discovered_by))
        current.curated = current.curated or candidate.curated
        current.relevance_score = max(current.relevance_score, candidate.relevance_score)
    return list(merged.values())


class ArxivHttpClient:
    """仅访问可信主机、限制响应大小并有限退避的轻量 HTTPS 客户端。"""

    def __init__(self, *, timeout: float, retries: int, delay: float, user_agent: str) -> None:
        self.timeout = timeout
        self.retries = retries
        self.delay = delay
        self.user_agent = user_agent

    def fetch_bytes(self, url: str, *, max_bytes: int) -> tuple[bytes, str]:
        """读取一次有界响应，并在存在时遵循 Retry-After。"""

        if not url.startswith(TRUSTED_URL_PREFIXES):
            raise ValueError(f"untrusted URL: {url}")
        request = urllib.request.Request(  # noqa: S310 - 上方已检查 URL 白名单
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/atom+xml,application/pdf",
            },
        )
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = cast(
                    HTTPResponse,
                    urllib.request.urlopen(request, timeout=self.timeout),  # noqa: S310
                )
                with response:
                    content_type = response.headers.get_content_type()
                    announced_size = response.headers.get("Content-Length")
                    if announced_size and int(announced_size) > max_bytes:
                        raise ValueError(f"response exceeds {max_bytes} bytes")
                    data = response.read(max_bytes + 1)
                if len(data) > max_bytes:
                    raise ValueError(f"response exceeds {max_bytes} bytes")
                return data, content_type
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in {408, 429, 500, 502, 503, 504}:
                    break
                retry_after = exc.headers.get("Retry-After")
                wait_seconds = float(retry_after) if retry_after and retry_after.isdigit() else 0.0
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                wait_seconds = 0.0
            if attempt + 1 < self.retries:
                time.sleep(max(wait_seconds, self.delay * (2**attempt)))
        raise RuntimeError(f"request failed after {self.retries} attempts: {url}") from last_error

    def query(self, parameters: Mapping[str, str | int]) -> list[ArxivPaper]:
        """执行一次有界 arXiv API 查询并解析全部返回条目。"""

        url = f"{ARXIV_API_BASE}?{urllib.parse.urlencode(parameters)}"
        data, _ = self.fetch_bytes(url, max_bytes=10 * 1024 * 1024)
        return parse_atom_feed(data)

    def download_pdf(self, paper: ArxivPaper, target: Path) -> DownloadResult:
        """校验并原子写入单篇论文 PDF。"""

        data, content_type = self.fetch_bytes(paper.pdf_url, max_bytes=50 * 1024 * 1024)
        if not data.startswith(b"%PDF-"):
            raise ValueError(f"response is not a PDF (content-type={content_type})")
        atomic_write(target, data)
        return DownloadResult(
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
            content_type=content_type,
        )


def atomic_write(path: Path, data: bytes) -> None:
    """先把完整内容写入同目录临时文件，再替换目标文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(data)
    temporary.replace(path)


def write_json(path: Path, payload: object) -> None:
    """原子写入稳定的 UTF-8 JSON，便于可复现恢复。"""

    atomic_write(
        path,
        (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )


def append_jsonl(path: Path, payload: object) -> None:
    """追加一条不含凭据和响应正文的采集审计事件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def build_metadata(paper: ArxivPaper, result: DownloadResult, pdf_path: Path) -> dict[str, object]:
    """将发现记录转换为项目规范的原始元数据结构。"""

    return {
        "paper_id": f"arxiv:{paper.versionless_id}",
        "title": paper.title,
        "abstract": paper.abstract,
        "year": paper.year,
        "authors": [{"author_id": None, "name": name} for name in paper.authors],
        "doi": paper.doi,
        "arxiv_id": paper.versionless_id,
        "venue": paper.journal_reference,
        "source_url": paper.source_url,
        "pdf_path": pdf_path.as_posix(),
        "references": [],
        "ingestion_version": "v1",
        "source_metadata": {
            "provider": paper.metadata_provider,
            "external_ids": paper.external_ids,
            "primary_category": paper.primary_category,
            "categories": paper.categories,
            "license_url": paper.license_url,
            "tags": paper.tags,
            "discovered_by": paper.discovered_by,
            "relevance_score": paper.relevance_score,
            "published": paper.published,
            "updated": paper.updated,
            "fetched_at": datetime.now(UTC).isoformat(),
        },
        "acquisition": {
            "pdf_url": paper.pdf_url,
            "sha256": result.sha256,
            "size_bytes": result.size_bytes,
            "content_type": result.content_type,
            "downloaded_at": datetime.now(UTC).isoformat(),
        },
    }


def index_local_corpus(metadata_dir: Path, pdf_dir: Path) -> LocalCorpusIndex:
    """只统计签名、大小和哈希均一致的元数据/PDF 文件对。"""

    arxiv_ids: set[str] = set()
    titles: set[str] = set()
    invalid_files: list[str] = []
    for metadata_path in sorted(metadata_dir.glob("*.json")):
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            arxiv_id = normalize_arxiv_id(str(payload["arxiv_id"]))
            title = normalize_title(str(payload["title"]))
            acquisition = payload["acquisition"]
            pdf_value = str(payload["pdf_path"])
            pdf_path = pdf_dir / Path(pdf_value).name
            data = pdf_path.read_bytes()
            if not data.startswith(b"%PDF-"):
                raise ValueError("invalid PDF signature")
            if not isinstance(acquisition, dict):
                raise ValueError("missing acquisition object")
            if acquisition.get("size_bytes") != len(data):
                raise ValueError("size mismatch")
            if acquisition.get("sha256") != hashlib.sha256(data).hexdigest():
                raise ValueError("SHA-256 mismatch")
            arxiv_ids.add(arxiv_id)
            titles.add(title)
        except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            invalid_files.append(f"{metadata_path.name}: {exc}")
    return LocalCorpusIndex(
        valid_count=len(arxiv_ids),
        arxiv_ids=frozenset(arxiv_ids),
        normalized_titles=frozenset(titles),
        invalid_metadata_files=tuple(invalid_files),
    )
