#!/usr/bin/env python3
"""获取显式且有界的 arXiv 论文集合及可复现元数据。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from http.client import HTTPResponse
from pathlib import Path
from typing import cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "seed_papers.json"
DEFAULT_PDF_DIR = PROJECT_ROOT / "data" / "raw" / "papers"
DEFAULT_METADATA_DIR = PROJECT_ROOT / "data" / "raw" / "metadata"
DEFAULT_LOG = PROJECT_ROOT / "data" / "raw" / "download_manifest.jsonl"
ARXIV_API = "https://export.arxiv.org/api/query?id_list={arxiv_id}"
ARXIV_PDF = "https://arxiv.org/pdf/{arxiv_id}"
ARXIV_ID_PATTERN = re.compile(r"^(?:[a-z-]+(?:\.[A-Z]{2})?/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?$", re.I)
SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,79}$")
ATOM = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
USER_AGENT = "kg-crag-course-project/0.1 (bounded academic corpus fetcher)"


@dataclass(frozen=True)
class SeedPaper:
    arxiv_id: str
    slug: str
    tags: list[str]


@dataclass(frozen=True)
class DownloadResult:
    sha256: str
    size_bytes: int
    content_type: str


def parse_args() -> argparse.Namespace:
    """解析有界下载参数；本脚本有意不支持候选发现。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--limit", type=int, default=10, choices=range(1, 21), metavar="1..20")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3, choices=range(1, 6), metavar="1..5")
    parser.add_argument("--delay", type=float, default=3.0, help="Seconds between HTTP requests")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Replace existing valid artifacts")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_manifest(path: Path, limit: int) -> list[SeedPaper]:
    """发出请求前加载并校验显式白名单。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_papers = payload.get("papers")
    if not isinstance(raw_papers, list):
        raise ValueError("manifest.papers must be a list")

    papers: list[SeedPaper] = []
    seen_ids: set[str] = set()
    for item in raw_papers[:limit]:
        if not isinstance(item, dict):
            raise ValueError("every paper entry must be an object")
        arxiv_id = str(item.get("arxiv_id", "")).strip()
        slug = str(item.get("slug", "")).strip()
        tags = item.get("tags", [])
        if not ARXIV_ID_PATTERN.fullmatch(arxiv_id):
            raise ValueError(f"invalid arXiv ID: {arxiv_id!r}")
        if not SLUG_PATTERN.fullmatch(slug):
            raise ValueError(f"invalid filename slug: {slug!r}")
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ValueError(f"tags for {arxiv_id} must be strings")
        if arxiv_id in seen_ids:
            raise ValueError(f"duplicate arXiv ID: {arxiv_id}")
        seen_ids.add(arxiv_id)
        papers.append(SeedPaper(arxiv_id=arxiv_id, slug=slug, tags=tags))
    return papers


def open_with_retry(url: str, *, timeout: float, retries: int, delay: float) -> HTTPResponse:
    """以有限指数退避访问一个可信 arXiv 端点。"""

    trusted_prefixes = ("https://arxiv.org/", "https://export.arxiv.org/")
    if not url.startswith(trusted_prefixes):
        raise ValueError("only trusted HTTPS arXiv endpoints are allowed")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return cast(HTTPResponse, urllib.request.urlopen(request, timeout=timeout))  # noqa: S310
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(delay * (2**attempt))
    raise RuntimeError(f"request failed after {retries} attempts: {url}") from last_error


def fetch_bytes(
    url: str,
    *,
    timeout: float,
    retries: int,
    delay: float,
    max_bytes: int,
) -> tuple[bytes, str]:
    """将限制大小的响应读入内存，并确保关闭连接。"""

    with open_with_retry(url, timeout=timeout, retries=retries, delay=delay) as response:
        content_type = response.headers.get_content_type()
        announced_size = response.headers.get("Content-Length")
        if announced_size and int(announced_size) > max_bytes:
            raise ValueError(f"response exceeds {max_bytes} bytes")
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"response exceeds {max_bytes} bytes")
    return data, content_type


def parse_arxiv_metadata(xml_bytes: bytes, paper: SeedPaper) -> dict[str, object]:
    """将 Atom 条目转换为项目稳定且带来源的记录结构。"""

    root = ET.fromstring(xml_bytes)  # noqa: S314 - 响应大小受限且不会展开外部实体
    entry = root.find("atom:entry", ATOM)
    if entry is None:
        raise ValueError(f"arXiv returned no entry for {paper.arxiv_id}")

    def text(name: str) -> str:
        element = entry.find(name, ATOM)
        return " ".join((element.text or "").split()) if element is not None else ""

    authors = [
        {"author_id": None, "name": " ".join((node.text or "").split())}
        for node in entry.findall("atom:author/atom:name", ATOM)
    ]
    published = text("atom:published")
    doi = text("arxiv:doi") or None
    primary_category = entry.find("arxiv:primary_category", ATOM)
    category = primary_category.get("term") if primary_category is not None else None
    fetched_at = datetime.now(UTC).isoformat()
    versionless_id = re.sub(r"v\d+$", "", paper.arxiv_id)
    return {
        "paper_id": f"arxiv:{versionless_id}",
        "title": text("atom:title"),
        "abstract": text("atom:summary"),
        "year": int(published[:4]) if published else None,
        "authors": authors,
        "doi": doi,
        "arxiv_id": paper.arxiv_id,
        "venue": text("arxiv:journal_ref") or None,
        "source_url": f"https://arxiv.org/abs/{urllib.parse.quote(paper.arxiv_id)}",
        "pdf_path": f"data/raw/papers/{paper.slug}.pdf",
        "references": [],
        "ingestion_version": "v1",
        "source_metadata": {
            "provider": "arXiv",
            "api_url": ARXIV_API.format(arxiv_id=paper.arxiv_id),
            "primary_category": category,
            "tags": paper.tags,
            "fetched_at": fetched_at,
        },
    }


def atomic_write(path: Path, data: bytes) -> None:
    """只在完整内容可用后替换目标产物。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(data)
    temporary.replace(path)


def download_pdf(url: str, target: Path, args: argparse.Namespace) -> DownloadResult:
    """下载、校验、计算哈希并原子保存单个 PDF。"""

    data, content_type = fetch_bytes(
        url,
        timeout=args.timeout,
        retries=args.retries,
        delay=args.delay,
        max_bytes=50 * 1024 * 1024,
    )
    if not data.startswith(b"%PDF-"):
        raise ValueError(f"response is not a PDF (content-type={content_type})")
    atomic_write(target, data)
    return DownloadResult(
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        content_type=content_type,
    )


def append_log(path: Path, record: dict[str, object]) -> None:
    """追加一条不含凭据的 JSON Lines 审计记录。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def existing_result(pdf_path: Path, metadata_path: Path) -> DownloadResult | None:
    """校验可复用产物，而不是只判断文件是否存在。"""

    if not pdf_path.is_file() or not metadata_path.is_file():
        return None
    data = pdf_path.read_bytes()
    if not data.startswith(b"%PDF-"):
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    acquisition = metadata.get("acquisition", {})
    sha256 = hashlib.sha256(data).hexdigest()
    if not isinstance(acquisition, dict) or acquisition.get("sha256") != sha256:
        return None
    return DownloadResult(
        sha256=sha256,
        size_bytes=len(data),
        content_type=str(acquisition.get("content_type", "application/pdf")),
    )


def process_paper(paper: SeedPaper, args: argparse.Namespace) -> str:
    """获取一篇白名单论文，并留下可审计的逐篇结果。"""

    pdf_path = args.pdf_dir / f"{paper.slug}.pdf"
    metadata_path = args.metadata_dir / f"{paper.slug}.json"
    if not args.force and not args.metadata_only:
        reusable = existing_result(pdf_path, metadata_path)
        if reusable is not None:
            append_log(
                args.log,
                {
                    "arxiv_id": paper.arxiv_id,
                    "status": "skipped_valid",
                    "sha256": reusable.sha256,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )
            return "skipped_valid"

    api_url = ARXIV_API.format(arxiv_id=urllib.parse.quote(paper.arxiv_id))
    xml_bytes, _ = fetch_bytes(
        api_url,
        timeout=args.timeout,
        retries=args.retries,
        delay=args.delay,
        max_bytes=2 * 1024 * 1024,
    )
    metadata = parse_arxiv_metadata(xml_bytes, paper)
    time.sleep(args.delay)

    if args.metadata_only:
        metadata["pdf_path"] = None
        result = None
    else:
        pdf_url = ARXIV_PDF.format(arxiv_id=urllib.parse.quote(paper.arxiv_id))
        result = download_pdf(pdf_url, pdf_path, args)
        metadata["acquisition"] = {
            "pdf_url": pdf_url,
            "sha256": result.sha256,
            "size_bytes": result.size_bytes,
            "content_type": result.content_type,
            "downloaded_at": datetime.now(UTC).isoformat(),
        }

    atomic_write(
        metadata_path,
        (json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )
    append_log(
        args.log,
        {
            "arxiv_id": paper.arxiv_id,
            "metadata_path": str(metadata_path),
            "pdf_path": None if args.metadata_only else str(pdf_path),
            "sha256": result.sha256 if result else None,
            "status": "metadata_only" if args.metadata_only else "downloaded",
            "timestamp": datetime.now(UTC).isoformat(),
        },
    )
    return "metadata_only" if args.metadata_only else "downloaded"


def main() -> int:
    """执行有界获取，跳过单篇失败并汇总结果。"""

    args = parse_args()
    papers = load_manifest(args.manifest.resolve(), args.limit)
    if args.dry_run:
        print(f"Would process {len(papers)} explicitly allowlisted papers:")
        for paper in papers:
            print(f"- {paper.arxiv_id}: {paper.slug} [{', '.join(paper.tags)}]")
        return 0

    statuses: dict[str, int] = {}
    for index, paper in enumerate(papers):
        try:
            status = process_paper(paper, args)
            print(f"[{index + 1}/{len(papers)}] {paper.arxiv_id}: {status}")
        except (OSError, ValueError, RuntimeError, ET.ParseError, json.JSONDecodeError) as exc:
            status = "failed"
            print(f"[{index + 1}/{len(papers)}] {paper.arxiv_id}: failed: {exc}", file=sys.stderr)
            append_log(
                args.log,
                {
                    "arxiv_id": paper.arxiv_id,
                    "error": str(exc),
                    "status": status,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )
        statuses[status] = statuses.get(status, 0) + 1
        if index + 1 < len(papers):
            time.sleep(args.delay)

    print("Summary: " + ", ".join(f"{key}={value}" for key, value in sorted(statuses.items())))
    return 1 if statuses.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
