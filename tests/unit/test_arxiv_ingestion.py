"""arXiv 确定性解析、排序与本地校验测试。"""

import hashlib
import json
from pathlib import Path

import pytest

from kg_crag.ingestion.arxiv import (
    ArxivPaper,
    index_local_corpus,
    merge_candidates,
    normalize_arxiv_id,
    parse_atom_feed,
    score_relevance,
)
from kg_crag.ingestion.openalex import extract_arxiv_id, parse_openalex_works

ATOM_FIXTURE = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>https://arxiv.org/abs/2401.00001v2</id>
    <updated>2024-02-01T00:00:00Z</updated>
    <published>2024-01-01T00:00:00Z</published>
    <title>An Agentic Retrieval-Augmented Generation System</title>
    <summary>A language model agent uses retrieval and reflection.</summary>
    <author><name>Example Author</name></author>
    <category term="cs.AI"/>
    <category term="cs.CL"/>
    <arxiv:primary_category term="cs.AI"/>
    <link rel="alternate" href="https://arxiv.org/abs/2401.00001v2"/>
    <link title="pdf" href="https://arxiv.org/pdf/2401.00001v2" type="application/pdf"/>
  </entry>
</feed>
"""


def make_paper(arxiv_id: str = "2401.00001") -> ArxivPaper:
    return ArxivPaper(
        arxiv_id=arxiv_id,
        title="Agentic RAG",
        abstract="A retrieval augmented generation agent.",
        published="2024-01-01T00:00:00Z",
        updated="2024-01-01T00:00:00Z",
        authors=["Example Author"],
        categories=["cs.AI"],
        primary_category="cs.AI",
        doi=None,
        journal_reference=None,
        source_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        license_url=None,
    )


def test_parse_atom_feed_normalizes_versioned_id() -> None:
    papers = parse_atom_feed(ATOM_FIXTURE)
    assert len(papers) == 1
    assert papers[0].arxiv_id == "2401.00001"
    assert papers[0].primary_category == "cs.AI"
    assert papers[0].authors == ["Example Author"]


def test_normalize_arxiv_id_rejects_non_arxiv_value() -> None:
    with pytest.raises(ValueError, match="invalid arXiv ID"):
        normalize_arxiv_id("../../secret")


def test_score_and_merge_preserve_discovery_provenance() -> None:
    first = make_paper()
    first.tags = ["rag"]
    first.discovered_by = ["rag_query"]
    second = make_paper("2401.00001v3")
    second.tags = ["agent"]
    second.discovered_by = ["agent_query"]
    merged = merge_candidates([first, second])
    assert len(merged) == 1
    assert merged[0].tags == ["agent", "rag"]
    assert merged[0].discovered_by == ["agent_query", "rag_query"]
    assert score_relevance(merged[0], {"agentic": 3.0, "retrieval augmented": 2.0}) == 8.0


def test_index_local_corpus_counts_only_checksum_valid_pairs(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "papers"
    metadata_dir = tmp_path / "metadata"
    pdf_dir.mkdir()
    metadata_dir.mkdir()
    pdf_data = b"%PDF-1.4\nfixture"
    (pdf_dir / "paper.pdf").write_bytes(pdf_data)
    metadata = {
        "arxiv_id": "2401.00001",
        "title": "Agentic RAG",
        "pdf_path": "data/raw/papers/paper.pdf",
        "acquisition": {
            "size_bytes": len(pdf_data),
            "sha256": hashlib.sha256(pdf_data).hexdigest(),
        },
    }
    (metadata_dir / "paper.json").write_text(json.dumps(metadata), encoding="utf-8")

    index = index_local_corpus(metadata_dir, pdf_dir)
    assert index.valid_count == 1
    assert index.invalid_metadata_files == ()


def test_parse_openalex_work_keeps_only_arxiv_backed_record() -> None:
    payload = {
        "results": [
            {
                "id": "https://openalex.org/W123",
                "doi": "https://doi.org/10.48550/arXiv.2401.00001",
                "title": "Agentic Retrieval",
                "publication_year": 2024,
                "publication_date": "2024-01-01",
                "ids": {"arxiv": "https://arxiv.org/abs/2401.00001v2"},
                "authorships": [{"author": {"display_name": "Ada Researcher"}}],
                "locations": [],
                "abstract_inverted_index": {"Agentic": [0], "retrieval": [1]},
            },
            {
                "id": "https://openalex.org/W456",
                "title": "No open preprint",
                "publication_year": 2024,
                "ids": {},
                "locations": [],
            },
        ]
    }
    papers = parse_openalex_works(payload)
    assert len(papers) == 1
    assert papers[0].arxiv_id == "2401.00001"
    assert papers[0].authors == ["Ada Researcher"]
    assert papers[0].abstract == "Agentic retrieval"
    assert papers[0].metadata_provider == "OpenAlex"


def test_extract_arxiv_id_falls_back_to_location() -> None:
    work = {
        "ids": {},
        "locations": [{"pdf_url": "https://arxiv.org/pdf/2401.00001v3.pdf"}],
    }
    assert extract_arxiv_id(work) == "2401.00001"
