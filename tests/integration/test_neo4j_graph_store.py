"""显式启用的真实 Neo4j 图存储 smoke。"""

import os
from pathlib import Path

import pytest

from kg_crag.graph.config import load_graph_config
from kg_crag.graph.metadata import plan_metadata_graph
from kg_crag.graph_store import Neo4jGraphStore
from kg_crag.indexing.processed import ProcessedPaper
from kg_crag.models import Chunk, Paper
from kg_crag.settings import get_settings

pytestmark = pytest.mark.skipif(
    os.getenv("KG_CRAG_RUN_NEO4J_TESTS") != "1",
    reason="set KG_CRAG_RUN_NEO4J_TESTS=1 to run the real Neo4j graph integration test",
)


async def test_neo4j_graph_store_is_idempotent_and_version_isolated() -> None:
    settings = get_settings()
    config = load_graph_config(Path("configs/retrieval.yaml"))
    paper = Paper(paper_id="neo4j-smoke", title="Neo4j Smoke")
    chunk = Chunk(
        chunk_id="neo4j-smoke-c1",
        paper_id=paper.paper_id,
        text="bounded graph smoke",
        token_count=3,
        content_hash="e" * 64,
        processing_version="v1",
    )
    processed = ProcessedPaper(paper.paper_id, "v1", (chunk,), Path("."), paper, "f" * 64)
    plan = plan_metadata_graph([processed], config, model="integration")
    store = Neo4jGraphStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    try:
        await store.ensure_graph(plan.identity, rebuild=True)
        assert (await store.sync_paper(plan.bundles[0])).status == "created"
        assert (await store.sync_paper(plan.bundles[0])).status == "unchanged"
    finally:
        await store.close()
