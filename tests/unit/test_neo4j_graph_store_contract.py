"""不连接数据库的 Neo4j 事务边界与错误脱敏测试。"""

from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from kg_crag.errors import KGCRAGError
from kg_crag.graph.config import load_graph_config
from kg_crag.graph.metadata import MetadataGraphPlan, plan_metadata_graph
from kg_crag.graph_store import Neo4jGraphStore
from kg_crag.indexing.processed import ProcessedPaper
from kg_crag.models import Chunk, Paper


class _Result:
    def __init__(self, row: object | None = None) -> None:
        self.row = row

    async def single(self) -> object | None:
        return self.row

    async def data(self) -> list[dict[str, object]]:
        return []


class _Transaction:
    def __init__(self) -> None:
        self.fail_fact_write = False
        self.queries: list[str] = []

    async def run(self, query: object, **parameters: object) -> _Result:
        text = str(query)
        self.queries.append(text)
        if self.fail_fact_write and "MERGE (a)-[r:" in text:
            raise RuntimeError("password=must-not-leak")
        return _Result()


class _Session:
    def __init__(self, transaction: _Transaction) -> None:
        self.transaction = transaction

    async def __aenter__(self) -> "_Session":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def execute_read(
        self, operation: Callable[[_Transaction], Awaitable[object]], **kwargs: object
    ) -> object:
        return await operation(self.transaction)

    async def execute_write(
        self, operation: Callable[[_Transaction], Awaitable[object]], **kwargs: object
    ) -> object:
        return await operation(self.transaction)


class _Driver:
    def __init__(self) -> None:
        self.transaction = _Transaction()

    def session(self, **kwargs: object) -> _Session:
        return _Session(self.transaction)

    async def close(self) -> None:
        return None


def _plan(paper_id: str) -> MetadataGraphPlan:
    config = load_graph_config(Path("configs/retrieval.yaml"))
    paper = Paper(paper_id=paper_id, title=f"Paper {paper_id}")
    chunk = Chunk(
        chunk_id=f"{paper_id}-c1",
        paper_id=paper_id,
        text="graph transaction fixture",
        token_count=3,
        content_hash="a" * 64,
        processing_version="v1",
    )
    processed = ProcessedPaper(paper_id, "v1", (chunk,), Path("."), paper, "b" * 64)
    return plan_metadata_graph([processed], config, model="mock")


async def test_neo4j_failure_is_sanitized_and_next_paper_can_continue() -> None:
    plan = _plan("p1")
    driver = _Driver()
    store = Neo4jGraphStore("bolt://unused", "unused", "secret", driver=driver)
    await store.ensure_graph(plan.identity)
    driver.transaction.fail_fact_write = True
    with pytest.raises(KGCRAGError) as caught:
        await store.sync_paper(plan.bundles[0])
    assert "must-not-leak" not in str(caught.value.detail)

    driver.transaction.fail_fact_write = False
    assert (await store.sync_paper(plan.bundles[0])).status == "created"
    assert any("GraphPaperState" in query for query in driver.transaction.queries)
