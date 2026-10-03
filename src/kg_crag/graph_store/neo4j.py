"""Neo4j 5.x 的异步、版本隔离知识图谱适配器。"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Protocol, cast

from kg_crag.errors import KGCRAGError
from kg_crag.graph.schema import bundle_hash
from kg_crag.graph_store.queries import query_text
from kg_crag.models import (
    Chunk,
    ErrorCode,
    ErrorDetail,
    Evidence,
    GraphBundle,
    GraphIdentity,
    GraphPathHit,
    GraphProvenance,
    GraphQueryRequest,
    GraphRecordState,
    GraphSyncResult,
    Paper,
)


class _Result(Protocol):
    async def single(self) -> object | None: ...

    async def data(self) -> list[dict[str, object]]: ...


class _Transaction(Protocol):
    async def run(self, query: object, **parameters: object) -> _Result: ...


class _Session(Protocol):
    async def __aenter__(self) -> _Session: ...

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None: ...

    async def execute_read(
        self, operation: Callable[[_Transaction], Awaitable[object]], **kwargs: object
    ) -> object: ...

    async def execute_write(
        self, operation: Callable[[_Transaction], Awaitable[object]], **kwargs: object
    ) -> object: ...


class _Driver(Protocol):
    def session(self, **kwargs: object) -> _Session: ...

    async def close(self) -> None: ...


def _store_error(stage: str, error: Exception) -> KGCRAGError:
    error_name = type(error).__name__
    lowered = error_name.casefold()
    timeout = "timeout" in lowered
    retryable = timeout or "serviceunavailable" in lowered or "sessionexpired" in lowered
    return KGCRAGError(
        ErrorDetail(
            code=ErrorCode.TIMEOUT if timeout else ErrorCode.EXTERNAL_SERVICE,
            message="Neo4j graph operation failed",
            retryable=retryable,
            context={"stage": stage, "error_type": error_name},
        )
    )


class Neo4jGraphStore:
    """仅通过固定模板读图，并以单篇事务同步完整目标状态。"""

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        *,
        database: str = "neo4j",
        driver: object | None = None,
    ) -> None:
        self.database = database
        if driver is None:
            # 惰性导入和构造，普通模块导入及默认测试不会连接数据库。
            from neo4j import AsyncGraphDatabase

            driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        self._driver = cast(_Driver, driver)
        self._identity: GraphIdentity | None = None

    async def ensure_graph(self, identity: GraphIdentity, *, rebuild: bool = False) -> None:
        async def initialize(tx: _Transaction) -> object:
            await tx.run(
                "CREATE CONSTRAINT graph_entity_key IF NOT EXISTS "
                "FOR (n:GraphEntity) REQUIRE n.graph_key IS UNIQUE"
            )
            if rebuild:
                await tx.run(
                    "MATCH (n {graph_version: $graph_version}) DETACH DELETE n",
                    graph_version=identity.graph_version,
                )
            await tx.run(
                "CREATE CONSTRAINT graph_identity_version IF NOT EXISTS "
                "FOR (n:GraphIdentity) REQUIRE n.graph_version IS UNIQUE"
            )
            result = await tx.run(
                "MATCH (n:GraphIdentity {graph_version: $graph_version}) "
                "RETURN n.payload AS payload",
                graph_version=identity.graph_version,
            )
            existing = await result.single()
            if existing is None:
                await tx.run(
                    "CREATE (:GraphIdentity {graph_version: $graph_version, payload: $payload})",
                    graph_version=identity.graph_version,
                    payload=identity.model_dump_json(),
                )
                return None
            payload = existing["payload"]  # type: ignore[index]
            if GraphIdentity.model_validate_json(payload) != identity:
                raise ValueError("stored graph identity is incompatible")
            return None

        try:
            async with self._driver.session(database=self.database) as session:
                await session.execute_write(initialize)
            self._identity = identity.model_copy(deep=True)
        except Exception as error:
            raise _store_error("ensure_graph", error) from error

    def _require_identity(self, version: str | None = None) -> GraphIdentity:
        if self._identity is None:
            raise KGCRAGError(
                ErrorDetail(
                    code=ErrorCode.CONFIGURATION,
                    message="Neo4j graph identity has not been initialized",
                    retryable=False,
                )
            )
        if version is not None and version != self._identity.graph_version:
            raise KGCRAGError(
                ErrorDetail(
                    code=ErrorCode.VALIDATION,
                    message="requested graph version is incompatible",
                    retryable=False,
                )
            )
        return self._identity

    async def record_state(self, paper_id: str) -> GraphRecordState | None:
        identity = self._require_identity()

        async def read(tx: _Transaction) -> object:
            result = await tx.run(
                "MATCH (s:GraphPaperState {graph_version: $graph_version, paper_id: $paper_id}) "
                "RETURN s.bundle_hash AS bundle_hash, s.entity_count AS entity_count, "
                "s.fact_count AS fact_count",
                graph_version=identity.graph_version,
                paper_id=paper_id,
            )
            return await result.single()

        try:
            async with self._driver.session(database=self.database) as session:
                row = await session.execute_read(read)
            if row is None:
                return None
            return GraphRecordState(
                paper_id=paper_id,
                bundle_hash=row["bundle_hash"],  # type: ignore[index]
                entity_count=row["entity_count"],  # type: ignore[index]
                fact_count=row["fact_count"],  # type: ignore[index]
            )
        except KGCRAGError:
            raise
        except Exception as error:
            raise _store_error("record_state", error) from error

    async def sync_paper(self, bundle: GraphBundle) -> GraphSyncResult:
        identity = self._require_identity(bundle.identity.graph_version)
        if identity != bundle.identity:
            raise ValueError("bundle identity is incompatible with Neo4j graph")
        digest = bundle_hash(bundle)
        previous = await self.record_state(bundle.paper_id)
        if previous is not None and previous.bundle_hash == digest:
            return GraphSyncResult(
                paper_id=bundle.paper_id,
                status="unchanged",
                entity_count=len(bundle.entities),
                fact_count=len(bundle.facts),
            )

        async def synchronize(tx: _Transaction) -> object:
            entity_rows = [
                {
                    "graph_key": f"{identity.graph_version}:{item.entity_id}",
                    "entity_id": item.entity_id,
                    "node_type": item.node_type.value,
                    "name": item.name,
                    "normalized_name": item.normalized_name,
                    "external_id": item.external_id,
                    "paper_scope": item.paper_scope,
                    "aliases": item.aliases,
                    "properties": json.dumps(item.properties, sort_keys=True),
                }
                for item in bundle.entities
            ]
            await tx.run(
                "UNWIND $rows AS row MERGE (n:GraphEntity {graph_key: row.graph_key}) "
                "SET n += row, n.graph_version = $graph_version",
                rows=entity_rows,
                graph_version=identity.graph_version,
            )
            for fact in bundle.facts:
                relation = fact.relation.value
                query = (
                    "MATCH (a:GraphEntity {graph_key: $source_key}), "
                    "(b:GraphEntity {graph_key: $target_key}) "
                    f"MERGE (a)-[r:{relation} "
                    "{graph_version: $graph_version, fact_id: $fact_id}]->(b) "
                    "SET r.source_paper_id = $paper_id, r.confidence = $confidence, "
                    "r.provenance = $provenance"
                )
                await tx.run(
                    query,
                    source_key=f"{identity.graph_version}:{fact.source_entity_id}",
                    target_key=f"{identity.graph_version}:{fact.target_entity_id}",
                    graph_version=identity.graph_version,
                    fact_id=fact.fact_id,
                    paper_id=bundle.paper_id,
                    confidence=fact.provenance.confidence,
                    provenance=fact.provenance.model_dump_json(),
                )
            await tx.run(
                "MATCH ()-[r {graph_version: $graph_version, source_paper_id: $paper_id}]->() "
                "WHERE NOT r.fact_id IN $fact_ids DELETE r",
                graph_version=identity.graph_version,
                paper_id=bundle.paper_id,
                fact_ids=[item.fact_id for item in bundle.facts],
            )
            await tx.run(
                "MATCH (n:GraphEntity {graph_version: $graph_version}) WHERE NOT (n)--() "
                "AND n.paper_scope = $paper_id DELETE n",
                graph_version=identity.graph_version,
                paper_id=bundle.paper_id,
            )
            await tx.run(
                "MERGE (s:GraphPaperState {graph_version: $graph_version, paper_id: $paper_id}) "
                "SET s.bundle_hash = $bundle_hash, s.entity_count = $entity_count, "
                "s.fact_count = $fact_count",
                graph_version=identity.graph_version,
                paper_id=bundle.paper_id,
                bundle_hash=digest,
                entity_count=len(bundle.entities),
                fact_count=len(bundle.facts),
            )
            return None

        try:
            async with self._driver.session(database=self.database) as session:
                await session.execute_write(synchronize)
        except Exception as error:
            raise _store_error("sync_paper", error) from error
        return GraphSyncResult(
            paper_id=bundle.paper_id,
            status="updated" if previous else "created",
            entity_count=len(bundle.entities),
            fact_count=len(bundle.facts),
        )

    async def query(self, request: GraphQueryRequest) -> list[GraphPathHit]:
        self._require_identity(request.graph_version)
        parameters: dict[str, object] = {
            "graph_version": request.graph_version,
            "entity_ids": request.entity_ids,
            "entity_names": [item.casefold().strip() for item in request.entity_names],
            "relation_types": [item.value for item in request.relation_types],
            "min_confidence": request.min_confidence,
            "max_hops": request.max_hops,
            "max_candidates": request.max_candidates,
            "paper_ids": request.paper_ids,
            "node_types": [item.value for item in request.node_types],
        }
        if not parameters["relation_types"]:
            from kg_crag.models import GraphRelationType

            parameters["relation_types"] = [item.value for item in GraphRelationType]

        async def read(tx: _Transaction) -> object:
            from neo4j import Query

            statement = Query(query_text(request.template), timeout=request.timeout_seconds)
            result = await tx.run(statement, **parameters)
            return await result.data()

        try:
            async with self._driver.session(
                database=self.database, default_access_mode="READ"
            ) as session:
                rows = await session.execute_read(read)
            hits: list[GraphPathHit] = []
            for row in cast(list[dict[str, object]], rows):
                fact_ids = cast(list[str], row["fact_ids"])
                names = cast(list[str], row["names"])
                provenances = [
                    GraphProvenance.model_validate_json(item)
                    for item in cast(list[str], row["provenances"])
                ]
                entity_ids = cast(list[str], row["entity_ids"])
                path_id = (
                    __import__("hashlib")
                    .sha256(json.dumps([request.graph_version, entity_ids, fact_ids]).encode())
                    .hexdigest()
                )
                score = row["score"]
                if not isinstance(score, int | float):
                    raise ValueError("Neo4j graph query returned a non-numeric score")
                hits.append(
                    GraphPathHit(
                        path_id=path_id,
                        graph_version=request.graph_version,
                        template=request.template,
                        entity_ids=entity_ids,
                        fact_ids=fact_ids,
                        provenances=provenances,
                        summary=" -> ".join(names),
                        score=float(score),
                        hop_count=len(fact_ids),
                    )
                )
            return hits[: request.max_candidates]
        except Exception as error:
            raise _store_error("query", error) from error

    async def close(self) -> None:
        await self._driver.close()

    async def upsert_paper(self, paper: Paper, chunks: list[Chunk]) -> None:
        raise NotImplementedError("legacy upsert is not supported by Neo4jGraphStore")

    async def retrieve(
        self, entity_names: list[str], *, max_hops: int, top_k: int
    ) -> list[Evidence]:
        raise NotImplementedError("legacy retrieval is not supported by Neo4jGraphStore")

    async def delete_paper(self, paper_id: str) -> None:
        raise NotImplementedError("legacy deletion is not supported by Neo4jGraphStore")
