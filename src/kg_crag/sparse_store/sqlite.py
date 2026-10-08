"""带身份校验和论文级事务的 SQLite FTS5 Sparse Store。"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sqlite3
import uuid
from pathlib import Path

from pydantic import ValidationError

from kg_crag.errors import KGCRAGError
from kg_crag.models import (
    Chunk,
    ErrorCode,
    ErrorDetail,
    Evidence,
    EvidenceLocation,
    EvidenceRanks,
    EvidenceScores,
    EvidenceSourceType,
    SparseIndexIdentity,
)
from kg_crag.sparse_store.base import SparseRecordState, SparseSyncResult
from kg_crag.sparse_store.tokenizer import prepare_sparse_query

FilterValue = str | int | bool
_ALLOWED_FILTERS = frozenset({"paper_id", "section", "processing_version"})
_REQUIRED_TABLES = frozenset({"metadata", "chunks", "chunks_fts"})

_SCHEMA_SQL = """
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE chunks (
    row_id INTEGER PRIMARY KEY,
    chunk_id TEXT NOT NULL UNIQUE,
    paper_id TEXT NOT NULL,
    section TEXT,
    page_start INTEGER,
    page_end INTEGER,
    text TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    processing_version TEXT NOT NULL
);
CREATE INDEX chunks_paper_id_idx ON chunks(paper_id);
CREATE INDEX chunks_processing_version_idx ON chunks(processing_version);
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='row_id',
    tokenize='unicode61 remove_diacritics 0'
);
CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.row_id, new.text);
END;
CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.row_id, old.text);
END;
CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.row_id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.row_id, new.text);
END;
"""


def build_sqlite_index_identity(
    *,
    schema_version: str,
    tokenizer: str,
    bm25_version: str,
    corpus_snapshot_hash: str,
) -> SparseIndexIdentity:
    """把实现版本和语料快照纳入可复现索引身份。"""

    payload = {
        "schema_version": schema_version,
        "tokenizer": tokenizer,
        "bm25_version": bm25_version,
        "python_version": platform.python_version(),
        "sqlite_version": sqlite3.sqlite_version,
        "fts5_enabled": True,
        "fts5_version": f"sqlite-builtin-{sqlite3.sqlite_version}",
        "corpus_snapshot_hash": corpus_snapshot_hash,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return SparseIndexIdentity(**payload, index_version=hashlib.sha256(encoded).hexdigest())


class SQLiteSparseStore:
    """使用独立 manifest 拒绝静默复用不兼容或损坏的 FTS5 索引。"""

    def __init__(
        self,
        db_path: Path,
        identity: SparseIndexIdentity,
        *,
        max_top_k: int = 100,
        max_candidates: int = 100,
        max_query_chars: int = 2000,
        max_query_tokens: int = 64,
    ) -> None:
        self.db_path = db_path
        self.manifest_path = db_path.with_suffix(f"{db_path.suffix}.manifest.json")
        self._identity = identity
        self.max_top_k = max_top_k
        self.max_candidates = max_candidates
        self.max_query_chars = max_query_chars
        self.max_query_tokens = max_query_tokens

    @property
    def identity(self) -> SparseIndexIdentity:
        return self._identity

    @property
    def is_initialized(self) -> bool:
        return self.db_path.is_file() and self.manifest_path.is_file()

    async def ensure_index(self, identity: SparseIndexIdentity, *, rebuild: bool = False) -> None:
        if identity != self._identity:
            raise _store_error(
                ErrorCode.CONFIGURATION, "requested sparse index identity does not match"
            )
        try:
            _probe_fts5()
            if rebuild:
                self._remove_exact_index_files()
            exists = self.db_path.exists()
            manifest_exists = self.manifest_path.exists()
            if exists != manifest_exists:
                raise _store_error(ErrorCode.DATA, "sparse index or manifest is missing")
            if not exists:
                self._create_index_atomically()
            self._verify_existing_index()
        except KGCRAGError:
            raise
        except (OSError, sqlite3.DatabaseError, ValidationError, ValueError) as error:
            raise _store_error(ErrorCode.DATA, "sparse index initialization failed") from error

    async def sync_paper(self, paper_id: str, chunks: list[Chunk]) -> SparseSyncResult:
        if not paper_id.strip():
            raise _store_error(ErrorCode.VALIDATION, "paper_id must not be empty")
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise _store_error(ErrorCode.VALIDATION, "chunk IDs must be unique within a paper sync")
        mismatch = next((chunk.chunk_id for chunk in chunks if chunk.paper_id != paper_id), None)
        if mismatch is not None:
            raise _store_error(
                ErrorCode.VALIDATION,
                "chunk paper_id must match the paper being synchronized",
                {"paper_id": paper_id, "chunk_id": mismatch},
            )

        connection = self._connect_verified()
        try:
            previous = _record_state(connection, paper_id)
            incoming = {chunk.chunk_id: chunk for chunk in chunks}
            added_ids = incoming.keys() - previous.keys()
            deleted_ids = previous.keys() - incoming.keys()
            updated_ids = {
                chunk_id
                for chunk_id in incoming.keys() & previous.keys()
                if incoming[chunk_id].content_hash != previous[chunk_id].content_hash
                or incoming[chunk_id].processing_version != previous[chunk_id].processing_version
            }
            skipped = len(incoming) - len(added_ids) - len(updated_ids)

            # 同一 with 块只有一次提交；任一写入失败会回滚本篇的全部变化。
            with connection:
                for chunk_id in sorted(added_ids | updated_ids):
                    self._write_chunk(connection, incoming[chunk_id])
                for chunk_id in sorted(deleted_ids):
                    connection.execute(
                        "DELETE FROM chunks WHERE paper_id = ? AND chunk_id = ?",
                        (paper_id, chunk_id),
                    )
            return SparseSyncResult(
                paper_id, len(added_ids), len(updated_ids), skipped, len(deleted_ids)
            )
        except KGCRAGError:
            raise
        except sqlite3.DatabaseError as error:
            raise _store_error(ErrorCode.DATA, "sparse paper transaction failed") from error
        finally:
            connection.close()

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        offset: int = 0,
        filters: dict[str, FilterValue] | None = None,
    ) -> list[Evidence]:
        if top_k <= 0 or top_k > self.max_top_k:
            raise _store_error(
                ErrorCode.VALIDATION,
                "top_k is outside the sparse search limit",
                {"top_k": top_k, "max_top_k": self.max_top_k},
            )
        if offset < 0 or offset + top_k > self.max_candidates:
            raise _store_error(
                ErrorCode.VALIDATION,
                "sparse result window exceeds candidate limit",
                {"offset": offset, "window": offset + top_k, "max_count": self.max_candidates},
            )
        selected_filters = dict(filters or {})
        unknown = set(selected_filters) - _ALLOWED_FILTERS
        if unknown:
            raise _store_error(
                ErrorCode.VALIDATION, "unknown sparse filter", {"filter": sorted(unknown)[0]}
            )
        prepared = prepare_sparse_query(
            query, max_chars=self.max_query_chars, max_tokens=self.max_query_tokens
        )
        clauses = ["chunks_fts MATCH ?"]
        parameters: list[FilterValue] = [prepared.fts_query]
        for name, value in sorted(selected_filters.items()):
            clauses.append(f"chunks.{name} = ?")
            parameters.append(value)
        parameters.extend([top_k, offset])
        sql = (
            "SELECT chunks.chunk_id, chunks.paper_id, chunks.section, "  # noqa: S608
            "chunks.page_start, chunks.text, chunks.content_hash, "
            "chunks.processing_version, -bm25(chunks_fts) AS sparse_score "
            "FROM chunks_fts JOIN chunks ON chunks.row_id = chunks_fts.rowid "
            f"WHERE {' AND '.join(clauses)} "
            "ORDER BY sparse_score DESC, chunks.chunk_id ASC LIMIT ? OFFSET ?"
        )
        connection = self._connect_verified()
        try:
            rows = connection.execute(sql, parameters).fetchall()
            return [
                _row_to_evidence(self.identity.index_version, row, rank)
                for rank, row in enumerate(rows, start=offset + 1)
            ]
        except sqlite3.DatabaseError as error:
            raise _store_error(ErrorCode.DATA, "sparse search failed") from error
        finally:
            connection.close()

    async def record_state(self, paper_id: str) -> dict[str, SparseRecordState]:
        connection = self._connect_verified()
        try:
            return _record_state(connection, paper_id)
        finally:
            connection.close()

    async def paper_context(self, paper_id: str, *, limit: int) -> list[Evidence]:
        """从当前冻结索引读取论文上下文，摘要优先且不执行无界 FTS 查询。"""

        if not paper_id.strip() or limit <= 0 or limit > 20:
            raise _store_error(ErrorCode.VALIDATION, "invalid paper context request")
        connection = self._connect_verified()
        try:
            rows = connection.execute(
                """
                SELECT chunk_id, paper_id, section, page_start, text, content_hash,
                       processing_version
                FROM chunks
                WHERE paper_id = ?
                ORDER BY
                    CASE
                        WHEN lower(COALESCE(section, '')) LIKE '%abstract%'
                          OR lower(ltrim(text)) LIKE 'abstract%' THEN 0
                        ELSE 1
                    END,
                    ordinal ASC,
                    chunk_id ASC
                LIMIT ?
                """,
                (paper_id, limit),
            ).fetchall()
            return [_row_to_context_evidence(self.identity.index_version, row) for row in rows]
        except sqlite3.DatabaseError as error:
            raise _store_error(ErrorCode.DATA, "sparse paper context lookup failed") from error
        finally:
            connection.close()

    def _write_chunk(self, connection: sqlite3.Connection, chunk: Chunk) -> None:
        connection.execute(
            """
            INSERT INTO chunks(
                chunk_id, paper_id, section, page_start, page_end, text,
                token_count, content_hash, ordinal, processing_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                paper_id=excluded.paper_id,
                section=excluded.section,
                page_start=excluded.page_start,
                page_end=excluded.page_end,
                text=excluded.text,
                token_count=excluded.token_count,
                content_hash=excluded.content_hash,
                ordinal=excluded.ordinal,
                processing_version=excluded.processing_version
            """,
            (
                chunk.chunk_id,
                chunk.paper_id,
                chunk.section,
                chunk.page_start,
                chunk.page_end,
                chunk.text,
                chunk.token_count,
                chunk.content_hash,
                chunk.ordinal,
                chunk.processing_version,
            ),
        )

    def _create_index_atomically(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.db_path.with_name(f".{self.db_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            connection = sqlite3.connect(temporary)
            try:
                connection.executescript(_SCHEMA_SQL)
                connection.execute(
                    "INSERT INTO metadata(key, value) VALUES ('identity', ?)",
                    (_identity_json(self.identity),),
                )
                connection.commit()
                _verify_database(connection, self.identity)
            finally:
                connection.close()
            os.replace(temporary, self.db_path)
            _write_manifest_atomically(self.manifest_path, self.identity)
        except Exception:
            temporary.unlink(missing_ok=True)
            if self.db_path.exists() and not self.manifest_path.exists():
                self.db_path.unlink(missing_ok=True)
            raise

    def _verify_existing_index(self) -> None:
        manifest_identity = _read_manifest(self.manifest_path)
        if manifest_identity != self.identity:
            raise _store_error(ErrorCode.CONFIGURATION, "sparse index manifest is incompatible")
        connection = sqlite3.connect(self.db_path)
        try:
            _verify_database(connection, self.identity)
        finally:
            connection.close()

    def _connect_verified(self) -> sqlite3.Connection:
        self._verify_existing_index()
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _remove_exact_index_files(self) -> None:
        for path in (
            self.db_path,
            self.manifest_path,
            Path(f"{self.db_path}-wal"),
            Path(f"{self.db_path}-shm"),
        ):
            path.unlink(missing_ok=True)


def _probe_fts5() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE VIRTUAL TABLE fts5_probe USING fts5(value)")
    except sqlite3.DatabaseError as error:
        raise _store_error(
            ErrorCode.CONFIGURATION, "Python SQLite build does not support FTS5"
        ) from error
    finally:
        connection.close()


def _verify_database(connection: sqlite3.Connection, identity: SparseIndexIdentity) -> None:
    integrity = connection.execute("PRAGMA quick_check").fetchone()
    if integrity is None or integrity[0] != "ok":
        raise _store_error(ErrorCode.DATA, "sparse index integrity check failed")
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        )
    }
    if not _REQUIRED_TABLES.issubset(tables):
        raise _store_error(ErrorCode.DATA, "sparse index schema is incomplete")
    row = connection.execute("SELECT value FROM metadata WHERE key = 'identity'").fetchone()
    if row is None:
        raise _store_error(ErrorCode.DATA, "sparse index identity is missing")
    try:
        stored = SparseIndexIdentity.model_validate_json(row[0])
    except ValidationError as error:
        raise _store_error(ErrorCode.DATA, "sparse index identity is invalid") from error
    if stored != identity:
        raise _store_error(ErrorCode.CONFIGURATION, "sparse database identity is incompatible")


def _identity_json(identity: SparseIndexIdentity) -> str:
    return json.dumps(identity.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)


def _write_manifest_atomically(path: Path, identity: SparseIndexIdentity) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = {"schema_version": "v1", "identity": identity.model_dump(mode="json")}
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_manifest(path: Path) -> SparseIndexIdentity:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "v1":
            raise ValueError("unsupported manifest schema")
        return SparseIndexIdentity.model_validate(payload["identity"])
    except (
        OSError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValidationError,
        ValueError,
    ) as error:
        raise _store_error(ErrorCode.DATA, "sparse index manifest is invalid") from error


def load_sqlite_index_identity(db_path: Path) -> SparseIndexIdentity:
    """只读取指定数据库旁的 manifest，不创建、重建或打开索引。"""

    return _read_manifest(db_path.with_suffix(f"{db_path.suffix}.manifest.json"))


def _record_state(connection: sqlite3.Connection, paper_id: str) -> dict[str, SparseRecordState]:
    rows = connection.execute(
        "SELECT chunk_id, content_hash, processing_version FROM chunks WHERE paper_id = ?",
        (paper_id,),
    )
    return {
        row[0]: SparseRecordState(content_hash=row[1], processing_version=row[2]) for row in rows
    }


def _row_to_evidence(index_version: str, row: sqlite3.Row, rank: int) -> Evidence:
    return Evidence(
        evidence_id=f"sparse:{index_version}:{row['chunk_id']}",
        content=row["text"],
        source_type=EvidenceSourceType.CHUNK,
        source_id=row["chunk_id"],
        paper_id=row["paper_id"],
        location=EvidenceLocation(section=row["section"], page=row["page_start"]),
        scores=EvidenceScores(sparse=row["sparse_score"]),
        ranks=EvidenceRanks(sparse=rank),
        external=False,
        metadata={
            "index_version": index_version,
            "content_hash": row["content_hash"],
            "processing_version": row["processing_version"],
        },
    )


def _row_to_context_evidence(index_version: str, row: sqlite3.Row) -> Evidence:
    return Evidence(
        evidence_id=f"paper-context:{index_version}:{row['chunk_id']}",
        content=row["text"],
        source_type=EvidenceSourceType.CHUNK,
        source_id=row["chunk_id"],
        paper_id=row["paper_id"],
        location=EvidenceLocation(section=row["section"], page=row["page_start"]),
        external=False,
        metadata={
            "index_version": index_version,
            "content_hash": row["content_hash"],
            "processing_version": row["processing_version"],
            "paper_context_expansion": True,
        },
    )


def _store_error(
    code: ErrorCode,
    message: str,
    context: dict[str, str | int | float | bool | None] | None = None,
) -> KGCRAGError:
    return KGCRAGError(
        ErrorDetail(code=code, message=message, retryable=False, context=context or {})
    )
