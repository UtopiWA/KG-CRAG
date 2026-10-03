"""Sparse Store 契约、内存替身和安全分词工具。"""

from kg_crag.models import SparseIndexIdentity
from kg_crag.sparse_store.base import SparseRecordState, SparseStore, SparseSyncResult
from kg_crag.sparse_store.memory import InMemorySparseStore, SparseSearchCall
from kg_crag.sparse_store.sqlite import (
    SQLiteSparseStore,
    build_sqlite_index_identity,
    load_sqlite_index_identity,
)
from kg_crag.sparse_store.tokenizer import (
    TOKENIZER_VERSION,
    PreparedSparseQuery,
    normalize_scientific_text,
    prepare_sparse_query,
    tokenize_scientific_text,
)

__all__ = [
    "TOKENIZER_VERSION",
    "InMemorySparseStore",
    "PreparedSparseQuery",
    "SQLiteSparseStore",
    "SparseIndexIdentity",
    "SparseRecordState",
    "SparseSearchCall",
    "SparseStore",
    "SparseSyncResult",
    "build_sqlite_index_identity",
    "load_sqlite_index_identity",
    "normalize_scientific_text",
    "prepare_sparse_query",
    "tokenize_scientific_text",
]
