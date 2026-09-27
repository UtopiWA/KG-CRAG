"""向量存储契约。"""

from kg_crag.vector_store.base import VectorStore
from kg_crag.vector_store.memory import InMemoryVectorStore

__all__ = ["InMemoryVectorStore", "VectorStore"]
