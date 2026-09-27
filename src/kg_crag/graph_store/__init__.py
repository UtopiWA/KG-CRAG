"""图存储契约。"""

from kg_crag.graph_store.base import GraphStore
from kg_crag.graph_store.memory import InMemoryGraphStore

__all__ = ["GraphStore", "InMemoryGraphStore"]
