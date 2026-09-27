"""离线实现的 Protocol 与无网络保证测试。"""

import socket
from typing import NoReturn

from pytest import MonkeyPatch

from kg_crag.graph_store import GraphStore, InMemoryGraphStore
from kg_crag.retrieval import MockReranker, MockRetriever, Reranker, Retriever
from kg_crag.vector_store import InMemoryVectorStore, VectorStore


def _deny_network(*args: object, **kwargs: object) -> NoReturn:
    del args, kwargs
    raise AssertionError("offline implementations must not access the network")


async def test_offline_implementations_conform_to_protocols_without_network(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(socket, "create_connection", _deny_network)

    retriever = MockRetriever([])
    reranker = MockReranker()
    vector_store = InMemoryVectorStore()
    graph_store = InMemoryGraphStore()

    assert isinstance(retriever, Retriever)
    assert isinstance(reranker, Reranker)
    assert isinstance(vector_store, VectorStore)
    assert isinstance(graph_store, GraphStore)

    assert await retriever.retrieve("query", top_k=1) == []
    assert await reranker.rerank("query", [], top_k=1) == []
    assert await vector_store.search([1.0], top_k=1) == []
    assert await graph_store.retrieve(["entity"], max_hops=0, top_k=1) == []
