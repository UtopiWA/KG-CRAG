# Architecture

KG-CRAG starts as a modular monolith. FastAPI exposes stable HTTP contracts; a future LangGraph workflow coordinates query analysis, routing, retrieval, correction, generation, and critique. Dense/sparse retrieval uses Qdrant, graph retrieval uses Neo4j, and every backend is accessed through a typed interface.

The authoritative end-to-end flow is:

```text
Query -> Analyze -> Route -> Retrieve -> Fuse/Rerank -> Evaluate
      -> Correct (bounded) -> Generate -> Critique (bounded) -> Finalize
```

No workflow branch may rely on parsing free-form prose for control flow. Decisions and loop counters live in `AgentState`. External SDKs are restricted to adapter modules so tests can substitute deterministic mocks.

