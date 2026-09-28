# Architecture

KG-CRAG starts as a modular monolith. FastAPI exposes stable HTTP contracts; a future LangGraph workflow coordinates query analysis, routing, retrieval, correction, generation, and critique. Dense/sparse retrieval uses Qdrant, graph retrieval uses Neo4j, and every backend is accessed through a typed interface.

The authoritative end-to-end flow is:

```text
Query -> Analyze -> Route -> Retrieve -> Fuse/Rerank -> Evaluate
      -> Correct (bounded) -> Generate -> Critique (bounded) -> Finalize
```

No workflow branch may rely on parsing free-form prose for control flow. Decisions and loop counters live in `AgentState`. External SDKs are restricted to adapter modules so tests can substitute deterministic mocks.

## 当前 Dense RAG 垂直切片

迭代 02 已实现一条不经过未来 Agent 路由的明确基线：

```text
processed Chunk -> 增量索引 -> Embedding 缓存 -> Qdrant 版本化集合
用户问题 -> 单次 DenseRetriever -> Evidence -> 有界上下文
         -> 至多一次 LLM 生成 -> AnswerClaim/Citation -> Trace/结果
固定问题集 -> 逐题隔离运行 -> Recall/MRR/nDCG/引用指标 -> 基线报告
```

`runtime.py` 只装配 Provider 和后端；索引、检索、生成、评测逻辑分别位于独立包内。Qdrant 与 OpenAI SDK 不进入领域模块，默认测试使用内存存储、伪客户端或 Mock。这个切片不包含 BM25、融合、重排、Graph、动态路由、反思、Web、API 或 UI。
