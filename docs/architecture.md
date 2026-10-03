# Architecture

KG-CRAG starts as a modular monolith. FastAPI exposes stable HTTP contracts; a future LangGraph workflow coordinates query analysis, routing, retrieval, correction, generation, and critique. Dense retrieval uses Qdrant, Sparse retrieval uses SQLite FTS5, graph retrieval uses Neo4j, and every backend is accessed through a typed interface.

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

`runtime.py` 只装配 Provider 和后端；索引、检索、生成、评测逻辑分别位于独立包内。Qdrant 与 OpenAI SDK 不进入领域模块，默认测试使用内存存储、伪客户端或 Mock。

## 当前 Hybrid 检索切片

迭代 03 在不改变 Dense-only 基线的前提下增加固定、有界路径：

```text
processed Chunk -> SQLite FTS5 Sparse 索引 -> SparseRetriever -> Evidence
问题 -> DenseRetriever ─┐
                       ├-> RRF/Weighted -> 可选 Cross-Encoder -> HybridRetrievalResult
       SparseRetriever ─┘                                      -> 可选一次回答生成
冻结开发集 -> 五策略零 LLM 矩阵 -> Recall/MRR/nDCG/覆盖率/阶段时延
```

`HybridRetrievalService` 每路最多调用一次；单路故障可显式降级，两路均故障则失败。融合器是无副作用纯函数，Cross-Encoder 惰性加载并允许显式回退。生成器只接收公共 Evidence，不接触 Qdrant、SQLite 或模型对象。本切片不包含动态路由、反思、Web 或 UI。

## 当前 Graph Retriever 切片

迭代 04 将知识图谱实现为与 Dense/Sparse 并列、按需调用的证据工具：

```text
published Paper/Chunk -> 无 LLM 基础图 ─┐
代表 Chunk -> 有预算结构化抽取 -> 复核 ├-> 论文级 GraphBundle -> Memory/Neo4j
固定模板请求 -> 单次 GraphStore 查询 -> 路径来源复验 -> Graph Evidence
冻结开发集 -> 零 LLM 关系/多跳矩阵 -> 路径、召回、排序和溯源指标
```

所有事实先通过公共模型和白名单端点校验，再以论文级事务同步。Neo4j 只接收固定参数化只读模板；默认测试使用版本化内存后端。Graph Retriever 不自动调用 Dense、Sparse、Web 或生成器，来源哈希漂移时拒绝整条路径。本切片仍不包含动态路由、纠错循环、Web、反思、API 或 UI。
