# Proposal

## Why

KG-CRAG 已有核心模型和后端 Protocol，但缺少统一错误语义、稳定 Trace schema 以及可执行的离线 Retriever/Reranker/Store，导致后续摄取与工作流 change 无法在不连接外部服务的情况下验证跨模块契约。迭代 00 需要先补齐这些工程基座，并把现有隐式行为固化为可测试、可导出的契约。

## What Changes

- 定义结构化错误代码、错误详情和可重试语义，供模块边界统一使用。
- 将最小 TraceEvent v1 固化为严格校验的公共模型，并让 AgentState 引用该模型。
- 增加确定性的 Mock Retriever、Mock Reranker、内存 Vector Store 和内存 Graph Store，记录调用并遵守现有 Protocol。
- 加强运行配置与健康响应的边界校验，同时保持健康检查不依赖外部服务且不暴露密钥。
- 导出公共模型的 JSON Schema 快照，并通过测试检测意外契约漂移。
- 补充基座测试、README 和项目地图，使后续 change 能直接复用这些能力。

## Non-goals

- 不实现真实 Dense/Sparse/Graph 检索算法或 Qdrant/Neo4j 后端。
- 不增加 LangGraph 节点、路由、纠错、生成或评测业务逻辑。
- 不接入在线模型、搜索服务或新的第三方运行时依赖。
- 不改变现有论文采集数据格式和原始语料。

## Capabilities

### New Capabilities

- `foundation-contracts`: 定义工程基座的严格公共契约、结构化错误、TraceEvent v1、离线测试替身、配置校验与 schema 快照行为。

### Modified Capabilities

无。当前 `openspec/specs/` 尚无已生效能力规格。

## Impact

- 关联项目规格：架构约束、可复现性、可靠性、安全与隐私，以及 `OBS-001`、`OBS-002`、`OBS-004`。
- 迭代依据：`../guidebooks/iteration-00-foundation.md`。
- 预计影响 `models/`、`workflow/state.py`、`retrieval/`、`vector_store/`、`graph_store/`、`settings.py`、`api/app.py`、测试、README、schema 文档和 `PROJECT_MAP.md`。
- 公共导出会增加，但现有 Provider、Store、Retriever 方法签名和健康检查 URL 保持兼容。
- 不新增外部依赖；默认测试继续完全离线运行。
