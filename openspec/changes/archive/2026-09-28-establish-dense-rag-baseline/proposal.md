# Proposal

## Why

迭代 01 已提供稳定、可追溯且可重建的 Chunk，但系统尚不能把这些产物转化为可查询的向量索引，也没有可复现的 Dense RAG 问答基线。现在需要建立一条最小而完整的向量检索与证据约束回答链，作为后续混合检索、知识图谱和 Agent 纠错效果的固定对照。

本 change 依据 `../PROJECT_SPEC.md` 的 `ING-006`、`RET-001`、`RET-004`、`RET-005`、`ANS-001`、`ANS-002`、`ANS-003`、`ANS-006`、`OBS-001`、`OBS-002`、`EVAL-003`、`EVAL-005`、`EVAL-006`、`EVAL-008` 以及 `../guidebooks/iteration-02-vector-rag.md`。

## What Changes

- 冻结 Dense RAG 的版本化配置，包括 Embedding 模型与维度、批大小、缓存键、Qdrant 集合 schema、距离度量、`top_k`、按论文去重、Prompt 和索引版本。
- 增加可批处理、可缓存且经过输出校验的 Embedding 编排；业务模块只依赖 `EmbeddingProvider`，默认单元测试继续使用确定性 Mock。
- 增加 Qdrant Vector Store 适配器以及幂等 upsert、按论文删除、允许字段过滤、相似度检索、集合兼容性检查和显式重建策略。
- 增加从已发布 Chunk 产物构建索引的单一命令，支持 dry-run、增量更新、有界批处理、显式重建、失败隔离和可审计运行清单。
- 增加 Dense Retriever，将查询向量和 Qdrant 命中转换为稳定、可追溯的 `Evidence`，保留原始 Dense 分数、排名、Chunk、论文、章节和页码。
- 增加仅依据选定 Evidence 生成回答的最小链路，返回引用、置信度、证据不足标记和最小结构化 Trace；没有足够证据时给出保守回答，而不补造事实。
- 建立不少于 20 个带目标 Chunk 标注的 pilot 问题，保存 Recall@K、MRR、nDCG@K、引用指标、逐题结果和失败样本，并冻结基线配置与运行元数据。
- **非目标**：不实现 BM25、融合、重排序、Graph Retriever、问题动态路由、纠错/反思循环、Web 兜底、通用问答 API 或 Streamlit 界面。

## Capabilities

### New Capabilities

- `dense-rag-baseline`: 定义从版本化 Chunk 向量索引、Dense Evidence 检索到证据约束回答和可复现 pilot 评测的完整基线行为。

### Modified Capabilities

无。现有 `foundation-contracts` 已覆盖公共 Protocol、统一错误和离线替身，`paper-ingestion` 已覆盖稳定 Chunk 与可重建产物；本 change 只消费这些契约，不修改其既有需求。

## Impact

- **代码与配置**：预计影响 `providers/`、`vector_store/`、`retrieval/`、`generation/`、`evaluation/`、公共模型和 Schema 快照，扩展 `configs/retrieval.yaml`、`configs/evaluation.yaml`、运行时设置，并增加索引、问答和基线评测薄脚本。
- **数据与运行产物**：读取 `data/processed` 中已发布 Chunk；Qdrant 集合、Embedding 缓存、索引/问答/评测运行清单和逐题结果均为可再生产物，不提交 Git。
- **依赖与系统**：需要固定兼容的 Qdrant Python 客户端和至少一个真实 Embedding/LLM Provider 适配器；Qdrant 集成测试与需要模型或凭据的验收必须和默认离线测试分离。
- **兼容性**：公共模型新增字段或模型时必须保持已有数据可读、更新导出与 Schema 快照；集合 schema、向量维度、Embedding 模型或索引输入版本变化时必须生成新索引版本或显式重建，不能静默复用不兼容集合。
- **风险**：Embedding 模型输出维度漂移、远端调用部分失败、Qdrant schema 不匹配、缓存污染和生成模型越过证据作答可能破坏可复现性或忠实度；设计与任务必须分别给出校验、失败隔离、回滚和离线替身策略。
