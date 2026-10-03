# Proposal

## Why

现有 Dense RAG 基线擅长语义相似检索，但对缩写、连字符模型名、数据集名和低频科研术语的精确匹配仍可能漏召回。迭代 03 需要在不引入 Agent 循环和额外默认 LLM 调用的前提下，补齐可复现的 Sparse、融合与受控重排能力，为后续缺口驱动的动作选择提供独立、可计量的检索路径。

本 change 依据 `../PROJECT_SPEC.md` 的 `RET-002`、`RET-004`、`RET-005`、`RET-006`、`EVAL-003`、`EVAL-004`、`EVAL-005`、`OBS-002`、`OBS-003`、研究假设 `H1`，以及 `../guidebooks/iteration-03-hybrid-retrieval.md`。

## What Changes

- 增加从已发布 Chunk 构建的版本化 Sparse/BM25 索引，支持 dry-run、增量同步、幂等重跑、断点恢复、白名单过滤和有界查询。
- 增加 Sparse Retriever，把命中转换为统一内部 `Evidence`，保留稳定 Chunk/Paper 溯源、Sparse 原始分数与排名。
- 增加确定性的 RRF 与可配置 Weighted Fusion，按稳定 Evidence 身份合并 Dense/Sparse 结果，保留原始和融合分数、排名以及稳定次级排序。
- 增加单路失败的显式降级策略：允许时使用仍成功的检索结果并记录降级原因，两路均失败时返回结构化错误，不伪造 Evidence。
- 接入惰性加载的小型本地 Cross-Encoder Reranker；默认只重排融合 top-20、最多输出 top-8，并提供确定性离线替身。LLM Reranker 不作为默认实现或完成条件。
- 增加有界 Hybrid 查询与检索评测入口，使用同一冻结开发集比较 Dense、Sparse、RRF、Weighted 和 Fusion+Rerank；默认运行零 LLM 检索指标，只允许入选配置执行至多一次证据约束回答生成。
- 为每次检索记录策略、输入/输出候选数、各阶段时延、后端状态、是否降级、Reranker 调用和可复现成本摘要，并原子保存运行结果。
- **非目标**：不实现知识图谱、动态路由、facet 缺口诊断、纠错循环、回答反思、Web 兜底、LLM Reranker 或 UI。

## Capabilities

### New Capabilities

- `hybrid-retrieval`: 定义版本化 Sparse 索引与检索、Dense/Sparse 确定性融合、受控 Cross-Encoder 重排、失败降级、统一 Evidence 溯源和零 LLM 检索评测行为。

### Modified Capabilities

无。现有 `dense-rag-baseline` 的 Dense-only 行为、调用上限和冻结基线保持不变；Hybrid 能力只复用其 Embedding、Dense Retriever、Evidence 和证据约束生成边界。

## Impact

- **代码与配置**：预计新增 Sparse 索引/存储、Sparse Retriever、融合器、Cross-Encoder 适配器、Hybrid 编排和检索评测模块；扩展严格配置、公共运行结果模型、Schema 快照、薄 CLI 与相邻测试。
- **数据与产物**：读取已发布 `data/processed` Chunk；Sparse 索引、Reranker 缓存、运行清单和评测结果均位于 Git 忽略的可再生数据层，并绑定语料快照与配置版本。
- **依赖与资源**：优先使用 Python/SQLite 内置能力实现轻量 Sparse 索引，复用现有 Transformers 依赖加载小型 Reranker；模型必须惰性加载、缓存位置可配置，BGE Embedding 与 Reranker 不得要求同时常驻 GPU。
- **兼容性**：不改变现有 Dense 命令和结果语义；新增公共字段必须有兼容默认值。Sparse 分词、索引 schema、融合公式、权重或 Reranker 模型/修订变化时必须产生新版本，不能静默复用旧产物。
- **依赖**：依赖已归档的 `paper-ingestion` 和 `dense-rag-baseline` 能力，不依赖后续 Graph 或 Agent 工作流。
- **风险**：Dense 与 Sparse 分数量纲不同、同分排序不稳定、单路故障被静默掩盖、模型缓存占用和重排延迟都可能损害可复现性；design 和 tasks 必须给出归一化、稳定排序、显式降级、资源上限和离线验证策略。
