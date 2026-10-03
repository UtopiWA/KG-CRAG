# Proposal

## Why

现有 Dense/Sparse/Hybrid 能力能够返回语义或词法相关 Chunk，但不能稳定回答“某方法使用何模型、在哪个数据集上评测、结果指标如何”一类关系型和多跳问题。迭代 04 需要增加一个按需调用、来源可回溯且资源有界的图证据工具，为后续证据缺口路由提供独立 Graph 动作，而不是把知识图谱重新变成所有查询的固定步骤。

本 change 依据 `../PROJECT_SPEC.md` 的 `RET-003`、`RET-004`、`RET-005`、`RET-007`、`OBS-002`、`OBS-003`、`EVAL-004`、`EVAL-005`、`EVAL-008`，以及 `../guidebooks/iteration-04-knowledge-graph.md`。

## What Changes

- 冻结 Paper、Author、Institution、Method、Model、Dataset、Task、Metric、Result、Chunk 的版本化图 schema、稳定标识、允许关系和约束，所有正文事实必须携带现存 `source_chunk_id`、置信度、抽取器版本和创建时间。
- 从已发布 Paper/Chunk 和现有显式引用元数据构建无需 LLM 的全语料基础图，支持 dry-run、pilot、显式论文、有界全量、逐篇事务、检查点、幂等重跑和原子运行清单；缺失引用不推测、不联网补齐。
- 对 15 篇 pilot 每篇确定性选择 3–5 个摘要、方法、实验或结论 Chunk，使用版本化结构化 Prompt 在全 change 最多 100 次 LLM 请求内抽取白名单实体与关系；支持批处理、断点复用、单 Chunk 失败隔离和零调用 dry-run。
- 增加保守实体规范化与别名规则：只有稳定外部 ID 或无歧义规范键才能自动合并；低置信度、同名冲突和类型冲突必须保持分离并进入可审计复核队列。
- 增加 Neo4j 生产适配器和确定性内存替身，所有图查询使用参数化只读白名单模板；拒绝任意 Cypher、写查询、未知模式和越界 hop/候选规模。
- 增加独立 Graph Retriever，将关系或路径命中转换为 `external=false` 的统一 `Evidence`，同时保留图路径、分数/排名及每条事实到原始 Chunk 的溯源；它不会自动触发 Dense、Sparse、Web 或回答生成。
- 增加绑定 pilot 快照的关系型/多跳开发子集和零 LLM 图检索评测，报告路径命中、Evidence Recall@K、MRR、nDCG、失败、时延与数据库调用；真实 Neo4j 和正文抽取测试必须显式启用并受预算约束。
- **非目标**：不实现动态路由、facet 缺口诊断、纠错循环、Graph 与 Hybrid 自动融合、社区摘要、生成式 Cypher、全 8,363 Chunk 正文抽取、Web、反思或 UI。

## Capabilities

### New Capabilities

- `provenance-aware-knowledge-graph`: 定义版本化图契约、基础图与有界正文抽取、保守实体归一化、受控图存储、可溯源 Graph Retriever 和关系/多跳检索评测。

### Modified Capabilities

无。现有 `paper-ingestion`、`dense-rag-baseline` 和 `hybrid-retrieval` 的行为保持不变；本能力只读取其已发布 Paper/Chunk 和统一 `Evidence` 契约。

## Impact

- **代码与配置**：预计新增图领域/运行模型、抽取与规范化流水线、Neo4j 适配器、Graph Retriever、图评测模块、严格配置、Prompt、薄 CLI、Schema 快照和相邻测试；现有骨架 `GraphStore`/`InMemoryGraphStore` 将扩展为正式契约。
- **数据与产物**：只读 `data/raw` 与 `data/processed`；图导入清单、抽取缓存、复核报告和评测结果进入 Git 忽略的可再生数据层，小型离线 fixture 与冻结问题清单可以提交。Neo4j 数据继续保存在独立持久卷。
- **依赖与资源**：增加固定版本 Neo4j Python Driver；元数据图无需 LLM，正文抽取仅限 pilot 代表 Chunk、最多 100 次请求，默认测试零网络、零真实 LLM、零真实 Neo4j。开发 Neo4j 继续采用 1 GiB heap 与 512 MiB page cache 上限。
- **兼容性**：不改变既有 Dense/Sparse/Hybrid CLI 和结果语义。图 schema、抽取 Prompt/模型、规范化规则、语料快照或查询模板变化必须形成新版本，禁止静默复用旧图或抽取缓存。
- **依赖**：依赖已归档的 `paper-ingestion`、`dense-rag-baseline` 和 `hybrid-retrieval`；不依赖后续动态路由、纠错或 Web change。
- **风险**：LLM 事实抽取可能幻觉、同名实体可能误合并、引用元数据可能缺失、图路径可能放大错误事实、Neo4j 状态可能与 processed 语料漂移。实现必须以白名单 schema、Chunk 溯源、保守合并、身份校验、事务导入、有限查询和人工复核缓解这些风险。
