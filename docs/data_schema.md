# 数据模式

公共领域契约位于 `src/kg_crag/models/domain.py`，摄取阶段契约位于 `models/ingestion.py`，Sparse/Hybrid 运行契约位于 `models/retrieval.py`，图契约位于 `models/graph.py`；这些模型都拒绝未知字段。可评审 JSON Schema 快照位于 `docs/schemas/`。

## 数据分层

- `data/raw` 是不可变输入：元数据记录 PDF 相对路径、来源、大小和 SHA-256。
- `data/interim` 保存 `ParsedDocument` 与 `CleanedDocument`。前者含一基页码、三位小数坐标、确定性 Block 顺序、Section 和显式警告；后者记录清洗统计及 `source_block_ids`。
- `data/processed` 保存公共 `Paper`、按 ordinal 排序的 `Chunk` JSONL、`QualityReport` 和批次 `IngestionRunManifest`。

所有派生记录都能由 raw 输入、固定解析器和版本化配置重建。目录布局和处理版本计算见 `docs/ingestion.md`。

## 稳定标识与溯源

新 Paper ID 的优先级为 DOI、无版本 arXiv ID、其他受支持外部 ID、规范化标题 SHA-256；现有合法 `arxiv:<id>` 标识保持兼容。Paper 目录使用 Windows 安全的 `paper-<sha256(paper_id)[:16]>`。

Block ID 绑定输入哈希、页码、顺序、坐标和原文哈希；Chunk ID 绑定 Paper ID、processing-version、Section 键、ordinal、页码范围和内容哈希。每个 Chunk 含 Paper ID、章节、起止页码、Token 数、内容 SHA-256 与处理版本。正文图事实必须关联现存 Chunk、内容哈希、抽取器/Prompt 版本、置信度与带时区创建时间；元数据图事实关联 Paper 和 `paper.json` 哈希。

`Evidence` 是 Dense、Sparse、Graph 和 Web 检索的统一下游边界。Dense 命中记录稳定 `source_id`、分数、排名、集合版本、内容哈希及 `external=false`，不暴露 Qdrant 私有对象。摄取流水线不创建索引，也不把解析器私有对象写入公共契约。

## Dense RAG 契约

- `VectorCollectionIdentity` 绑定 schema、Embedding Provider/模型/revision、维度、距离和 payload 字段；集合名由其哈希派生。
- `IndexRunManifest` 绑定配置哈希、语料快照、集合身份和逐篇新增/更新/跳过/删除/失败状态。
- `AnswerClaim` 区分事实、综合、推断、不确定与冲突；除不确定声明外必须带引用。
- `Citation` 将展示编号解析回 Evidence、Chunk、Paper、章节和页码。
- `DenseRAGResult` 保存 Dense 路径、内部 Evidence、回答、引用、置信度、版本标识和最小 Trace。
- `PilotQuestionSet` 绑定语料快照与实际 Chunk 目标；`DenseEvaluationReport` 保存逐题状态及汇总指标。

对应快照位于 `docs/schemas/`。运行产物只保存工作区相对路径，错误详情不得包含 Chunk 正文、Prompt、密钥或绝对路径。

## Sparse 与 Hybrid 契约

- `SparseIndexIdentity` 绑定 schema、分词器、BM25、Python/SQLite/FTS5、语料快照与索引版本。
- `SparseIndexRunManifest` 保存逐篇新增、更新、跳过、删除、失败状态及检查点对应身份。
- Sparse 命中仍使用公共 `Evidence`，记录 `scores.sparse`、`ranks.sparse` 和可回溯 Chunk payload，不泄漏 SQLite 行对象。
- 融合结果以稳定 Chunk ID 合并来源，在 `Evidence` 中同时保留 Dense、Sparse、Fusion 和 Rerank 分数/排名。
- `HybridRetrievalResult` 保存各阶段状态、候选数、时延、调用次数、降级路径及语料/集合/索引/融合/重排版本。
- `HybridQueryResult` 包含嵌套检索结果、实际 Dense/Sparse 路径及可选回答；`with_answer=false` 时回答字段为空且 `llm_calls=0`。
- `HybridEvaluationReport` 绑定评测版本，保留逐题失败、策略指标和阶段平均时延；同版本逐题检查点可安全复用。

## Graph 契约

- `GraphIdentity` 绑定 schema、语料快照、基础构建器、Chunk 选择器、抽取器、Prompt/模型修订、规范化和查询模板版本。
- `GraphEntity` 和 `GraphFact` 只允许固定节点、关系及端点组合；事实 ID 同时绑定关系端点和来源身份。
- `GraphBundle` 是单篇论文的后端无关完整目标状态，写库前检查重复端点、悬空事实、来源 Paper 与图身份。
- `GraphQueryRequest` 仅允许四种枚举模板及有界参数，不存在 raw Cypher 字段；`GraphPathHit` 不暴露 Driver、Session 或数据库节点。
- Graph Evidence 使用 `source_type=graph`、`external=false`，在 metadata 标量中保留 graph/template/path/fact/hop 版本信息；正文路径必须回读当前 Chunk，元数据路径不得伪造章节或页码。
- `GraphEvaluationQuestionSet` 绑定唯一语料快照和图版本，逐题目标可包含事实、路径和来源，重复问题 ID 或版本混用会被拒绝。
