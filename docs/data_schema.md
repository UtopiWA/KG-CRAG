# 数据模式

公共领域契约位于 `src/kg_crag/models/domain.py`，摄取阶段契约位于 `models/ingestion.py`；两者都拒绝未知字段。可评审 JSON Schema 快照位于 `docs/schemas/`。

## 数据分层

- `data/raw` 是不可变输入：元数据记录 PDF 相对路径、来源、大小和 SHA-256。
- `data/interim` 保存 `ParsedDocument` 与 `CleanedDocument`。前者含一基页码、三位小数坐标、确定性 Block 顺序、Section 和显式警告；后者记录清洗统计及 `source_block_ids`。
- `data/processed` 保存公共 `Paper`、按 ordinal 排序的 `Chunk` JSONL、`QualityReport` 和批次 `IngestionRunManifest`。

所有派生记录都能由 raw 输入、固定解析器和版本化配置重建。目录布局和处理版本计算见 `docs/ingestion.md`。

## 稳定标识与溯源

新 Paper ID 的优先级为 DOI、无版本 arXiv ID、其他受支持外部 ID、规范化标题 SHA-256；现有合法 `arxiv:<id>` 标识保持兼容。Paper 目录使用 Windows 安全的 `paper-<sha256(paper_id)[:16]>`。

Block ID 绑定输入哈希、页码、顺序、坐标和原文哈希；Chunk ID 绑定 Paper ID、processing-version、Section 键、ordinal、页码范围和内容哈希。每个 Chunk 含 Paper ID、章节、起止页码、Token 数、内容 SHA-256 与处理版本。图谱功能实现后，每条图事实仍必须关联来源 Chunk、抽取器版本、置信度与创建时间。

`Evidence` 是 Dense、Sparse、Graph 和 Web 检索的统一下游边界。Dense 命中记录稳定 `source_id`、分数、排名、集合版本、内容哈希及 `external=false`，不暴露 Qdrant 私有对象。摄取流水线不创建索引，也不把解析器私有对象写入公共契约。

## Dense RAG 契约

- `VectorCollectionIdentity` 绑定 schema、Embedding Provider/模型/revision、维度、距离和 payload 字段；集合名由其哈希派生。
- `IndexRunManifest` 绑定配置哈希、语料快照、集合身份和逐篇新增/更新/跳过/删除/失败状态。
- `AnswerClaim` 区分事实、综合、推断、不确定与冲突；除不确定声明外必须带引用。
- `Citation` 将展示编号解析回 Evidence、Chunk、Paper、章节和页码。
- `DenseRAGResult` 保存 Dense 路径、内部 Evidence、回答、引用、置信度、版本标识和最小 Trace。
- `PilotQuestionSet` 绑定语料快照与实际 Chunk 目标；`DenseEvaluationReport` 保存逐题状态及汇总指标。

对应快照位于 `docs/schemas/`。运行产物只保存工作区相对路径，错误详情不得包含 Chunk 正文、Prompt、密钥或绝对路径。
