# Design

## Context

动机与范围见 `proposal.md`，行为契约见 `specs/hybrid-retrieval/spec.md`。当前系统已有稳定 Chunk、内容寻址 Embedding 缓存、Qdrant Dense 索引、`Retriever`/`Reranker` Protocol、统一 `Evidence` 及 Dense 检索评测。`EvidenceScores` 和 `EvidenceRanks` 已预留 sparse、fusion、rerank 字段，Mock Reranker 也已存在，但生产 Sparse 索引、融合器、本地 Reranker、Hybrid 编排和对应运行模型仍未实现。

当前语料约为 8 千级 Chunk，宿主机按 16 GB 内存和 4 GB 显存约束设计。默认测试必须零网络，候选方案评测必须零 LLM；已有 Dense-only CLI、结果和规格必须继续保持原行为。

## Goals / Non-Goals

**Goals:**

- 以低磁盘和内存开销建立可事务更新的 Sparse 索引，并保持与 processed Chunk 相同的稳定身份和溯源。
- 把 Sparse、RRF、Weighted 和 Cross-Encoder Rerank 实现为可独立测试、固定上限且可被后续控制器计量的组件。
- 对融合公式、排序、降级和运行版本作出确定性定义，使检索实验能够重放和比较。
- 复用现有证据约束生成规则，对唯一入选 Hybrid 配置提供显式、至多一次的回答验证入口。

**Non-Goals:**

- 不把 Sparse 或 Hybrid 接入动态路由、facet 判断、纠错循环、Graph 或 Web；这些由后续 change 完成。
- 不训练检索器或 Reranker，不引入 LLM Reranker，也不扩大为完整评测平台。
- 不修改 Dense-only 命令的默认行为、冻结问题集或 `DenseRAGResult` 语义。
- 不要求 BGE Embedding 与 Cross-Encoder 同时驻留 GPU，不把索引、权重或运行结果提交 Git。

## Decisions

### 1. 使用 SQLite FTS5 作为轻量、版本化的 Sparse 后端

新增 `sparse_store/` 边界，包含后端无关 Protocol、确定性内存替身和 SQLite FTS5 实现。生产索引位于 `data/processed/sparse-index/<index-version>/index.sqlite3`，以 Chunk ID 为稳定记录身份，并保存重建 Evidence 所需的论文、章节、页码、正文、内容哈希和处理版本。独立 manifest 保存 schema、Python/SQLite 版本、FTS5 tokenizer 配置、BM25 参数、语料快照和最后完成的论文。

默认 tokenizer 版本为 `scientific-unicode-v1`：NFC、case-fold，保留字母数字以及标识符内部的连字符和下划线；查询只由 tokenizer 产生的有界 Token 构造转义后的 OR 表达式，不把原始用户文本直接作为 FTS 查询语法。FTS5 `bm25()` 的方向统一转换为“分数越大越相关”，同分按 Chunk ID 排序。

首次构建在临时数据库中完成完整校验后原子发布；增量运行以单篇论文为事务，先写新记录再删除该论文陈旧记录，并在事务提交后更新检查点。重复运行通过内容哈希和处理版本跳过。启动时先验证 Python 构建包含 FTS5；缺失时返回配置错误，不静默切换算法。

备选方案是引入 `rank-bm25` 并把完整 postings 序列化为 JSON。它实现简单，但增量事务、过滤、崩溃恢复和全量内存占用较弱；当前规模下 SQLite FTS5 更符合资源与恢复约束，因此不采用该方案。

### 2. 保持 Retriever Protocol，新增显式运行结果模型

`SparseRetriever` 实现现有 `Retriever` Protocol，和 `DenseRetriever` 一样只返回统一 `Evidence`。新增严格模型承载跨阶段信息：

- `SparseIndexIdentity`、Sparse 索引逐篇结果与运行清单；
- `RetrievalStageSummary`：策略、状态、候选数、时延、错误和调用次数；
- `RetrievalCostSummary`：Embedding、Sparse、Reranker 调用数和本地推理时延，不虚构货币成本；
- `HybridRetrievalResult`：最终 Evidence、融合/Reranker 版本、降级状态和阶段摘要；
- Hybrid 开发集逐题结果与汇总报告。

这些模型放入新的 `models/retrieval.py` 并从 `kg_crag.models` 导出；现有 `Evidence` 字段足够表达各阶段分数和排名，不增加后端私有对象。Hybrid 业务编排返回 `HybridRetrievalResult`，而 Dense/Sparse 仍可通过公共 Protocol 独立调用，便于迭代 05 把它们注册为不同成本的动作。

备选方案是扩展 `Retriever.retrieve()` 返回列表和运行元数据的联合类型；这会破坏 Dense、Mock 与未来 Graph 实现的现有调用方，因此不采用。

### 3. 融合是无副作用纯函数，按 Chunk 合并并固定公式

融合输入是 Dense 和 Sparse Evidence 的拷贝，稳定合并键为内部 Chunk 的 `source_id`。合并结果使用 `hybrid:<fusion-version>:<chunk-id>` 作为 Evidence ID，并保留两路已有的 score/rank。

- RRF：`sum(1 / (rrf_k + source_rank))`，缺失来源不贡献分数。
- Weighted：分别在本次有界候选池中对 Dense 和 Sparse 原始分数做 min-max 归一化，再计算 `dense_weight * dense_norm + sparse_weight * sparse_norm`；某一路所有分数相等时，该路已返回候选的归一化值设为 `1.0`，缺失来源为 `0.0`。
- 稳定排序：融合分数降序，其次采用最佳来源排名、Chunk ID 升序。

融合版本绑定方法、归一化版本、RRF 常数、权重、候选上限和去重策略。跨路 Chunk 去重始终执行；可选按论文去重在 Rerank 之后、最终截断之前执行，使 Reranker 可以从同一论文的多个候选中选择更适合的一条。

直接加权原始 Dense/BM25 分数会混合不同量纲；只使用排名加权又会让 Weighted 与 RRF 缺乏区分，因此选择版本化的候选池归一化。

### 4. Hybrid 固定并发调用两路，并把降级作为结果状态

`HybridRetrievalService` 对 Dense 和 Sparse 各调用至多一次，可用 `asyncio.gather` 并发执行，但每路仍遵守自己的候选和超时上限。默认 `allow_partial=true`：单路失败时使用成功一路的真实 Evidence，并记录 `degraded=true`、失败阶段和安全 `ErrorDetail`；两路失败直接失败。配置可关闭部分降级，此时任一路失败均终止。

空结果不是后端故障：一路为空、另一路有结果时仍按正常公式融合；两路均为空时返回成功的空 Evidence。服务不隐藏重试、不调用 Graph/Web，也不触发回答生成。

备选方案是单路失败时静默返回另一条路径，这会污染后续动作成本和实验结论，因此降级必须进入公共结果与报告。

### 5. Reranker 复用 Sentence Transformers 依赖并默认在 CPU 惰性加载

新增 Cross-Encoder 适配器实现现有 `Reranker` Protocol，默认模型为 `BAAI/bge-reranker-base`，默认修订固定为 `2cfc18c9415c912f9d8155881c133215df768a70`；模型标识、修订、设备、批大小和缓存路径全部来自严格配置。模块导入、dry-run、禁用路径和默认测试均不得加载或下载权重。

执行顺序固定为：融合 → 截取最多 20 个候选 → Cross-Encoder 批量评分 → 可选按论文去重 → 最多输出 8 个结果。默认设备为 CPU，以避免与 BGE-M3 争用 4 GB 显存；显式 GPU 配置必须由运行者确保 Embedding 和 Reranker 不同时驻留。Provider 返回数量不符、非有限分数或部分批次时拒绝整批。

默认 `allow_rerank_fallback=true`，失败时保留未重排融合结果并标记降级；严格实验可关闭回退。LLM Reranker 不接入运行时，仅可在未来独立 change 中提出。

### 6. Hybrid 回答验证复用公共证据约束组件，不改 Dense 服务

把 Evidence 编号、上下文上限、结构化回答解析和引用校验提取为兼容的公共组件，保留现有 Dense 导入与行为。新增 Hybrid 查询服务先取得 `HybridRetrievalResult`；只有显式 `with_answer=true` 且证据达到门槛时才调用现有 LLM Provider 一次，结果记录实际检索路径、融合版本、Reranker 状态和引用。

默认 CLI 和全部检索矩阵使用 `with_answer=false`。这样可以验证入选 Hybrid RAG，同时不把迭代 03 变成新的多轮生成工作流。复制一套 Dense 生成逻辑会导致引用规则漂移，因此不采用。

### 7. 开发集评测分为零 LLM 矩阵和一次入选配置验证

新增绑定 pilot 语料快照的 `hybrid_dev_questions.json`，保留现有问题并补充缩写、连字符模型名、数据集名和低频实体样本，总量控制在 20–40 题。检索评测器对 Dense、Sparse、RRF、Weighted、Fusion+Rerank 使用相同问题、目标 Chunk 和 K 值，计算 Recall@K、MRR、nDCG@K、Evidence coverage、失败数和各阶段时延；逐题结果和汇总使用版本哈希目录原子发布。

候选矩阵不调用 LLM。开发集选定权重、阈值和唯一 Hybrid 配置后，显式命令才可对该配置运行至多一次逐题回答生成；回答结果与检索报告分开保存，避免汇总失败导致重复计费。默认测试使用内存 Sparse Store、Mock Retriever 和 Mock Reranker；SQLite 使用临时数据库做离线集成测试，真实 Cross-Encoder 验收必须显式启用且限制候选和问题数。

### 8. 配置、资源和产物全部纳入版本边界

扩展 `configs/retrieval.yaml`，新增 Sparse schema/tokenizer/BM25/路径/查询上限、融合归一化与权重、部分降级、Reranker 模型修订/设备/批量/候选上限和 Hybrid 输出上限；扩展 `configs/evaluation.yaml` 增加开发集路径、策略矩阵和结果根目录。所有路径必须位于工作区可再生数据层，配置哈希不包含密钥或机器绝对路径。

每次运行记录模型与索引版本、候选数、阶段时延和调用数。索引、模型缓存和结果默认不进 Git；文档明确缓存优先迁到 D 盘。新增、删除或改变模块职责时同步更新 `PROJECT_MAP.md`。

## Risks / Trade-offs

- **[SQLite 构建缺少 FTS5 或不同版本排序细节有差异]** → 启动时能力探测，manifest 记录 SQLite 版本，以固定 fixture 校验分词、BM25 方向和同分排序；不兼容时明确失败。
- **[Weighted min-max 受当前候选池影响]** → 候选上限与归一化版本进入配置哈希，和 RRF 分开报告，不宣称不同候选池间原始融合分数可直接比较。
- **[部分降级掩盖质量下降]** → `degraded`、失败阶段和后端状态成为必填运行信息，严格评测可以关闭降级。
- **[Cross-Encoder 占用内存或延迟过高]** → 默认 CPU、惰性加载、候选 20/输出 8、有限批量、可禁用和可回退；真实 smoke 只运行少量问题。
- **[开发集调权导致结论过拟合]** → 本 change 只冻结开发配置，不接触后续冻结测试集；最终结论留给迭代 07/09。
- **[抽取公共生成组件影响 Dense 基线]** → 保留原导出和 Dense 行为，以现有 Dense 全套测试和 Schema 快照防止回归。

## Migration Plan

1. 先增加向后兼容的配置和公共模型，现有 Dense 配置缺少新字段时使用禁用 Hybrid 的安全默认值。
2. 增加 Sparse Store、索引器和离线测试，在独立目录构建索引，不修改 Qdrant 集合或 Dense 运行产物。
3. 增加 Sparse Retriever、融合、Hybrid 编排和 Reranker；先使用 Mock/内存实现完成确定性矩阵，再显式运行 SQLite 与小型 Cross-Encoder smoke。
4. 冻结开发集和选定配置后才运行可选回答验证，保存所有版本与逐题结果。
5. 更新 README、架构/实验文档和 `PROJECT_MAP.md`。回滚时禁用 Hybrid、删除明确版本的可再生 Sparse 索引与结果目录，继续使用未改变的 Dense-only 命令和集合。
