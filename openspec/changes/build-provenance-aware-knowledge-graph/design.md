# Design

## Context

动机与范围见 `proposal.md`，行为契约见 `specs/provenance-aware-knowledge-graph/spec.md`。当前系统已有稳定 Paper/Chunk、15 篇 pilot、全语料 processed 产物、统一 `Evidence`、LLM Provider、GraphStore Protocol、确定性内存替身、Neo4j Compose 服务与连接设置；但现有 GraphStore 只按正文词法匹配，不表达实体、关系、图身份、事实溯源、事务状态或白名单查询。`Paper.references` 已能承载来源元数据中明确提供的引用，但当前语料许多记录为空，因此引用缺失必须是正常状态。

当前 processed 语料约 110 篇、8,363 个 Chunk，宿主机约束为 16 GiB 内存和 4 GiB 显存。默认测试必须零网络、零真实 LLM、零真实 Neo4j；正文抽取限定 15 篇 pilot、每篇 3–5 个 Chunk、最多 100 次请求。Graph 是后续控制器可选择的证据工具，本 change 不实现自动路由或 facet 覆盖矩阵。

## Goals / Non-Goals

**Goals:**

- 用严格公共模型冻结图 schema、身份、事实溯源、抽取预算、复核记录、运行清单和评测报告。
- 让基础图完全由本地确定性数据构建，让昂贵抽取只覆盖少量代表 Chunk，并能缓存、恢复和审计。
- 让内存与 Neo4j 后端遵守同一事务、身份和只读查询契约，使默认测试可完整覆盖关键语义。
- 将关系和多跳命中转换为可引用的公共 Evidence，同时明确区分正文与元数据来源。

**Non-Goals:**

- 不解析任意生成式 Cypher，不训练实体链接或关系抽取模型，不自动合并所有同名实体。
- 不对 8,363 个 Chunk 执行 LLM 抽取，不构建社区摘要、图向量索引或 Graph+Hybrid 融合器。
- 不增加动态路由、facet 诊断、纠错循环、Web、回答反思、API 或 UI。

## Decisions

### 1. 图 schema 使用严格记录模型和显式白名单

新增 `models/graph.py`，定义 `GraphNodeType`、`GraphRelationType`、`GraphEntity`、`GraphFact`、`GraphProvenance`、`GraphBundle`、图身份、逐项运行结果、复核项、查询请求/路径命中和评测报告，并从 `kg_crag.models` 导出及生成 Schema 快照。

节点白名单固定为 Paper、Author、Institution、Method、Model、Dataset、Task、Metric、Result、Chunk。关系与合法端点固定为：Paper→Author `AUTHORED_BY`、Author→Institution `AFFILIATED_WITH`、Paper→Paper `CITES`、Paper→Chunk `CONTAINS`、Paper→Method `HAS_METHOD`、Method→Model `USES_MODEL`、Method→Dataset `EVALUATED_ON`、Method→Task `ADDRESSES_TASK`、Method→Result `REPORTS_RESULT`、Result→Metric `USES_METRIC`、Result→Dataset `ON_DATASET`、Result→Task `FOR_TASK`。Result 保存规范化文本值和可选数值/单位；不能可靠结构化的值保留文本而不猜测单位。

实体 ID 优先使用类型加规范外部 ID；没有外部 ID 时使用类型、Unicode NFC/case-fold 后规范名与必要作用域的 SHA-256。事实 ID 绑定图 schema、关系、端点、来源身份和来源哈希。图身份绑定 schema、语料快照、元数据构建器、选择器、抽取 Prompt/模型修订和规范化版本。

备选方案是在 Neo4j 中直接接收任意属性和关系。它开发快但无法在写入前发现模式漂移，也难以用内存实现做契约测试，因此不采用。

### 2. 元数据事实与正文事实使用统一 Provenance 联合类型

`GraphProvenance` 区分 `metadata` 与 `chunk`：元数据来源保存 Paper ID、元数据/processed Paper 哈希和构建器版本，置信度固定为 1；正文来源额外要求 Chunk ID、内容哈希、抽取器版本、Prompt 版本、候选置信度和带时区创建时间。正文候选只有在 Chunk resolver 证明来源仍存在且哈希一致后才能成为 GraphFact。

GraphBundle 是后端无关的完整论文级目标状态，先在内存完成类型、端点、来源、重复和预算校验，再交给 Graph Store。这样非法 LLM 输出不会进入数据库，也不会依赖数据库约束承担业务校验。

备选方案是在每条关系上保存一段来源文本。它会复制大量正文并在 Chunk 更新后形成陈旧证据，因此仅保存稳定来源身份和必要短标签，查询时回读 processed Chunk。

### 3. 基础图读取已发布 Paper/Chunk，引用只接受现有显式标识

复用 `discover_processed()` 的哈希复验和选择边界，从每篇最新已发布版本构建 Paper、Author、Chunk、AUTHORED_BY、CONTAINS。引用解析只读取 `Paper.references`，按 DOI、arXiv 或已知内部 Paper ID 的同一规范规则匹配当前选择或已发布语料；空列表是合法输入，无法唯一匹配的值进入运行清单，不创建占位 Paper。

元数据 plan/run 支持 pilot、重复 `--paper-id` 拒绝、有界 `--all --limit`、dry-run、论文级状态和检查点。所有 Paper/Chunk 构建不调用 LLM，也不读取在线 OpenAlex 或其他服务。

备选方案是运行时调用 OpenAlex 补引用。它会引入网络、许可、快照漂移和额外成本，偏离可重建输入边界，因此留给独立语料增强 change。

### 4. 代表 Chunk 选择器先过滤章节，再以稳定覆盖规则选 3–5 条

选择器优先匹配 Abstract、Method/Approach、Experiment/Evaluation/Result、Conclusion 等规范化章节键；每类先取最早且满足最小字符数的 Chunk，再按信息密度和 ordinal 补足，最终按 Chunk ID 去重并按论文内顺序输出。若论文只有不足 3 个合格 Chunk，则使用全部合格项并在计划中明确不足，不为达到数量复制 Chunk。

配置固定每篇最少 3、最多 5、单 Chunk 最大输入字符、每次响应字符、批大小、并发、总调用 100、累计输入 Token 160,000 和输出 Token 40,000。Token 优先使用 Provider usage；离线 Mock 或 Provider 不返回 usage 时，以固定版本的保守字符估算器计费。预算判断发生在每批调用前。

备选方案是用 Embedding/LLM 为全量 Chunk 打相关性分。它增加模型驻留或调用且选择难以复现；章节和位置启发式已足够覆盖本迭代代表样本，因此不采用。

### 5. 抽取使用独立适配器、版本化 Prompt 和逐 Chunk内容寻址缓存

新增 `GraphExtractionProvider` Protocol，返回严格抽取载荷与 usage；生产适配器复用现有 LLM Provider 的单次生成能力，Mock 直接返回确定性 fixture。Prompt 只允许当前 Chunk 正文和 schema 摘要，要求实体、关系、文本证据片段与置信度，不允许模型生成来源 ID。代码在响应外部注入 Paper/Chunk/版本信息，并校验证据片段确实存在于规范化 Chunk 文本。

缓存键绑定 Chunk ID/内容哈希、schema、Prompt、模型/修订、抽取配置和输出契约版本；缓存回读必须重新校验公共模型。单 Chunk 失败不缓存成功状态、不发布部分候选，也不隐藏重试。检查点记录完成、失败和预算耗尽，重跑只处理缺失或身份变化项。

备选方案是扩展通用 `LLMProvider` 返回项目特定实体。它会把 Graph 语义泄漏到 Dense 回答 Provider；独立适配器能保持现有调用兼容。

### 6. 规范化分为确定性自动合并和只生成建议的歧义检测

规范化先做 NFC、空白、标点和大小写统一，并保存原始别名。外部 ID 相同可合并；无外部 ID 时，仅同类型且规范键在当前版本唯一的候选自动合并。缩写展开、编辑距离和跨类型同名只产生 `EntityReviewItem`，不会改变实体或事实。人工决定以小型版本化 JSON 保存，绑定候选 ID、来源和规则版本；应用时再次检查外部 ID 与类型约束。

复核报告只保存名称、类型、候选 ID、分数、原因和来源标识，不保存完整 Chunk 正文。这样后续可人工复核，又不把相似度阈值误当作真值。

备选方案是对近似名称统一做 fuzzy merge。其召回较高但会把方法、模型或数据集误合并，破坏关系可信度，因此不采用。

### 7. Graph Store 以论文级目标状态同步，Neo4j 用版本命名空间隔离

扩展 `GraphStore` Protocol：`ensure_graph(identity, rebuild=False)`、`record_state(paper_id)`、`sync_paper(bundle)`、`query(request)` 和 `close()`；保留当前内存实现的兼容入口只作为迁移层，生产编排使用新接口。新增 Neo4j 异步适配器并固定 Driver 版本。

Community Neo4j 中所有节点和关系携带 `graph_version`，内部 `graph_key` 将版本与稳定 ID 组合，唯一约束落在 graph_key 上，从而允许新旧版本并存。初始化校验 schema identity 节点和约束；每篇同步在单事务中 MERGE 目标节点/关系，验证数量后删除该 graph_version/paper 下陈旧关系及孤立的论文作用域节点。共享实体只有在没有任何关系后才删除。凭据仅来自 Settings，错误仅暴露错误类型、阶段和 retryable。

备选方案是每次重建清空 Neo4j。它不可安全回滚且可能误删并行版本，因此仅允许带精确 graph_version 的显式清理/重建。

### 8. 查询是枚举模板请求，不暴露 Cypher

公开 `GraphQueryRequest` 只接受模板枚举、实体 ID/规范名、允许类型、Paper 过滤、top_k、max_candidates、max_hops 和 timeout。v1 模板为 `entity_neighbors`、`relation_lookup`、`bounded_path`、`method_evidence`；其中 bounded_path 只允许 1–3 hop 和白名单关系组合。模板模块生成固定 Cypher，所有值参数化；调用方没有 raw query 字段。

内存和 Neo4j 必须通过同一合同测试。Neo4j 适配器使用只读 session/transaction、驱动超时和 `LIMIT`；模板静态测试拒绝写关键字、过程调用、未绑定输入和无界关系模式。每个 GraphRetriever 请求只调用 Graph Store 一次，不重试、不触发其他检索器。

备选方案是让 LLM 生成 Cypher 再做字符串过滤。字符串过滤无法可靠证明只读和有界，也难以固定实验语义，因此明确不采用。

### 9. 路径命中按支持 Chunk 转为 Evidence

Graph Store 返回公共 `GraphPathHit`，包含稳定 path/fact/entity ID 和来源，不返回 Driver 对象。GraphRetriever 用 processed Chunk resolver 回读并复验来源：同一路径按唯一正文 Chunk 拆分 Evidence，使用 `graph:<graph-version>:<path-id>:<source-id>` 作为 ID；metadata-only 路径以 Paper/元数据记录为 source_id 并生成有限事实描述。Evidence metadata 只放 path ID、紧凑事实 ID、hop、模板和版本等有限标量，`scores.graph` 与稳定 rank 必填，`external=false`。

排序固定为模板相关分数降序、hop 数升序、最佳事实置信度降序、path ID 和 source ID。任何路径含失效正文来源时整条拒绝并返回数据漂移错误，不能用 Neo4j 中的陈旧属性降级生成证据。

### 10. 运行产物与评测全部版本化且默认离线

配置新增 graph schema/路径/选择器/抽取预算/规范化阈值/查询上限和开发集矩阵，路径必须是工作区内相对路径。清单与缓存置于 `data/processed/graph-*`，评测置于 `data/evaluation/results/graph-retrieval`，全部 Git 忽略并原子写入。CLI 分为 `build_metadata_graph.py`、`extract_graph_facts.py`、`query_graph_retrieval.py` 和 `evaluate_graph_retrieval.py`，仅做参数和运行时装配。

提交 12–20 题关系/多跳开发集，绑定 pilot 快照、graph_version、目标事实/路径/Chunk。指标计算和失败隔离复用现有评测模式，但候选矩阵不调用 LLM。默认单元/集成测试使用 Mock 抽取器与内存后端；真实 Neo4j 由 `KG_CRAG_RUN_NEO4J_TESTS=1` 显式启用，真实正文抽取由独立开关和预算确认启用，二者不进入默认 `pytest`。

## Risks / Trade-offs

- **[抽取模型生成不存在或越界事实]** → 严格 schema、证据片段包含校验、来源由代码注入、整 Chunk 原子拒绝和抽样复核。
- **[保守归一化降低图召回]** → 保存别名和复核建议，用开发集报告歧义失败；优先避免不可逆误合并。
- **[引用元数据普遍为空]** → 把无引用作为合法状态并报告覆盖，不联网补齐；关系/多跳验收主要依赖可溯源正文事实。
- **[共享实体的论文级回滚复杂]** → 关系记录来源 Paper，事务只删除目标论文陈旧事实，共享实体仅在无引用时清理。
- **[Neo4j Community 单数据库导致版本数据共存]** → graph_key/graph_version 强制隔离、所有模板必须带版本参数、清理要求精确版本。
- **[最多 100 次调用仍产生 Token 和时间成本]** → pilot 代表 Chunk、调用与 Token 双预算、批前停止、内容寻址缓存、默认 Mock 和可中断检查点。
- **[图路径放大低置信度事实]** → 查询配置最小事实置信度，多跳分数受最弱边限制，Evidence 保留每条事实与来源以供下游判断。

## Migration Plan

1. 先增加向后兼容的严格配置和公共图模型，更新 Schema 快照；现有 Dense/Sparse/Hybrid 行为不变。
2. 扩展内存 Graph Store 合同并实现基础图构建、选择器、Mock 抽取、规范化和离线测试，再增加 Neo4j Driver/适配器。
3. 先对 pilot 执行基础图 dry-run 和实际导入，验证幂等与版本隔离；再以小批量 Mock/真实抽取 smoke 验证缓存、预算和复核报告。
4. 冻结 Graph 开发集后运行零 LLM 的内存矩阵；Neo4j 可用时显式运行集成 smoke，并记录未验证的在线抽取或数据库项。
5. 更新 README、架构、数据 schema、实验手册和 `PROJECT_MAP.md`。回滚时停用 Graph 动作并继续使用现有 Dense/Sparse/Hybrid；图数据只按精确 graph_version 清理，抽取缓存和 processed/raw 输入不受影响。
