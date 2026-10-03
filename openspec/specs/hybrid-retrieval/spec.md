# hybrid-retrieval Specification

## Purpose

为科研论文 Chunk 提供可重建、可追溯且资源有界的 Sparse、Dense/Sparse 融合与本地重排能力，使精确术语召回能够补充语义检索，并为后续预算纠错提供独立、可审计的检索动作。

## Requirements

### Requirement: 版本化且可恢复的 Sparse 索引

系统 MUST 从通过质量门禁发布的 Chunk 构建版本化 Sparse 索引，并 MUST 将索引 schema、分词规则、评分参数和输入语料快照记录在身份或运行清单中。索引入口 MUST 支持 dry-run、pilot、显式论文和有界全量选择，并 MUST 以论文为隔离单元执行增量同步、幂等重跑和检查点恢复；不兼容或损坏的索引 MUST 被拒绝，不能静默复用或自动删除。

#### Scenario: dry-run 不写入索引

- **WHEN** 调用方对合法的受控论文选择执行 Sparse 索引 dry-run
- **THEN** 系统 MUST 报告索引版本、语料快照和预计新增、更新、跳过、删除数量，且 MUST NOT 创建、修改或删除索引与检查点

#### Scenario: 增量同步保持幂等

- **WHEN** 同一 Chunk ID、内容哈希、处理版本和 Sparse 配置被重复同步
- **THEN** 系统 MUST 跳过未变化记录；新增或变化记录 MUST 原子更新，论文中已不存在的记录 MUST 只在新记录成功提交后删除，重复执行后的可查询内容 MUST 一致

#### Scenario: 单篇失败可从检查点继续

- **WHEN** 有界批次中的一篇论文产物损坏、契约不匹配或索引事务失败
- **THEN** 系统 MUST 回滚该论文的本次修改、记录有限错误并继续其他论文，且后续运行 MUST 能从已完成论文之后恢复

#### Scenario: 不兼容索引被拒绝

- **WHEN** 已有索引的 schema、分词规则、评分参数或绑定语料状态与当前请求不兼容
- **THEN** 系统 MUST 在查询或写入前返回不可重试的结构化错误，除非调用方显式指定经过展示的精确重建目标

### Requirement: 有界且可溯源的 Sparse 检索

系统 MUST 对非空查询执行一次有界 Sparse/BM25 检索，支持科研缩写、连字符标识、模型名、数据集名和低频实体的确定性分词，并支持配置化 `top_k`、候选上限和白名单元数据过滤。每条命中 MUST 转换为统一内部 `Evidence`，保留稳定 Evidence/Chunk/Paper 标识、章节、页码、内容哈希、Sparse 原始分数和排名，并设置 `external=false`。

#### Scenario: 精确科研术语返回稳定 Evidence

- **WHEN** 查询包含索引中存在的缩写、连字符模型名、数据集名或低频实体
- **THEN** 系统 MUST 按 Sparse 分数降序和稳定次级键返回不超过 `top_k` 的 Evidence，且每条结果都能回溯到唯一的已发布 Chunk

#### Scenario: 未知过滤和非法边界提前失败

- **WHEN** 查询为空、`top_k` 非正数或超过上限，或过滤字段不在允许列表
- **THEN** 系统 MUST 在访问 Sparse 索引前返回不可重试的输入错误

#### Scenario: 无命中与后端故障不伪造证据

- **WHEN** 合法查询没有命中，或 Sparse 索引不可读、超时或暂时不可用
- **THEN** 系统 MUST 分别返回空结果或结构化后端错误，且 MUST NOT 伪造 Evidence、触发 LLM 或自动访问 Graph/Web

### Requirement: 可解释且确定性的 Dense/Sparse 融合

系统 MUST 提供 RRF 和 Weighted 两种版本化融合策略，按稳定来源身份合并 Dense 与 Sparse Evidence，并 MUST 保留各来源的原始分数、原始排名、融合分数和融合排名。Weighted 策略 MUST 使用版本化的分数归一化规则和总和为 1 的非负权重；所有策略 MUST 使用稳定次级键解决同分，且输入不得被原地修改。

#### Scenario: 重叠结果合并且保留来源排名

- **WHEN** 同一 Chunk 同时出现在 Dense 和 Sparse 候选中
- **THEN** 系统 MUST 只输出一条融合 Evidence，并同时保留 Dense、Sparse 的原始分数与排名以及新计算的融合分数与排名

#### Scenario: 单路独有结果参与融合

- **WHEN** 某条 Evidence 只来自 Dense 或 Sparse 一路
- **THEN** 系统 MUST 按选定融合公式使用已有一路的信息计算结果，MUST NOT 为缺失来源伪造分数或排名

#### Scenario: 非法融合配置被拒绝

- **WHEN** 融合方法未知、RRF 常数非法、权重为负、权重和不为 1 或候选/输出规模超过配置上限
- **THEN** 系统 MUST 在调用检索器或 Reranker 前返回不可重试的配置错误

### Requirement: Hybrid 检索和单路失败降级

系统 MUST 将 Dense 与 Sparse 保持为可独立调用的有界检索动作，并提供固定 Hybrid 编排以各调用至多一次、融合结果和可选去重。失败策略 MUST 由配置显式决定；允许部分降级时只能使用成功一路的真实 Evidence，并 MUST 在结果中记录失败后端和降级原因，两路均失败时 MUST 返回结构化错误。

#### Scenario: 两路成功生成 Hybrid 结果

- **WHEN** Dense 与 Sparse 检索均成功
- **THEN** 系统 MUST 按固定配置融合候选、执行稳定排序与可选去重，并记录两路的输入输出候选数和状态

#### Scenario: 单路失败按配置降级

- **WHEN** 一路返回结构化故障且配置允许部分降级，另一路成功返回 Evidence
- **THEN** 系统 MUST 只使用成功一路的 Evidence 生成结果，标记结果已降级并保留失败详情，MUST NOT 把该结果记录为完整双路融合

#### Scenario: 禁止降级或两路失败

- **WHEN** 配置禁止部分降级，或 Dense 与 Sparse 两路均失败
- **THEN** 系统 MUST 返回结构化检索错误且不输出成功结果，不得隐藏错误、重试成无界循环或切换到 Graph/Web

### Requirement: 受控本地 Cross-Encoder 重排

系统 MUST 通过公共 `Reranker` 契约接入固定模型标识与修订的本地 Cross-Encoder，并 MUST 惰性加载模型、使用可配置缓存路径和记录推理成本。默认候选上限 MUST 为 20，最终输出上限 MUST 为 8；重排 MUST 保留原始 Dense、Sparse、融合分数与排名，补充 Rerank 分数与排名，并继续保持完整 Evidence 溯源。LLM Reranker MUST NOT 成为默认路径或完成条件。

#### Scenario: 启用重排时只处理有界候选

- **WHEN** Hybrid 配置启用 Reranker 且融合候选可用
- **THEN** 系统 MUST 最多向模型提交融合 top-20，按 Rerank 分数和稳定次级键排序，并最多返回 top-8，同时保留重排前的全部排名字段

#### Scenario: 禁用重排时不加载模型

- **WHEN** 配置禁用 Reranker
- **THEN** 系统 MUST 直接返回有界融合结果，不得加载或下载 Reranker 权重，也不得产生 Rerank 分数或排名

#### Scenario: 重排失败按显式策略处理

- **WHEN** 模型加载、推理或输出校验失败
- **THEN** 系统 MUST 返回结构化错误；仅在配置允许重排降级时返回未重排的融合结果，并明确记录降级原因，MUST NOT 返回部分或顺序不明的重排批次

### Requirement: 检索运行记录和原子发布

系统 MUST 为 Sparse 与 Hybrid 检索生成可审计运行摘要，至少记录策略与版本、查询标识、语料/索引版本、各阶段候选数、时延、后端状态、降级状态、Reranker 调用次数和本地推理成本摘要。持久化结果 MUST 使用 UTF-8 稳定序列化和原子发布，且 MUST NOT 包含凭据、模型内部对象、完整 Prompt 或未选中的大段候选正文。

#### Scenario: 成功运行可重放和下钻

- **WHEN** 固定查询、语料、配置、索引、融合和 Reranker 版本完成检索
- **THEN** 系统 MUST 保存足以重放排序的版本标识、候选计数、阶段时延、最终 Evidence 与分数排名，并使同一确定性输入产生相同顺序

#### Scenario: 发布失败不留下成功产物

- **WHEN** 运行摘要或结果在写入、回读校验或目录替换期间失败
- **THEN** 系统 MUST 清理本次临时产物并返回失败，且 MUST NOT 留下可被识别为成功运行的半写目录

#### Scenario: 诊断信息保持脱敏和有界

- **WHEN** 后端、模型或文件系统错误被记录
- **THEN** 系统 MUST 只保存有限的错误代码、重试性、阶段和安全上下文，不得保存密钥、认证头、原始后端响应或无界候选内容

### Requirement: 零 LLM 的 Hybrid 检索评测

系统 MUST 在绑定同一语料快照的冻结开发集上比较 Dense、Sparse、RRF、Weighted 和 Fusion+Rerank。开发集 MUST 包含不少于 20 个具有稳定问题 ID、目标 Chunk 和标注说明的问题，并覆盖缩写、连字符标识、模型/数据集名和低频实体；所有候选配置 MUST 先以零 LLM 方式计算 Recall@K、MRR、nDCG@K、证据覆盖和时延，失败题 MUST 保留在分母与逐题结果中。

#### Scenario: 所有检索策略使用相同冻结输入

- **WHEN** 执行 Hybrid 开发集评测
- **THEN** 系统 MUST 对所有策略使用相同问题、目标 Chunk、语料快照和 K 值，保存逐题排名与失败，并拒绝目标缺失、问题重复或快照漂移的清单

#### Scenario: 候选方案评测不调用 LLM

- **WHEN** 运行 Dense、Sparse、RRF、Weighted 或 Fusion+Rerank 的检索指标矩阵
- **THEN** 系统 MUST NOT 调用回答生成或 LLM Judge，且 MUST 记录 Reranker 是否启用及其本地推理次数和时延

#### Scenario: 只有入选配置运行有界回答验证

- **WHEN** 开发集指标选定一个 Hybrid 配置并显式启用回答验证
- **THEN** 系统 MUST 复用现有证据约束生成规则，每题最多调用一次 LLM，保存所用配置和 Evidence；未显式启用时 MUST 保持零 LLM

#### Scenario: 单题失败仍计入报告

- **WHEN** 某题的任一检索、融合或重排阶段失败
- **THEN** 系统 MUST 保存该题的失败阶段与有限错误、继续其余问题，并在失败统计和指标分母中保留该题

