# Spec Delta

## Purpose

为已发布的科研论文 Chunk 建立版本化、可重建且可审计的 Dense Vector RAG 基线，使系统能够执行语义检索、返回可追溯证据，并在严格证据边界内生成带引用的保守回答与可复现实验结果。

## ADDED Requirements

### Requirement: 版本化 Embedding 与安全缓存

系统 MUST 通过 `EmbeddingProvider` 对 Chunk 和查询执行有界批量向量化，并 MUST 在配置中固定 Provider、模型标识、模型修订、输入规范化规则、向量维度、批大小和归一化策略。缓存键 MUST 绑定完整规范化输入哈希、模型标识与修订以及影响输出的配置哈希；系统 MUST 校验返回数量、顺序、维度和全部数值均为有限值，且 MUST NOT 缓存不完整或未通过校验的结果。

#### Scenario: 相同输入复用有效缓存

- **WHEN** 相同规范化文本、模型修订和 Embedding 配置再次请求向量
- **THEN** 系统 MUST 返回与首次成功结果一致的向量而不重复调用 Provider，并记录缓存命中数量

#### Scenario: 配置变化隔离缓存

- **WHEN** 模型标识、模型修订、输入规范化规则、向量维度或归一化策略发生变化
- **THEN** 系统 MUST 使用不同缓存命名空间或缓存键，且 MUST NOT 把旧配置生成的向量作为新配置结果复用

#### Scenario: Provider 返回非法批次

- **WHEN** Provider 返回的向量数量与输入不一致、顺序无法对应、维度错误或包含非有限数值
- **THEN** 系统 MUST 以结构化错误拒绝整个批次，MUST NOT 写入该批缓存或向量集合，并允许上层按论文隔离失败

### Requirement: 版本化且兼容性受控的向量集合

系统 MUST 为 Dense 索引定义版本化集合契约，至少包含集合 schema 版本、Embedding 模型与修订、向量维度、距离度量和允许过滤的 payload 字段。集合名称或版本 MUST 由该契约稳定确定；每个向量记录 MUST 使用稳定 Chunk ID 作为幂等标识，并保存重建 Evidence 所需的论文标识、章节、页码范围、Chunk 顺序、内容、内容哈希和处理版本。

#### Scenario: 兼容集合被复用

- **WHEN** 目标集合的 schema、模型修订、维度、距离度量和 payload 契约与当前配置完全一致
- **THEN** 系统 MUST 复用该集合执行增量写入，并在运行清单中记录集合版本

#### Scenario: 不兼容集合默认失败

- **WHEN** 已有集合与当前向量维度、距离度量、模型修订或 payload schema 不兼容，且调用方未显式请求重建
- **THEN** 系统 MUST 在写入前返回不可重试配置错误，且 MUST NOT 删除、修改或混用已有集合

#### Scenario: 显式重建限定目标

- **WHEN** 调用方显式请求重建一个经过解析和展示的目标集合
- **THEN** 系统 MUST 只替换该集合，在新集合通过 schema 校验后才将其视为可用，并记录原版本、目标版本和重建结果

### Requirement: 有界、幂等且可审计的索引编排

系统 MUST 提供从已发布 `data/processed` Chunk 产物构建 Dense 索引的统一入口，支持 dry-run、pilot、显式论文、有界批量、增量更新和显式重建。入口 MUST 在任何写入前校验 Chunk 契约、产物哈希、选择范围、配置和集合兼容性；相同 Chunk ID、内容哈希和索引版本的重复运行 MUST 不产生重复记录或无意义的重新向量化。

#### Scenario: dry-run 零写入

- **WHEN** 调用方以 dry-run 选择 pilot、指定论文或有界批量
- **THEN** 系统 MUST 报告输入快照、集合版本以及预计新增、更新、跳过和删除数量，但 MUST NOT 调用写入、删除或缓存发布操作

#### Scenario: 增量更新保持幂等

- **WHEN** 索引中已存在相同 Chunk ID、内容哈希、处理版本和 Embedding 配置的记录
- **THEN** 系统 MUST 校验后跳过该记录；新增或内容变化的 Chunk MUST 以稳定 ID 写入，重复执行后集合内容 MUST 保持一致

#### Scenario: 单篇失败不终止有界批次

- **WHEN** 有界批次中的一篇论文缺少产物、哈希不匹配或 Embedding 失败，而其他论文有效
- **THEN** 系统 MUST 记录该论文失败并继续处理其他论文，MUST NOT 把失败论文标记为已索引，且最终退出状态和运行清单 MUST 反映部分失败

#### Scenario: 无界选择被拒绝

- **WHEN** 调用方未提供受控选择，或请求的论文数、批大小或候选规模超过配置上限
- **THEN** 系统 MUST 在访问 Embedding Provider 或 Qdrant 前以不可重试输入错误拒绝请求

### Requirement: 可追溯且有界的 Dense 检索

系统 MUST 将非空查询向量化并执行一次有界 Dense 检索，不得在本基线中触发查询重写、检索器切换或纠错循环。检索 MUST 支持配置化 `top_k`、候选上限、允许字段的元数据过滤和可选按论文去重；返回的每条 `Evidence` MUST 使用稳定标识，标记为内部 Chunk 来源，并保留原始 Dense 分数、原始排名、Chunk ID、Paper ID、章节、页码和内容哈希，不得泄漏后端私有对象。

#### Scenario: 查询返回有序 Evidence

- **WHEN** 查询有效且兼容集合中存在匹配 Chunk
- **THEN** 系统 MUST 按 Dense 分数降序和稳定次级键返回不超过 `top_k` 条 Evidence，并使每条 Evidence 都能解析回唯一 Chunk、论文和原始页码

#### Scenario: 过滤与按论文去重受上限约束

- **WHEN** 调用方提供允许的过滤条件或启用按论文去重
- **THEN** 系统 MUST 只返回满足过滤条件的 Evidence，并通过一次不超过配置候选上限的检索选择每篇论文的最高排名结果

#### Scenario: 未知过滤或非法边界被拒绝

- **WHEN** 过滤字段不在允许列表、查询为空、`top_k` 非正数或超过配置上限
- **THEN** 系统 MUST 在查询向量化或数据库访问前返回不可重试输入错误

#### Scenario: 无命中或后端失败显式降级

- **WHEN** 集合无匹配结果，或 Embedding/Qdrant 返回超时或暂时性失败
- **THEN** 系统 MUST 分别返回空证据状态或可重试结构化错误，MUST NOT 伪造 Evidence，也不得自动切换到 Sparse、Graph 或 Web 来源

### Requirement: 仅依据证据的保守回答与引用

系统 MUST 使用版本化 Prompt 和有界上下文从选定 Dense Evidence 生成结构化回答。生成输入 MUST 只包含用户问题、回答规则和选定 Evidence；每个事实性结论 MUST 引用一个或多个已提供 Evidence，引用 MUST 可解析到 Chunk、论文、章节和页码。结果 MUST 包含回答、结构化结论类型、引用、置信度、证据不足标记、检索路径、外部证据标记和 Trace ID；Dense 基线的检索路径 MUST 为 Dense，外部证据标记 MUST 为 false。

#### Scenario: 充分证据生成带引用回答

- **WHEN** 选定 Evidence 达到配置的最低数量和相关性门槛，且生成结果通过结构和引用校验
- **THEN** 系统 MUST 返回只引用已选 Evidence 的回答，并使所有引用标识和位置可被程序化解析

#### Scenario: 证据不足时不调用生成或补造事实

- **WHEN** 检索结果为空或所有候选均未达到配置的证据门槛
- **THEN** 系统 MUST 返回确定性的证据不足结果、低置信度和空引用，且 MUST NOT 调用 LLM、陈述具体论文事实或触发外部检索

#### Scenario: 无效生成结果安全失败

- **WHEN** LLM 输出无法解析、包含未知引用、事实性结论没有引用或响应超出配置大小上限
- **THEN** 系统 MUST 拒绝该输出并返回结构化生成错误或保守失败结果，MUST NOT 将未验证文本作为最终回答

#### Scenario: 基线调用次数受硬上限约束

- **WHEN** 执行一次 Dense RAG 问答
- **THEN** 系统 MUST 最多执行一次检索和一次生成调用，不得在本 change 内进行隐藏重试、反思或循环再生成

### Requirement: 单命令入口与原子运行记录

系统 MUST 为建索引、Dense 问答和 pilot 评测分别提供单一命令入口；命令 MUST 支持显式配置路径和安全的运行上限，并在执行前显示解析后的操作目标、配置哈希、数据快照、模型版本和索引版本。非 dry-run 执行 MUST 生成 UTF-8 运行清单，记录运行标识、开始/结束时间、状态、时延、选择范围、计数、有限错误详情和产物相对路径；清单与结果 MUST 原子发布。

#### Scenario: 固定配置可重放问答

- **WHEN** 相同问题、数据快照、索引版本、Prompt 版本、模型配置和确定性 Provider 被再次执行
- **THEN** 系统 MUST 产生相同 Evidence 顺序、引用映射和结构化结果，并在运行记录中保存全部重放标识

#### Scenario: 配置或目标无效时不留下成功记录

- **WHEN** 配置校验失败、索引不存在或不兼容、输出目标越界，或执行在原子发布前失败
- **THEN** 系统 MUST 返回非零状态且不得留下被视为成功的半写清单或结果文件

### Requirement: 冻结 pilot 与可复现基线评测

系统 MUST 维护不少于 20 个经过人工复核的 pilot 问题，每题 MUST 具有稳定问题 ID、问题文本、目标 Chunk ID 集合和标注说明，并绑定语料快照。评测 MUST 使用冻结的 Dense 配置、索引、Prompt 和模型版本，至少计算 Recall@K、MRR、nDCG@K、引用准确率与引用召回率，保存逐题 Evidence、回答、引用、时延、状态和失败原因；失败样本 MUST 计入汇总而不能被丢弃。

#### Scenario: 完整 pilot 生成可审计报告

- **WHEN** 所有 pilot 标注都能解析到当前语料快照中的 Chunk，且索引与评测配置兼容
- **THEN** 系统 MUST 运行全部问题并原子保存逐题结果、指标汇总、配置哈希、数据快照、索引版本、Prompt/模型版本和运行时间

#### Scenario: 单题失败仍保留在结果中

- **WHEN** 某题发生检索、生成、引用校验或超时失败
- **THEN** 系统 MUST 保存该题的失败状态和有限错误详情，继续其余问题，并在指标分母与失败统计中保留该题

#### Scenario: 标注或冻结条件漂移被拒绝

- **WHEN** 目标 Chunk 不存在、问题 ID 重复、语料快照不匹配，或调用方试图在同一基线标识下静默改变配置、Prompt、模型或索引版本
- **THEN** 系统 MUST 在评测开始前拒绝运行，并要求生成新的版本化基线标识或修复标注

### Requirement: 离线默认测试与受控外部集成

系统 MUST 为 Embedding、LLM、Vector Store、Dense Retriever 和问答链保留确定性离线替身，使默认单元测试和离线集成测试不访问网络、在线模型或真实 Qdrant。真实 Qdrant、真实 Embedding 和真实 LLM 验收 MUST 通过显式标记或命令启用，并 MUST 使用有界 fixture、超时和无敏感信息的失败记录。

#### Scenario: 默认测试完全离线

- **WHEN** 开发者在没有凭据、模型服务和 Qdrant 的环境中运行默认测试
- **THEN** 系统 MUST 使用 Mock 与内存实现覆盖正常、证据不足、非法输入、外部失败和上限场景，且不得发起网络连接

#### Scenario: 显式集成测试隔离外部失败

- **WHEN** 开发者显式启用真实 Qdrant 或 Provider 集成测试
- **THEN** 系统 MUST 使用独立测试集合和有界数据，测试结束后只清理明确命名的测试资源，并把服务不可用报告为跳过或受控失败而不影响默认离线测试
