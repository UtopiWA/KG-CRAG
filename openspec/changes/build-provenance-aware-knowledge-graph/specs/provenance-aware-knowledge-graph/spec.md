# Spec Delta

## Purpose

为科研论文提供可重建、来源可追溯且资源有界的知识图谱和 Graph Retriever，使关系型与多跳问题能够按需获取图证据，同时避免无条件图检索和全语料正文 LLM 抽取。

## ADDED Requirements

### Requirement: 版本化图 schema 与稳定事实身份

系统 MUST 为 Paper、Author、Institution、Method、Model、Dataset、Task、Metric、Result、Chunk 定义版本化节点类型、稳定标识、允许关系、端点约束和必填属性。每个图节点与关系 MUST 绑定图 schema 版本和语料快照；正文事实 MUST 绑定现存 `source_chunk_id`、来源内容哈希、置信度、抽取器版本和创建时间，元数据事实 MUST 绑定稳定 Paper 或元数据记录及其内容哈希。

#### Scenario: 合法图事实具有完整溯源

- **WHEN** 白名单关系的端点类型、稳定标识和来源满足当前图 schema
- **THEN** 系统 MUST 接受该事实并生成可重放的稳定事实 ID，且调用方能够从事实解析到唯一 Chunk 或高确定性元数据记录

#### Scenario: 非法模式或悬空来源被拒绝

- **WHEN** 节点类型、关系类型、关系端点组合不在白名单中，或正文事实引用不存在、哈希不匹配或不属于该 Paper 的 Chunk
- **THEN** 系统 MUST 在写入图数据库前返回不可重试的结构化数据错误，且 MUST NOT 发布部分事实

### Requirement: 无 LLM 的基础图构建

系统 MUST 从已发布且通过质量门禁的 Paper/Chunk 以及原始记录中已有的显式引用标识构建 Paper、Author、Chunk、作者关系、包含关系和可解析引用关系，不得为缺失作者 ID、机构或引用目标调用 LLM、访问网络或推测事实。基础图构建 MUST 支持 dry-run、pilot、显式论文和带明确上限的全量选择，并 MUST 记录语料快照、图版本、逐篇差异和无法解析的引用。

#### Scenario: 重复导入保持幂等

- **WHEN** 相同 Paper/Chunk、元数据哈希、图 schema 和规范化版本被重复构建
- **THEN** 系统 MUST 跳过未变化节点与关系，不产生重复记录，并使查询结果和稳定 ID 保持一致

#### Scenario: 引用缺失或无法解析

- **WHEN** Paper 没有引用列表，或引用值不能唯一解析到当前语料中的 Paper
- **THEN** 系统 MUST 分别记录无引用或未解析引用，继续处理其他确定性元数据事实，且 MUST NOT 创建猜测的 Paper 或引用边

#### Scenario: dry-run 与无界选择

- **WHEN** 调用方执行合法 dry-run，或请求未给出上限的全量构建
- **THEN** 合法 dry-run MUST 只返回图身份和差异计划且零数据库写入；无界请求 MUST 在数据库和 LLM 调用前被拒绝

### Requirement: 代表 Chunk 的确定性选择与有界结构化抽取

系统 MUST 对 pilot 每篇论文从摘要、方法、实验、结果或结论中确定性选择 3–5 个代表 Chunk，并使用版本化 Prompt、模型标识、模型修订、白名单 schema 和严格结构化输出执行正文关系抽取。一次 change 的正文抽取 MUST 不超过 100 次 LLM 请求，并 MUST 同时受每 Chunk 输入、单次输出、累计输入 Token、累计输出 Token、批大小和并发上限约束；dry-run、缓存命中和禁用抽取时 MUST 零 LLM 调用。

#### Scenario: pilot 抽取在预算内完成

- **WHEN** 代表 Chunk 选择和剩余调用/Token 预算均合法
- **THEN** 系统 MUST 每个 Chunk 至多调用一次抽取 Provider，校验完整结构化输出，并记录 Prompt、模型、Chunk 内容和图 schema 版本形成的缓存身份及实际或保守估算用量

#### Scenario: 预算不足时停止而不超支

- **WHEN** 下一批请求将超过调用、输入 Token、输出 Token、批大小或并发上限
- **THEN** 系统 MUST 在发起该批请求前停止，保存已完成检查点并把未处理 Chunk 标记为预算耗尽，MUST NOT 隐藏重试、扩大范围或自动改用其他在线 Provider

#### Scenario: 单 Chunk 输出无效或 Provider 失败

- **WHEN** 抽取响应超长、不是合法结构、包含未知类型、引用当前 Chunk 之外的文本，或 Provider 调用失败
- **THEN** 系统 MUST 拒绝该 Chunk 的整批候选事实、记录脱敏错误并继续其他 Chunk；不得发布部分解析结果或没有来源的事实

### Requirement: 保守实体规范化与人工复核队列

系统 MUST 以版本化规则规范实体名称和别名，并 MUST 仅在稳定外部 ID 相同，或实体类型相同且规范键唯一、没有冲突证据时自动合并。低置信度候选、同名不同类型、冲突外部 ID、缩写歧义和近似字符串匹配 MUST 保持为不同实体并进入包含候选、理由、分数和来源的复核队列。

#### Scenario: 高确定性实体安全合并

- **WHEN** 两个同类型候选具有相同稳定外部 ID，或在当前图版本中具有唯一且无冲突的规范键
- **THEN** 系统 MUST 生成同一规范实体 ID，合并去重别名并保留每个原候选和来源事实的映射

#### Scenario: 歧义实体保持分离

- **WHEN** 候选同名但类型不同、外部 ID 冲突、规范键对应多个实体或置信度低于自动合并阈值
- **THEN** 系统 MUST 保持独立实体、禁止自动重连既有事实，并在复核报告中给出有限且可审计的冲突原因

#### Scenario: 复核决定可重放

- **WHEN** 人工对特定候选对给出合并或保持分离决定
- **THEN** 系统 MUST 以版本化决定文件重放该结论并记录决定依据；未知实体、失效来源或与外部 ID 冲突的决定 MUST 在导入前被拒绝

### Requirement: 版本化图存储与逐篇事务生命周期

系统 MUST 通过后端无关 Graph Store 契约和生产图数据库适配器管理图身份、约束、逐篇记录状态、事务同步、受控查询和精确版本清理。新图版本 MUST 与旧版本隔离；逐篇同步 MUST 在同一事务中写入完整新状态后删除该论文的陈旧事实，失败时回滚该论文且允许批次继续。

#### Scenario: 图身份兼容时安全复用

- **WHEN** 已有图的 schema、语料快照、规范化规则、抽取版本和约束与当前请求完全一致
- **THEN** 系统 MUST 校验约束和记录状态后复用该图版本，并仅同步新增、变化或删除的论文事实

#### Scenario: 不兼容或损坏图被拒绝

- **WHEN** 图身份缺失、不兼容，约束缺失，或已记录来源无法回溯到当前 processed 语料
- **THEN** 系统 MUST 在查询和写入前返回结构化错误，除非调用方显式提供 dry-run 展示的精确重建版本；系统 MUST NOT 自动删除其他图版本

#### Scenario: 单篇事务失败隔离

- **WHEN** 一篇论文的节点或关系违反约束或后端事务失败
- **THEN** 系统 MUST 回滚该论文的全部本次变化、记录有限错误、继续其他论文，并使上一次已提交状态继续可查询

### Requirement: 只读白名单图查询

系统 MUST 只允许版本化、参数化的只读图查询模板，并 MUST 在访问图数据库前校验模板标识、实体/关系类型、过滤字段、`top_k`、候选规模、hop 上限、路径长度和超时。系统 MUST NOT 接受用户或 LLM 提供的任意查询语言文本，且 MUST NOT 通过查询入口执行创建、更新、删除、过程调用或无界路径遍历。

#### Scenario: 合法关系或多跳模板

- **WHEN** 调用方选择允许的实体邻接、关系查找或有界多跳模板并提供合法参数
- **THEN** 系统 MUST 以参数化查询执行一次只读请求，返回不超过候选上限的稳定排序路径及其事实溯源，并记录模板版本、数据库调用数和时延

#### Scenario: 非法查询在后端调用前失败

- **WHEN** 模板未知、参数包含查询语言片段、类型或过滤字段不在白名单、hop/候选/超时越界，或请求写操作
- **THEN** 系统 MUST 返回不可重试的输入错误，数据库调用次数 MUST 为零，且错误不得回显凭据、完整查询或无界参数

#### Scenario: 图后端为空或不可用

- **WHEN** 合法模板无匹配路径，或图数据库超时、认证失败或暂时不可用
- **THEN** 系统 MUST 分别返回成功的空结果或脱敏的结构化后端错误，且 MUST NOT 伪造路径、隐藏重试或自动切换 Dense、Sparse、Web 或回答生成

### Requirement: Graph Retriever 统一 Evidence 与路径溯源

系统 MUST 将图查询命中转换为 `external=false` 的统一 `Evidence`。每条 Evidence MUST 具有稳定 ID、Graph 分数与排名、图/模板版本和路径标识；正文支持的关系 MUST 按支持 Chunk 拆分为可引用 Evidence，并保留组成路径的稳定事实 ID、hop 数、Paper/Chunk、章节、页码和内容哈希。仅由高确定性元数据支持的关系 MUST 明确标记元数据来源，不得伪装为正文 Chunk。

#### Scenario: 多跳路径返回可引用证据

- **WHEN** 一条有界路径由一个或多个正文事实支持且所有来源仍有效
- **THEN** 系统 MUST 为唯一支持 Chunk 返回稳定排序的 Evidence，使每条关系都可从路径事实回溯到至少一条 Evidence，并且不得泄漏数据库节点、会话或驱动对象

#### Scenario: 元数据事实保持来源区别

- **WHEN** 命中作者、包含或显式引用等仅由规范元数据支持的关系
- **THEN** 系统 MUST 返回标记为内部元数据来源的 Graph Evidence，保存 Paper/元数据哈希和事实 ID，且 MUST NOT 填造不存在的章节、页码或 `source_chunk_id`

#### Scenario: 溯源失效时拒绝命中

- **WHEN** 路径任一正文事实的 Chunk 缺失、内容哈希漂移或 Paper 身份不匹配
- **THEN** 系统 MUST 从结果中拒绝整条受影响路径、记录数据漂移错误，且 MUST NOT 仅凭图数据库中的陈旧文本生成 Evidence

### Requirement: 可恢复运行记录与原子产物

系统 MUST 为基础图、正文抽取、规范化和导入生成可恢复检查点及 UTF-8 稳定运行清单，记录配置哈希、语料/图/Prompt/模型/规范化版本、选择范围、调用与 Token 预算、实际用量、逐篇或逐 Chunk 状态、数据库调用、时延、复核统计和有限错误。缓存、清单、复核报告与评测结果 MUST 原子发布，且不得包含密钥、认证头、完整 Prompt、模型内部对象或无界原文。

#### Scenario: 同版本断点重跑不重复计费

- **WHEN** 同一 Chunk 内容、Prompt、模型修订、schema 和配置已有通过校验的抽取缓存或完成检查点
- **THEN** 系统 MUST 复用结果且不重复调用 LLM；版本任一部分变化时 MUST 形成新身份而不是静默复用

#### Scenario: 发布失败不留下成功状态

- **WHEN** 检查点、清单、复核报告或评测结果在写入、回读校验或替换期间失败
- **THEN** 系统 MUST 清理本次临时产物并返回失败，且 MUST NOT 留下可被识别为成功的半写文件或目录

### Requirement: 关系与多跳图检索评测

系统 MUST 维护绑定 pilot 语料快照和图版本的人工标注开发子集，覆盖关系查找、实体歧义、单跳、多跳、无路径和错误来源场景。评测 MUST 默认零 LLM，并报告路径命中率、Evidence Recall@K、MRR、nDCG@K、溯源完整率、失败数、数据库调用数和阶段时延；失败题 MUST 保留在逐题结果和指标分母中。

#### Scenario: 图策略使用冻结输入

- **WHEN** 执行 Graph Retriever 开发集评测
- **THEN** 系统 MUST 校验问题 ID、目标事实/路径/Chunk、语料快照、图版本和 K 值，拒绝重复标识、缺失来源或版本漂移，并原子保存逐题路径、Evidence 和失败

#### Scenario: 默认评测零在线调用

- **WHEN** 使用已冻结图版本运行 Graph 检索指标
- **THEN** 系统 MUST NOT 调用 LLM、Embedding、Web、Dense 或 Sparse Retriever，并 MUST 记录每题恰当的零或一次图数据库调用及受控时延

#### Scenario: 真实依赖仅显式启用

- **WHEN** 默认测试未设置真实抽取或 Neo4j 集成开关
- **THEN** 系统 MUST 使用确定性 Mock 抽取器和内存图存储，真实测试 MUST 明确跳过且不得连接网络、启动数据库或消耗 Token
