# Foundation Contracts Specification

## Purpose

为 KG-CRAG 的后续摄取、检索和 Agent 迭代提供严格、可导出且可离线验证的公共工程契约，使模块在接入真实模型与数据库前即可稳定协作和诊断失败。

## Requirements

### Requirement: 严格公共数据契约

系统 MUST 使用拒绝未知字段的校验模型表达跨模块公共数据，并 MUST 为约定的公共模型生成确定性 JSON Schema 快照。相同代码版本重复导出 MUST 产生语义和字节均一致的快照；公共字段、类型或约束发生变化时，快照差异 MUST 可供评审。

#### Scenario: 未知字段被拒绝

- **WHEN** 调用方使用包含未声明字段的数据构造公共模型
- **THEN** 系统 MUST 返回结构化校验错误，而不是静默丢弃该字段

#### Scenario: Schema 快照可复现

- **WHEN** 在相同代码版本和运行环境中连续两次导出公共模型 Schema
- **THEN** 两次输出 MUST 完全一致，并使用稳定的 UTF-8 JSON 格式

### Requirement: 统一错误语义

系统 MUST 提供跨模块统一的结构化错误详情，至少包含稳定错误代码、安全错误消息、是否可重试和有限上下文。错误详情 MUST 拒绝未知字段，且 MUST 不包含密钥、认证头或完整外部响应正文。

#### Scenario: 可重试外部错误

- **WHEN** 适配器报告暂时性外部服务失败
- **THEN** 错误详情 MUST 使用稳定的外部服务错误代码并将 `retryable` 标记为 true

#### Scenario: 不可重试输入错误

- **WHEN** 调用方提供不符合公共契约的输入
- **THEN** 错误详情 MUST 使用稳定的校验错误代码并将 `retryable` 标记为 false

#### Scenario: 敏感上下文被拒绝

- **WHEN** 错误上下文包含表示密码、Token、API Key 或认证信息的字段名
- **THEN** 系统 MUST 在错误详情创建阶段拒绝该上下文

### Requirement: TraceEvent v1

系统 MUST 提供严格校验的 TraceEvent v1，至少包含 schema 版本、Trace 标识、非负序号、节点、事件、发生时间和有限详情。AgentState MUST 直接引用该公共模型；Trace 详情 MUST 拒绝敏感字段名和未知顶层字段。

#### Scenario: 有效事件进入 Agent 状态

- **WHEN** 调用方提供完整且合法的 TraceEvent v1
- **THEN** 系统 MUST 接受该事件，并允许 AgentState 按事件顺序保存它

#### Scenario: 非法序号被拒绝

- **WHEN** TraceEvent 的序号为负数
- **THEN** 系统 MUST 在事件进入工作流状态前拒绝它

#### Scenario: Trace 不接受凭据

- **WHEN** Trace 详情包含密码、Token、API Key 或认证信息字段
- **THEN** 系统 MUST 拒绝该事件，且 MUST 不把敏感值写入日志

### Requirement: 确定性离线测试替身

系统 MUST 为 Retriever、Reranker、Vector Store 和 Graph Store 提供不访问网络或真实数据库的确定性实现。这些实现 MUST 遵守公开 Protocol、记录可断言的调用信息、尊重 `top_k` 和过滤参数，并对相同状态与输入返回相同结果。

#### Scenario: 离线检索可重复

- **WHEN** 测试以相同候选、查询、过滤条件和 `top_k` 连续调用离线 Retriever
- **THEN** 两次结果及其顺序 MUST 相同，且调用记录 MUST 保留查询参数

#### Scenario: 离线重排序保持溯源

- **WHEN** 离线 Reranker 对一组 Evidence 排序
- **THEN** 输出 MUST 保留每条 Evidence 的来源、位置和元数据，并 MUST 稳定截断到 `top_k`

#### Scenario: 内存存储幂等更新和删除

- **WHEN** 相同稳定标识的记录被重复写入后再按论文删除
- **THEN** 存储 MUST 不产生重复记录，并 MUST 只移除属于目标论文的数据

#### Scenario: 非法边界参数被拒绝

- **WHEN** 离线实现收到非正数 `top_k`、负数图跳数或不匹配的向量维度
- **THEN** 系统 MUST 在执行检索前返回明确的不可重试输入错误

### Requirement: 运行配置与健康响应

系统 MUST 在进程启动或配置对象创建时校验端口、循环上限和必要连接地址。健康检查 MUST 不依赖在线模型、搜索、Qdrant 或 Neo4j，并 MUST 只返回状态、应用版本和环境名称。

#### Scenario: 合法配置可加载

- **WHEN** 环境变量使用合法端口、连接地址和循环上限
- **THEN** 系统 MUST 生成通过校验的运行配置

#### Scenario: 非法配置尽早失败

- **WHEN** 端口越界、连接地址格式非法或循环次数超出约束
- **THEN** 系统 MUST 在启动外部服务调用前返回配置校验错误

#### Scenario: 健康检查离线可用

- **WHEN** Qdrant、Neo4j、在线模型和 Web 搜索均未启动
- **THEN** 健康检查 MUST 返回成功状态、应用版本和环境名称，且响应中 MUST 不出现任何密钥字段
