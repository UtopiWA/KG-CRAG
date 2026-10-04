# evidence-sufficiency-corrective-workflow Specification

## Purpose

为复杂科研文献问题提供可验证的证据需求、可解释的充分性诊断和资源有界的内部纠错工作流，使系统能够定位首次检索的具体缺口、选择针对性动作，并在证据充分或继续检索无收益时可靠停止。

## Requirements

### Requirement: 稳定且可验证的证据需求 facet

系统 MUST 将比较、多跳、关系和其他复杂问题表示为有稳定标识的 `EvidenceRequirement` 集合；每个 facet MUST 至少包含规范化描述、是否必需、期望证据类型、可机器验证的满足条件和生成来源。规则分析 MUST 是默认基线；系统 MUST 仅在规则结果标记为复杂或低置信度且预算允许时调用版本化 LLM 生成结构化候选，并 MUST 在接受前执行相同的严格校验。相同问题、配置、Prompt、模型和规则版本 MUST 产生相同身份与顺序。

#### Scenario: 比较问题生成两侧必需 facet

- **WHEN** 问题要求比较两个明确对象的架构和实验结果
- **THEN** 系统 MUST 生成覆盖两个对象及两个比较维度的稳定必需 facet，并为每项给出可验证条件，而不是只保存自由文本问题类型

#### Scenario: 简单问题使用规则结果且零 LLM

- **WHEN** 规则分析对问题产生完整且高置信度的证据需求
- **THEN** 系统 MUST 接受规则结果，记录生成方式，并且 MUST NOT 调用 LLM

#### Scenario: 模型候选无效时保守回退

- **WHEN** LLM 输出未知字段、重复或越界 facet、不可验证条件、超长内容或非结构化响应
- **THEN** 系统 MUST 原子拒绝该响应、记录脱敏失败，并使用确定性规则结果或返回可解释的需求生成失败，不得让自由文本进入条件边

### Requirement: 可追溯的 Evidence—facet 覆盖矩阵

系统 MUST 对每条内部 `Evidence` 与每个 facet 生成结构化覆盖项，至少记录 Evidence/facet 稳定标识、是否匹配、支撑强度、匹配依据、来源质量和冲突标记；覆盖判断 MUST 只使用统一 Evidence 字段、版本化规则及受控结构化输出，并 MUST 保留到原始 Chunk 或高确定性图元数据的来源。`external=true` 的 Evidence MUST NOT 被计为本迭代的内部覆盖。

#### Scenario: 一条证据覆盖多个可验证 facet

- **WHEN** 同一内部 Evidence 同时满足多个 facet 的实体、关系或结果条件
- **THEN** 系统 MUST 为每个满足关系生成独立覆盖项，保留同一来源身份和各自匹配依据，且不得复制或改写 Evidence 正文

#### Scenario: 来源漂移或外部证据不计内部覆盖

- **WHEN** Evidence 的来源标识/内容哈希不能由当前语料复核，或 Evidence 标记为外部来源
- **THEN** 系统 MUST 将对应覆盖项标记为无效或排除，记录有限原因，并 MUST NOT 用其满足内部必需 facet

#### Scenario: 同一状态的矩阵可重复

- **WHEN** Evidence 顺序变化但问题、facet、证据集合和版本均未变化
- **THEN** 系统 MUST 产生语义和排序均一致的覆盖矩阵与稳定矩阵身份

### Requirement: 结构化充分性、冲突与缺口诊断

系统 MUST 基于覆盖矩阵输出严格的充分性评估，至少包含已覆盖 facet、缺失必需 facet、可选 facet、冲突、每项支撑强度、总体置信度和充分/不足结论。必需 facet 只有在满足配置化最小证据数、支撑阈值、来源约束且没有未解决的阻断冲突时才算覆盖；总体相关性高 MUST NOT 替代逐 facet 判断。

#### Scenario: 首轮证据已经充分

- **WHEN** 所有必需 facet 均满足最小证据和支撑阈值且不存在阻断冲突
- **THEN** 系统 MUST 将状态标记为充分、保留选定 Evidence，并直接进入成功停止，不生成或执行纠错动作

#### Scenario: 比较缺侧定位到具体 facet

- **WHEN** 检索结果只支持比较对象的一侧而另一侧没有合格来源
- **THEN** 系统 MUST 将缺失侧对应的必需 facet 标记为未覆盖，给出有限诊断依据，而不得把总体相关性判为充分

#### Scenario: 冲突证据阻止虚假充分

- **WHEN** 两条有效 Evidence 对同一必需 facet 给出不能同时成立的结果且没有配置允许的解决依据
- **THEN** 系统 MUST 记录冲突来源、降低该 facet 的满足状态并保持证据不足，而不得静默选择分数较高的一条

### Requirement: 白名单内部纠错动作目录

系统 MUST 以版本化目录注册 Dense、Sparse、Hybrid、Graph、查询重写、问题分解和参数调整动作；每项 MUST 声明可处理的缺口类型、输入契约、调用的受控工具、候选/子问题上限、确定性成本估计和失败语义。动作执行 MUST 通过现有公共 Retriever/Graph 边界返回统一内部 Evidence，不得接收任意工具名、动态代码、任意 Cypher、Web 动作或回答生成动作。

#### Scenario: 术语错配选择精确召回动作

- **WHEN** 缺失 facet 的诊断为缩写、模型名或低频实体术语错配，且 Sparse/Hybrid 动作在剩余预算内
- **THEN** 候选目录 MUST 提供以该 facet 为目标的 Sparse 或 Hybrid 动作及其查询、成本和预期增益，不得无条件调用全部检索器

#### Scenario: 多跳断链选择 Graph 或受控分解

- **WHEN** 已有证据覆盖路径两端但缺少可验证的关系链
- **THEN** 候选目录 MUST 只产生有界 Graph 模板或最多三个子问题的分解动作，并保留目标 facet 和路径/hop 上限

#### Scenario: 未注册动作在执行前失败

- **WHEN** 决策引用未知动作、Web、自由文本工具名或越界参数
- **THEN** 系统 MUST 在任何 Retriever、数据库或模型调用前返回不可重试的结构化错误，工具调用次数 MUST 为零

### Requirement: 可解释且确定性的纠错决策

系统 MUST 为每个合法候选动作计算 `CorrectionDecision`，记录目标缺失 facet、预期覆盖增益、预计检索/模型调用、Token、候选量、时延、历史重复惩罚、总效用和选择理由。默认策略 MUST 使用版本化规则和成本表稳定排序；系统 MUST 仅在明确启用且预算允许时使用结构化 LLM 建议，最终动作仍 MUST 通过白名单、预算与正收益校验。相同结构化状态和配置 MUST 选择相同动作或相同停止原因。

#### Scenario: 正收益动作被稳定选择

- **WHEN** 多个动作合法且至少一个动作在剩余预算内具有正的预期覆盖净增益
- **THEN** 系统 MUST 按效用、成本和稳定次级键选择唯一动作，记录其他候选及未选择理由

#### Scenario: 无正收益时停止

- **WHEN** 所有候选动作预期增益非正、重复已执行动作且状态未改善，或成本超过剩余预算
- **THEN** 系统 MUST 产生结构化保守停止决定，MUST NOT 为消耗剩余预算而执行无意义调用

#### Scenario: 建议模型失败不切换 Provider

- **WHEN** 可选 LLM 决策建议超时、失败或返回非法结构
- **THEN** 系统 MUST 记录一次受预算约束的失败并回退到确定性规则决策或停止，MUST NOT 隐藏重试或自动切换 Provider

### Requirement: 多维硬预算和原子计费

系统 MUST 使用结构化预算账本同时限制内部检索轮次、子问题数、LLM 调用、输入/输出 Token、候选数、上下文规模和墙钟时延。单题内部检索总轮次 MUST 不超过 2，子问题 MUST 不超过 3，LLM 调用 MUST 不超过 4，输入输出 Token 合计 MUST 不超过 20,000；所有其他上限 MUST 是有界配置。每个动作 MUST 在调用前原子预留最坏成本，在完成或失败后记录实际或保守估算用量，任何维度不足时不得发起调用。

#### Scenario: 下一动作超过任一预算

- **WHEN** 候选动作的预留成本会超过任一剩余上限
- **THEN** 系统 MUST 在调用工具或模型前拒绝该动作，将原因记录为预算不足，并选择其他合法正收益动作或停止

#### Scenario: 失败调用仍被计费

- **WHEN** 已发起的 Retriever、数据库或 LLM 调用超时或失败
- **THEN** 系统 MUST 以实际用量或预留的保守估算扣减预算，不得通过失败重试绕过调用、Token 或时延上限

#### Scenario: 预算耗尽仍有缺失 facet

- **WHEN** 达到任一硬上限时仍存在未满足的必需 facet
- **THEN** 系统 MUST 以 `budget_exhausted` 停止，返回缺失 facet、已选 Evidence 和预算明细，不得自动放宽上限或触发 Web/回答生成

### Requirement: 结构化单 Agent 工作流与可靠停止

系统 MUST 以显式 `AgentState` 编排“分析需求、初始内部检索、覆盖评估、动作决策、纠错执行、重新评估、停止”阶段；状态 MUST 包含问题、facet、覆盖矩阵、充分性、候选/已选 Evidence、候选/已选动作、动作历史、检索轮次、已用/剩余预算、错误、Trace 和最终停止结果。所有条件边 MUST 只读取受校验的枚举、布尔值、计数和预算字段；循环 MUST 有配置上限且终止结果 MUST 明确区分充分、无收益、预算耗尽、内部知识缺失和执行失败。

#### Scenario: 纠错后覆盖提升并停止

- **WHEN** 首轮不足、一个合法动作补足全部必需 facet且仍在预算内
- **THEN** 工作流 MUST 合并去重的新 Evidence、重新计算完整覆盖矩阵、记录覆盖变化，并以 `sufficient` 停止且不再调用其他动作

#### Scenario: 重复状态触发停滞保护

- **WHEN** 纠错前后 facet 覆盖身份和有效 Evidence 集均未变化，或相同动作身份已执行
- **THEN** 工作流 MUST 将状态标记为停滞并以 `no_positive_gain` 停止，不得形成循环

#### Scenario: 工具异常与知识缺失可区分

- **WHEN** 合法动作因后端故障失败，或所有内部工具成功但语料中没有满足必需 facet 的证据
- **THEN** 工作流 MUST 分别输出 `execution_failed` 或 `internal_knowledge_missing`，保留有限错误与缺失 facet，不得把两者都归类为普通无命中

### Requirement: 可恢复缓存、检查点与结构化 Trace

系统 MUST 以问题、facet、Evidence 集、配置、Prompt、模型、语料和索引/图版本构造内容寻址身份，并为覆盖矩阵、动作决策和逐阶段状态生成 UTF-8 原子检查点。每次运行 MUST 使用唯一 Trace 标识，按稳定序号记录需求生成、路由、工具调用、覆盖变化、纠错决定、预算消费、异常和停止；Trace MUST 有界、脱敏，且不得包含密钥、认证信息、完整 Prompt、模型私有推理或未选中的无界正文。

#### Scenario: 同版本恢复不重复计费

- **WHEN** 同一运行身份已有通过校验的需求、覆盖或动作检查点
- **THEN** 系统 MUST 复用相应结果并从首个缺失阶段继续，缓存命中 MUST 不增加 Retriever 或 LLM 调用与 Token 用量

#### Scenario: 身份变化使缓存失效

- **WHEN** 问题、Evidence 内容、配置、Prompt、模型、语料或任一索引版本发生变化
- **THEN** 系统 MUST 形成新的缓存/运行身份并重新计算受影响阶段，不得静默复用旧覆盖或决策

#### Scenario: 检查点发布失败

- **WHEN** 状态或 Trace 在序列化、回读校验或原子替换期间失败
- **THEN** 系统 MUST 返回结构化持久化错误并清理临时产物，且 MUST NOT 留下可被识别为成功的半写检查点

### Requirement: 首次检索失败压力评测

系统 MUST 维护 20–40 题冻结开发压力集，覆盖术语错配、实体别名、比较缺侧、多跳断链、关键指标缺失、冲突证据和内部知识缺失，并绑定稳定问题/facet/目标 Evidence、语料快照、索引/图版本和预期停止类型。评测 MUST 比较固定 Hybrid、问题类型路由和 facet 缺口纠错，报告首次充分率、facet 诊断准确率、必需 facet 覆盖率、首次失败恢复率、纠错触发率、无意义动作率、平均轮次、工具/LLM 调用、Token、时延、失败数和停止原因；失败题 MUST 保留在逐题结果和指标分母中。

#### Scenario: 默认矩阵完全离线

- **WHEN** 运行默认压力集评测
- **THEN** 系统 MUST 使用 Mock、录制 fixture 或同版本缓存，不得访问网络、真实数据库或真实 LLM，并 MUST 对所有策略使用相同冻结输入和资源口径

#### Scenario: 在线验证显式且有界

- **WHEN** 调用方显式启用真实模型 smoke 或在线验证
- **THEN** 系统 MUST 先展示问题数与最坏预算并要求确认，最多运行 20 题，逐题继续遵守 4 次 LLM 和 20k Token 上限，且结果必须标记 Provider、模型、Prompt 与实际/估算用量

#### Scenario: 测试集和回答生成不参与调参

- **WHEN** 比较成本表、阈值或动作效用配置
- **THEN** 系统 MUST 只使用开发压力集和检索/覆盖指标，不得读取最终测试集、触发回答生成或以回答质量反向选择本迭代配置

