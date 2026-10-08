# Spec Delta

## Purpose

消除实时应用验收发现的跨段落伪冲突，使充分性判断只阻塞真正属于同一 facet、同一可比命题的相互矛盾 Evidence。

## MODIFIED Requirements

### Requirement: 结构化充分性、冲突与缺口诊断

系统 MUST 根据每个必需 facet 的有效匹配、最低支撑强度、独立来源数和阻断冲突生成结构化充分性结果。冲突 MUST 绑定同一 facet 下可比较的命题：系统 SHALL 优先使用 Evidence 中以相同 `normalized_value_key` 显式绑定的规范化值；只有指标或数值条件 facet 才可从文本中比较有单位或明确位置的指标值。普通正文中未绑定命题的年份、实体编号、实验数字、版本号或否定词 MUST NOT 单独构成阻断冲突。对于论文实体问题，系统 MUST 允许由显式实体命中锚定同论文的答案片段，并在有界数量内按论文上下文、重排、融合和检索排名保留多个匹配片段，而不得仅以实体词重合选择单个片段；论文上下文 MUST 来自本次运行锁定的索引快照，不得扫描可漂移的处理目录。结果 MUST 区分已覆盖、缺失、冲突和需要人工复核，并保存参与判断的 Evidence 标识与有限原因。

#### Scenario: 首轮证据已经充分

- **WHEN** 所有必需 facet 均满足最小证据和支撑阈值且不存在阻断冲突
- **THEN** 系统 MUST 将状态标记为充分、保留选定 Evidence，并直接进入成功停止，不生成或执行纠错动作

#### Scenario: 比较缺侧定位到具体 facet

- **WHEN** 检索结果只支持比较对象的一侧而另一侧没有合格来源
- **THEN** 系统 MUST 将缺失侧对应的必需 facet 标记为未覆盖，给出有限诊断依据，而不得把总体相关性判为充分

#### Scenario: 冲突证据阻止虚假充分

- **WHEN** 两条有效 Evidence 对同一必需 facet 给出不能同时成立的结果且没有配置允许的解决依据
- **THEN** 系统 MUST 记录冲突来源、降低该 facet 的满足状态并保持证据不足，而不得静默选择分数较高的一条

#### Scenario: 指标 Evidence 报告不同数值

- **WHEN** 同一指标 facet 的两条有效 Evidence 分别报告可比的 80% 与 90%
- **THEN** 系统记录阻断数值冲突、保留两侧 Evidence，并不得把该 facet 判为充分

#### Scenario: 组件正文包含无关数字和否定词

- **WHEN** 同一组件 facet 的相关 Chunk 分别包含实验倍率、年份、编号或 `without` 等与组件命题无关的文本
- **THEN** 系统不得据此生成数值或否定冲突；满足支撑阈值与来源要求时该 facet 仍可判为充分

#### Scenario: Evidence 提供显式规范化值

- **WHEN** 两条 Evidence 为同一 facet 和相同 `normalized_value_key` 提供不同的 `normalized_value`
- **THEN** 系统按布尔、数值或离散值类型记录可审计冲突，不依赖整段文本中其他无关词项

#### Scenario: 答案片段省略论文简称

- **WHEN** 一个高排名 Chunk 直接包含问题所需答案但省略论文简称，而同一检索候选集中该论文的其他 Chunk 已明确命中实体名
- **THEN** 系统 MUST 通过论文锚点保留答案 Chunk，并在证据上限内把高排名答案片段置于次要页面之前交给回答阶段

#### Scenario: 首个命中片段来自附录或实验页

- **WHEN** 初始 Hybrid Evidence 明确命中论文实体但没有承载问题所需定义、组件或机制
- **THEN** 实时应用 MUST 从当前冻结 Sparse 索引为首个锚定论文补入至多 3 个摘要优先片段、按稳定 Chunk 来源去重后重新评估，且不得为此调用额外 Embedding、LLM 或扫描当前处理目录
