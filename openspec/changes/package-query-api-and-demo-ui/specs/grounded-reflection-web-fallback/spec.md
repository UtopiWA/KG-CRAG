# Spec Delta

## Purpose

让真实模型的结构化输出失败可诊断、资源用量可解释，并确保回答阶段内部补救后的最终状态可以被应用层无损展示。

## MODIFIED Requirements

### Requirement: 逐结论可追溯的结构化回答

系统 MUST 只使用当前运行选定且通过校验的 `Evidence` 生成回答，并严格校验回答 Schema、结论类型、引用 ID、facet ID 和结论—Evidence 绑定。解析器 MAY 接受包裹单个 JSON 对象的单层 Markdown JSON 围栏，但 MUST 拒绝围栏外说明、多个对象、未知字段、未知引用或不受 Evidence 支撑的 facet 绑定。把“无可信结论”“证据不足”或等价拒答绑定到已覆盖必需 facet 的候选 MUST NOT 被接受为事实回答；系统 MAY 在相同引用和 facet 绑定内执行唯一一次重写。Provider 请求与响应校验失败 MUST 分类为有限、安全的失败类别，且不得把原始模型响应写入公共结果或 Trace。

#### Scenario: 内部证据充分时生成可追溯回答

- **WHEN** 内部纠错结果为充分且全部必需 facet 均有有效覆盖
- **THEN** 系统 MUST 仅基于选定内部 Evidence 生成逐结论引用的回答，`used_external` MUST 为 false，且每个关键结论 MUST 可追溯到相应 facet 覆盖项和原始 Chunk 或图事实

#### Scenario: 未覆盖 facet 不得伪装成完整结论

- **WHEN** 最终证据仍未覆盖一个或多个必需 facet
- **THEN** 系统 MUST 在最终结果中列出缺失 facet，并拒绝将其表述为已证实结论；若输出部分回答，MUST 明确限定其证据范围和不确定性

#### Scenario: 未知引用使候选回答失效

- **WHEN** 候选回答引用当前有界上下文之外的 Evidence、缺失来源位置，或非不确定结论没有引用
- **THEN** 系统 MUST 拒绝该候选回答且不得将其作为最终结果发布

#### Scenario: 模型使用 JSON 代码围栏

- **WHEN** 模型只用单层 `json` Markdown 围栏包裹一个满足 Schema 和绑定约束的对象
- **THEN** 系统移除围栏后执行完整校验，并可接受通过校验的回答

#### Scenario: 结构化回答包含未知引用

- **WHEN** JSON 可解析但结论引用当前上下文不存在的 Evidence 编号
- **THEN** 系统以 `unknown_citation` 类别拒绝回答并保守停止，Trace 不包含原始响应正文

#### Scenario: 拒答文本伪装成已覆盖结论

- **WHEN** 候选把“暂无可信结论”或等价拒答绑定到一个已覆盖的必需 facet
- **THEN** 确定性检查 MUST 将其标记为 `abstention` 且不得接受；若预算允许，系统只能在原有引用和 facet 绑定内重写一次

### Requirement: 共享预算下的可靠回答工作流

系统 MUST 在调用前预留最坏预算，并在 Provider 返回逐请求 usage 时优先记录可安全结算的实际输入/输出 Token；Provider 未返回 usage、请求失败或实际值超过既有预留边界时 MUST 使用有界保守估算。结果与公共预算摘要 MUST 明确区分 actual、estimated 和 mixed，不得把固定预留值无说明地表示为实际消耗。阶段输入输出总上限保持 12,000 Token 时，配置 MUST 为允许的两次生成分别保留完整单次输出额度。二次内部检索仍未补足必需 facet 时 MUST 直接进入保守停止，不得再执行必然无法通过完整性检查的回答生成。

#### Scenario: 任一维度不足时调用前停止

- **WHEN** 下一次生成、Critic、内部检索或 Web 调用会超过任一共享或阶段硬上限
- **THEN** 系统 MUST 在外部调用前拒绝该动作，并生成带缺失 facet、有效部分结论和预算摘要的保守结果

#### Scenario: 一次补救后强制终止

- **WHEN** 系统已经执行一次重新生成、一次预算内重新检索或一次 Web 兜底中的任一补救
- **THEN** 系统 MUST 在重新检查后接受合法结果或保守终止，不得选择第二个补救动作或形成循环

#### Scenario: 失败调用不能绕过预算

- **WHEN** 已发起的生成、Critic、检索或搜索调用超时或失败
- **THEN** 系统 MUST 记录实际用量或预留的保守估算并保留失败语义；超时不得重试，其他明确标记为 retryable 的回答 Provider 故障最多允许一次计入预算且写入 Trace 的重试，不得隐藏重试或通过恢复重复消费同一调用配额

#### Scenario: 推理内容耗尽输出额度

- **WHEN** 兼容 Provider 返回空正文、`finish_reason=length` 和非零推理内容长度
- **THEN** 系统 MUST 记录安全的结束原因、推理字符数和保守用量，将其分类为不可重试的 Provider 失败，且不得保存或公开私有推理正文

#### Scenario: Provider 返回实际 usage

- **WHEN** 一次回答调用成功返回合法的 prompt/completion Token 计数且不超过预留
- **THEN** 预算按该次实际值结算，Trace 和公共摘要将其标记为 actual

#### Scenario: Provider 请求未返回 usage

- **WHEN** 请求失败或兼容 Provider 没有返回合法 usage
- **THEN** 系统按预留或字符估算保守结算，并在 Trace 和公共摘要中标记 estimated

#### Scenario: 二次检索仍未覆盖必需 facet

- **WHEN** 回答阶段完成唯一一次内部二次检索后充分性结果仍为不足
- **THEN** 系统保留最终覆盖、冲突和动作历史并直接保守停止，回答模型调用次数不增加

### Requirement: 保守停止、检查点与脱敏 Trace

最终结果 MUST 保存回答终止时的 Evidence requirement、facet assessment、阻断冲突和内部动作历史。Trace MUST 以有限类别区分 Provider 请求、JSON、Schema、引用和绑定失败，并记录本次用量是否估算；Provider 请求失败还 MUST 在可获得时记录安全的 retryable、HTTP 状态码和异常类型，不得保存完整 Prompt、原始 Provider 响应、凭据或私有推理。回答生成失败的最终停止类型 MUST 为 `generation_failed`，内部详情不得被泛化为无从定位的 `remediation_failed`。

#### Scenario: Web 禁用或失败时明确知识边界

- **WHEN** 内部知识缺失但 Web 被禁用、无可信结果、超时或 Provider 失败
- **THEN** 系统 MUST 返回可验证的保守结果，列出已支持结论、缺失 facet、Web 状态和有限错误，不得伪造答案或隐藏为普通无命中

#### Scenario: 有效检查点恢复不重复调用

- **WHEN** 同一运行身份已有通过哈希和模式校验的生成、Critic 或 Web 检查点
- **THEN** 系统 MUST 复用已完成阶段并从首个缺失阶段继续，对应 Provider 调用和 Token 计数不得再次增加

#### Scenario: 检查点不兼容或发布失败

- **WHEN** 输入、配置、Prompt、Provider、模型或数据版本变化，或检查点序列化、回读校验和原子替换失败
- **THEN** 系统 MUST 拒绝不兼容缓存或返回结构化持久化错误，清理临时产物且不得留下可识别为成功的半写结果

#### Scenario: 回答生成无法解析

- **WHEN** Provider 已返回内容但该内容不是合法 JSON
- **THEN** 最终停止类型为 `generation_failed`，Trace 记录 `invalid_json` 和用量来源，公共结果不包含原始内容

#### Scenario: 回答阶段二次检索更新状态

- **WHEN** 回答反思执行内部二次检索并更新覆盖矩阵、冲突、选定 Evidence 或动作历史
- **THEN** 最终结果保存更新后的状态，使 API 和 UI 能展示与停止原因一致的 facet 和动作
