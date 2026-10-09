# Grounded Reflection and Web Fallback Specification

## Purpose

为内部纠错后的科研问答提供逐结论可追溯的回答检查、单轮受控反思和可信 Web 兜底，使系统在有限资源内补充明确缺失的证据，同时始终区分内部与外部来源并诚实呈现剩余知识边界。

## Requirements

### Requirement: 逐结论可追溯的结构化回答

系统 MUST 只使用当前运行选定且通过校验的 `Evidence` 生成回答，并严格校验回答 Schema、结论类型、引用 ID、facet ID 和结论—Evidence 绑定。解析器 MAY 接受包裹单个 JSON 对象的单层 Markdown JSON 围栏，但 MUST 拒绝围栏外说明、多个对象、未知字段、未知引用或不受 Evidence 支撑的 facet 绑定。把“无可信结论”“证据不足”“现有 Evidence 未提供所需信息”或等价拒答绑定到已覆盖必需 facet 的候选 MUST NOT 被接受为事实回答；证据覆盖状态 MUST NOT 代替回答中的带引用结论覆盖必需 facet。系统 MAY 在完全相同的有界回答上下文内执行唯一一次重写，并允许把复合结论拆为原子结论、补出首稿遗漏的已支持 facet、合并既有结论或增加不绑定事实的 `uncertain` 边界说明；重写仍由未知引用、未知 facet 和逐结论 Evidence—facet 支持校验约束，MUST NOT 引入上下文外来源、新 facet 或无支持的事实绑定。Provider 请求与响应校验失败 MUST 分类为有限、安全的失败类别，且不得把原始模型响应写入公共结果或 Trace。

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
- **THEN** 确定性检查 MUST 将其标记为 `abstention` 且不得接受；若预算允许，系统只能在当前有界上下文已有且受支持的 Evidence—facet 范围内重写一次

#### Scenario: 重写拆分复合结论

- **WHEN** 首稿把多个已支持 facet 写入同一条结论，重写将其拆成分别引用原 Evidence 的多个原子结论
- **THEN** 系统 MUST 允许这种范围内重组并继续执行逐条 Evidence—facet 校验，不得仅因引用集合与 facet 集合不再完全相同而报 `new_fact_binding`

#### Scenario: 补证后把来源元数据误当作不可回答

- **WHEN** Web 已返回并绑定能够支持必需 facet 的正文，但首次回答声称“Evidence 未提供或未包含所需信息”
- **THEN** 确定性检查 MUST 将其识别为拒答；若仍有第二次生成预算，系统 MAY 在相同有界 Evidence—facet 范围内重写一次，且不得再次搜索、检索或扩大上下文

#### Scenario: 外部证据存在但回答遗漏事实

- **WHEN** Web Evidence 已覆盖必需 facet，但候选只输出元数据、拒答或没有形成绑定该 facet 的事实结论
- **THEN** 确定性检查 MUST 将候选判为不完整，不得仅凭外部覆盖矩阵将其接受

### Requirement: 确定性优先的回答检查

系统 MUST 在任何语义 Critic 之前确定性检查回答结构、长度、结论与引用标识、引用存在性、来源位置、内外部标记、结论—facet 映射、明显无引用数值以及与已知阻断冲突的一致性。只有通过确定性检查且仍需判断语义支撑或归因的候选回答才能提交给版本化 Critic；Critic MUST 只接收问题、结论、对应的有限 Evidence 摘录、facet 和冲突摘要，返回严格的结构化结果，且单次运行最多调用一次。

#### Scenario: 确定性错误不消耗 Critic 调用

- **WHEN** 候选回答含未知引用、重复标识、越界内容或明显无引用数值
- **THEN** 系统 MUST 直接给出结构化失败项，Critic 调用次数 MUST 为零

#### Scenario: Critic 识别语义不支撑或错误归因

- **WHEN** 结论具有合法引用，但引用摘录不支持该结论或结论把结果归因给错误论文、方法或实验对象
- **THEN** Critic MUST 将对应结论标记为不支撑或错误归因，并返回有限理由和相关结论、引用及 facet 标识

#### Scenario: Critic 失败时不放行未验证回答

- **WHEN** Critic 超时、Provider 失败、输出不可解析或引用未知标识
- **THEN** 系统 MUST 记录一次失败并保守停止或使用已通过的确定性结论范围，MUST NOT 隐藏重试、切换 Provider 或把 Critic 失败解释为检查通过

### Requirement: 结构化评估与单轮反思决策

系统 MUST 将完整性、忠实度、引用支撑、证据冲突、错误归因和必需 facet 覆盖汇总为结构化 `AnswerEvaluation`，并从“接受、重新生成、预算内重新检索、Web 兜底、保守终止”中产生唯一决策。硬性引用或来源错误 MUST 优先于模型评分；未覆盖必需 facet MUST 阻止“接受”。同一结构化状态、配置、Prompt 和 Provider 版本 MUST 产生相同的规则决策，且反思补救总计最多执行一轮。

#### Scenario: 检查通过后直接接受

- **WHEN** 候选回答通过全部硬性检查、没有不支撑关键结论，并覆盖所有必需 facet
- **THEN** 系统 MUST 接受该回答并停止，不再调用 Critic、检索器、搜索或生成模型

#### Scenario: 可修复表达问题只重新生成一次

- **WHEN** Evidence 和 facet 覆盖充分，但回答存在可通过删除、收缩或重写结论修复的支撑问题
- **THEN** 系统 MUST 最多重新生成一次，将失败项作为有限约束传入，并对新候选重新执行确定性检查而不得开始第二轮反思

#### Scenario: 证据缺口不能靠重新生成修复

- **WHEN** 回答失败源于必需 facet 未覆盖或阻断冲突未解决
- **THEN** 系统 MUST 选择仍在共享预算内的唯一合法检索分支，或保守终止，MUST NOT 仅通过改写 Prompt 宣称问题已解决

### Requirement: 严格门控的 Web 搜索

系统 MUST 默认禁用 Web。Web 只有在调用方显式允许、内部工作流以 `internal_knowledge_missing` 结束、内部检索预算已经按原身份结算、存在明确缺失的必需 facet、可信来源策略有效且 Web 预算充足时才能触发。搜索查询 MUST 由原问题和缺失 facet 确定性构造；单题最多执行一次查询和一次 Provider 调用，不得因无结果、超时或限流而自动重试或切换 Provider。

#### Scenario: 内部证据充分时禁止 Web

- **WHEN** 内部工作流已覆盖全部必需 facet，或回答问题仅是可重新生成的表达错误
- **THEN** 系统 MUST NOT 调用 Search Provider，即使 Web 已启用且尚有预算

#### Scenario: 内部知识缺失时执行一次受控搜索

- **WHEN** 内部工作流以 `internal_knowledge_missing` 结束、调用方允许 Web，且查询、可信来源和预算校验全部通过
- **THEN** 系统 MUST 只针对缺失 facet 发起一次有界搜索，并记录触发原因、查询身份、Provider、调用次数和预算消费

#### Scenario: 非知识缺失停止原因不能触发 Web

- **WHEN** 内部工作流因执行失败、配置错误、预算预留失败或状态损坏而停止
- **THEN** 系统 MUST 保留原停止语义并禁止 Web，MUST NOT 用外部搜索掩盖内部故障

### Requirement: 可信来源策略与有界结果选择

系统 MUST 只接受配置化允许的 HTTPS 来源类型和规范化域名，默认可信类型 MUST 限于论文官网、arXiv、开放学术数据库和项目主页。允许域名数量 MUST 不超过 20，单次搜索候选 MUST 不超过 5，系统 MUST 在获取或转换正文前校验 URL、重定向后的最终域名、来源类型和结果大小；不得访问登录页、付费墙、任意用户 URL 或非白名单域名。

#### Scenario: 可信学术来源进入候选集

- **WHEN** 搜索结果具有合法 HTTPS URL、最终域名在允许列表、来源类型受支持且字段完整
- **THEN** 系统 MUST 按来源优先级、相关性和稳定次级键选择不超过 5 条候选

#### Scenario: 不可信或重定向越界结果被拒绝

- **WHEN** 结果使用非 HTTPS URL、域名不在允许列表、重定向离开允许域名、需要认证或声明的来源类型不匹配
- **THEN** 系统 MUST 在把内容交给模型前拒绝该结果、记录有限原因，且不得自动扩展白名单

### Requirement: 外部结果转换为独立可追溯 Evidence

系统 MUST 将通过可信策略的 Web 结果转换为统一 `Evidence`，其来源类型 MUST 为 Web、`external` MUST 为 true，并至少保留稳定 Evidence 标识、页面标题、规范化 URL、访问时间、有限摘录、来源类型、内容哈希和 Provider 结果标识。每条摘录 MUST 不超过 2,000 字符，外部上下文合计 MUST 不超过 8,000 字符。外部 Evidence MUST 与内部 Evidence 分开评估和计数，不得修改既有内部覆盖结论、伪造 Paper/Chunk 标识或覆盖同名内部来源。

#### Scenario: 合法结果生成稳定外部 Evidence

- **WHEN** 同一规范化 URL、摘录内容、Provider 结果和转换版本被再次处理
- **THEN** 系统 MUST 产生相同 Evidence 身份、显式外部标记和相同溯源字段，并可由引用解析回该 URL 和摘录

#### Scenario: 字段缺失或内容越界时拒绝转换

- **WHEN** Web 结果缺少标题、URL、访问时间、摘录或 Provider 身份，或内容超过上限且无法安全截断
- **THEN** 系统 MUST 拒绝该结果并不得生成部分、内部或无来源 Evidence

#### Scenario: 外部证据不能改写内部状态

- **WHEN** 外部 Evidence 覆盖了原先缺失的 facet
- **THEN** 系统 MUST 单独记录外部覆盖和最终混合证据范围，原内部充分性与停止原因 MUST 保持可审计且不被改写为内部充分

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

#### Scenario: 补证后的首次结构化输出非法

- **WHEN** Web 或内部补检已经完成，但首次回答为非法 JSON、非法 Schema、未知引用、未知 facet 或不受支持的 facet 绑定，且第二次生成预算仍可预留
- **THEN** 系统 MAY 以更严格的结构约束重写一次并记录重试类别；该重写不重复搜索或检索，累计生成次数不得超过 2，第二次仍失败时必须保守停止

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

#### Scenario: Web 返回目标论文的英文摘要

- **WHEN** 中文复合问题的可信 Web 结果以英文标题或摘要明确命中可区分的目标项目名，但单条摘要没有重复问题中的全部子组件名或中文术语
- **THEN** 系统 MUST 允许以目标主实体建立待检查的外部 facet 绑定，将标题和摘录一并交给回答与 Critic；普通分类词、相似项目名或不满足短缩写消歧的结果不得获得该绑定

#### Scenario: 学术主来源与项目页同时命中

- **WHEN** 同一次 Web 搜索中，arXiv、论文站或学术数据库与 GitHub 等项目页都被判定可覆盖同一 facet
- **THEN** 系统 MUST 只让学术主来源获得该 facet 的回答绑定，并从生成上下文排除未绑定的次要结果；只有没有匹配主来源时项目页才可作为受检查的兜底证据

#### Scenario: 搜索摘要只有页面元数据

- **WHEN** 可信搜索结果命中目标论文，但普通搜索摘要只包含标题或页面元数据
- **THEN** 单次受控搜索请求 MUST 优先携带目标实体和 facet 证据词，并 MAY 请求同一结果的有界纯文本正文；转换器只截取目标实体附近正文和相关搜索片段，仍受单条与总外部上下文上限约束

#### Scenario: Critic 识别内部来源不支持结论

- **WHEN** 内部充分性曾被判为通过，但语义 Critic 指出候选存在不支持、来源错配、facet 错配或错误归因，且请求已允许可用 Web
- **THEN** 系统 MUST 把受影响的候选 facet 恢复为证据缺口，并 MAY 将唯一补救轮次用于受控 Web，而不得直接把同一错误内部证据重新表述后接受
- **AND** Critic 的有界上下文 MUST 优先保留候选实际引用的 Evidence 及原引用编号，不得因截取未引用的前序结果而遗漏被检查来源
- **AND** 若首轮 Critic 已消费唯一 Critic 预算，Web 后 MUST 执行完整确定性引用与 facet 检查，不得再次调用 Critic 并伪装成预算内操作

#### Scenario: Critic 使用单层 JSON 围栏

- **WHEN** Critic 只用单层 `json` Markdown 围栏包裹一个满足 Schema 与 ID 约束的 findings 对象
- **THEN** 系统移除围栏后执行完整校验；围栏外说明、未知 ID、未知 code 或非法字段仍必须失败

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

### Requirement: 默认离线且资源有界的回答与 Web 评测

系统 MUST 使用冻结的离线 fixture 覆盖充分回答、无引用数值、错误归因、不支撑结论、冲突、内部知识缺失、Web 禁用/失败/不可信来源和预算耗尽。评测 MUST 报告正确性或任务得分、完整性、忠实度、引用准确率、引用召回率、无依据结论比例、Web 触发率、Web 使用率、搜索/模型调用数、Token、时延、失败数和停止原因；失败题 MUST 保留在逐题结果和指标分母中。默认测试和评测 MUST 使用 Mock 或录制响应且网络调用数为零。

#### Scenario: 默认离线评测不访问外部服务

- **WHEN** 开发者运行默认测试或回答反思评测
- **THEN** 系统 MUST 使用确定性 LLM/Search 替身或录制 fixture，覆盖正常、失败和上限场景，且不得访问真实模型、搜索服务或数据库

#### Scenario: 显式真实 Web 验收严格限量

- **WHEN** 调用方显式选择真实 Web 验收
- **THEN** 系统 MUST 在执行前展示 5–10 个知识缺失问题、Provider、可信域名和最坏调用/Token/上下文预算并要求确认，只运行该批问题且逐题遵守全部硬上限

#### Scenario: 相同成功配置不重复在线运行

- **WHEN** 相同问题集、配置、Prompt、Provider、模型和数据版本已有通过校验的成功在线结果
- **THEN** 系统 MUST 复用该结果或要求显式创建新实验身份，不得默认重复真实搜索与模型调用；失败恢复只能继续未完成题目
