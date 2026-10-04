# Design

## Context

动机见 [proposal.md](./proposal.md)，行为边界见 [spec.md](./specs/grounded-reflection-web-fallback/spec.md)。当前迭代 5 工作流以严格的 `CorrectionRunResult` 结束，能够给出 facet、内部覆盖矩阵、选定 Evidence、停止原因、预算账本和 Trace，但不生成最终 Agent 回答。现有 Dense/Hybrid 回答模型和解析器可处理论文引用，却没有稳定结论标识、facet 关联、回答评估或 Web URL 引用，并且 Dense 基线明确禁止反思和 Web。

`AgentState` 已预留草稿回答、回答评估、引用和最终回答字段，但这些字段仍是宽泛字典或字符串。本设计在不改变 Dense 基线和内部纠错行为的前提下，把回答阶段作为消费纠错结果的独立、可组合工作流实现。

## Goals / Non-Goals

**Goals:**

- 保持内部纠错、回答反思和 Web 兜底的模型、预算和检查点边界清晰，能够分别评测。
- 用严格模型使所有条件边只读取枚举、计数和布尔值，并使内外部引用都可解析。
- 默认路径完全离线；真实搜索和模型调用均通过可替换 Provider、显式开关和调用前预算门禁。
- 优先复用现有 Evidence、facet、覆盖矩阵、Trace、原子产物和 Provider 模式。

**Non-Goals:**

- 不把新回答流程塞回 Dense/Hybrid 基线服务，也不改变其单次生成语义。
- 不实现通用浏览器、页面抓取器、URL 任意访问或多 Provider 自动切换。
- 不以 Critic 分数替代确定性契约，不保存模型私有推理或完整网页正文。
- 不在本迭代统一全部历史回答结果模型；旧模型继续兼容原入口。

## Decisions

### 1. 使用独立的下游回答工作流组合内部纠错

新增回答工作流入口接收已经完成的 `CorrectionRunResult`，也可由一个薄编排入口先调用现有内部纠错再进入回答阶段。内部纠错图本身不增加 Web 或回答节点；回答工作流持有原 `RunIdentity`、内部停止结果和预算账本的校验副本，并在需要内部重新检索时调用现有白名单 Action Executor，而不是另建检索旁路。

建议模块边界：

- `models/answer.py`：回答、检查、反思、Web 结果和阶段预算契约。
- `answering/`：上下文构造、确定性检查、Critic、决策和保守回答。
- `web/`：可信来源策略、结果归一化和外部 Evidence 转换。
- `workflow/answer.py` 与 `workflow/nodes/answer.py`：状态编排和依赖注入。
- `evaluation/answer.py`：离线与显式在线评测。

选择独立工作流是为了保留迭代 5 的无回答评测和检查点身份，也便于在后续 API 层按请求关闭 Web。备选方案是扩展现有纠错图；该方案会把回答失败与内部检索失败混在同一停止语义中，并使现有压力集产生行为漂移，因此不采用。

### 2. 新增回答契约，不放宽旧 Dense 引用模型

保留 `AnswerClaim`、`Citation` 和 `DenseRAGResult` 的现有字段与验证，避免把允许 URL 且可无 `paper_id` 的外部引用倒灌到 Dense 基线。新增契约预计包含：

- `GroundedClaim`：`claim_id`、文本、结论类型、`citation_ids`、`facet_ids`。
- `GroundedCitation`：局部引用 ID、Evidence/来源 ID、来源类型、`external`、可选 Paper/章节/页码和可选 URL。
- `AnswerFinding`：检查代码、严重度、结论/引用/facet 标识和有限理由。
- `AnswerEvaluation`：六个检查维度、发现项、是否可接受和建议动作。
- `ReflectionDecision`：唯一动作、原因、目标 facet 和预算估算。
- `GroundedAnswerResult`：最终回答、缺失 facet、内外部 Evidence、评估、预算、停止原因和 Trace。

模型验证器按来源强制互斥溯源：内部 Chunk 引用必须有 Paper/位置且 URL 可空；Web 引用必须有 HTTPS URL、`external=true` 且不得伪造 Chunk/Paper 身份。结论 ID、引用 ID 和运行结果身份均由规范化内容及版本生成稳定哈希。

备选方案是直接把 `Citation.paper_id` 改为可空；这会弱化已归档 Dense 规格的保证并扩大回归面，因此不采用。

### 3. 结论支撑同时绑定引用和 facet 覆盖

生成上下文给每条选定 Evidence 分配稳定的运行内引用 ID，并附上该 Evidence 能支撑的 facet ID。模型只能从这些标识中选择，不能输出自由格式 URL 或来源。解析后按以下顺序验证：

1. Pydantic 结构、长度和唯一性。
2. 引用是否属于本次上下文，来源字段是否与 Evidence 一致。
3. 结论引用的 Evidence 是否在对应 facet 覆盖中有效。
4. 未覆盖必需 facet、阻断冲突和明显无引用数值。
5. 只有仍需语义判断的结论才进入 Critic。

内部 `CoverageMatrix` 保持原样。Web 结果使用单独的 `ExternalFacetCoverage` 集合记录对原缺失 facet 的支撑；最终结果并列保存内部评估和外部补充覆盖，绝不把原停止原因重写成内部充分。

### 4. Critic 只补充语义判断，规则拥有最终否决权

新增两个独立版本化 Prompt：`grounded-answer-v1.txt` 和 `answer-critic-v1.txt`。回答 Prompt 输出严格 JSON；Critic Prompt 只接收有界结论、其引用摘录、facet 与冲突摘要，并输出发现项列表，不要求也不保存推理过程。

确定性检查失败时不调用 Critic。Critic 输出必须再次经过严格解析、标识集合校验和大小限制；非法输出按失败处理。Critic 不能将确定性失败改为通过，也不能建议任意工具名。

重新生成时，把首次 Critic 已确认支持的结论—引用—facet 绑定作为允许集合，并要求删除或收缩失败结论；第二候选不得引入新事实绑定。这样只需一次 Critic，重生成结果可通过确定性集合包含关系校验完成收口。

备选方案是每次生成后再次调用 Critic；它更直观，但至少增加一次模型调用且仍可能形成评审循环，不符合迭代资源边界。

### 5. 用固定优先级决策表替代 LLM 路由

`ReflectionDecision` 由版本化规则生成，优先级如下：

1. 输入、来源、预算或持久化错误：保守终止。
2. 内部 `execution_failed`、`invalid_input` 或 `budget_exhausted`：保留原语义并终止，禁止 Web。
3. `internal_knowledge_missing`：若 Web 门禁全部通过则 `WEB_SEARCH`，否则保守终止。
4. 证据充分且候选回答通过：`ACCEPT`。
5. 证据充分但仅有可删除/收缩的回答问题：`REGENERATE`。
6. 有明确缺失 facet、原内部预算仍可原子预留且存在合法白名单动作：`RERETRIEVE`。
7. 其他情况：保守终止。

`REGENERATE`、`RERETRIEVE` 和 `WEB_SEARCH` 共用一个 `remediation_used` 门闩，任一动作执行后只能重新检查并接受或终止。规则决策记录全部拒绝原因，测试以表驱动方式覆盖。

备选方案是让 LLM 选择工具；它增加 Token、不可重复性和注入风险，且没有必要，因此不采用。

### 6. Search Provider 是最小、固定配置的外部边界

在 `providers/base.py` 增加异步 `SearchProvider` Protocol，输入为已规范化的单一查询和有界结果数，输出严格的 `SearchResult` 列表。实现包括：

- `MockSearchProvider`：单元测试与离线评测使用。
- `RecordedSearchProvider`：读取版本化 fixture，禁止网络。
- 一个显式启用的真实 JSON-over-HTTPS 适配器，初始选择固定端点的 Tavily Search；使用现有标准库/AnyIO，不引入搜索 SDK。请求禁用 SDK 隐藏重试，固定超时并只请求最多 5 条结果。

真实适配器的端点由 Provider 类型固定，不接受每题传入 base URL。业务层在任何正文进入 Evidence 前执行 HTTPS、IDNA/小写域名、默认端口、片段、来源类型和允许域名校验；若 Provider 返回最终 URL/重定向信息，则最终域名也必须通过同一策略。系统不主动抓取结果页面，因此不会产生额外页面请求、登录或付费墙访问。

选择 Tavily 是因为它提供结构化标题、URL、片段和结果标识，适合严格适配层；核心能力不依赖其字段，后续可新增其他适配器。直接调用任意 URL 或解析搜索 HTML 的方案存在 SSRF、版式漂移和请求失控风险，不采用。

### 7. 可信来源策略在配置加载时冻结

`grounded_answer.web` 配置包含 Provider、启用开关、最多 20 个规范化域名、允许来源类型、结果/摘录/上下文上限和超时。默认关闭 Web；默认域名只包含明确列出的学术站点，不使用宽泛后缀、正则或运行时自动扩展。项目主页必须由配置显式加入，不能仅因排名靠前而信任。

搜索查询只由原问题和缺失 facet 的规范化描述构造，长度有界且不拼接现有网页正文。外部 Evidence ID 绑定规范化 URL、摘录哈希、Provider 结果 ID 和转换版本；访问时间作为溯源字段但不参与内容身份，从而使同一录制结果可复现。

### 8. 使用组合预算而不是扩展内部 BudgetUsage

新增 `AnswerBudgetLedger` 保存阶段级 `limit/used/reserved`，维度包括回答生成、Critic、Web、反思轮次、Token、候选、上下文字符和时延；同时在最终摘要中嵌入只读的内部 `BudgetLedger`。内部重新检索仍由原账本预留和结算，因此无法重置 2 轮内部上限。

默认值直接采用 spec 硬边界或更低值：2 次生成、1 次 Critic、1 次 Web、1 次反思、12k 总 Token、16k 回答上下文、5 条 Web 结果和 8k 外部上下文。每个 Provider 适配器返回实际或保守估算用量；超时和解析失败按预留上限结算。阶段 LLM 总调用数通过生成与 Critic 计数之和验证不超过 3。

不直接给现有 `BudgetUsage` 增加字段，因为其严格 JSON 契约已用于迭代 5 检查点和评测；组合模型可以保持旧产物可读并清楚区分成本来源。

### 9. 每个外部副作用之后原子检查点

回答运行身份在原内部身份之上增加回答/检查 Prompt、规则、Web 策略、Search Provider 和模型修订哈希。阶段枚举预计为 `PREPARE`、`GENERATE`、`CHECK`、`DECIDE`、`REMEDIATE`、`FINALIZE`。每个模型、搜索或内部检索调用完成并结算后立即写入 UTF-8 临时目录，回读校验成功后原子替换正式检查点。

恢复逻辑只复用身份完全相同且序号连续的检查点。调用已发起但结果/结算未成功落盘时，恢复默认保守终止，不能猜测“未消费”并自动重放；用户可用新运行身份显式重试。Trace 仅记录哈希、数量、域名、状态、原因码和用量，不记录 Prompt、摘录正文或密钥。

### 10. 离线 fixture 是完成条件，真实 Web 是显式验收层

离线开发集使用约 12–20 个小型冻结案例覆盖所有决策分支和检查错误；其中 Web 响应保存为最小化 JSON fixture，不保存整页。单元测试注入 Mock LLM/Search/Retriever，断言默认网络调用为零、预算预留/结算、外部标记、引用/facet 回溯和恢复幂等。

评测报告逐题保留结果或错误，并汇总 spec 所列回答、引用、Web 和成本指标。真实 Web CLI 需要 `--online --confirm`、显式问题数 5–10 和已有凭据；执行前打印最坏调用/Token/上下文预算。成功运行以完整身份缓存，重复命令默认复用；失败运行只继续未完成题目。

### 11. 配置、兼容性和文档布局

在 `configs/default.yaml` 增加默认禁用的 `grounded_answer` 段，在 `configs/evaluation.yaml` 增加回答反思评测段；敏感值仍只由 `.env` 的现有 `KG_CRAG_WEB_SEARCH_API_KEY` 读取。Prompt、策略和配置均参与身份哈希。重要新路径和职责在实施时更新根目录 `PROJECT_MAP.md`，运行方法、真实服务边界和资源上限更新 `README.md` 及专题文档。

所有新字段通过新模型和新入口提供，现有 Dense/Hybrid/Corrective 配置、CLI、产物和测试无需迁移。

## Failure Strategy

| 失败位置 | 行为 | 是否允许自动重试 |
|---|---|---|
| 配置、Prompt 或输入契约 | 外部调用前失败，返回结构化错误 | 否 |
| 初始生成 | 结算预留成本；无合法候选则保守终止 | 否 |
| 确定性检查 | 生成结构化发现；可按决策表使用一次补救 | 仅规则允许的单次补救 |
| Critic | 失败不等于通过；保留确定性安全子集或终止 | 否 |
| 内部重新检索 | 保留内部失败语义和原预算 | 否 |
| Web 搜索/结果校验 | 明确记录禁用、失败、无可信结果或越界 | 否 |
| 检查点发布 | 清理临时目录并返回持久化错误 | 否 |

## Risks / Trade-offs

- [单次 Critic 可能漏掉重生成后的新语义错误] → 禁止重生成引入首次检查未确认的新结论绑定，并再次执行全部确定性检查。
- [域名白名单提高安全性但降低召回] → 把无可信结果视为正常知识边界，白名单只能通过配置变更和测试扩展。
- [搜索摘要可能截断上下文或二次转述] → 明确标记来源类型和外部属性，限制其只支撑可从摘录直接验证的结论，不推断全文事实。
- [组合预算增加状态复杂度] → 保持内部账本只读引用，统一预留/结算辅助函数并用不变量测试覆盖总调用数。
- [Tavily 可用性和 API 变化] → Provider Protocol、录制 fixture 和适配器契约隔离供应商；不自动切换供应商。
- [检查点可能包含受版权约束的外部摘录] → 只保存已选择且长度受限的摘录与哈希，不保存全文或未选择结果，并保持产物在 Git 忽略目录。

## Migration Plan

1. 先增加新模型、配置和 Prompt，默认 `enabled=false`，验证旧配置与旧产物仍可加载。
2. 实现纯函数检查、决策、预算和 Web 转换，再接入 Mock/录制 Provider，完成离线测试。
3. 新增独立回答工作流和检查点入口；保留现有 Dense/Hybrid/Corrective CLI 不变。
4. 增加离线评测、文档和项目地图，完成全量静态检查与默认离线测试。
5. 仅在凭据可用时运行一次 5–10 题显式真实 Web 验收；失败不阻塞默认离线能力，但必须记录为未验证或受控失败。

回滚时关闭或移除新回答入口和 `grounded_answer` 配置即可，原内部纠错与基线入口不依赖新工作流。已生成产物位于 Git 忽略目录，可保留用于审计，无需修改或删除旧索引与语料。
