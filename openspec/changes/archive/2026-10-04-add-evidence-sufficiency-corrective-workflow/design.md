# Design

## Context

动机和范围见 `proposal.md`，行为契约见 `specs/evidence-sufficiency-corrective-workflow/spec.md`。当前系统已有稳定 `Evidence`、Dense/Sparse/Hybrid/Graph 工具、统一错误与 `TraceEvent v1`、LLM Provider/Mock、版本化配置和多套离线评测模式；`workflow/state.py` 仍是早期 `TypedDict` 骨架，节点与条件边尚未实现。现有 `RouteDecision` 和 `RetrievalEvaluation` 只表达问题类型路由和总体评分，不能表示 facet 覆盖、动作收益、多维预算或可靠停止。

宿主机资源边界仍按 16 GiB 内存、4 GiB 显存和本地单机开发考虑。默认测试不得加载真实模型、连接 Qdrant/Neo4j 或访问网络；单题总 LLM 预算上限为 4 次、20k Token，内部检索总轮次为 2，在线验证最多 20 题。本 change 位于 Web、回答反思、API/UI 之前，因此终点是“可供后续回答阶段消费的证据与停止诊断”，不是最终自然语言回答。

## Goals / Non-Goals

**Goals:**

- 用严格公共契约冻结 facet、覆盖、充分性、动作、成本、预算、状态、停止结果和评测记录。
- 让规则基线在零 LLM 下完整运行，并把可选模型调用限制在结构化边界和同一总预算内。
- 复用现有独立 Retriever/Graph 能力，通过适配器统一动作输入输出，不修改其后端语义。
- 让动作选择、条件边、循环和停止只依赖结构化状态，并能从 Trace 解释覆盖增益与成本。
- 以离线压力集证明“首次相关但不充分”能够被诊断、恢复或可靠停止。

**Non-Goals:**

- 不实现 Web 检索、回答生成、引用编排、回答后反思、HTTP API 或 UI。
- 不训练分类器、强化学习策略或端到端路由模型，不让 LLM 输出直接控制节点名或工具调用。
- 不修改 Dense/Hybrid/Graph 的索引、评分、查询模板或独立评测契约。
- 不把开发压力集扩大为完整最终测试集，不用最终测试结果调阈值或动作成本。

## Decisions

### 1. 新公共模型集中在 correction 领域，旧路由模型保留兼容

新增 `models/correction.py`，定义 `EvidenceRequirement`、满足条件联合类型、`FacetCoverage`、`CoverageMatrix`、`SufficiencyAssessment`、`CorrectionAction`/`ActionEstimate`/`CorrectionDecision`、`BudgetLimit`/`BudgetUsage`/`BudgetLedger`、`StopReason`、`CorrectionRunResult` 和评测模型。所有模型继承严格公共基类、拒绝未知字段并进入 Schema 快照。

`workflow/state.py` 的 `AgentState` 扩展为 LangGraph 使用的显式 TypedDict，但以严格 `CorrectionState` 校验模型作为节点输入输出和检查点边界；构建器负责一次性初始化全部字段。现有 `RouteDecision`、`RetrievalEvaluation` 及字段继续导出，提供只读迁移适配器，不让新条件边解析其中的 `reason`、`missing_aspects` 等自由文本。

备选方案是直接向 `domain.py` 继续增加字段。它会让已稳定的论文/Evidence 模型与实验性策略强耦合，也不利于后续 Web/反思扩展，因此不采用。

### 2. Facet 由规则优先生成，可选 LLM 只提出严格候选

`correction/requirements.py` 先以版本化规则识别对象、比较维度、关系、指标和多跳约束，生成稳定 facet ID；简单问题至少生成一个内容 facet，比较问题按对象×维度生成必需项。只有规则标记 `complex` 或置信度低于配置阈值时，才调用独立 `RequirementProvider`。生产适配器复用通用 `LLMProvider`，Prompt 版本化并只接受严格 JSON；模型不能生成问题 ID、预算、版本或工具名，这些由代码注入。

Provider 没有可信 usage 时使用固定字符估算器保守计费。模型响应整批校验：facet 数、长度、枚举、重复、满足条件和与原问题的词项边界任一非法即整批拒绝，回退到规则结果；不隐藏重试或切换 Provider。

备选方案是所有问题都调用 LLM 分解。它会让默认测试、成本和复现依赖在线模型，并在简单问题上没有收益，因此不采用。

### 3. 覆盖矩阵使用纯函数规则，低置信度模型判断只能补充说明

`correction/coverage.py` 以规范化词项、实体/论文 ID、Evidence 类型、Graph 路径元数据、章节/页码、分数和内容哈希计算覆盖项。规则按 facet 类型选择匹配器，通过 `FacetMatcher` Protocol 隔离；结果按 facet ID、Evidence ID 稳定排序。Coverage ID 绑定问题、facet、Evidence 内容身份、规则版本和阈值。

支撑强度由匹配质量、Retriever 分数和来源质量的版本化公式得到，不跨来源臆造缺失分数。冲突检测先覆盖同一 facet 的离散值、数值/单位和显式否定；无法可靠归一化的内容只标记待复核。可选结构化模型评估只能在规则低置信度时补充 `match/conflict` 候选，仍需引用现有 Evidence ID 和有限证据片段，不能成为条件边的自由文本输入。

备选方案是用单次 LLM 对拼接上下文给总体“充分/不足”标签。它无法下钻到 facet，输入顺序敏感且难以做压力集归因，因此不采用。

### 4. 充分性由逐 facet 门槛聚合，不使用单一平均分

`correction/sufficiency.py` 对每个 facet 检查最小独立来源数、最小支撑强度、允许来源类型和阻断冲突，再聚合为 `SufficiencyAssessment`。所有必需 facet 通过才可 `sufficient=true`；可选 facet 只影响诊断，不阻止停止。总体置信度是展示与排序字段，不能覆盖必需项失败。

选定 Evidence 使用集合覆盖式稳定选择：优先覆盖更多尚未满足的必需 facet，再比较最弱支撑、来源质量、成本和 Evidence ID；上限来自配置。来源解析器复核内部 Chunk/Graph 元数据，`external=true` 在本 change 中被排除并记录。

备选方案是沿用 `RetrievalEvaluationScores.coverage` 阈值。该标量不能表达比较缺侧和冲突，保留它只作兼容输出而不作新工作流条件。

### 5. 动作目录与执行器分离，所有工具通过适配器归一

`correction/actions.py` 保存静态、版本化动作描述和适用缺口；`correction/executor.py` 通过 `CorrectionTool` Protocol 执行。Dense、Sparse、Hybrid 和 Graph 分别使用已有服务的薄适配器，统一接受 `ActionRequest` 并返回 `ActionResult(Evidence, cost, diagnostics)`。查询重写、问题分解和参数调整是复合动作：先由确定性规则或受预算的结构化 Provider 生成最多三个查询，再调用目录中固定的内部 Retriever；它们不能动态选择未注册工具。

目录不注册 Web、生成或任意 Cypher。Graph 适配器只能构造现有 `GraphQueryRequest` 白名单模板；Hybrid 适配器保留原有降级标记。动作结果在进入状态前复核 Evidence、按稳定 ID 去重并截断，后端对象不能进入状态。

备选方案是让路由 LLM返回函数名和参数。它扩大注入面、使预算预留不可靠，也违反结构化条件边约束，因此不采用。

### 6. 决策采用版本化净效用公式和稳定次级键

`correction/policy.py` 为每个候选计算：

`utility = expected_required_facet_gain - retrieval_cost - model_cost - token_cost - latency_cost - repeat_penalty`

各权重、动作基础成本、历史成功率先验、最低正收益和适用规则来自严格配置并进入配置哈希。预期增益只针对当前缺失必需 facet，初版使用可解释规则与开发集统计，不在线学习。候选先经过白名单和最坏预算过滤，再按 utility、预期增益、总成本、动作 ID 稳定排序。utility 不为正或状态指纹未改善时停止。

问题类型路由作为评测基线单独实现，不参与核心策略的缺口推断。可选 LLM 只能给出结构化候选排序建议；最终仍由同一验证器和净效用门槛决定。

备选方案是按固定问题类型直接选择 Retriever。它保留为对照，但不能根据首次证据的实际缺口调整动作，因此不作为主策略。

### 7. 预算先预留、后结算，失败调用也消费预算

`BudgetLedger` 同时维护总上限、已用、预留和剩余值。每个节点调用前以 `ActionEstimate` 原子预留检索轮次、子问题、模型调用、输入/输出 Token、候选、上下文字符和时延；任一维度不足则整项拒绝。完成后以真实 usage 结算；Provider/后端没有 usage 或调用失败时保留预留估算。缓存命中不预留外部调用，只记录缓存命中事件。

硬上限固定为总内部检索轮次 2、子问题 3、LLM 调用 4、输入输出合计 20,000 Token；配置再收紧单动作候选、累计唯一 Evidence、选定 Evidence、上下文字符和墙钟时延，不能超过模型定义的安全上限。时延测试注入单调时钟，避免真实 sleep。

备选方案是操作完成后才统计成本。失败、超时或并发请求可能先超支，无法满足硬预算，因此不采用。

### 8. 使用固定 LangGraph 状态图，但节点和边保持框架无关纯函数

增加经 Python 3.11 验证并固定版本的 LangGraph 依赖，以 `StateGraph` 装配以下固定流程：

```text
requirements -> initial_retrieve -> assess -> decide
                                      │ sufficient/stop -> finalize
                                      └ execute -> reassess -> decide/finalize
```

节点实现放在 `workflow/nodes/`，只接收校验后的状态和显式依赖；条件边放在 `workflow/edges/`，只返回固定枚举节点名。工作流编译和依赖装配放在 `workflow/corrective.py`。初始检索策略通过配置选择固定 Hybrid 或问题类型路由，以便同一 runner 支持基线；核心 facet 策略只改变纠错决策。

执行器串行运行，一次只执行一个动作，简化预算原子性和可重放性。每轮合并 Evidence 后必须完整重算矩阵；状态指纹绑定有效 Evidence 集、覆盖和动作历史，连续不变立即 `no_positive_gain`。终止枚举固定为 `sufficient`、`no_positive_gain`、`budget_exhausted`、`internal_knowledge_missing`、`execution_failed` 和 `invalid_input`。

备选方案是自行实现 while 循环。虽然依赖更少，但项目既定架构需要可视化状态图和固定边；LangGraph 只承担编排，领域逻辑仍可脱离框架单测，资源增量可控。

### 9. 检查点、Trace 与缓存共享一个运行身份

运行身份绑定规范化问题、facet/coverage/policy 版本、配置哈希、Prompt/模型修订、语料快照及 Dense/Sparse/Graph 版本。缓存分别保存严格的需求响应、覆盖矩阵和动作决策；检查点保存阶段、完整可序列化状态、序号和身份。写入沿用项目原子临时文件—回读校验—替换模式，路径限制在工作区内。

Trace 继续使用 `TraceEvent v1`，details 只保存 ID、枚举、计数、分数、预算标量和有限错误；完整 Prompt、未选正文、模型响应和私有推理不进入 Trace。运行产物放在 `data/processed/corrective-runs/` 与 `data/evaluation/results/corrective-workflow/`，保持 Git 忽略。

备选方案是依赖 LangGraph 默认内存 checkpoint。它不能跨进程恢复或绑定索引版本，也不满足原子产物审计，因此仅用于单元测试，不作为产品检查点。

### 10. 压力评测优先离线录制，在线只做最多 20 题 smoke

新增 20–40 题 `corrective_dev_questions.json`，每题冻结初始 Evidence fixture、facet 真值、缺口类型、允许目标 Evidence/动作和期望停止原因。默认矩阵用相同 fixture 比较 fixed Hybrid、type router、facet corrective 三个策略，不调用回答生成；指标和逐题失败使用现有评测的原子报告模式。

真实后端/模型只通过独立环境开关启用，先 dry-run 展示最坏调用与 Token，再运行不超过 3 题 smoke；smoke 成功且预算记录可信后才允许扩大到最多 20 题。线上结果与离线指标分开标记，缓存命中、估算 usage 和真实 usage 不混为一类。

备选方案是直接在线跑完整 40 题。它对模型 Token、数据库和时间开销不必要，也会把外部波动混入策略验收，因此不采用。

## Risks / Trade-offs

- **[规则 facet 漏掉隐含需求]** → 以比较/多跳压力 fixture 扩充规则，低置信度才调用结构化 Provider，并把未解析需求显式暴露而非假装充分。
- **[词法覆盖误判语义支撑]** → 按 facet 类型使用不同匹配器、最低来源与支撑阈值；将边界样本保留在逐题诊断中，后续可替换匹配器而不改公共契约。
- **[成本先验与真实时延漂移]** → 同时保存估计和实际成本，版本化更新开发集统计；决策始终受硬预算而不是成本预测单独约束。
- **[不同 Retriever 的调用接口不一致]** → 仅在薄适配器中转换，严格校验统一 `ActionResult`，不把后端类型泄漏到状态。
- **[LangGraph 增加依赖和调试复杂度]** → 固定兼容版本，节点/边保持纯函数并提供无框架单测；编译图只在运行入口发生。
- **[缓存复用陈旧判断]** → 身份绑定问题、Evidence 内容、配置、Prompt、模型、语料和所有索引版本；缺失任一版本时拒绝持久缓存。
- **[两轮检索限制降低恢复上限]** → 优先验证一次针对性纠错的单位成本收益；更长循环需独立 change 和新预算证据。

## Migration Plan

1. 先加入向后兼容的 correction 公共模型、Schema、严格配置和确定性纯函数；旧模型与脚本不变。
2. 增加 Retriever/Graph 适配器、动作目录、预算账本和离线 fixture，默认不装配真实后端。
3. 扩展 `AgentState` 并加入固定状态图；用兼容构建器映射旧 route/evaluation 字段，现有垂直切片继续可独立运行。
4. 运行 3–5 题离线 smoke，再冻结 20–40 题开发集并执行三策略矩阵；仅在显式开关和预算确认后进行在线 smoke。
5. 更新 README、架构、实验说明和 `PROJECT_MAP.md`。回滚时停用 corrective 入口并继续使用现有 Dense/Hybrid/Graph 脚本；新缓存和运行产物可按精确运行身份删除，不影响索引与语料。
