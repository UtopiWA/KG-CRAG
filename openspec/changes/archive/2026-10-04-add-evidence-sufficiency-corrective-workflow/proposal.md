# Proposal

## Why

当前系统已经具备 Dense、Sparse、Hybrid 与 Graph 等独立证据工具，但仍缺少把复杂问题表示为可验证证据需求、识别首次检索具体缺口并在硬预算内选择纠错动作的统一工作流。迭代 05 需要把项目核心研究问题落到可重放的结构化决策链上，使“为什么继续检索、补什么、何时停止”能够被测试和审计。

## What Changes

- 新增稳定的证据需求 facet、满足条件、Evidence—facet 覆盖项、冲突、充分性评估、纠错动作、成本估计、预算账本、停止原因和运行结果契约，并导出可评审 Schema。
- 新增规则优先的问题特征分析与 facet 生成；只有复杂或低置信度输入才允许在明确预算内调用版本化 LLM Provider，失败时使用确定性保守结果。
- 新增可解释的充分性评估器，以结构化覆盖矩阵输出已覆盖/缺失必需 facet、冲突、支撑强度、置信度及是否充分，不以单一相关性分数代替覆盖判断。
- 将 Dense、Sparse、Hybrid、Graph、查询重写、问题分解和参数调整注册为白名单纠错动作；依据目标 facet、预期覆盖增益、历史动作、剩余预算和版本化成本表确定性选择动作。
- 扩展 `AgentState` 并实现只读取结构化字段的有界单 Agent 工作流，最多执行 2 轮内部检索、3 个子问题、4 次 LLM 调用和合计 20k Token；充分、无正收益、重复动作、异常或预算耗尽时可靠停止。
- 新增首次检索失败压力集、离线评测与受控在线 smoke，比较固定 Hybrid、问题类型路由和 facet 缺口纠错的首次充分率、诊断准确率、恢复率、无意义动作率及资源消耗。
- 更新运行配置、Prompt、Schema 快照、工作流/评测文档和项目地图。
- 不接入 Web、回答生成、回答后反思、API 或 UI；不训练路由模型，不允许条件边解析自由文本，也不改变现有 Retriever 的单独行为契约。

关联 `PROJECT_SPEC.md`：`ROU-001`–`ROU-005`、`COR-001`–`COR-008`、`OBS-001`–`OBS-004`、`EVAL-005`、`EVAL-007`–`EVAL-009`、6.4 Agent 状态、7 架构约束及 8.1 可复现性。规划依据为 `guidebooks/iteration-05-corrective-workflow.md`。

## Capabilities

### New Capabilities

- `evidence-sufficiency-corrective-workflow`: 定义证据需求、facet 覆盖与冲突评估、白名单纠错动作、预算账本、结构化状态机、可靠停止以及首次检索失败压力评测。

### Modified Capabilities

无。现有 Dense、Hybrid 和 Graph 规格继续作为独立工具契约；本 change 只在新的工作流能力中编排它们。

## Impact

- **代码与公共契约**：预计新增 `models/correction.py`、`correction/`、工作流节点/条件边/编排器和纠错评测模块，并扩展 `workflow/state.py`、公共模型导出及 Schema 快照。
- **配置与 Prompt**：扩展 `configs/default.yaml` 与 `configs/evaluation.yaml` 的预算、阈值、动作成本和压力集边界；新增版本化 facet 生成 Prompt，但默认路径保持规则与 Mock 可运行。
- **数据与产物**：新增 20–40 题开发压力集及 Git 忽略的检查点/逐题结果；数据绑定问题、Evidence、配置、Prompt、模型和索引版本。
- **依赖与兼容性**：增加经 Python 3.11 验证的固定版本 LangGraph 工作流编排依赖；现有检索/问答脚本和公共 Evidence 字段保持兼容，旧 `RouteDecision`/`RetrievalEvaluation` 仅提供迁移适配，不作为新条件边的自由文本输入。
- **资源**：默认单元和矩阵评测零网络、零真实数据库、零真实 LLM；真实在线验收最多 20 题，单题最多 4 次 LLM 调用、20k Token、2 轮内部检索与 3 个子问题。
- **风险**：规则 facet 召回不足会导致过早停止，成本估计漂移会影响动作选择，跨 Retriever 调用语义不一致可能破坏确定性；通过保守默认、严格适配器、版本化成本表、录制 fixture 和逐题 Trace 控制。
