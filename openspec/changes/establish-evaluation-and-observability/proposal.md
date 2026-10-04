# Proposal

## Why

现有 Dense、Hybrid、纠错与回答评测各自维护数据、运行身份和报告，难以在同一冻结输入上证明 facet 覆盖提升、失败恢复收益及其资源代价。迭代 7 需要建立统一、可恢复且可下钻的评测与脱敏观测层，使项目核心贡献能够以可复现实验而非平均分或个别样例呈现。

本 change 对应 `../guidebooks/iteration-07-evaluation.md`，落实 `../PROJECT_SPEC.md` 的 OBS-001～OBS-004、EVAL-001～EVAL-009，以及可复现性、资源边界和成本统计要求。

## Goals

- 在同一冻结输入和统一分母上比较固定 Hybrid、类型路由与 facet 纠错，并量化覆盖、恢复和资源代价。
- 使任一研究结论可从汇总指标下钻至逐题 Evidence、facet、决策、失败和 Trace。
- 以离线优先、硬预算和断点恢复控制时间、磁盘与 LLM Token 消耗。

## Non-goals

- 不优化检索器或 Agent 策略，不批量为所有配置生成回答，也不新增 UI 或在线服务接口。
- 不把单一 LLM Judge 当作事实真值，不引入多轮/多模型 Judge，不使用测试集调参。

## What Changes

- 建立 60～80 题的统一冻结评测集契约，划分 40～50 题开发集与 20～30 题测试集；标注问题类型、答案要点、必需/可选 facet、最小充分 Evidence、知识充分性和首次检索失败类型。
- 在运行前校验标注完整性、稳定 ID、内容哈希、版本漂移、跨划分重复和泄漏；测试集仅允许一次显式正式运行且不得参与调参。
- 建立统一指标与逐题记录，覆盖检索、Evidence/facet 覆盖、充分性判断、恢复收益、无效纠错、回答与引用质量、循环、Web、延迟、调用量、Token 和成本。
- 建立有硬上限、可断点恢复的统一 runner：全部配置只运行零 LLM 的检索/覆盖评测，回答与 Judge 仅作用于入选系统和限定人工复核子集；单题失败仍进入分母。
- 生成可由汇总指标下钻到问题、facet、Evidence、决策、失败、路由、循环和引用的原子报告，并冻结数据、语料、索引、配置、模型、Prompt、指标与 Judge 版本。
- 在不破坏 `TraceEvent v1` 读取能力的前提下增加版本化评测 Trace 契约；新评测运行使用结构化、脱敏且有界的事件，旧产物通过显式适配器读取。
- 默认测试和正式检索矩阵保持离线；真实模型、Judge、搜索和数据库只在显式开关、预算确认与缓存约束下启用。

## Capabilities

### New Capabilities

- `evaluation-observability`: 统一评测标注、冻结划分、指标计算、可恢复运行、资源门禁、逐题可观测记录与可下钻报告。

### Modified Capabilities

- `foundation-contracts`: 增加向后兼容的版本化评测 Trace 契约，同时保留既有 `TraceEvent v1` 的读取语义和敏感信息禁令。

## Impact

- 主要影响 `src/kg_crag/evaluation/`、`src/kg_crag/observability/`、公共模型与 schema 导出、评测脚本、`configs/evaluation.yaml`、`data/evaluation/`、测试及实验文档；现有分项 evaluator 通过适配层复用，不重写检索和回答工作流。
- 依赖当前 `foundation-contracts`、`dense-rag-baseline`、`hybrid-retrieval`、`provenance-aware-knowledge-graph`、`evidence-sufficiency-corrective-workflow` 与 `grounded-reflection-web-fallback` 规格，以及冻结的语料、索引和 Prompt/模型版本。
- 不新增必需的在线依赖；可选 Judge 复用现有模型 Provider，并由调用数、题数、Token 和预算上限共同约束。
- 主要风险是标注成本与一致性、数据泄漏、Trace/报告体积、旧产物兼容、不同 evaluator 的分母口径以及 Judge 偏差；通过严格 schema、哈希与泄漏门禁、有界字段、适配器、失败计入分母和人工抽检降低风险。
