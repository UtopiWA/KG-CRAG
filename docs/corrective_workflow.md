# 证据充分性与有界纠错工作流

## 公共语义

`EvidenceRequirement` 把问题拆成可机器验证的 facet。`CoverageMatrix` 对每个 facet—Evidence 组合记录匹配、支撑强度、来源质量、依据和冲突；外部或无法由当前语料复核的 Evidence 不计内部覆盖。只有全部必需 facet 达到最小来源数和支撑阈值、且没有阻断冲突时，`SufficiencyAssessment.sufficient` 才为真。冲突必须比较同一命题：显式值需共享 `normalized_value_key`，文本数值比较只适用于指标 facet；普通正文中的年份、编号、其他实验数字或无关否定词不构成阻断冲突。

论文实体问题先用问题中的显式拉丁实体名锚定候选论文，再允许同论文中省略简称的摘要或续接 Chunk 参与覆盖。实时应用会从本次运行已经锁定版本的 SQLite Sparse 索引读取首个锚定论文至多 3 个摘要优先片段，不扫描 `data/processed`，也不会混入索引快照之外的新数据。选证据时优先采用该论文上下文及重排、融合和路由排名，并在 `max_selected_evidence` 上限内保留多个匹配片段，而不是让一个只出现实体名的附录页独占回答上下文；稳定 Chunk 来源仍可由 Evidence 回溯。

旧 `RouteDecision` 和 `RetrievalEvaluation` 保持兼容，但新条件边不读取它们的自由文本理由或缺失描述。

## 动作与预算

动作目录只允许 `dense`、`sparse`、`hybrid`、`graph`、`rewrite`、`decompose` 和 `adjust`。策略按缺失 facet 生成候选，以预期必需覆盖增益减去检索、模型、Token、时延与重复成本，并用稳定次级键选择唯一正收益动作。

调用前必须原子预留最坏成本，成功后按实际用量结算，失败按预留量保守计费。硬上限是两轮内部检索、三个子问题、四次 LLM 调用和合计 20,000 Token；配置只能收紧。停止原因固定为 `sufficient`、`no_positive_gain`、`budget_exhausted`、`internal_knowledge_missing`、`execution_failed` 和 `invalid_input`。

## 运行、产物与恢复

```bash
python scripts/run_corrective_workflow.py --dry-run
python scripts/run_corrective_workflow.py --question-id q01
python scripts/evaluate_corrective_workflow.py --smoke
python scripts/evaluate_corrective_workflow.py
```

需求生成缓存位于 `data/processed/corrective-cache/requirements/`，单题阶段检查点位于 `data/processed/corrective-runs/<run-id>/`；评测逐题检查点和报告位于 `data/evaluation/results/corrective-workflow/<evaluation-id>/`。文件采用 UTF-8 暂存、回读校验和原子替换；问题、规则、Prompt、模型或其他身份版本漂移会形成新缓存键或目录。Trace 只记录 ID、枚举、计数、分数、预算和有限错误，不保存凭据、完整 Prompt、模型响应、私有推理或无界正文。

默认命令使用冻结 fixture 和 Mock Retriever，零网络、零真实模型、零真实数据库。在线模式必须显式添加 `--online --confirm-budget`，先运行不超过三题的 smoke；只有 smoke 成功且 usage 可审计时才可扩到最多二十题。

## 回滚

停用两个 corrective 脚本即可回到独立 Dense/Hybrid/Graph 流程，不需要修改现有索引或语料。清理时只能按明确 `run-id` 或 `evaluation-id` 处理可再生产物；不得删除 `data/raw`。依赖回滚可移除固定 LangGraph 包，但必须同时停用 `workflow/corrective.py` 入口。
