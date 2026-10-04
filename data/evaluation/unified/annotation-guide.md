# 统一评测标注指南 v4

## 目标与来源

统一集用于比较固定 Hybrid、问题类型路由和 facet 纠错，不替代原始分项 fixture。v4 包含 dev 40 题、test 20 题：17 道纠错题来自 `corrective_dev_questions.json`，43 道科研问答来自 15 篇论文的冻结 Chunk。

原纠错 q07～q09 只有比较占位句，Graph fixture 只有未展开 Fact ID，grounded fixture 只有行为场景而没有真值 Evidence，均不进入统一候选集。后两类仍可用于各模块行为测试，但不得冒充统一问答真值。

## 文件与字段规则

- `evidence.json` 固化题目引用的 Evidence 正文、来源位置、论文级来源组和内容哈希；题目不得引用目录外 ID。
- `question-sources.json` 固化当前题目文本及其来源。原题迁移记录指向同题 ID 的 fixture；新命题记录为 `chunk_derived` 并指向实际 Evidence。
- `source_fixture` 只表示当前问题文本的直接来源，不能用同论文的另一道旧题代替；答案证据来源由 `evidence.json` 独立表达。
- `question_id`、facet、答案要点和 Evidence ID 一经人工冻结不得原地改义；修订时提升数据集版本。
- `answer_points` 必须是 Evidence 直接支持的具体命题，不能复用“摘要说明了……”一类选段理由。
- 必需 facet 描述完整答案必须覆盖的信息；并列组件、数值或比较侧应拆成独立 facet。
- facet 的 `evidence_match=any` 表示任一目标 Evidence 即可覆盖，`all` 表示所有目标必须共同出现；冲突综合判断必须使用 `all`。
- `relevant_evidence` 使用 1～3 级相关性，3 表示直接支撑核心答案。
- 每个最小充分集必须覆盖全部必需 facet；冲突题和内部知识缺失题保持为空。
- `knowledge_sufficiency` 区分内部充分、证据冲突和内部不足；冲突不能因出现第三个数值而自动消解。
- `source_group_ids` 使用稳定 Paper ID。共享同一论文来源的所有问题必须位于同一 split。
- `leakage_group_id` 绑定同题改写；它与论文级来源组均不得跨 dev/test。

## 压力类型判定

- `terminology_mismatch` 要求问题与 Evidence 使用可明确记录的不同表达，普通翻译或近乎逐字改写不算术语错配。v4 中仅 `camel-research-use` 保留该标签，预期词面映射为问题的“研究资源”对应 Evidence 的“conversational data for studying behaviors and capabilities”。
- `entity_alias` 要求问题与 Evidence 之间存在可核对的缩写、别名或拼写变体，稀有实体或精确名称检索不等于别名。
- `multi_hop_gap` 要求回答显式连接中间实体、关系或阶段；从单段文本直接枚举组件或模式应标为 `control`。
- 只有问题明确询问的信息才能作为必需 facet；背景数量、额外范围或相关事实可省略时，不得用它们压低完整性。

## 人工复核步骤

v4 已于 2026-10-04 完成 dev/test 全量人工复核，当前 `manifest.json` 为 `reviewed=true`。以下步骤供后续新版候选集复核时重复执行：

1. 新建或修订候选集时先确认 `reviewed=false`，避免把未复核数据误当作已冻结数据。
2. 按题打开 `dev.json` 或 `test.json`，先在 `question-sources.json` 核对题目来源，再按 Evidence ID 查找 `evidence.json` 的正文和来源位置。
3. 逐项确认问题对象明确、每个答案要点可由正文直接推出、每个必需 facet 都是完整回答所必需。
4. 对 sufficient 题确认每个最小充分集覆盖全部必需 facet；对 conflict 题确认单侧事实只绑定对应 Evidence，综合冲突 facet 使用 `all` 且没有虚假裁决。
5. 对 internal_missing 题确认答案要点和 Evidence 为空，但 facet 明确指出缺失的具体信息。
6. 按论文 Paper ID 检查 dev/test 不共享答案来源，再运行统一数据校验。
7. 只有 60 题全部人工通过后，才把 `reviewed` 改为 `true`，在 `review_note` 记录复核人、日期和范围，并重新运行完整检查。

```powershell
python scripts/validate_evaluation_dataset.py
python scripts/export_model_schemas.py --check
pytest -W error
```

测试集只允许在开发选择冻结后正式运行一次。若人工复核发现任何题目或 Evidence 需要修改，必须继续保持 `reviewed=false`、提升数据集版本并重新计算全部哈希。
