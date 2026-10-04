# 统一评测标注指南 v1

## 目标与来源

统一集用于比较固定 Hybrid、问题类型路由和 facet 纠错，不替代原始分项 fixture。v1 从现有纠错、Hybrid、Graph 与 grounded 数据迁移：开发集 40 题，测试集 20 题。`source_fixture` 必须保留原题位置，任何人工修订都要发布新数据集版本并重算清单哈希。

## 字段规则

- `question_id`、facet、答案要点和 Evidence ID 一经发布不得原地改义。
- `question_types` 描述题目任务；`stress_category` 只描述预先构造的首次检索缺口，不能根据系统输出反标。
- 必需 facet 是完整回答不可缺少的证据需求；可选 facet 只增加信息量。
- `relevant_evidence` 使用 1～3 级相关性，3 表示直接支撑核心答案。
- 每个 `minimum_sufficient_evidence_sets` 必须覆盖所有必需 facet；内部知识缺失题必须为空。
- `knowledge_sufficiency` 分为内部充分、证据冲突和内部不足。冲突不能因为命中更多 Evidence 自动算充分。
- `leakage_group_id` 绑定同题改写；`source_group_ids` 绑定共享答案来源。两者均不得跨 dev/test。

## 复核顺序

1. 对照 `source_fixture` 检查问题和目标 Evidence。
2. 检查答案要点是否只陈述 Evidence 可支持的内容。
3. 检查每个必需 facet 在每个最小充分集中至少有一个目标 Evidence。
4. 检查压力类型与预期缺口、知识状态一致。
5. 运行 `python scripts/validate_evaluation_dataset.py`，确认规模、哈希、版本和泄漏门禁。

测试集仅在开发集选择被冻结后运行一次。修正测试标注时必须提升 `dataset_version`，保留旧结果并说明失效原因。
