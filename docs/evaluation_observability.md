# 统一评测与可观测性

## 数据与边界

统一数据位于 `data/evaluation/unified/`：`manifest.json` 绑定版本、规模、哈希和人工复核状态，`dev.json`/`test.json` 保存严格题目标注，`annotation-guide.md` 说明复核规则。当前候选集为 `reviewed=false`；真实评测会在调用策略前拒绝它，只有 fixture 管线验收可继续。迁移脚本默认 dry-run；只有显式 `--write` 才创建数据，已有文件还需 `--force` 才能重建。每次改动题目、facet、Evidence 或划分都必须提升版本并更新清单，不能手工只改哈希。

默认配置在 `configs/evaluation.yaml` 的 `unified_evaluation` 段。题数绝对上限 100，Judge 回答绝对上限 50；YAML 只能收紧，不能放宽代码边界。自动化测试与 fixture 回放不构造真实模型、搜索或数据库客户端。

## 运行模式

- `--dry-run`：只加载配置和冻结数据，执行 schema、哈希、规模、压力覆盖与泄漏检查。
- `--fixture-mode`：使用确定性标注回放验证 runner、指标、Trace、检查点和报告；产物明确标记 `fixture_mode=true`，不能冻结开发选择或消耗正式 test。
- `--observations FILE`：读取真实工作流预先发布的完整 `StrategyObservationSet`。文件必须与数据集/split 哈希一致，并完整覆盖每题 × 三策略；缺项、多项或版本漂移都会在评测前失败。
- 程序化模式：把现有 Hybrid、路由和纠错入口包装成 `CallableStrategyAdapter` 注入 `UnifiedEvaluationRunner`；在线依赖必须先通过显式开关、预算确认和健康检查。

运行目录按 `unified-run-<hash>` 隔离，包含 `items/` 检查点、`report.json`、`failures.json` 和 `run-manifest.json`。数据、配置、索引、策略、模型、Prompt、指标、Judge、随机种子、fixture/真实模式、入选系统或 Judge 子集变化都会形成新身份。损坏检查点重算，不兼容身份不会复用。

## 开发选择与正式测试

开发集允许反复运行。完成真实三策略矩阵后，使用非 fixture dev 报告冻结唯一入选系统；选择文件绑定数据、配置、Prompt、模型和报告哈希，不能覆盖。test 默认不可运行：必须已有匹配选择、显式 `--confirm-test-run`、无调参/选策略参数且当前数据集版本没有正式锁。

正式运行成功后写不可覆盖测试锁。修改命令参数不能绕过同版本一次性限制；需要修正测试题时发布新数据集版本，并保留旧结果和失效说明。当前迭代的 fixture 验收没有创建开发选择或测试锁。

## Judge 与人工复核

确定性答案要点和引用检查先运行。Judge 是独立附加信号：只接受入选系统、预声明非空子集、一次单模型调用流程和最多 50 个回答；必须显式开启在线模式并确认预算。缓存键绑定 Provider、Prompt、问题和逐题结果，失败调用仍记录调用量和错误。Judge—人工一致率单独展示，Judge 不改写冻结真值或自动选择策略。

## 常用检查

```powershell
python scripts/validate_evaluation_dataset.py
python scripts/evaluate_unified.py --dry-run
python scripts/evaluate_unified.py --fixture-mode
python scripts/export_model_schemas.py --check
pytest -W error
ruff check .
ruff format --check .
mypy
```

报告禁止保存凭据、完整 Prompt、私有推理、Provider 原始响应和 Evidence 正文。终端摘要只显示运行身份、数量、失败数和报告路径。
