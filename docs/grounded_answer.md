# 回答反思与受控 Web 兜底

回答阶段只消费已经终止的纠错工作流结果。内部证据充分时生成带 Claim、Citation 和 facet 绑定的回答；内部知识确实缺失时，才允许在显式开关、可信域名和独立预算都满足的情况下调用一次 Web。所有来源最终统一为 `Evidence`，外部证据始终标记 `external=true`。

## 默认边界

- `grounded_answer.enabled` 与 `grounded_answer.web.enabled` 默认关闭；请求侧还需显式允许 Web。
- 单题最多生成 2 次、Critic 1 次、反思 1 次、Web 1 次，本阶段 LLM 合计最多 3 次。
- 输入输出合计最多 12,000 Token（输入 8,800、输出 3,200），回答上下文最多 16,000 字符；该分配可容纳两次各 1,600 输出 Token 的有界生成。
- Web 最多返回 5 条，单条摘录最多 2,000 字符，外部上下文合计最多 8,000 字符。
- 内部重新检索复用纠错工作流的原预算，总检索轮次仍不得超过 2；二次检索仍不足时直接保守停止，不再浪费回答生成调用。

运行参数位于 `configs/default.yaml` 的 `grounded_answer` 节。真实模型和搜索密钥只通过 `.env` 或进程环境传入；Tavily 还需设置 `KG_CRAG_WEB_SEARCH_PROVIDER=tavily`、`KG_CRAG_WEB_SEARCH_API_KEY` 和 `KG_CRAG_ENABLE_WEB_FALLBACK=true`。`glm-5.3-flash` 不能关闭思考，建议设置 `KG_CRAG_LLM_REASONING_EFFORT=low`；否则大量 `reasoning_content` 可能在结构化正文生成前耗尽输出额度。

## 检查与失败语义

候选回答先经过确定性结构、引用、来源、facet、数值和冲突检查。把“暂无可信结论”“证据不足”等拒答绑定到已覆盖 facet 时，确定性检查会产生 `abstention`，不会再把格式正确的拒答当作事实答案。只有确定性检查通过时才调用一次语义 Critic。检查失败后只能选择一次重新生成、内部重新检索或 Web 补证；拒答重写只能复用原引用和 facet 绑定，补救完成后只能接受或保守停止，不会进入第二轮反思。

系统关闭 OpenAI SDK 的隐式重试，也不会切换服务。Provider 超时立即停止；429、5xx、连接故障或空响应等明确标记为 retryable 的快速故障，在剩余预算允许时最多显式重试一次。`finish_reason=length` 且正文为空属于确定的额度耗尽，不重复提交相同请求。两次调用均计入预算和 Trace，Trace 只记录安全的 retryable、HTTP 状态码、异常类型、结束原因和推理字符数，不记录推理正文。回答解析只额外容忍包裹单个对象的 Markdown JSON 围栏，Schema、引用和 facet 绑定仍严格校验；失败按 Provider 请求、JSON、Schema、未知 ID 和绑定错误分类，且不保存原始响应。最终结果会保留终止时的 facet 评估、冲突、内部动作、缺失 facet、内外证据、停止原因、预算和脱敏 Trace，从而明确知识边界。

OpenAI-compatible Provider 返回合法 usage 时按逐请求实际 Token 结算；无 usage、请求失败或实际值超过既有预留边界时采用保守估算。应用响应通过 `token_usage_source` 区分 `actual`、`estimated`、`mixed` 和 `none`。

## 恢复与产物

检查点和最终报告使用输入、配置、Prompt、模型与 Provider 版本组成的内容身份，并原子写入配置的工作区相对路径。身份一致且最终结果存在时直接复用。恢复只继续能够证明已经完成的纯计算阶段；若检查点位于可能已经发起外部调用的边界，则保守终止，避免重复消费调用额度。

运行产物和评测结果位于 `data/processed/`，属于可再生内容，不提交 Git。公共结果与评测报告 Schema 位于 `docs/schemas/`。

## 评测命令

```bash
# 只校验 12 题冻结集、录制 Web fixture 和最坏预算，不访问网络
python scripts/evaluate_grounded_answer.py --dry-run

# 默认离线录制评测；结果按身份缓存
python scripts/evaluate_grounded_answer.py

# 真实 Provider 验收必须同时确认，且显式限制为 5～10 题
python scripts/evaluate_grounded_answer.py --online --confirm --limit 5
```

默认测试全部使用 Mock 或录制响应。真实验收测试只有设置 `KG_CRAG_RUN_GROUNDED_ONLINE=1` 后才会运行，并且仍受 CLI 双重门禁和严格超时限制。
