# Hybrid 检索操作手册

本模块在既有 Dense 基线上增加 SQLite FTS5 Sparse、确定性融合和可选本地 Cross-Encoder。检索、评测默认零 LLM；只有显式回答验证会调用已配置的 LLM Provider。

## 固定边界

- Sparse 使用 `scientific-unicode-v1`：NFC、case-fold，并保留标识符内部的连字符和下划线。
- RRF：每路贡献 `1 / (rrf_k + rank)`，缺失来源贡献 0。
- Weighted：分别对两路候选做 min-max；等分列表归一化为 1，缺失来源为 0，再按配置权重求和。
- 排序依次使用最终分数降序、最佳来源排名、Chunk ID；跨路按 Chunk 去重，可在重排后按论文去重。
- Dense/Sparse 各调用至多一次；默认单路失败显式降级，两路失败则返回结构化错误。
- Reranker 默认 CPU、惰性加载、输入最多 20、输出最多 8；失败可回退到融合结果并标记降级。

## 建索引与查询

```bash
python scripts/build_sparse_index.py --pilot --dry-run
python scripts/build_sparse_index.py --pilot
python scripts/build_sparse_index.py --pilot \
  --rebuild-index-version <精确的64位index_version>

python scripts/query_hybrid_retrieval.py "CAMEL 如何组织角色扮演智能体？" \
  --sparse-index-version <index_version>
python scripts/query_hybrid_retrieval.py "AgentBench" \
  --method weighted --dense-weight 0.4 --sparse-weight 0.6 \
  --filter paper_id=arxiv:2308.03688 --no-reranker \
  --sparse-index-version <index_version>
```

先运行 dry-run。`--all` 必须带显式 `--limit`；重建必须填写当前计划的完整版本，防止误删其他索引。查询 dry-run 不打开数据库、不连接 Qdrant、不加载模型。

## 开发集评测

```bash
python scripts/evaluate_hybrid_retrieval.py --smoke --dry-run
python scripts/evaluate_hybrid_retrieval.py --smoke \
  --sparse-index-version <index_version>
python scripts/evaluate_hybrid_retrieval.py --max-questions 24 \
  --sparse-index-version <index_version>
```

默认矩阵为 Dense、Sparse、RRF、Weighted、Fusion+Rerank，且 `llm_calls=0`。回答验证必须显式限制唯一配置和调用数：

```bash
python scripts/evaluate_hybrid_retrieval.py --smoke \
  --sparse-index-version <index_version> \
  --with-answer --selected-strategy fusion_rerank --answer-budget 5
```

## 产物、恢复与回滚

- 索引：`data/processed/sparse-index/<index_version>/index.sqlite3` 及身份 manifest。
- 索引运行：`data/processed/sparse-index-runs/<run_id>/manifest.json`；检查点位于 `checkpoints/`。
- 查询：`data/processed/hybrid-query-runs/<run_id>/`。
- 评测：`data/evaluation/results/hybrid-retrieval/<evaluation_version>/`，含逐题检查点、报告和可选回答。

以上均被 Git 忽略，可由 processed Chunk 重建。更新内容时直接增量运行；身份不兼容时先 dry-run，再用精确版本重建。回滚只需停用 Hybrid 或改回旧配置/索引版本，Dense-only 命令和集合不受影响。删除产物前必须确认精确版本目录，不能删除整个 `data/processed`。

## 资源建议与验收状态

SQLite pilot 索引仅处理 970 Chunk，构建通常为秒级。Embedding 和 Cross-Encoder 不应同时常驻小显存 GPU；默认 Reranker 用 CPU。若 C 盘空间紧张，把 `dense_embedding.cache_root` 与 `reranker.cache_root` 配到工作区 D 盘相对目录后再建缓存，配置变更会进入版本边界，但机器绝对路径和密钥不进入哈希。

2026-10-03 本机已验证 Sparse dry-run、首次构建、幂等重跑、精确重建和过滤查询。真实五策略 smoke 因 Docker/Qdrant 守护进程未响应而保留为未验证项；离线 SQLite/Mock 全链路和默认无网络门禁已通过。真实 Cross-Encoder smoke 需设置 `KG_CRAG_RUN_RERANKER_TESTS=1`，最多 5 题、20 候选，默认测试不会下载模型。
