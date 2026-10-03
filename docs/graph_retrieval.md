# 可溯源知识图谱与 Graph Retriever

## 边界

基础图由已发布 `paper.json`、`chunks.jsonl` 和其中已有的显式引用确定性构建，不联网补齐元数据。正文事实只覆盖 15 篇 pilot 的代表 Chunk；Graph Retriever 是与 Dense、Sparse、Web 并列的可选工具，本模块不做自动路由、融合、回答生成或生成式 Cypher。

节点和关系白名单由 `models/graph.py` 冻结。图身份绑定 schema、语料快照、构建器、选择器、抽取器、Prompt、模型修订、规范化和查询模板版本；任一组成变化都产生新 `graph_version`。Neo4j 以 `graph_version` 和 `graph_key` 隔离版本，只允许精确版本的显式重建。

## 运行顺序

```bash
python scripts/build_metadata_graph.py --dry-run
python scripts/build_metadata_graph.py
python scripts/query_graph_retrieval.py "完整实体名称" --top-k 10
python scripts/evaluate_graph_retrieval.py
```

生产 Neo4j 导入需先启动服务，再显式选择后端：

```bash
docker compose up -d neo4j
python scripts/build_metadata_graph.py --backend neo4j
set KG_CRAG_RUN_NEO4J_TESTS=1
pytest -q tests/integration/test_neo4j_graph_store.py
```

不要清空整个数据库。重建只允许作用于 dry-run 已显示的精确 `graph_version`；回滚时停用 Graph 动作即可，Dense/Sparse/Hybrid 不受影响。

## 正文抽取预算

真实抽取必须同时提供 `--online --confirm-budget`。建议先使用 `--paper-id ... --max-chunks 3` 做 smoke，再决定是否扩到 pilot。默认限制为每篇 3–5 个 Chunk、最多 100 次请求、160,000 输入 Token、40,000 输出 Token、单并发；每批调用前检查预算。缓存键绑定 Chunk 内容、schema、Prompt、模型修订和输出契约，命中缓存时零调用。

抽取成功后会先生成完整论文级 Bundle 和复核报告，再同步到所选后端。默认 `--backend memory` 便于低成本验收且进程退出后不保留数据库状态；只有显式传入 `--backend neo4j` 才写入 Neo4j。失败调用也会计入请求和保守 Token 预留，单个 Chunk 的非法输出不会部分发布。

2026-10-03 使用本地配置的 `glm-5.3-flash` 对 1 篇、最多 3 个 Chunk 执行真实 smoke。Provider 数分钟内未返回任何可验证响应，运行被人工中止；没有启动完整 pilot，也没有生成虚假成功清单。在线抽取和真实 Neo4j 集成仍属于显式环境验收项，离线 Mock、预算、缓存、来源校验和内存后端已纳入默认测试。

## 产物与复核

- `data/processed/graph-runs/`：基础图、抽取运行清单及逐篇完整 Bundle。
- `data/processed/graph-extraction-cache/`：逐 Chunk 内容寻址缓存。
- `data/processed/graph-review/`：歧义实体复核报告和版本化人工决定。
- `data/evaluation/results/graph-retrieval/`：逐题检查点与汇总报告。
- `data/evaluation/graph_dev_questions.json`：可提交的 12 题冻结开发集。

正文 Evidence 必须回读当前 processed Chunk 并复验 Paper、Chunk 和内容哈希；任一来源漂移时拒绝整条路径。元数据关系只返回有限事实描述，不伪造正文、章节或页码。复核报告只保存名称、候选 ID、理由、分数和来源标识，不保存完整 Chunk。

人工决定文件固定为 `data/processed/graph-review/decisions.json`。每项决定必须绑定当前 `normalization_version`、复核 ID、两个实体 ID 和来源 ID；导入前会拒绝未知候选、失效来源、跨类型合并和冲突外部 ID。

## 验收

默认评测不会调用 LLM、Embedding、Dense、Sparse 或 Web，每题只访问图存储零或一次。当前 15 篇 pilot 基础图 dry-run 和内存构建成功，冻结 12 题矩阵的路径命中率、Evidence Recall@K 和溯源完整率均可由 `evaluate_graph_retrieval.py` 重放。生成结果目录受 Git 忽略，不应提交。
