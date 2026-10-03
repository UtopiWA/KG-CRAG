# Experiments

Planned experiment families are: LLM-only, dense RAG, dense+sparse hybrid RAG, fixed vector+graph RAG, and full KG-CRAG. Ablations remove graph retrieval, sparse retrieval, adaptive routing, corrective retrieval, answer critique, or web fallback; RRF and weighted fusion are compared separately.

Every run must record the dataset split, random seed, configuration hash, prompt/model/index versions, latency, tool calls, and token or inference cost. Generated result files belong under `data/evaluation/results/` and are not committed by default.

## Dense RAG v1 基线

迭代 02 的冻结基线使用 `configs/pilot_corpus.json` 对应的 15 篇 processed 论文、`data/evaluation/dense_pilot_questions.json` 的 20 个真实 Chunk 标注、固定 BGE-M3 revision、版本化 Prompt 和 Qdrant 集合身份。指标为 Recall@5/10/20、MRR、nDCG@5/10/20、citation precision 与 citation recall；失败题保留在分母中。

问题清单中的 `corpus_snapshot_hash` 不匹配、目标 Chunk 缺失或题数少于 20 时，评测必须在任何模型或数据库调用前失败。相同语料、配置、Prompt、集合和 K 列表形成相同 baseline ID，但每次运行使用独立目录，禁止静默覆盖结果。探索集可以用于调试，冻结 pilot 结果不得被反向用于调参。

## Hybrid Retrieval v1 开发矩阵

`data/evaluation/hybrid_dev_questions.json` 绑定同一 pilot 快照，共 24 题；除原有语义问题外，加入大写缩写、连字符模型名、数据集名和低频实体。矩阵使用完全相同的问题、目标 Chunk 和 K 值比较 Dense、Sparse、RRF、Weighted 与 Fusion+Rerank，计算 Recall@K、MRR、nDCG@K、Evidence coverage、失败数、总时延与阶段时延。

矩阵默认零 LLM。配置、语料快照、Dense 集合、Sparse 索引、融合版本、Reranker 版本、策略和 K 值共同形成 `evaluation_version`；逐题检查点只有身份严格匹配时才复用。回答验证与检索报告分开保存，必须显式选择唯一 `fusion_rerank` 配置并给出与题数完全一致的调用预算。

2026-10-03 验收记录：pilot Sparse dry-run、970 Chunk 首次构建、全量幂等跳过、精确版本重建和带论文过滤查询均通过；真实五策略 smoke 因本机 Docker/Qdrant 守护进程未响应而未完成，未生成或伪造质量指标。SQLite + Mock Dense/Reranker 的零网络集成矩阵已覆盖建索引、重跑、单路降级、重排回退、报告和断点复用。
