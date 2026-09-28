# Experiments

Planned experiment families are: LLM-only, dense RAG, dense+sparse hybrid RAG, fixed vector+graph RAG, and full KG-CRAG. Ablations remove graph retrieval, sparse retrieval, adaptive routing, corrective retrieval, answer critique, or web fallback; RRF and weighted fusion are compared separately.

Every run must record the dataset split, random seed, configuration hash, prompt/model/index versions, latency, tool calls, and token or inference cost. Generated result files belong under `data/evaluation/results/` and are not committed by default.

## Dense RAG v1 基线

迭代 02 的冻结基线使用 `configs/pilot_corpus.json` 对应的 15 篇 processed 论文、`data/evaluation/dense_pilot_questions.json` 的 20 个真实 Chunk 标注、固定 BGE-M3 revision、版本化 Prompt 和 Qdrant 集合身份。指标为 Recall@5/10/20、MRR、nDCG@5/10/20、citation precision 与 citation recall；失败题保留在分母中。

问题清单中的 `corpus_snapshot_hash` 不匹配、目标 Chunk 缺失或题数少于 20 时，评测必须在任何模型或数据库调用前失败。相同语料、配置、Prompt、集合和 K 列表形成相同 baseline ID，但每次运行使用独立目录，禁止静默覆盖结果。探索集可以用于调试，冻结 pilot 结果不得被反向用于调参。
