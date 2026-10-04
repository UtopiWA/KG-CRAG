# KG-CRAG

面向科研文献问答的证据自适应检索 Agent。当前已完成工程基座、论文摄取、Dense Vector RAG、Hybrid/Graph 检索，以及证据 facet 充分性诊断与有界纠错工作流。系统可把 Dense、Sparse、Graph 命中统一为 Evidence，定位首次检索的具体证据缺口，在两轮内部检索和统一预算内选择白名单动作，并给出可审计停止原因。

## 环境要求

- Python 3.11（当前依赖范围也兼容 3.12）
- Docker 与 Docker Compose（运行 Qdrant / Neo4j 时需要）

## 初始化

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

不要把真实 API Key 写入仓库；仅在本地 `.env` 中设置。

## 常用命令

```bash
# 单元测试、代码检查与类型检查
pytest
ruff check .
mypy

# 基础数据服务
docker compose up -d qdrant neo4j
docker compose ps

# API（当前仅包含健康检查）
uvicorn kg_crag.api.app:app --reload
# GET http://127.0.0.1:8000/health
```

## 工程基座契约

- `kg_crag.models` 统一导出论文领域模型，以及 `ErrorDetail`、`TraceEvent` 和
  `HealthResponse` 等基础运行契约；这些 Pydantic 模型拒绝未知顶层字段。
- `KGCRAGError` 通过稳定 `ErrorCode` 和 `retryable` 表达调用方可处理的失败；错误
  `context` 与 Trace `details` 只允许有限标量，并拒绝密码、Token、API Key 和认证字段。
- `MockRetriever`、`MockReranker`、`InMemoryVectorStore` 和 `InMemoryGraphStore` 是确定性、
  不联网的测试替身。它们遵守公开 Protocol 并记录调用，但不模拟生产检索质量、Qdrant
  或 Neo4j 的完整语义。
- `Settings` 要求端口位于 1–65535，Qdrant 使用带主机的 HTTP(S) URL，Neo4j 使用带主机
  的 `bolt://` 或 `neo4j://` URI；工作流循环次数继续受各字段上限约束。
- `/health` 只返回 `status`、`version` 和 `environment`，不会探测外部服务或返回密钥。

公共模型 Schema 快照位于 `docs/schemas/`：

```bash
# 模型有意变化后重建快照
python scripts/export_model_schemas.py

# CI 或评审时只检查，不修改文件
python scripts/export_model_schemas.py --check
```

## 获取论文语料

脚本只处理 `configs/seed_papers.json` 中的显式 arXiv ID，默认最多 10 篇，串行、限速并记录 SHA-256。它不会进行关键词爬取或无限发现。

```bash
# 先看将执行什么
python scripts/fetch_seed_papers.py --dry-run

# 下载元数据和 PDF；可安全重复执行，已有且校验有效的文件会跳过
python scripts/fetch_seed_papers.py

# 只拉取元数据，或限制数量
python scripts/fetch_seed_papers.py --metadata-only --limit 3

# 校验本地语料及清单
python scripts/verify_seed_corpus.py
```

当需要达到完整语料规模时，使用批量采集脚本。它把显式种子和 `configs/corpus_collection.json` 中的多组主题查询合并，默认通过 OpenAlex 发现带 arXiv 标识的候选，并始终从 arXiv 官方地址下载 PDF。采集过程执行年份/分类/相关性筛选、ID 和标题去重、候选缓存、断点续跑，并在达到目标数后停止：

```bash
# 只显示当前已验证数量和缺口，不联网
python scripts/collect_arxiv_corpus.py --dry-run

# 默认补齐到 110 篇；已有有效 PDF 不会重复下载
python scripts/collect_arxiv_corpus.py

# 重新执行 OpenAlex 发现，或显式设置目标（安全上限为 200）
python scripts/collect_arxiv_corpus.py --refresh-discovery --target 110

# 仅刷新候选缓存，不下载 PDF
python scripts/collect_arxiv_corpus.py --refresh-discovery --discovery-only

# arXiv Atom API 可用时，也可显式切换或合并发现源
python scripts/collect_arxiv_corpus.py --refresh-discovery --discovery-source both
```

候选缓存、逐项审计日志和运行报告分别位于 `data/raw/arxiv_candidates.json`、`collection_manifest.jsonl` 和 `collection_report.json`。这些运行产物与 PDF 一样默认不进入 Git。

产物位置：

- `data/raw/papers/*.pdf`：原始 PDF
- `data/raw/metadata/*.json`：标准化元数据与获取来源
- `data/raw/download_manifest.jsonl`：逐次获取/跳过/失败记录

下载前应确认使用场景符合对应论文的授权条件。原始 PDF 和运行时元数据默认被 Git 忽略。

## 摄取本地论文

摄取只读取 `data/raw`，默认安全选择 15 篇已人工核验版面的 pilot。先 dry-run，再实际执行并重读产物：

```bash
# 默认等价于 --pilot，不创建任何输出
python scripts/ingest_corpus.py --dry-run

# 生成 pilot 产物；相同哈希与处理版本会校验后跳过
python scripts/ingest_corpus.py --pilot
python scripts/verify_ingestion_outputs.py --pilot

# 单篇调试
python scripts/ingest_corpus.py --paper-id arxiv:2312.07559

# 仅当 pilot 自动质量与 configs/pilot_review.json 人工门禁都有效时可运行
python scripts/ingest_corpus.py --all --limit 110
python scripts/verify_ingestion_outputs.py --all
```

`--force` 可原子重建同一处理版本，`--interim-root` 和 `--processed-root` 可覆盖输出位置。退出码 `0` 表示全部成功或合法跳过，`1` 表示参数/配置/预校验失败，`2` 表示批次完成但至少一篇失败。完整目录、版本算法、门禁和故障恢复说明见 `docs/ingestion.md`。

PDF 解析依赖固定为 `pymupdf==1.28.2`。该依赖采用 AGPL v3/商业双许可证；分发或部署前必须阅读 `docs/dependencies.md` 并确认许可方案。

## Dense RAG 基线

先启动 Qdrant，再按“规划、索引、查询、评测”的顺序运行：

```bash
docker compose up -d qdrant
python scripts/build_vector_index.py --pilot --dry-run
python scripts/build_vector_index.py --pilot
python scripts/query_dense_rag.py "Voyager 的三个核心组件是什么？" --top-k 10
python scripts/evaluate_dense_rag.py --dry-run
python scripts/evaluate_dense_rag.py
```

索引 dry-run 只读取已发布 Chunk 和 Qdrant 点状态，不加载 Embedding 模型、不写缓存或集合；查询 dry-run 只校验配置、Prompt、过滤和上限。默认 `KG_CRAG_LLM_PROVIDER=mock` 便于离线验收；真实运行需显式配置 `openai-compatible` Provider 和本地密钥。首次实际索引会按固定 revision 下载 BGE-M3，具体版本、产物布局、重建/回滚和验收状态见 `docs/dense_rag.md`。

## Hybrid 检索

Hybrid 默认并发调用 Dense 与 Sparse 各一次，以统一 `Evidence` 融合，且不会调用 LLM。先构建轻量 Sparse 索引，再查询或评测：

```bash
python scripts/build_sparse_index.py --pilot --dry-run
python scripts/build_sparse_index.py --pilot
python scripts/query_hybrid_retrieval.py "CAMEL 如何组织角色扮演智能体？" \
  --sparse-index-version <dry-run 中的 index_version>
python scripts/evaluate_hybrid_retrieval.py --smoke --dry-run
python scripts/evaluate_hybrid_retrieval.py --smoke \
  --sparse-index-version <index_version>
```

评测矩阵比较 Dense、Sparse、RRF、Weighted 与 Fusion+Rerank，默认零 LLM。只有显式提供 `--with-answer --selected-strategy fusion_rerank --answer-budget <题数>` 才会对唯一入选配置逐题生成一次回答。完整公式、降级语义、重建/回滚、产物和资源上限见 `docs/hybrid_retrieval.md`。

## 可溯源知识图谱

基础图只读取已发布 Paper/Chunk 和显式引用，不访问网络或 LLM。查询只接受四种固定模板，不接收任意 Cypher：

```bash
# 先检查 15 篇 pilot 的图身份和规模，再写入内存后端运行清单
python scripts/build_metadata_graph.py --dry-run
python scripts/build_metadata_graph.py

# 查询和固定开发集评测均使用内存图，默认零在线调用
python scripts/query_graph_retrieval.py "CAMEL: Communicative Agents for Mind Exploration of Large Language Model Society"
python scripts/evaluate_graph_retrieval.py

# 正文抽取必须同时显式允许在线调用并确认预算；可先限制 1 篇、3 个 Chunk
python scripts/extract_graph_facts.py --online --confirm-budget \
  --paper-id arxiv:2303.17760 --max-chunks 3

# smoke 通过后才显式写入 Neo4j；默认 memory 仍会保存可审计 Bundle
python scripts/extract_graph_facts.py --online --confirm-budget --backend neo4j
```

默认硬上限为 15 篇、每篇 3–5 个代表 Chunk、100 次请求、160,000 输入 Token 和 40,000 输出 Token。抽取缓存、清单和评测结果均为可再生产物并被 Git 忽略；真实 Neo4j 测试仅在 `KG_CRAG_RUN_NEO4J_TESTS=1` 时运行。图身份、Schema、复核、回滚及验收记录见 `docs/graph_retrieval.md`。

## 证据充分性与有界纠错

纠错工作流默认完全离线：规则生成 facet，以冻结 Evidence fixture 比较 fixed Hybrid、问题类型路由和 facet 缺口纠错，不访问模型、网络、Qdrant 或 Neo4j。

```bash
python scripts/run_corrective_workflow.py --question-id q01 --dry-run
python scripts/run_corrective_workflow.py --question-id q01
python scripts/evaluate_corrective_workflow.py --smoke
python scripts/evaluate_corrective_workflow.py

# 真实 LLM 只在显式开关和预算确认后启用
python scripts/run_corrective_workflow.py --question-id q07 --online --confirm-budget
python scripts/evaluate_corrective_workflow.py --online --confirm-budget --limit 3
```

单题硬上限为 2 轮内部检索、3 个子问题、4 次 LLM 调用和 20,000 输入输出 Token。动作目录不包含 Web、回答生成或动态工具名。缓存、检查点和评测结果分别位于 `data/processed/corrective-*` 与 `data/evaluation/results/corrective-workflow/`，均由 Git 忽略。完整语义、恢复和回滚说明见 `docs/corrective_workflow.md`。

## 回答反思与受控 Web 兜底

回答阶段使用严格 Claim/Citation/facet 绑定、确定性检查和最多一次语义 Critic。Web 默认关闭，只有内部停止原因为 `internal_knowledge_missing`、请求显式允许且可信来源策略通过时，才会搜索一次；任何失败均返回可审计的保守知识边界。

```bash
# 冻结题集与录制 fixture 校验，零网络
python scripts/evaluate_grounded_answer.py --dry-run

# 默认离线评测
python scripts/evaluate_grounded_answer.py

# 真实 LLM/Tavily 验收：必须双重确认并限制 5～10 题
python scripts/evaluate_grounded_answer.py --online --confirm --limit 5
```

单题最多 2 次生成、1 次 Critic、1 次 Web 和 1 次反思，输入输出合计不超过 12k Token。配置、可信来源、恢复与失败语义详见 `docs/grounded_answer.md`。

## 统一评测与可观测性

迭代 7 将具有语义真值的纠错 fixture 与冻结论文 Chunk 迁移为 60 题 v4 统一集（40 dev、20 test），统一记录问题类型、具体答案要点、必需/可选 facet、最小充分 Evidence、压力类型和知识充分性。`evidence.json` 保存可核验正文和内容哈希，`question-sources.json` 区分原题迁移与 Chunk 派生命题，论文级来源组阻止同一论文跨 dev/test。当前 v4 已于 2026-10-04 完成 dev/test 全量人工复核并通过自动校验；正式测试仍须在开发选择冻结后显式执行且同版本最多一次：

```powershell
# 数据迁移默认 dry-run；冻结数据已存在时不会静默覆盖
python scripts/migrate_unified_evaluation_dataset.py
python scripts/validate_evaluation_dataset.py

# 仅检查配置、哈希、规模和泄漏门禁
python scripts/evaluate_unified.py --dry-run

# 验收 runner、检查点、Trace 和报告；结果带 fixture_mode=true，不能冻结研究选择
python scripts/evaluate_unified.py --fixture-mode
python scripts/evaluate_unified.py --fixture-mode --selected-answer-strategy facet_corrective
```

真实策略应先通过 `UnifiedEvaluationRunner` 适配器或 `StrategyObservationSet` 发布完整三策略观察，再由 `--observations <file>` 生成可比较报告。开发选择只能由非 fixture 的 dev 报告一次性冻结；正式 test 还要求 `--confirm-test-run`，每个数据集版本只允许一次且不能携带调参/选策略参数。Judge 默认关闭，只能用于已选系统、显式预算确认和最多 50 个预声明回答，结果不覆盖人工真值。详细契约、产物和恢复方式见 `docs/evaluation_observability.md`。

冻结 v4 开发集可使用本地观察入口：论文题实际查询 BGE-M3 内存 Dense、SQLite Sparse 和处理后语料内存 Graph，合成纠错题则显式标记为冻结受控回放。该入口禁止 LLM、Web 和隐式模型联网探测，也不依赖 Qdrant、Neo4j 或 Docker：

```powershell
python scripts/collect_unified_observations.py --dry-run
python scripts/collect_unified_observations.py
python scripts/evaluate_unified.py --observations data/evaluation/results/unified/observations/dev-v4-local.json --index-version local-bundle-ed1d9059-51809a40 --strategy-version local-observation-v1
```

当前 v4 开发矩阵已冻结 `facet_corrective`；选择记录同时保存数据/报告哈希、观察哈希、版本声明及关键阈值。观察文件与报告位于 Git 忽略的结果目录，`development-selection.json` 是正式 test 的可提交门禁。

如需复核入选策略的零 token 抽取式回答代理，可在上述评测命令后增加 `--selected-answer-strategy facet_corrective --extractive-answer-validation`。它只核算 Top-8 Evidence 对冻结答案要点和引用的支持，不等同于自然语言生成质量；当前 Judge 保持未启用。

仅在用户明确确认消耗一次性正式 test 后，先执行 `python scripts/collect_unified_observations.py --split test --confirm-test-observation`；它会读取冻结选择并只为入选策略准备抽取式回答。随后以同一观察文件执行 `python scripts/evaluate_unified.py --split test --observations data/evaluation/results/unified/observations/test-v4-local.json --index-version local-bundle-ed1d9059-51809a40 --strategy-version local-observation-v1 --confirm-test-run`。任一步发现已有同版本正式锁都会拒绝继续。

## 项目结构

完整文件树、模块职责、关键脚本和按任务定位说明见仓库上级目录的 `../PROJECT_MAP.md`。新增、删除、重命名重要路径或改变模块职责时，应在同一变更中维护该地图。
