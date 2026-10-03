# KG-CRAG

面向科研文献问答的证据自适应检索 Agent。当前已完成工程基座、论文摄取、Dense Vector RAG 与 Hybrid 检索：可从受控 PDF 发布稳定 Chunk，构建版本化 Qdrant/SQLite FTS5 索引，以 RRF 或加权方式融合 Dense/Sparse Evidence，并在有界本地 Cross-Encoder 重排后执行可选的证据约束回答。Graph、动态路由和纠错工作流仍属于后续迭代。

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

## 项目结构

完整文件树、模块职责、关键脚本和按任务定位说明见仓库上级目录的 `../PROJECT_MAP.md`。新增、删除、重命名重要路径或改变模块职责时，应在同一变更中维护该地图。
