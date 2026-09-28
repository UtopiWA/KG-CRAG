# Dense RAG 基线操作手册

本文档描述迭代 02 的 Dense-only 可复现基线。它只消费通过质量门禁发布的 `data/processed` Chunk，不修改 `data/raw`，也不实现后续迭代的 Sparse、Graph、融合、路由、纠错或 Web 能力。

## 版本与外部边界

- Embedding：`BAAI/bge-m3`，revision `5617a9f61b028005a4858fdac845db406aefb181`，1024 维、余弦距离。
- 向量后端：Qdrant；集合身份绑定模型、revision、维度、距离、payload schema 和 schema version。
- LLM：默认离线 Mock；真实请求仅经 OpenAI-compatible Provider，并从 `KG_CRAG_LLM_*` 设置读取凭据。
- Prompt：`prompts/dense-rag-v1.txt`，版本为模板 UTF-8 字节的 SHA-256。
- 配置：`configs/retrieval.yaml` 与 `configs/evaluation.yaml`；Dense 配置以 canonical JSON 计算哈希，凭据不参与。

Sentence Transformers 在第一次实际 embed 时才导入并可能下载权重。dry-run、导入模块和默认测试均不下载模型、不连接 LLM；Qdrant 集成测试只有设置 `KG_CRAG_RUN_QDRANT_TESTS=1` 才连接数据库。

## 初始化与 dry-run

```powershell
python -m pip install -e ".[dev]"
docker compose up -d qdrant

# 读取 processed 产物和 Qdrant 状态，只输出预计差异
python scripts/build_vector_index.py --pilot --dry-run

# 不连接 Qdrant、模型或 LLM，只校验查询参数和版本
python scripts/query_dense_rag.py "CAMEL 如何组织智能体协作？" --top-k 10 --dry-run

# 不连接外部服务，校验 20 题、语料快照和目标 Chunk
python scripts/evaluate_dense_rag.py --dry-run
```

`--paper-id` 可重复；完整选择必须写成 `--all --limit N` 且 `N` 不超过配置上限。过滤使用 `--filter paper_id=...`、`section=...` 或 `processing_version=...`，未知字段在 Embedding 前拒绝。

## 建索引、查询与评测

```powershell
# 首次运行会加载/下载固定 revision；重复运行跳过未变化 Chunk
python scripts/build_vector_index.py --pilot

# 只有明确接受删除旧版本测试集合时才使用精确重建
python scripts/build_vector_index.py --pilot --rebuild

# 默认保存完整结果与最小 Trace；调试可加 --no-persist
python scripts/query_dense_rag.py "Voyager 的三个核心组件是什么？" --top-k 10

# 使用冻结的 20 题清单；单题失败不终止批次
python scripts/evaluate_dense_rag.py
```

退出码 `0` 表示成功，`1` 表示配置/预校验或整体运行失败，索引/评测的 `2` 表示批次完成但存在逐项失败。入口脚本只编排参数，公共逻辑都在 `src/kg_crag/`。

## 产物与恢复

- Embedding 缓存：`data/processed/embedding-cache/<namespace>/*.json`。
- 索引清单：`data/processed/index-runs/<run-id>/manifest.json`。
- 单次问答：`data/processed/query-runs/<run-id>/result.json`。
- 评测报告：`data/evaluation/results/dense-rag/<run-id>/report.json`。

缓存和结果采用临时路径校验后原子发布。逐篇索引先完成全部向量校验，再 upsert，最后只删除该论文的陈旧点；某篇失败不会阻止其他论文。集合不兼容时默认拒绝复用，不会隐式删除。回滚时保留旧集合及清单，恢复原配置后重新查询；只有显式 `--rebuild` 才重建当前精确集合。

## 验证与验收记录

```powershell
pytest -W error
ruff check .
ruff format --check .
mypy
python scripts/export_model_schemas.py --check
python scripts/verify_seed_corpus.py
docker compose config --quiet

# 连接本机测试 Qdrant，测试只清理唯一 qtest_* 集合
$env:KG_CRAG_RUN_QDRANT_TESTS="1"
pytest tests/integration/test_qdrant_dense_store.py -W error
```

截至 2026-09-28，固定 Python 依赖已安装并通过导入/冲突检查，离线 Mock 与 Qdrant 客户端内存模式闭环已验证。Docker Desktop daemon 后续已启动，但 Docker Hub 拉取 `qdrant/qdrant:v1.13.2` 连续返回 EOF；因此改用 Qdrant 官方 GitHub Release 的同版本 Linux 静态二进制，在本机 WSL 临时目录启动真实服务。验收覆盖唯一 `qtest_*` 集合的 schema、pilot dry-run、首次索引、幂等重跑、内容更新、过滤、论文去重、证据约束查询、陈旧点删除、精确重建和限定清理，结束时确认没有测试集合残留并停止临时服务。Compose 镜像启动仍受 Docker Hub 网络限制，不等同于失败的 Qdrant 适配器验收。

真实 BGE-M3 权重未在默认门禁下载，真实 LLM pilot 因未配置 `KG_CRAG_LLM_API_KEY` 未验证；不得把 Mock 指标表述成真实模型质量。完整门禁及 OpenSpec 验证的最终执行结果以 change 任务清单和交付说明为准。
