# KG-CRAG

面向科研文献问答的知识图谱增强自纠错 RAG Agent。当前仓库处于阶段 0：提供可演进的工程骨架、核心数据契约、外部服务接口、Mock 实现、基础 API、基础设施配置，以及可审计的公开论文语料获取工具。

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

## 项目结构

完整文件树、模块职责、关键脚本和按任务定位说明见仓库上级目录的 `../PROJECT_MAP.md`。新增、删除、重命名重要路径或改变模块职责时，应在同一变更中维护该地图。
