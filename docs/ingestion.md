# 论文摄取操作手册

## 边界与前提

摄取入口只读取本地 `data/raw/metadata/*.json` 与配对 PDF，不联网，也不会修改原始文件。开始前先运行：

```bash
python scripts/verify_seed_corpus.py
python scripts/ingest_corpus.py --pilot --dry-run
```

每条输入必须通过普通文件、可信根目录 containment、`%PDF-` 签名、大小和 SHA-256 校验。`configs/default.yaml` 限制单批最多 200 篇、单文件 100 MiB 和单文档 500 页。

## 选择与运行

选择器恰好采用一种：无选择器或 `--pilot` 处理固定试点；可重复 `--paper-id` 处理明确论文；`--all` 处理完整语料。`--limit` 必须在 1–200 之间。

```bash
python scripts/ingest_corpus.py --dry-run
python scripts/ingest_corpus.py --pilot
python scripts/ingest_corpus.py --paper-id arxiv:2312.07559 --force
python scripts/ingest_corpus.py --all --limit 110
```

`--all` 在开始前验证 `configs/pilot_corpus.json`、`configs/pilot_review.json`、每篇 pilot 的 processing-version、自动质量报告及产物哈希；任一项过期都会拒绝完整语料运行。更改解析器、内容配置或输入 PDF 后，应先重跑 pilot、人工抽检并更新 review。

## 数据流和产物

处理顺序固定为：raw 预校验 → 元数据规范化 → PyMuPDF 解析 → 正文清洗 → 章节内切分 → 自动质量统计 → 原子发布。单篇失败会写为有限 `ErrorDetail`，不阻断其余论文。

```text
data/interim/paper-<sha256(paper_id)[:16]>/<processing-version>/
├── parsed_document.json
└── cleaned_document.json

data/processed/paper-<sha256(paper_id)[:16]>/<processing-version>/
├── paper.json
├── chunks.jsonl
└── quality_report.json

data/processed/runs/<run-id>/manifest.json
```

JSON 使用 UTF-8、排序键、固定缩进和末尾换行；JSONL 按 Chunk ordinal 排序。发布前先写同级临时目录并重读校验，完成后才替换目标版本；强制重建失败会恢复旧版本。

`processing_version = sha256(input_sha256 + parser_name + parser_version + config_hash)`。配置哈希只包含影响内容的规范化配置，不含机器路径和 CLI 选择范围。Paper ID 对新记录按 DOI、无版本 arXiv ID、其他外部 ID、规范化标题哈希取值；已有 `arxiv:<id>` raw 标识保持兼容。Block、Section 和 Chunk ID 都由内容与结构字段确定，不含时间或随机数。

## 质量与人工复核

`quality_report.json` 记录页数、字符数、空文本页、Block/Section/Chunk 数、警告分布、Chunk Token 分布、页码覆盖率和各核心产物哈希。使用以下只读命令重读所有模型、JSONL、哈希和页码范围：

```bash
python scripts/verify_ingestion_outputs.py --pilot
python scripts/verify_ingestion_outputs.py --all
```

人工复核必须同时检查原 PDF 与 parsed/cleaned/chunk，至少覆盖普通文本、双栏、表格、公式、跨栏元素和异常版面。`scripts/render_pilot_candidates.py` 可把指定 arXiv ID 的代表页渲染到 `tmp/pdfs/`；这些临时 PNG 检查后应删除。

## 最近验收记录

2026-09-28 的 `pymupdf 1.28.2 + adapter-v1` 验收结果：

- pilot dry-run 后实际生成 15/15，修复独立空格 span 被过滤的问题后，新版本再次 15/15 成功；紧接的重跑 15/15 均经哈希验证后跳过。
- pilot 共 458 页、8361 个清洗 Block、1014 个 Chunk，页码覆盖及模型/哈希重读无失败。
- 人工抽检全部必需版面标签，未发现系统性页码错位、正文缺失或跨栏错序；review 已绑定 15 个处理版本并通过。
- 门禁后运行 `--all --limit 110`：15 篇跳过、95 篇新成功、0 失败；完整重读共 2848 页、55229 个 Block、8407 个 Chunk，无无效哈希或页码范围。
- MuPDF 对两篇文件输出了颜色空间语法诊断，但文本解析、质量报告和最终产物验证均成功；若未来需要抽取图片，应重新评估这些 PDF。

原始 PDF、interim、processed 和运行清单均为本地可再生数据，默认不提交 Git。若要清理派生层，只删除明确的 `data/interim`/`data/processed` 版本目录，不触碰 `data/raw`。
