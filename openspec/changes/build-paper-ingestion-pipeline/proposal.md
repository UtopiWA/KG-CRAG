# Proposal

## Why

KG-CRAG 已有 110 组通过哈希校验的原始论文 PDF 与元数据，但尚不能把它们确定性地转换为可供检索使用、能回溯到论文页码的结构化文档和 Chunk。迭代 01 需要先在代表性 pilot 子集上建立可审计的解析、清洗、切分与失败隔离流水线，再验证对完整语料的有界扩展能力。

## What Changes

- 从现有 110 篇校验通过的语料中冻结 10–20 篇代表性 pilot 清单，记录普通文本、双栏、表格、公式和异常版面等人工标签与选择依据。
- 定义解析输入、页面、Section、Block、ParseWarning、ParsedDocument、摄取结果和运行清单等严格公共契约，以及可替换的 Parser Protocol。
- 增加基于 PyMuPDF 的首个 PDF 解析适配器，保留页码、块顺序、位置和解析警告；对加密、损坏、空文本及疑似复杂版面显式失败或降级。
- 增加元数据规范化与稳定 Paper ID 规则，并把原始元数据、PDF SHA-256 和外部标识保留在可审计产物中。
- 增加确定性的正文清洗和章节感知切分，支持页眉页脚清理、软连字符修复、异常字符处理、相邻重复内容清理、重叠窗口、版本化 Token 统计、稳定 Chunk ID 与内容哈希。
- 增加可重复执行的摄取编排和 `ingest_corpus.py` CLI，支持 `--dry-run`、pilot、单篇、批量、有界完整语料试跑、增量跳过及逐篇失败报告。
- 将中间态、处理结果和运行清单写入 `data/interim` 与 `data/processed` 的版本化路径；相同输入与配置安全复用，输入、解析器或配置变化生成新处理版本而不覆盖原结果。
- 补充模型 Schema 快照、离线 fixture、单元/集成测试、人工抽检清单、README、数据模式文档和项目地图。

## Non-goals

- 不建立 Dense、Sparse/BM25、向量数据库或知识图谱索引。
- 不承诺完美恢复复杂公式、表格语义、图片内容或所有多栏版面；无法可靠恢复时必须保留警告和定位信息。
- 不执行 OCR，也不接入在线解析服务；扫描件只记录可诊断的低文本或空文本结果。
- 不修改、覆盖或删除 `data/raw` 中的 PDF 与原始元数据。
- 不在默认单元测试中处理完整 110 篇真实语料或访问网络。

## Capabilities

### New Capabilities

- `paper-ingestion`: 定义 pilot 语料治理、PDF 解析、元数据规范化、正文清洗、章节感知切分、稳定溯源、版本化产物、幂等重跑和逐篇失败隔离行为。

### Modified Capabilities

无。现有 `foundation-contracts` 继续提供严格模型、结构化错误和确定性离线测试要求，本 change 在其上新增摄取领域能力。

## Impact

- 关联项目规格：`ING-001`–`ING-007`、稳定标识、架构约束、可复现性、可靠性和安全与隐私要求。
- 迭代依据：`../guidebooks/iteration-01-ingestion.md`；依赖已归档并生效的 `foundation-contracts`。
- 预计影响 `models/`、`ingestion/`、`configs/`、`scripts/`、`data/interim`、`data/processed`、测试、Schema 快照、README、`docs/data_schema.md` 和 `PROJECT_MAP.md`。
- 新增固定版本的 PyMuPDF 运行时依赖；切分 Token 统计采用仓库内版本化、无需下载模型的确定性计数器，避免默认测试依赖网络或模型缓存。
- 不改变现有论文采集脚本、原始元数据格式、Retriever/Store Protocol 或 API 路径。

## Dependencies and Risks

- pilot 选择依赖当前 110 组 PDF/元数据继续通过独立哈希校验；缺失或损坏记录必须在摄取前失败，不能由解析器绕过。
- PyMuPDF 的块顺序不能覆盖所有复杂版面，尤其是双栏、跨栏标题、公式和表格；实现需采用确定性排序、复杂版面告警和逐页人工抽检，而不是静默生成低可信文本。
- 清洗规则可能误删有效重复内容或错误连接连字符；每类规则必须有输入输出 fixture，并在产物中记录规则版本和警告。
- 完整语料试跑可能暴露长文档、内存或异常 PDF 问题；CLI 必须提供论文数量和单文件页数/大小上限，并让单篇失败不影响其余论文。
