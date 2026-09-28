# Design

## Context

当前 `data/raw` 已有 110 组 PDF/元数据，采集阶段能够验证 PDF 签名、大小和 SHA-256；`ingestion/` 只包含 arXiv/OpenAlex 发现与本地索引辅助逻辑，尚无正文 Parser、清洗、切分或摄取运行模型。`Paper` 与 `Chunk` 已是跨模块公共模型，`Chunk` 具备论文、章节、页码、文本、Token 数和内容哈希，但缺少显式顺序和处理版本。参见 `proposal.md` 的动机以及 `specs/paper-ingestion/spec.md` 的行为要求。

实现必须保持 `data/raw` 只读、Windows/Linux 路径兼容、Python 3.11/3.12、严格 mypy 和默认测试完全离线。现有采集脚本及 raw 元数据格式继续兼容；中间态和处理产物均是可删除重建、默认不提交 Git 的派生数据。

## Goals / Non-Goals

**Goals:**

- 以小而明确的领域模型分离原始输入、解析结果、清洗结果、Chunk 和运行审计状态。
- 对相同输入、解析器版本和规范化配置产生确定性内容与稳定 ID。
- 对复杂版面采用“保留文本与位置、显式警告、人工门禁”的可诊断降级策略。
- 让单篇、pilot 和有界批量共享同一流水线，并通过版本目录和原子写入安全重跑。
- 为后续索引 change 提供只依赖公开 `Paper`/`Chunk` 契约的 UTF-8 JSON/JSONL 产物。

**Non-Goals:**

- 不把内存中的版面启发式抽象为通用排版引擎，也不加入 OCR、在线解析 API 或模型下载。
- 不让摄取流水线直接调用 Vector Store、Graph Store、Embedding 或 Retriever。
- 不保证跨不同 PyMuPDF/MuPDF 版本产生相同解析字节；解析器版本是处理版本的一部分。
- 不把运行时间戳等非确定字段写入可复用的 Paper、ParsedDocument 或 Chunk 内容；它们只进入运行清单。

## Decisions

### 1. 新增独立摄取契约并兼容扩展 Chunk

新增 `src/kg_crag/models/ingestion.py`，由 `models/__init__.py` 导出以下严格模型和枚举：

- `LayoutLabel`、`PilotPaper`、`PilotManifest`：pilot 的稳定标识、版面标签和选择依据；
- `BoundingBox`、`BlockKind`、`DocumentBlock`、`DocumentPage`、`DocumentSection`：一基页码、稳定顺序、坐标与结构；
- `WarningSeverity`、`ParseWarningCode`、`ParseWarning`：警告代码、页码、Block ID 和安全详情；
- `ParsedDocument` 与 `CleanedDocument`：输入哈希、解析器信息、页面/章节、清洗规则版本、Block 溯源和统计；
- `IngestionStatus`、`IngestionItemResult`、`IngestionRunManifest`、`QualityReport`：逐篇状态、产物哈希、有限错误、批次统计和门禁结果。

`DocumentBlock.block_id` 由输入哈希、页码、块顺序、规范化坐标和原始文本哈希计算；清洗后的块保存 `source_block_ids`，Chunk 通过页码范围和处理版本继续回溯。坐标统一为 PDF page space 的 `(x0, y0, x1, y1)`，输出前四舍五入到三位小数，避免无意义浮点漂移。

现有 `Chunk` 增加带兼容默认值的 `ordinal >= 0` 和非空 `processing_version`，摄取流水线始终显式赋值；已有测试构造器和离线 Store 无需立刻修改调用签名。`Paper.ingestion_version` 保存本次处理版本。模型变化后同步公共导出、JSON Schema 注册表与快照。

选择独立文件而不是继续扩大 `domain.py`，是因为解析页面/块和运行清单属于摄取边界，后续检索只需要 `Paper`/`Chunk`。不新建另一套 `IngestedChunk`，避免索引阶段在相似模型之间转换和产生字段漂移。

### 2. pilot 清单固定为 15 篇并与运行时质量记录分离

新增 `configs/pilot_corpus.json`，从现有哈希有效语料中选择 15 篇，使用 `arxiv_id` 作为清单引用键；每条记录包含 `layout_labels`、`selection_reason` 和人工复核状态。标签枚举至少包含 `plain_text`、`two_column`、`table`、`formula`、`full_width_element` 与 `anomalous_layout`，允许一篇具有多个标签。实施时必须先渲染候选页面确认标签，不能仅凭题目猜测。

选择 15 篇是 10–20 范围的中点，足以覆盖版面组合，同时适合每次本地回归和人工抽检。原始 PDF 不提交；清单只保存稳定标识，不保存机器绝对路径。

自动质量统计进入对应运行目录的 `quality_report.json`，人工结论写入版本化 `configs/pilot_review.json`，以 `processing_version` 和清单哈希绑定。完整语料模式要求两者匹配且结论为通过；这样自动生成的数据仍可删除重建，而放行决定可评审、不会因清空 `processed` 丢失。

### 3. Parser Protocol 隔离 PyMuPDF 适配器

新增 `ingestion/base.py` 中的 `DocumentParser` runtime-checkable Protocol，核心方法为同步的 `parse(pdf_path: Path, request: ParseRequest) -> ParsedDocument`。PDF 解析是本地 CPU/文件操作，无需为了尚不存在的异步编排引入线程或异步接口。`ParseRequest` 包含 Paper ID、输入 SHA-256 和已校验的页数/大小上限；路径在进入 Parser 前已经解析并确认位于 raw PDF 根目录。

首个实现位于 `ingestion/pdf.py`，固定使用 `pymupdf==1.28.2` 和 `import pymupdf`。该版本要求 Python 3.10+，官方为 Python 3.10–3.14 提供 wheels，符合项目范围；依赖及其 AGPL/商业双许可证必须在依赖评审中显式记录。版本依据见 [PyMuPDF PyPI](https://pypi.org/project/pymupdf/) 和 [官方安装说明](https://pymupdf.readthedocs.io/en/latest/installation.html)。

适配器使用文本字典中的块、行和 span 构造 Block，并遵循以下稳定规则：

1. 页码转为一基；忽略纯空白 span，但保留非空块坐标和字体统计。
2. 先识别横跨大部分页宽的 full-width 块，再按配置的列间距阈值把其他块分为至多两列；同一区域以列、`y0`、`x0`、原始块序号排序。
3. 检测到三列以上、重叠块、跨栏歧义、图像占比高、疑似表格或公式时保留文本并写警告，不尝试声称语义结构已恢复。
4. 根据编号模式、字号相对中位数、粗体和短行规则识别标题；无法识别时使用单一 `__document__` Section，不丢失正文。
5. 在关闭文档前读取全部必要信息；不把 PyMuPDF 对象或库专有字段写入公共模型。

选择轻量的确定性版面启发式而非 PyMuPDF4LLM、深度学习解析器或在线服务，是为了控制依赖、保证离线测试并保留适配器替换空间。未来复杂解析器必须实现相同 Protocol，并以新的解析器名称/版本产生独立处理版本。

### 4. 元数据规范化独立于采集格式

新增 `ingestion/metadata.py`，把现有 raw JSON 映射到 `Paper` 和摄取来源记录：

- DOI 去除 `https://doi.org/`/`doi:` 前缀并小写；arXiv ID 去版本；作者名和标题执行 Unicode NFC、空白折叠但保留显示大小写；
- Paper ID 优先使用 `doi:<canonical-doi>`，其次 `arxiv:<versionless-id>`、受支持外部 ID，最后 `title:<sha256>`；
- raw 中现有 `paper_id` 视为采集阶段的旧标识并保留在来源记录中，不因其使用 `arxiv:` 而拒绝含 DOI 的合法记录；只有外部 ID 彼此矛盾、文件配对错误或哈希不符才失败；
- 未知 raw 字段不直接进入 `Paper`，但完整 raw 元数据 SHA-256 和允许的来源字段进入运行审计。

稳定 ID 函数必须是纯函数并单独测试。选择重新派生 canonical ID 而非信任 raw `paper_id`，是为了满足项目级 ID 优先级且兼容早期采集脚本生成的 `arxiv:` 标识。

### 5. 清洗生成新文档，不原地改写解析结果

新增 `ingestion/cleaning.py`，输入 `ParsedDocument`、输出 `CleanedDocument`。原始解析产物单独保存，便于定位规则错误。规则按固定顺序执行：Unicode NFC 与受控字符替换、页级行重建、页眉页脚检测、软连字符移除、受限换行连字符连接、相邻重复块折叠、空白规范化。

默认页眉页脚检测只检查页面顶部/底部 10% 区域；同一规范化签名至少出现在 3 页且覆盖不低于 60% 的可用页面才删除。换行连字符仅在前后均为字母、下一片段以小写字母开始且不命中公式/代码特征时连接。相邻去重仅折叠连续且文本签名相同的块，不执行全文全局去重。

每条规则返回修改计数、源 Block ID 和警告；清洗后块保存全部源 Block ID。选择显式阶段模型而不是对字符串做不可见原地处理，使规则能用小 fixture 独立测试，也让人工抽检能对比 parsed/cleaned 产物。

### 6. 切分采用版本化正则 Token 计数和章节内窗口

新增 `ingestion/chunking.py`。首版计数器标识为 `regex-v1`，使用 Unicode 单词或单个非空白标点单元进行计数，不依赖模型文件或网络。它不是未来 Embedding 模型的真实 tokenizer，因此配置与产物必须记录标识；后续替换计数器会自然产生新处理版本。

切分配置沿用当前 `target_tokens=500`、`min_tokens=300`、`max_tokens=700`、`overlap_tokens=75`，并校验 `0 <= overlap < min <= target <= max`。算法逐 Section 处理，优先按段落、其次句子、最后计数单元边界组装窗口；短小相邻单元在同一 Section 内合并，普通 Chunk 不跨 Section。单个不可分割单元超过上限时单独保留并写警告，不能截断或丢弃。

`content_hash = sha256(cleaned_text_utf8)`；`chunk_id` 对 canonical Paper ID、处理版本、Section 稳定键、序号、页码范围和内容哈希的规范串计算 SHA-256。Chunk 顺序只由文档结构和配置决定，不使用字典遍历、当前时间或随机数。

### 7. 单一规范化配置哈希决定处理版本

扩展 `configs/default.yaml` 的 `ingestion`、`parsing`、`cleaning` 与现有 `chunking` 段，并新增严格配置加载器。配置包含：

- `max_papers=200`、`max_pdf_bytes=104857600`、`max_pages=500`、最低每页/每文档字符数；
- 列检测、标题检测、页眉页脚和字符清洗阈值；
- Token 计数器和切分参数；
- pilot/review 清单相对路径及 interim/processed 根目录。

配置哈希基于校验后、排序键、无机器绝对路径的 canonical JSON；`processing_version = sha256(input_sha256 + parser_name + parser_version + config_hash)`。输出根目录和 CLI 选择范围不进入处理版本，避免同一内容因机器路径或批次大小产生不同 ID。

### 8. 版本目录使用 Paper ID 哈希键并原子发布

Paper ID 可能包含 `:`、`/` 等不适合 Windows 的字符，因此目录键固定为 `paper-<sha256(paper_id)[:16]>`，清单保存键到完整 Paper ID 的映射。目录布局为：

```text
data/interim/<paper-key>/<processing-version>/
├── parsed_document.json
└── cleaned_document.json

data/processed/<paper-key>/<processing-version>/
├── paper.json
├── chunks.jsonl
└── quality_report.json

data/processed/runs/<run-id>/manifest.json
```

JSON 使用 UTF-8、排序键、固定缩进和末尾换行；JSONL 按 `ordinal` 每行一个紧凑对象。每组论文产物先写到目标同级的唯一临时目录，逐文件重读校验和计算 SHA-256 后，以 `os.replace` 发布；失败时清理本次临时目录，不删除已发布版本。运行清单最后写入，只有其中逐篇状态为 `succeeded` 或哈希匹配的 `skipped` 才表示可复用。

运行 ID 使用 UTC 时间、配置哈希前缀和随机无敏感标识后缀，因此运行清单本身不承诺跨运行字节一致；同一处理版本下的 Parsed/Cleaned/Paper/Chunk/Quality 内容不得包含运行时间并必须一致。

### 9. Pipeline 与 CLI 共用选择、校验和失败策略

新增 `ingestion/pipeline.py` 负责单篇处理，`scripts/ingest_corpus.py` 只解析参数和汇总结果。选择模式恰好一个：`--pilot`（无选择器时的安全默认）、可重复 `--paper-id` 或 `--all`；共同参数包括 `--config`、`--limit`、`--dry-run`、`--force` 和输出根覆盖。`--all` 必须同时满足 pilot review 与自动质量报告通过，`--limit` 必须在 1 到 `max_papers` 之间。

执行顺序为：加载配置与选择器 → 校验全部被选 raw 输入 → dry-run 返回，或逐篇执行规范化、解析、清洗、切分、质量统计与原子发布 → 写运行清单。预校验避免错误清单造成部分写入；解析期的单篇错误由 Pipeline 捕获并转换为安全 `ErrorDetail`，批次继续。未知内部异常只在本地调试日志保留堆栈，清单保存稳定代码、安全消息和论文标识，不保存 PDF 内容、绝对路径或第三方响应正文。

退出码固定为：`0` 全部成功/合法跳过，`1` 参数、配置或选择预校验失败，`2` 批次完成但至少一篇失败。dry-run 不创建运行目录。脚本不访问网络，所有输入均来自本地显式目录。

### 10. 测试分层和 pilot 门禁

单元测试使用小型文本/模型 fixture 覆盖 ID、配置哈希、清洗规则、章节切分、边界值、路径约束和原子写入。解析适配器测试使用 `tests/fixtures/ingestion/` 中可提交的小型 PDF，另在测试临时目录构造加密、损坏和空文本输入；这些测试不读取 110 篇本地语料，也不联网。

集成测试从 raw 风格 fixture 运行到 interim/processed，验证重复运行、配置变化、单篇失败隔离和删除派生层后重建。真实 pilot 运行与人工页面抽检是操作验收，不进入默认 pytest；结果摘要记录于文档，但 PDF 和大体积产物继续由 Git 忽略。完整 110 篇只执行显式有界试跑，不作为 change 的默认测试前置条件。

## Risks / Trade-offs

- [PyMuPDF/MuPDF 小版本会改变块边界或排序] → 固定 `pymupdf==1.28.2`，把解析器版本纳入处理版本，升级时重新跑 pilot 并评审产物差异。
- [PyMuPDF 采用 AGPL/商业双许可证] → 在依赖与交付文档中记录许可证，课程分发方式需满足 AGPL；若后续交付策略不兼容，在实施前改用许可证合适且满足同一 Parser Protocol 的替代实现。
- [双栏启发式可能误排跨栏图注或侧栏] → 保留坐标、生成警告、要求覆盖标签的人工抽检；失败不阻塞其他论文，但不能通过全量门禁。
- [正则 Token 数与未来 Embedding tokenizer 不一致] → 显式记录 `regex-v1`，切分上限仅作为本迭代的确定窗口；索引迭代若更换 tokenizer 必须生成新处理版本。
- [严格 canonical Paper ID 与历史 raw `paper_id` 不一致] → 保留旧标识和全部外部 ID，目录使用 canonical ID 哈希；不重写 raw 元数据。
- [原子替换无法跨文件提供数据库事务] → 以版本临时目录、逐文件哈希和最后写入的运行清单作为发布协议；缺失清单或哈希不符一律重建。
- [pilot 人工放行可能过时] → review 文件绑定 pilot 清单哈希和处理版本；任一变化使门禁自动失效。

## Migration Plan

1. 加入固定 PyMuPDF 依赖、摄取模型、配置加载和 Schema 快照，不改变现有采集或运行入口。
2. 加入 Parser、规范化、清洗、切分及 fixture 测试，再实现 Pipeline 和 CLI；此时仍不处理真实 raw 语料。
3. 渲染并冻结 15 篇 pilot 清单，先 dry-run，再生成版本化产物和自动质量报告；人工抽检后写入与处理版本绑定的 review。
4. pilot 门禁通过后，以显式 `--all --limit 110` 执行完整语料试跑，记录失败、警告和统计，不提交 PDF 或派生产物。
5. 更新 README、数据模式、操作说明和项目地图，并执行 pytest、Ruff、mypy、Schema 与 OpenSpec 严格验证。

回滚时可移除新增入口、依赖和模块，并删除可再生的 `data/interim`/`data/processed` 产物；`data/raw` 不受影响。若只回滚某个解析/清洗版本，保留旧代码可读的运行清单，删除对应 processing-version 目录即可，不需要迁移原始语料。
