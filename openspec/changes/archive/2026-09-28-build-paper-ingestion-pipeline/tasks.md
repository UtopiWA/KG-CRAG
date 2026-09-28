# Tasks

## 1. 依赖、公共契约与配置

- [x] 1.1 在 `pyproject.toml` 固定 `pymupdf==1.28.2`，记录其 AGPL/商业双许可证及 Python 3.11/3.12 wheel 约束，并通过全新或现有虚拟环境安装及 `import pymupdf` 验证依赖可用
- [x] 1.2 新增 `models/ingestion.py` 的 pilot、页面/Block/Section、警告、解析/清洗文档、逐篇结果、运行清单和质量报告严格模型及公共导出，并用单元测试验证未知字段、枚举、页码/坐标、顺序、哈希和敏感详情边界
- [x] 1.3 为 `Chunk` 增加兼容默认的 `ordinal` 与 `processing_version`，更新全部公共使用方、Schema 导出注册表和快照，并运行模型、内存 Store 与 schema `--check` 测试确认兼容性
- [x] 1.4 扩展 `configs/default.yaml` 并实现严格摄取配置加载、跨字段约束、canonical JSON 配置哈希和 processing-version 计算，使用测试验证默认值、非法上限、切分参数关系、绝对路径排除及相同配置哈希稳定

## 2. 原始输入、元数据与 pilot 治理

- [x] 2.1 实现 raw PDF/元数据发现和预校验，复用现有签名/大小/SHA-256 规则并加入普通文件、根目录 containment 和全选预校验；用临时目录测试合法输入、路径穿越、符号链接/目录、缺失配对和哈希冲突，且运行 `scripts/verify_seed_corpus.py` 确认 110 组现有输入仍有效
- [x] 2.2 实现标题、作者、DOI、arXiv ID 与外部 ID 规范化及固定优先级 Paper ID 纯函数，使用表驱动测试验证 Unicode/空白、DOI URL、版本化 arXiv ID、历史 raw `paper_id` 兼容和冲突元数据失败
- [x] 2.3 渲染现有核心语料的候选页面并人工确认版面，不凭标题推断；创建恰好 15 篇、覆盖全部必需标签的 `configs/pilot_corpus.json`，通过清单模型、唯一性、标签覆盖和 raw 哈希解析测试
- [x] 2.4 新增 `configs/pilot_review.json` 的初始待复核结构及加载/门禁校验，测试清单哈希、processing-version、逐篇结论或必需标签覆盖任一不匹配时完整语料门禁失败

## 3. Parser Protocol 与 PDF 适配器

- [x] 3.1 定义 runtime-checkable `DocumentParser`、`ParseRequest` 及确定性测试替身，在 `tests/fixtures/ingestion/` 增加可提交的小型 PDF/raw 元数据 fixture，并以运行时 Protocol 检查和离线测试验证接口
- [x] 3.2 实现 PyMuPDF 文档打开、加密/损坏/页数/大小/低文本限制、页级文本块和三位小数坐标提取，使用有效、损坏、加密、空文本和超限 fixture 验证逐篇结构化错误及资源正确关闭
- [x] 3.3 实现 full-width/至多双栏的确定性 Block 排序、稳定 Block ID、字体统计与复杂版面警告，使用合成双栏、跨栏、重叠、图像/表格/公式 fixture 验证顺序、位置溯源和受控降级
- [x] 3.4 实现标题/章节识别和 `__document__` 回退，测试编号标题、字号/粗体标题、跨页章节、无标题文档及重复运行的 Section/Block 字节稳定性

## 4. 清洗与章节感知切分

- [x] 4.1 实现不修改 ParsedDocument 的清洗阶段骨架、Unicode NFC/受控字符处理、修改统计和 `source_block_ids` 映射，用 fixture 验证输入对象不变、输出确定及非法控制字符不泄漏
- [x] 4.2 实现仅作用于顶部/底部 10% 且满足至少 3 页和 60% 覆盖率的页眉页脚规则，测试高置信重复噪声被移除、正文重复和低覆盖内容被保留并记录统计/警告
- [x] 4.3 实现软连字符、受限换行连字符、空白规范化和仅相邻 Block 去重，使用自然语言、代码、公式、非 ASCII 和非相邻重复 fixture 验证无损回退与溯源保留
- [x] 4.4 实现 `regex-v1` 确定性 Token 计数器及章节内段落/句子/Token 边界窗口，测试目标/最小/最大/重叠参数、短段合并、禁止跨 Section 和不可分割超长单元警告
- [x] 4.5 实现内容 SHA-256、稳定 Section 键、Chunk ordinal/页码范围/processing-version 与 Chunk ID，使用重复运行和单项输入/解析器/配置变化测试验证稳定性、版本隔离、无空 Chunk 及页码单调

## 5. 版本化产物、Pipeline 与 CLI

- [x] 5.1 实现 Windows 安全的 `paper-<hash>` 路径、排序键 UTF-8 JSON/JSONL 序列化、同级临时目录和原子发布，测试文件名无非法字符、写入失败不留下成功产物且已发布旧版本不受影响
- [x] 5.2 实现产物哈希校验、processing-version 目录复用与 `--force` 重建语义，测试完整匹配时幂等跳过、缺失/损坏/非成功清单时重建，以及删除 interim/processed 后产物字节一致
- [x] 5.3 实现单篇 Pipeline 的规范化→解析→清洗→切分→质量统计→发布流程，使用 Mock Parser 和 PDF fixture 验证 Paper、Parsed/Cleaned、Chunk、Quality 全链路字段及 raw 目录在运行前后字节不变
- [x] 5.4 实现有界批处理、逐篇异常到安全 ErrorDetail 的转换、成功/跳过/警告/失败统计及最后写入的运行清单，测试一篇失败不终止其余论文、清单无绝对路径/凭据/PDF 正文且部分失败返回预期状态
- [x] 5.5 实现 pilot 自动质量指标和人工 review 门禁，测试页数、字符、空页、警告、Section/Block/Chunk、Token 分布、页码覆盖统计，以及过期/失败 review 默认阻止完整语料选择
- [x] 5.6 新增薄入口 `scripts/ingest_corpus.py`，实现默认 pilot、`--pilot`、可重复 `--paper-id`、`--all`、`--limit`、`--config`、`--dry-run`、`--force` 和输出根覆盖；用 CLI 测试精确验证互斥选择、1/200 安全边界、dry-run 零写入及退出码 0/1/2

## 6. 集成验证与真实语料验收

- [x] 6.1 增加从 raw 风格 fixture 到 interim/processed 的离线集成测试，验证首次运行、幂等重跑、配置变化的新版本、单篇失败隔离和删除派生层重建；确认默认 pytest 不读取 110 篇语料、不访问网络
- [x] 6.2 对 15 篇 pilot 先运行 CLI dry-run，再执行实际摄取；逐篇校验 JSON/JSONL、产物哈希、页码覆盖和稳定重跑，要求全部生成有效 ParsedDocument、Paper 与 Chunk，否则修复根因并重跑后才完成任务
- [x] 6.3 按每种必需版面标签抽检原 PDF 与 parsed/cleaned/chunk 产物，记录页码错位、正文缺失、跨栏乱序和警告准确性；只有未发现系统性问题时才把 `configs/pilot_review.json` 更新为通过并验证门禁
- [x] 6.4 在 pilot 门禁通过后显式执行有界 `--all --limit 110` 试跑，保存并复核逐篇失败/警告和总体统计，确认批次能够完成且失败隔离有效；任何系统性版面或资源问题必须修复或在 change 中明确为阻塞，不能只忽略失败样本

## 7. 文档、地图与完整门禁

- [x] 7.1 更新 README、`docs/data_schema.md` 和新增摄取操作说明，准确记录初始化、pilot/dry-run/单篇/全量命令、目录布局、ID/版本算法、质量门禁、可再生数据和 PyMuPDF 许可证，并逐条执行示例中的只读或 dry-run 命令
- [x] 7.2 更新 `../PROJECT_MAP.md`，列出新增模型、ingestion 模块、配置、CLI、fixtures、测试和文档，并对照实际文件系统验证所有路径与模块职责
- [x] 7.3 运行 `pytest -W error`、`ruff check .`、`ruff format --check .`、`mypy`、`scripts/export_model_schemas.py --check`、`scripts/verify_seed_corpus.py` 和 `docker compose config --quiet`，记录真实语料或 Docker 环境警告且确认默认测试完全离线
- [x] 7.4 运行 `openspec validate build-paper-ingestion-pipeline --type change --strict --no-interactive`，复核 proposal、paper-ingestion spec、design、tasks 与实现/验收结果一致，并确认没有把向量、稀疏或图索引吸收到本 change
