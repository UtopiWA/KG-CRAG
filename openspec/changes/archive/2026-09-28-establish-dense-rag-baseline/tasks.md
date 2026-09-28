# Tasks

## 1. 依赖、配置与公共契约

- [x] 1.1 选定并在 `pyproject.toml` 固定兼容的 Qdrant 客户端、Sentence Transformers 和 OpenAI-compatible Provider 依赖，补充 `docs/dependencies.md` 的许可证、模型修订、Python 3.11/3.12 与首次下载约束，并在项目虚拟环境完成安装、导入及依赖冲突检查
- [x] 1.2 扩展 `Settings`、`.env.example`、`configs/retrieval.yaml` 和 `configs/evaluation.yaml`，实现拒绝未知字段的 Dense/Embedding/集合/索引/生成/评测配置模型、跨字段上限及 canonical JSON 哈希；用单元测试验证默认离线配置、凭据不进入哈希、非法维度/批量/候选/上下文上限和配置漂移
- [x] 1.3 新增集合身份、索引运行、Evidence 排名/内外部标记、Citation、AnswerClaim、DenseRAGResult、pilot 问题和评测报告严格模型，保持现有 Chunk/Evidence 输入兼容；更新公共导出与 JSON Schema 快照，并通过模型、历史 fixture 和 `export_model_schemas.py --check` 测试
- [x] 1.4 增加仓库内版本化 Dense RAG Prompt 及其稳定版本标识，测试模板只接受问题和带编号 Evidence、输出格式约束完整且 Prompt 变更会产生新版本

## 2. Embedding Provider、批处理与缓存

- [x] 2.1 实现惰性加载的本地 Sentence Transformers Embedding 适配器，固定模型修订、query/document 前缀、归一化与输出顺序；用替身模型测试导入零下载、批量顺序、空输入、加载失败和维度错误，而不在默认测试下载权重
- [x] 2.2 实现 OpenAI-compatible LLM Provider 适配器，确保 SDK 只存在于 provider 层、凭据只来自设置且异常转换为不含请求正文/认证信息的 `ErrorDetail`；用伪客户端测试成功、超时、限流、非法响应和日志脱敏
- [x] 2.3 实现 Embedding 有界批处理与完整批次校验，测试数量、顺序、维度、NaN/Infinity、批大小上限及某篇失败不会产生部分可用向量
- [x] 2.4 实现位于可再生数据层的内容寻址 Embedding 缓存，采用模型/修订/规范化/配置隔离和临时文件原子发布；测试命中不调用 Provider、配置变化未命中、损坏缓存重算、写入失败不留下成功项和并发目标冲突安全失败

## 3. Vector Store 契约与 Qdrant 后端

- [x] 3.1 按集合身份、点状态检查和论文级同步需要扩展后端无关 Vector Store 契约及内存实现，保持原有调用方兼容；以共享契约测试验证幂等 upsert、稳定点标识、精确过滤、按论文陈旧点删除和非法参数原子失败
- [x] 3.2 实现集合版本、名称、payload schema 与语料快照纯函数，测试 canonical 输入顺序不影响哈希，模型/修订/维度/距离/payload 任一变化会改变集合版本，而语料变化只改变快照
- [x] 3.3 实现 Qdrant 集合创建、兼容性检查、payload 索引和显式精确重建，测试兼容复用、不兼容默认拒绝、禁止通配/任意删除、建成后校验及重建失败不把不完整集合标为可用
- [x] 3.4 实现 Qdrant 的批量 upsert、点状态读取、论文级陈旧点删除、白名单过滤和相似度检索，将 SDK 异常映射为统一可重试/不可重试错误；用伪客户端覆盖分页、同分排序、超时、部分响应、未知过滤和敏感错误正文不泄漏
- [x] 3.5 增加显式启用的 Qdrant 集成测试，使用唯一测试集合验证 schema、重复 upsert、更新、过滤、搜索与限定清理；确认未启用环境变量时默认 pytest 不连接数据库且测试明确跳过

## 4. 已发布 Chunk 读取与索引流水线

- [x] 4.1 实现只读 processed Chunk 发现、pilot/显式论文/有界全量选择和产物哈希重验，复用摄取目录与严格模型；用临时目录测试合法选择、重复/未知论文、路径越界、缺失/损坏 JSONL、哈希冲突和最大论文数
- [x] 4.2 实现索引 dry-run 规划，准确计算语料快照、集合身份以及预计新增/更新/跳过/删除数量；以记录调用的 Mock 证明 dry-run 对缓存、Provider 和 Vector Store 零写入
- [x] 4.3 实现以论文为隔离单元的增量索引服务，先完整校验与向量化、再 upsert、最后删除该论文陈旧点；测试首次运行、相同内容幂等跳过、内容变化更新、陈旧点收敛、单篇失败继续和 Qdrant 删除失败被记录
- [x] 4.4 实现索引运行清单、逐篇状态、计数、时延、有限错误和相对产物路径的稳定序列化与原子发布，测试部分失败退出状态、重读校验、绝对路径/凭据/Chunk 正文不进入错误详情及半写回滚
- [x] 4.5 新增薄入口 `scripts/build_vector_index.py`，支持 dry-run、pilot、可重复 `--paper-id`、有界 `--all --limit`、`--config` 和精确 `--rebuild`；CLI 测试验证互斥选择、无界请求拒绝、目标预览、退出码和脚本不复制包内逻辑

## 5. Dense Retriever 与 Evidence 归一化

- [x] 5.1 实现 Dense Retriever 的查询校验、Embedding、单次候选检索、允许字段过滤、稳定排序与 `top_k`，用 Mock/内存测试验证一次调用、空查询/非法 K/未知过滤提前失败及后端超时的可重试错误
- [x] 5.2 实现受 `candidate_multiplier` 和 `max_candidates` 约束的可选按论文去重，测试每篇只保留最高排名、同分按 Chunk ID 稳定排序、候选不足不循环补取且最终数量不超过 `top_k`
- [x] 5.3 将命中规范化为稳定内部 `Evidence`，保留 Dense 分数/排名、集合版本、Chunk/Paper、章节、页码、内容哈希并排除 SDK 私有字段；通过 round-trip 测试确认所有引用可解析回 processed Chunk 且 `external=false`

## 6. 证据约束回答、引用与 Trace

- [x] 6.1 实现稳定 Evidence 编号和受 `max_evidence`、单条字符数、总字符数约束的上下文构建器，测试输入不被修改、排序确定、截断可审计、零证据和超限场景
- [x] 6.2 实现严格结构化 LLM 输出解析、AnswerClaim 类型和 Citation 映射校验，测试合法多引用、未知/重复引用、无引用事实、非法 JSON、超长响应和引用位置完整性
- [x] 6.3 实现 Dense RAG 查询服务：一次检索、门槛筛选、至多一次生成、确定性证据不足结果和最小 `TraceEvent v1`；用 Mock 测试充分证据、空/低分证据、生成失败、无隐藏重试、Dense 路径、低置信度与 `external=false`
- [x] 6.4 实现问答结果和 Trace 的稳定原子保存及重放标识，测试固定 Mock 下 Evidence/引用/结果字节一致，配置/Prompt/模型/索引变化形成新标识，失败不留下成功清单且 Trace 不含 Prompt 全文、正文或凭据
- [x] 6.5 新增薄入口 `scripts/query_dense_rag.py`，支持问题、配置、过滤、`top_k`、去重、dry-run/结果输出选项；CLI 测试验证解析后摘要、离线 Mock 注入、证据不足退出语义、非法目标拒绝和非交互调用

## 7. pilot 标注、指标与基线运行

- [x] 7.1 实现 pilot 问题清单加载、语料快照绑定和目标 Chunk 解析校验，测试不少于 20 题、唯一 ID、非空目标、重复目标归一、缺失 Chunk 与快照漂移在评测前失败
- [x] 7.2 从已验证 pilot 论文的实际 processed Chunk 人工编写并复核至少 20 个问题及目标 Chunk 标注，覆盖事实和语义检索且不复制大段正文；运行清单校验命令并抽查每个目标的论文、章节和页码可回溯
- [x] 7.3 实现确定性的 Recall@K、MRR、nDCG@K、citation precision 和 citation recall，使用手算 fixture 测试多目标、无命中、空引用、重复命中、失败题和 K 边界
- [x] 7.4 实现有界评测运行器和逐题失败隔离，保存每题 Evidence、回答、引用、时延、状态以及含失败题的汇总；测试一题失败后继续、指标分母不丢样本、baseline ID 版本化和结果目录原子发布
- [x] 7.5 新增薄入口 `scripts/evaluate_dense_rag.py`，支持配置、问题清单、最大题数和 dry-run；CLI 测试验证默认有界、冻结条件摘要、无效标注零调用、退出码以及同一 baseline ID 禁止静默覆盖
- [x] 7.6 增加从临时 processed fixture 经内存索引、Dense 检索、Mock 生成到评测报告的离线集成测试，验证首次运行、幂等重跑、证据不足、单题失败和默认零网络

## 8. 真实服务验收、文档与完整门禁

- [x] 8.1 更新 README、`docs/architecture.md`、`docs/data_schema.md`、`docs/experiments.md` 和新增 Dense RAG 操作说明，准确记录初始化、dry-run/索引/查询/评测命令、版本算法、产物布局、外部服务边界、重建与回滚，并逐条执行文档中的只读或 dry-run 示例
- [x] 8.2 更新 `../PROJECT_MAP.md`，列出新增模型、Provider、索引/检索/生成/评测模块、Prompt、配置、脚本、fixture、测试和文档，并对照实际文件系统复核职责与实现状态
- [x] 8.3 运行 `pytest -W error`、`ruff check .`、`ruff format --check .`、`mypy`、`scripts/export_model_schemas.py --check`、`scripts/verify_seed_corpus.py` 和 `docker compose config --quiet`，确认默认测试不访问网络/模型/Qdrant并记录任何环境警告
- [x] 8.4 显式启动或连接测试 Qdrant，先运行索引 dry-run，再对有界 fixture/pilot 执行首次索引、重复索引、查询、过滤、去重和精确重建验收；核对点数、payload、引用回溯、运行清单并只清理明确测试集合
- [x] 8.5 在可用环境中显式运行固定修订的真实 Embedding 与 LLM pilot 基线，保存至少 20 题的逐题结果和指标；若模型下载、凭据或在线服务不可用，记录未验证项而不伪造通过结论，且离线完成条件仍须全部满足
- [x] 8.6 运行 `openspec validate establish-dense-rag-baseline --type change --strict --no-interactive`，逐项复核 proposal、spec、design、tasks 与实现/验收记录一致，并确认没有引入 BM25、融合、重排、Graph、动态路由、纠错/反思、Web、API 或 UI 范围
