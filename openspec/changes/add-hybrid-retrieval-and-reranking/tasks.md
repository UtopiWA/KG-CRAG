# Tasks

## 1. 配置、公共模型与版本边界

- [x] 1.1 扩展严格检索配置，加入 Sparse schema/tokenizer/BM25/路径/查询上限、RRF/Weighted/归一化/权重、部分降级、Reranker 模型修订/设备/批量/候选和 Hybrid 输出上限；验证旧 Dense 配置仍可加载，非法权重、越界候选、工作区外路径和未知字段均在运行前失败
- [x] 1.2 更新 `configs/retrieval.yaml`、`configs/evaluation.yaml` 和非敏感环境模板，固定 `BAAI/bge-reranker-base` 修订 `2cfc18c9415c912f9d8155881c133215df768a70`、默认 CPU、top-20/top-8 与开发集矩阵；用配置哈希测试证明模型、分词、融合或上限变化会形成新版本且密钥/绝对缓存路径不参与哈希
- [x] 1.3 新增 Sparse 索引身份、逐篇索引结果、阶段摘要、成本摘要、Hybrid 检索结果和开发集报告严格模型，并更新公共导出；用模型测试覆盖状态组合、脱敏 `ErrorDetail`、候选/时延非负、降级信息和未知字段拒绝
- [x] 1.4 更新公共 JSON Schema 快照并运行 `python scripts/export_model_schemas.py --check`，确认既有 Dense Schema 与默认字段保持兼容且新增 Hybrid 模型可稳定重导出

## 2. Sparse Store 与索引生命周期

- [x] 2.1 新增 `SparseStore` Protocol 和确定性内存实现，支持集合身份、论文级记录状态、事务式同步、白名单过滤和有界搜索；以共享契约测试验证稳定排序、幂等更新、陈旧记录删除、空结果和非法参数原子失败
- [x] 2.2 实现 `scientific-unicode-v1` 规范化与分词、查询 Token/字符硬上限和安全转义 OR 查询构造；用缩写、`GPT-4`、`LLaMA-2`、下划线、数据集名、Unicode、空输入和 FTS 运算符注入 fixture 验证确定性
- [x] 2.3 实现 SQLite FTS5 schema、能力探测、索引身份与 manifest 校验，记录 Python/SQLite/FTS5、tokenizer 和 BM25 版本；用临时数据库测试首次创建、兼容复用、身份缺失/损坏/不兼容拒绝和显式精确重建
- [x] 2.4 实现 SQLite 论文级事务同步与分页/有界检索，先写新 Chunk 再删除该论文陈旧记录，并把 BM25 统一为分数越高越相关；测试事务回滚、重复同步、内容更新、过滤、同分 Chunk ID 排序和 Evidence 所需 payload 完整性
- [x] 2.5 实现 Sparse 索引 plan/run 服务，复用 processed Chunk 发现与哈希复验，支持 dry-run、pilot、重复 `--paper-id`、有界 `--all --limit`、检查点、单篇失败隔离和原子运行清单；测试 dry-run 零写入、部分失败继续及断点重跑只处理未完成或变化论文
- [x] 2.6 新增薄入口 `scripts/build_sparse_index.py`，验证参数互斥、无界全量请求拒绝、操作目标预览、退出码、精确重建确认和脚本不复制包内业务逻辑

## 3. Sparse Retriever 与统一 Evidence

- [x] 3.1 实现 `SparseRetriever` 的查询校验、一次有界搜索、允许字段过滤和稳定 `top_k`，用 Mock/内存测试证明空查询、非法 K、未知过滤会在后端调用前失败且不会触发 LLM、Graph 或 Web
- [x] 3.2 把 Sparse 命中转换为 `external=false` 的统一 Evidence，填充稳定 ID、Chunk/Paper、章节、页码、内容哈希、Sparse 分数和排名；通过 round-trip 测试确认每条结果可解析回 processed Chunk且不泄漏 SQLite 对象
- [x] 3.3 使用科研术语 fixture 验证 Sparse 对缩写、连字符模型名、数据集名和低频实体的召回，并覆盖无命中、索引不可读、暂时失败和同分排序场景

## 4. RRF、Weighted Fusion 与稳定去重

- [x] 4.1 实现无副作用的 Chunk 级 Evidence 合并和稳定 Hybrid Evidence ID，测试重叠结果只保留一条、单路独有结果不伪造缺失字段、输入对象不被修改且所有来源溯源保留
- [x] 4.2 实现版本化 RRF 公式和按融合分数、最佳来源排名、Chunk ID 的稳定排序；用手算 fixture 验证常数边界、单路/双路候选、同分和截断结果
- [x] 4.3 实现 Dense/Sparse 候选池 min-max 归一化与 Weighted Fusion，明确等分列表为 1、缺失来源为 0；用手算 fixture 验证权重边界、不同量纲、空列表和结果重放
- [x] 4.4 实现跨路 Chunk 去重和 Rerank 后可选论文去重，测试每篇保留最高最终排名、候选不足不循环补取、输出不超过上限且 Dense/Sparse/Fusion score/rank 同时保留

## 5. Hybrid 编排、失败降级与运行记录

- [x] 5.1 实现 `HybridRetrievalService`，对 Dense/Sparse 各调用至多一次并收集阶段计数、时延和调用成本；测试两路成功、一路/两路空结果、固定 Mock 重放和不触发隐藏重试或回答生成
- [x] 5.2 实现 `allow_partial` 策略：单路失败时仅使用成功 Evidence 并标记降级，关闭降级或两路均失败时返回结构化错误；测试失败后端、重试性和安全错误上下文均进入结果且不会被记录为完整融合
- [x] 5.3 实现 Hybrid 结果与阶段摘要的稳定序列化和临时目录原子发布，测试写入/回读/替换失败会清理半写产物，且结果不包含凭据、后端对象、完整 Prompt 或未选中候选正文
- [x] 5.4 新增薄入口 `scripts/query_hybrid_retrieval.py`，支持方法、权重、过滤、去重、Reranker 开关、降级策略、dry-run 和结果路径；CLI 测试验证解析摘要、上限、退出码和默认零 LLM

## 6. 本地 Cross-Encoder Reranker

- [x] 6.1 实现惰性 `CrossEncoderReranker` 适配器，固定模型/修订并从配置读取 CPU/GPU、缓存与批大小；用伪模型测试模块导入、dry-run和禁用路径零加载，加载/推理异常转换为不含文本或路径的安全错误
- [x] 6.2 实现 top-20 输入、批量完整性与有限数值校验、稳定同分排序和 top-8 输出，测试原始 Dense/Sparse/Fusion 字段保留、输入不变、空候选零模型调用以及数量/NaN/部分批次被整批拒绝
- [x] 6.3 把 Reranker 接入 Hybrid 服务并实现 `allow_rerank_fallback`，测试成功时记录模型版本、调用数和时延，失败回退时返回未重排融合结果并标记降级，严格模式则整体失败
- [x] 6.4 增加显式启用、最多 5 题和 top-20 候选的真实 Cross-Encoder smoke，记录峰值内存、设备、缓存位置和时延；默认 pytest 未启用时必须明确跳过且不得下载模型

## 7. Hybrid 回答验证与检索评测

- [x] 7.1 将 Evidence 上下文构建、结构化回答解析和引用校验提取为公共证据约束组件，同时保留 Dense 导入和行为；运行现有 Dense 问答、Schema 和离线端到端测试证明结果契约、一次生成上限与证据不足路径无回归
- [x] 7.2 实现显式 `with_answer` 的 Hybrid 查询服务：默认零 LLM，启用且证据充分时至多生成一次并记录实际路径、融合/Reranker 版本和引用；测试禁用、证据不足、合法回答、未知引用和 Provider 失败
- [x] 7.3 构建绑定当前 pilot 快照的 20–40 题 `hybrid_dev_questions.json`，覆盖原有语义问题以及缩写、连字符模型名、数据集名和低频实体；运行校验确认问题 ID 唯一、目标 Chunk 存在、标注可回溯且不使用后续冻结测试集
- [x] 7.4 实现 Dense、Sparse、RRF、Weighted、Fusion+Rerank 共用的检索评测器与 Recall@K、MRR、nDCG@K、Evidence coverage、失败数和阶段时延指标；用手算 fixture 验证失败题保留在分母、重复命中去重和 K 边界
- [x] 7.5 新增原子逐题报告、配置/语料/索引/融合/Reranker 版本哈希和断点复用，测试单题失败继续、版本漂移拒绝、同版本结果不重复计费以及候选矩阵从未调用 LLM
- [x] 7.6 新增薄入口 `scripts/evaluate_hybrid_retrieval.py`，支持 5 题 smoke、完整开发集、策略选择和唯一入选配置的显式回答验证；CLI 测试验证默认零 LLM、题量上限、冻结摘要和退出码
- [x] 7.7 增加从临时 processed fixture 经内存/SQLite Sparse、Mock Dense、融合、Mock Reranker 到评测报告的离线集成测试，验证首次构建、幂等重跑、单路降级、重排回退和默认零网络闭环

## 8. 文档、真实验收与完成门禁

- [x] 8.1 更新 README、架构、数据 schema、实验文档并新增 Hybrid 操作手册，记录 dry-run/建索引/查询/评测命令、产物布局、融合公式、降级语义、缓存迁移到 D 盘、重建/回滚和资源上限；逐条执行文档中的只读或 dry-run 示例
- [x] 8.2 更新 `../PROJECT_MAP.md`，列出新增模型、Sparse Store、索引/检索/融合/Reranker/评测模块、脚本、fixture、测试和文档，并对照实际文件系统复核职责与实现状态
- [x] 8.3 运行 `pytest -W error`、`ruff check .`、`ruff format --check .`、`mypy`、Schema 快照检查和 `docker compose config --quiet`，确认默认门禁不访问网络、真实模型、LLM、Qdrant 或 Neo4j
- [x] 8.4 对 pilot 先运行 Sparse dry-run，再执行首次构建、幂等重跑、内容更新、过滤、显式重建和查询验收；核对索引/manifest/检查点/Chunk 回溯，且只清理明确命名的测试产物
- [x] 8.5 运行零 LLM 的 5 题 smoke 与完整开发集矩阵，冻结一个入选 Hybrid 配置并报告 Dense/Sparse/RRF/Weighted/Rerank 的检索指标、失败、时延和资源；仅在显式预算内对入选配置执行一次回答验证，若真实模型环境不可用则记录未验证项而不伪造结果
- [x] 8.6 运行 `openspec validate add-hybrid-retrieval-and-reranking --type change --strict --no-interactive`，逐项复核 proposal、spec、design、tasks 与实现一致，并确认未引入 Graph、动态路由、facet 纠错、反思、Web、LLM Reranker 或 UI 范围
