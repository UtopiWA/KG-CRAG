# Tasks

## 1. 冻结配置、模型与 Schema

- [ ] 1.1 在 `src/kg_crag/models/graph.py` 定义严格的节点、关系、溯源、图身份、事实、Bundle、复核项、查询、路径命中和运行结果模型，并用单元测试验证白名单端点、稳定 ID、正文/元数据溯源及非法组合拒绝。
- [ ] 1.2 从 `kg_crag.models` 导出公共图模型并扩展 Schema 快照生成与测试，验证快照可重复且现有 Paper、Chunk、Evidence Schema 不变。
- [ ] 1.3 扩展严格配置模型和 `configs/retrieval.yaml`，覆盖 schema、产物路径、代表 Chunk、抽取预算、规范化、查询与评测上限，并测试未知字段、越界值和工作区外路径在执行前失败。
- [ ] 1.4 添加与 Compose 中 Neo4j 5.x 兼容的固定 Python Driver 依赖并更新锁定环境，验证干净环境能够安装且默认导入不建立数据库连接。
- [ ] 1.5 建立版本化图 schema 注册表、关系端点矩阵和实体/事实身份函数，并以固定向量测试验证 Unicode、别名、外部 ID、来源哈希及版本变化的行为。

## 2. 构建确定性的基础图

- [ ] 2.1 实现从已发布 processed Paper/Chunk 生成 Paper、Author、Chunk、`AUTHORED_BY` 与 `CONTAINS` 的基础 Bundle，并测试空作者、缺失可选字段和多版本 Paper 的处理。
- [ ] 2.2 实现仅基于 `Paper.references` 的 DOI、arXiv 和内部 Paper ID 引用解析，并测试唯一命中、空引用、未解析、歧义以及不会创建占位 Paper 或联网查询。
- [ ] 2.3 实现基础图选择与计划阶段，支持 pilot、去重 `--paper-id`、有界 `--all --limit` 和 dry-run，并测试无界全量请求在任何后端或模型调用前被拒绝。
- [ ] 2.4 实现基础图逐篇差异、状态、检查点和 UTF-8 原子清单，测试中断续跑、相同快照跳过、身份变化重算以及半写文件不会被识别为成功。
- [ ] 2.5 新增薄脚本 `scripts/build_metadata_graph.py`，验证 dry-run 输出图身份、数量和逐篇计划且数据库写入次数为零。

## 3. 有界正文事实抽取

- [ ] 3.1 实现按章节、最小长度、信息密度和 ordinal 稳定选择每篇 3–5 个代表 Chunk 的选择器，并用 fixture 测试覆盖不足、去重、确定排序和不复制 Chunk。
- [ ] 3.2 定义独立 `GraphExtractionProvider`、结构化响应和确定性 Mock，实现版本化 Prompt，使测试证明模型不能提供 Paper、Chunk 或版本来源标识。
- [ ] 3.3 实现生产抽取适配器及基于 Chunk 内容、schema、Prompt、模型修订和配置的内容寻址缓存，并测试缓存命中为零模型调用、缓存版本漂移失效及无 usage 时使用固定保守估算。
- [ ] 3.4 实现抽取候选的完整校验和来源注入，验证未知类型、非法端点、超长输出、非结构化输出或不在当前 Chunk 中的证据片段会原子拒绝该 Chunk 的全部候选。
- [ ] 3.5 实现请求数、累计输入/输出 Token、单次输入/输出、批大小和并发的批前预算闸门，测试下一批超限时不发出调用、保存检查点且不会隐藏重试或切换 Provider。
- [ ] 3.6 实现逐 Chunk 成功、失败、缓存、预算耗尽状态和脱敏错误记录，测试失败隔离、续跑仅处理缺失项且清单不含密钥、完整 Prompt 或无界正文。
- [ ] 3.7 新增薄脚本 `scripts/extract_graph_facts.py`，提供 dry-run、禁用抽取、显式在线开关与预算确认；测试 dry-run/禁用模式均为零 LLM 调用，pilot 计划不超过 15 篇、每篇 3–5 个 Chunk 和 100 次请求。

## 4. 保守规范化与人工复核

- [ ] 4.1 实现 NFC、空白、标点、大小写和别名的版本化确定性规范化，并测试外部 ID 相同或同类型唯一无冲突键才自动合并且保留候选映射。
- [ ] 4.2 实现歧义检测与 `EntityReviewItem` 生成，测试跨类型同名、冲突外部 ID、缩写、近似名称和低置信候选保持分离并给出有限原因、分数和来源。
- [ ] 4.3 实现版本化人工合并/保持分离决定的加载与重放，测试未知候选、失效来源和外部 ID 冲突的决定在图导入前被拒绝。
- [ ] 4.4 实现不含完整 Chunk 正文的原子复核报告和统计，并用快照测试验证稳定排序、脱敏字段和重复运行一致性。

## 5. Graph Store 与 Neo4j 适配器

- [ ] 5.1 将 `GraphStore` 扩展为图初始化、逐篇状态、事务同步、受控查询和关闭契约，同时保留旧入口的兼容迁移层，并让现有及新增 Protocol 测试通过。
- [ ] 5.2 重构内存 Graph Store 以实现版本隔离、幂等逐篇目标状态、陈旧事实清理和共享实体保留，并通过与生产适配器共用的契约测试。
- [ ] 5.3 实现 Neo4j 异步连接生命周期、图身份记录、`graph_version`/`graph_key` 约束检查与脱敏结构化错误，测试缺失或不兼容身份不会自动清库或接触其他版本。
- [ ] 5.4 实现单篇单事务的 Neo4j `sync_paper`，在写入完整新状态后清理该篇陈旧事实和无引用作用域实体；用事务替身测试失败全回滚且其他 Paper 可继续。
- [ ] 5.5 为 `entity_neighbors`、`relation_lookup`、`bounded_path` 和 `method_evidence` 实现固定参数化只读模板，静态测试验证始终带 graph version、`LIMIT`、超时，无写关键字、过程调用、未绑定输入或无界关系模式。
- [ ] 5.6 让内存与 Neo4j 查询输出同一 `GraphPathHit` 契约和稳定排序，并用参数化契约测试覆盖空命中、1–3 hop、关系/类型过滤、候选上限及非法请求零后端调用。
- [ ] 5.7 添加 `KG_CRAG_RUN_NEO4J_TESTS=1` 控制的隔离集成测试，验证约束、幂等同步、逐篇回滚、版本隔离和只读查询；默认 `pytest` 必须明确跳过且不启动或连接 Neo4j。

## 6. Graph Retriever 与统一 Evidence

- [ ] 6.1 实现 Graph Retriever 的请求校验和单次 Store 调用边界，测试未知模板、非法过滤、越界 top-k/候选/hop/超时在调用前失败，后端错误不重试且不触发其他 Retriever。
- [ ] 6.2 实现 processed Chunk/Paper resolver 与来源哈希复核，测试正文来源缺失、跨 Paper 或漂移时拒绝整条路径而不是使用数据库中的陈旧文本降级。
- [ ] 6.3 将有效路径按唯一支持 Chunk 转换为稳定 `external=false` Graph Evidence，并测试正文与 metadata-only 来源区分、事实 ID/hop/版本保留、无伪造页码章节以及 Evidence ID 可重复。
- [ ] 6.4 实现基于模板相关度、hop、最弱事实置信度、path/source ID 的稳定排序和 top-k 截断，并测试输入顺序变化不改变结果。
- [ ] 6.5 新增薄脚本 `scripts/query_graph_retrieval.py`，验证合法模板返回可序列化 Evidence/诊断，空命中成功返回空列表，认证或超时错误仅输出脱敏结构化信息。
- [ ] 6.6 添加 Mock 抽取、内存图和 Graph Retriever 的离线端到端测试，验证从 processed fixture 到关系/多跳 Evidence 全程零网络、零真实 LLM、零真实数据库调用。

## 7. 关系与多跳检索评测

- [ ] 7.1 创建 12–20 题人工标注的 Graph 开发子集，覆盖关系查找、实体歧义、单跳、多跳、无路径和错误来源，并由校验器检查题目 ID、pilot 快照、graph version、目标事实/路径/Chunk 和 K 值。
- [ ] 7.2 实现路径命中率、Evidence Recall@K、MRR、nDCG@K、溯源完整率及失败计数，使用手算 fixture 测试分母包含失败题和各指标边界。
- [ ] 7.3 实现逐题结果、数据库调用次数、阶段时延、错误与汇总的检查点和原子报告，测试中断续跑不会丢题、重复题和版本漂移会拒绝运行。
- [ ] 7.4 新增薄脚本 `scripts/evaluate_graph_retrieval.py`，测试默认内存矩阵不会调用 LLM、Embedding、Web、Dense 或 Sparse，且每题 Graph Store 调用恰为零或一次。
- [ ] 7.5 运行固定 Graph 开发集的离线评测并提交小型标注集与可复现配置，核对逐题结果能定位失败路径且生成产物目录保持 Git 忽略。

## 8. 文档、受控验收与质量门禁

- [ ] 8.1 更新 `README.md` 和相关数据/实验说明，记录四个脚本、图身份、dry-run、预算、复核流程、Neo4j/真实抽取显式开关、产物位置和精确版本清理方式，并人工验证命令与实际 `--help` 一致。
- [ ] 8.2 新增或调整重要目录、入口和模块职责后同步更新根目录 `PROJECT_MAP.md`，核对文件树中的路径均存在且注明地图需随结构变化维护。
- [ ] 8.3 对 15 篇 pilot 先执行基础图 dry-run，再用内存后端完成构建和重复运行，核对第二次无重复、零 LLM 调用、引用缺失正常且清单可回溯到语料快照。
- [ ] 8.4 先对不超过 3 个代表 Chunk 执行显式在线抽取 smoke，再在请求数不超过 100、输入 160,000 Token、输出 40,000 Token 的硬上限内完成 pilot；记录实际调用、Token、时间、缓存命中和失败，若 Provider 不可用则如实保留未验证项而不得伪造产物。
- [ ] 8.5 在 Neo4j 可用时显式运行集成 smoke 并核对版本隔离、幂等同步和查询；不可用时保留默认内存验收并记录未验证的真实数据库边界。
- [ ] 8.6 运行 Graph 定向测试及 `pytest -W error`、`ruff check .`、`ruff format --check .`、`mypy`，涉及 Compose 时再运行 `docker compose config --quiet`，修复本 change 引入的问题并记录仅因显式在线开关产生的跳过。
- [ ] 8.7 运行 `openspec validate build-provenance-aware-knowledge-graph --strict`，复核 proposal、spec、design、tasks 和实际差异一致，并确认未引入动态路由、facet、Web、Graph+Hybrid 融合、API 或 UI 范围。
