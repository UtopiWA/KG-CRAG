# Design

## Context

动机与范围见 `proposal.md`，行为边界见 `specs/dense-rag-baseline/spec.md`。当前摄取流水线已在 `data/processed/paper-<hash>/<processing-version>/chunks.jsonl` 发布稳定 `Chunk`；公共 `EmbeddingProvider`、`LLMProvider`、`Retriever`、`VectorStore`、`Evidence`、Mock Provider 和 `InMemoryVectorStore` 已存在。Qdrant 容器已固定为 `v1.13.2`，但 Python 客户端、生产适配器、索引编排、Dense Retriever、生成链和评测实现仍为空缺。

现有 `Evidence` 能保存 Dense 分数和位置，但没有显式原始排名或内部/外部标记；`generation/`、`evaluation/` 和 `observability/` 只有包骨架。`configs/retrieval.yaml` 只有初步 `top_k` 与去重开关，`configs/evaluation.yaml` 只有指标名。设计必须兼容已发布 Chunk，默认测试不得联网，并避免把后续 Sparse、Graph 或 Agent 工作流提前纳入。

## Goals / Non-Goals

**Goals:**

- 建立可从已发布 Chunk 重建、增量更新和明确回滚的 Dense 索引数据流。
- 让真实 Qdrant、真实 Provider 与确定性 Mock/内存实现共享相同公共契约。
- 为 Dense 检索、证据上下文、结构化回答、引用和基线运行记录定义稳定边界。
- 冻结一个可重复执行的小型 pilot，用于证明检索、引用和失败记录链路，而不是替代迭代 07 的完整评测平台。

**Non-Goals:**

- 不引入 LangGraph，不改变 `AgentState`，也不实现路由、融合、重排、纠错、反思或 Web 分支。
- 不提供生产 API/UI；本迭代以包内服务和薄 CLI 作为入口。
- 不把 Qdrant 数据卷、Embedding 缓存、运行结果或模型权重纳入 Git。
- 不承诺跨不同真实 LLM 的回答文本字节一致；可重放性通过冻结输入与版本、保存原始结构化结果并用 Mock 建立确定性测试保证。

## Decisions

### 1. 以一个纵向 `dense-rag-baseline` 能力连接四个内部阶段

数据流固定为：

```text
published chunks -> embed/cache -> Qdrant collection
question -> embed -> dense search -> Evidence -> bounded context
         -> structured LLM output -> citation validation -> DenseRAGResult
pilot questions -> repeated query pipeline -> per-item results + summary
```

阶段由包内服务组合，`scripts/build_vector_index.py`、`scripts/query_dense_rag.py` 和 `scripts/evaluate_dense_rag.py` 只解析参数、加载依赖并返回退出码。这样既满足单命令入口，又避免脚本复制业务逻辑。备选方案是把全部逻辑放入一个脚本或直接加入 Agent 工作流；前者不可复用，后者会提前耦合迭代 05，均不采用。

### 2. 扩展公共模型而不改变既有 Chunk 语义

在独立的 Dense RAG 模型模块中定义严格模型，至少包括：

- `VectorCollectionIdentity`：schema 版本、模型/修订、维度、距离和稳定版本；
- `IndexRunManifest` 与逐篇结果：输入快照、配置哈希、集合、计数、状态、时延、错误和产物；
- `EvidenceRanks` 或等价公共字段：先加入 Dense 原始排名，并预留后续阶段的可选排名；
- `Citation`、`AnswerClaim`、`DenseRAGResult`：引用映射、结论类型、置信度、证据不足、检索路径、外部证据标记和 Trace ID；
- pilot 问题、逐题结果和汇总报告。

现有 `Chunk` 字段和 ID 算法保持不变；新增字段优先使用兼容默认值，所有公共模型加入统一导出和 JSON Schema 快照。备选方案是把排名和引用都塞入 `metadata` 字典；这会削弱严格校验和跨后端一致性，因此不采用。

### 3. Embedding 分为 Provider、校验批处理器和内容寻址缓存

业务流程继续只依赖 `EmbeddingProvider`。新增批处理器负责：按配置切批、保持输入顺序、校验数量/维度/有限值、统计命中和把异常转换为安全 `ErrorDetail`。缓存存放在 Git 忽略的 `data/processed/embedding-cache/<namespace>/`，以规范化文本 SHA-256、Provider、模型修订和配置哈希生成键；先完整写临时文件并重读校验，再原子发布。

实现阶段固定一个可实际运行的本地 Sentence Transformers 适配器，并记录精确模型修订、归一化和 query/document 前缀；它不得在导入时下载模型。模型下载或加载只发生于显式命令，默认测试使用 `MockEmbeddingProvider`。若课程环境更适合已部署的 OpenAI-compatible Embedding 服务，可增加第二个适配器，但不能让 SDK 类型进入业务模块。

备选方案是只依赖 Qdrant 服务端向量化或只缓存 Chunk ID。服务端向量化会把模型版本隐藏在数据库外部，Chunk ID 又不能区分模型和规范化变化，因此均不采用。

### 4. 集合版本与语料快照分离

`collection_version` 由向量 schema 版本、Embedding Provider/模型修订、维度、距离度量和 payload schema 的 canonical JSON SHA-256 决定，集合名使用固定前缀与安全的短哈希。`corpus_snapshot_hash` 则由按稳定顺序排列的 Paper ID、processing-version、Chunk ID 和内容哈希决定。

前者变化意味着集合不兼容，必须新建或显式 `--rebuild` 精确目标；后者变化只触发同一兼容集合的增量同步，并记录新的输入快照。这样既支持增量更新，也不会把不同维度或模型的向量混入同一集合。备选方案是把语料哈希写进集合名；每次增量都会复制完整集合，成本不必要。

### 5. Qdrant payload 和删除语义采用白名单

向量点 ID 由稳定 Chunk ID 确定；payload 只保存重建 Evidence 所需的 `chunk_id`、`paper_id`、`section`、`page_start`、`page_end`、`ordinal`、`text`、`content_hash` 和 `processing_version`。只为显式允许的过滤字段创建索引，过滤映射集中在适配器中，未知字段在访问 Qdrant 前失败。

增量同步以论文为隔离单元：先校验并向量化该论文的全部目标 Chunk，再 upsert 新版本，随后仅删除同一 Paper ID 下不属于目标 Chunk 集合的陈旧点；若准备阶段失败则不修改该论文。Qdrant 批次失败记录为该论文失败，其他论文继续。`--rebuild` 只接受由当前配置解析出的精确集合名，先展示目标并要求显式参数，禁止通配符或任意集合删除。

### 6. Dense Retriever 单次过取样并稳定归一为 Evidence

查询先经过同一输入规范和 Embedding 校验，再调用 Vector Store 一次。未去重时请求 `top_k`；按论文去重时请求 `min(top_k * candidate_multiplier, max_candidates)`，在内存中保留每篇最高排名项并稳定截断，不通过循环补取。Dense 分数保留后端返回值，排名从 1 开始，同分使用 Chunk ID 作为次级键。

Evidence ID 由集合版本和 Chunk ID 稳定生成；来源固定为内部 Chunk，`external=false`，后端专有对象不外泄。空结果是正常的证据不足状态；超时和连接错误映射为可重试外部服务错误；输入、过滤和集合不兼容映射为不可重试错误。

### 7. 生成使用版本化结构化 Prompt 和一次调用

Prompt 以仓库内版本化模板保存，内容上下文使用稳定 `[E1]`、`[E2]` 编号并受 `max_evidence`、单条字符数和总字符数三重上限。生成输出要求为严格 JSON：回答由带类型和引用 ID 的 claims 构成，并包含模型给出的有限置信度；程序只接受已提供引用，拒绝无引用事实、未知引用、重复冲突映射和超长结果，再渲染为最终回答。

证据为空或最高分低于阈值时直接返回确定性不足结果，不调用 LLM。一次问答只允许一次 Dense 检索和一次生成，不进行 JSON 修复调用或隐藏重试；无效生成返回结构化失败。这条限制保持基线可解释，修复/反思留给后续 change。

真实生成通过一个 OpenAI-compatible `LLMProvider` 适配器接入，凭据只来自 `Settings`，请求/响应日志不保存认证信息或完整未验证响应。默认和绝大多数测试使用 `MockLLMProvider`；真实模型验收为显式、可跳过步骤。

### 8. Trace 只记录最小公共事件

每次问答生成唯一 Trace ID，并顺序记录 query accepted、embedding completed/cache status、retrieval completed、evidence selected、generation completed/skipped 和 result finalized 等 `TraceEvent v1`。details 只含版本、数量、时延和稳定标识，不包含 Prompt 全文、论文正文、密钥或模型私有推理。Trace 与结果一起原子保存；本迭代不建立跨请求可观察性后端。

### 9. pilot 是提交的标注，小结果是可再生产物

在 `data/evaluation/` 保存不少于 20 个可提交的问题与目标 Chunk 标注，问题来源于当前已验证 pilot 语料并绑定 `corpus_snapshot_hash`。标注只包含必要问题文本、稳定目标 Chunk ID 和简短依据，不复制大段论文正文。实现一个最小确定性指标模块计算 Recall@K、MRR、nDCG@K、citation precision/recall；迭代 07 再扩展完整数据划分、人工评分和 Agent 指标。

逐题结果和汇总写入已被 Git 忽略的 `data/evaluation/results/dense-rag/<run-id>/`，先写同级临时目录并校验后原子发布。失败题保留在分母和逐题结果中。任何配置、Prompt、模型、索引或语料快照变化都产生新 baseline ID，不能覆盖旧运行。

### 10. 配置、依赖和测试边界显式分层

扩展 `configs/retrieval.yaml` 保存 Dense/Embedding/集合/索引/上下文上限，扩展 `configs/evaluation.yaml` 保存 pilot 路径、K 值和输出根；敏感地址和凭据继续来自 `Settings`/`.env`。配置加载使用严格模型和跨字段校验，canonical JSON 产生配置哈希。

新增依赖必须固定兼容版本并在 `docs/dependencies.md` 记录许可证、模型许可、Python 3.11/3.12 wheel 与离线限制。单元测试只使用 Mock 和 `InMemoryVectorStore`；离线集成测试以临时 processed fixture 覆盖 index→retrieve→answer→evaluate。真实 Qdrant 测试使用唯一前缀的测试集合并由显式环境开关启用，真实模型验收另行显式执行。默认 `pytest` 不因外部服务缺失而失败或联网。

## Risks / Trade-offs

- [本地 Embedding 模型体积大、首次加载慢且可能需要联网下载] → 固定模型修订与缓存目录，导入不下载；将真实模型验收与默认测试分离，并允许以后增加兼容的远端适配器。
- [先 upsert 后删除可能在删除失败时暂时保留陈旧点] → 查询 payload 绑定当前 processing-version/输入快照，清单把该论文标为失败；重跑幂等收敛，且不采用可能先丢失可用索引的“先删后写”。
- [Qdrant 不提供跨整篇 upsert/delete 的通用事务] → 以论文为隔离单位完成预校验与向量化，记录部分失败并提供精确重建；不声称数据库级原子性。
- [余弦分数阈值依赖模型，跨模型不可比较] → 阈值纳入版本化配置，只在同一模型/修订的基线内解释，报告同时保存排名指标。
- [结构化引用验证不能证明语义上完全忠实] → 限制输入、要求逐 claim 引用、拒绝未知引用并保存逐题结果；语义忠实度批判器留给迭代 06/07。
- [公共 Evidence/结果模型扩展可能影响已有测试] → 新字段使用兼容默认值，先搜索全部引用，更新 Schema 快照并运行完整离线门禁。
- [真实 Provider 或 Qdrant 暂时不可用] → 返回可重试错误并保留已成功论文/题目；默认 Mock/内存测试继续可用，不自动回退到未经声明的来源。

## Migration Plan

1. 固定并安装 Qdrant、Embedding 和 LLM 适配器依赖，扩展配置但保持默认 Mock/离线启动可用。
2. 增加公共模型、配置加载、Embedding 批处理与缓存，并更新 Schema 快照和单元测试。
3. 实现 Qdrant 适配器和索引服务；先用内存实现验证，再用唯一测试集合执行显式集成测试。
4. 使用 dry-run 核对当前 processed Chunk 快照，创建版本化集合并对 pilot 建索引；验证点数、payload、抽样回溯和幂等重跑。
5. 接入 Dense Retriever、上下文构建、结构化生成、引用校验和 Trace；以 Mock 完成离线端到端测试，再显式验收真实 Provider。
6. 冻结 pilot 问题和基线配置，运行并保存报告；更新 README、数据模式、实验说明、依赖说明和 `../PROJECT_MAP.md`。
7. 回滚时停止使用新命令与集合即可，已有摄取产物和主规格不受影响；只在用户明确指定精确集合后删除本 change 创建的 Qdrant 集合与可再生缓存/结果。
