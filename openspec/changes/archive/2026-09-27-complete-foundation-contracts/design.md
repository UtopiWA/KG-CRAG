# Design

## Context

当前仓库已经通过 Pydantic `StrictModel`、多个 `Protocol` 和确定性 LLM/Embedding Mock 建立了部分基座，但 TraceEvent 仍是无运行时校验的 `TypedDict`，错误没有统一表达，Retriever/Reranker/Store 也没有可执行的离线实现。详见 `proposal.md`；行为边界见 `specs/foundation-contracts/spec.md`。

实现必须保持 Python 3.11、严格 mypy、默认测试离线和无新增运行时依赖。现有公开 Protocol 方法签名、`/health` 路径、论文采集格式及 `AgentState` 字段名称保持兼容。

## Goals / Non-Goals

**Goals:**

- 把错误、Trace 和健康响应变成严格且可导出的公共数据模型。
- 为现有 Retriever、Reranker、Vector Store 和 Graph Store Protocol 提供轻量确定性实现。
- 让配置错误在外部调用前失败，并为公共模型建立可评审的 schema 快照。
- 使后续迭代能够只使用公开接口和离线实现编写工作流测试。

**Non-Goals:**

- 离线 Store 不模拟 Qdrant/Neo4j 的全部语法、事务或性能特征。
- 不增加通用日志框架、持久化 Trace 后端或 API 全局异常处理器。
- 不实现真实 Embedding、检索融合、图查询或 LangGraph 编排。
- 不迁移或重写现有 raw/interim/processed 数据。

## Decisions

### 1. 基座模型与异常对象分层

新增 `models/foundation.py`，保存 `ErrorCode`、`ErrorDetail`、`TraceEvent` 和 `HealthResponse`，并由 `models/__init__.py` 统一导出。新增顶层 `errors.py`，只提供携带 `ErrorDetail` 的 `KGCRAGError`，使可序列化数据契约与 Python 控制流异常解耦。

选择该方案而不是继续扩充 `models/domain.py`，因为基础运行契约会被 API、工作流和适配器共同引用，与论文领域模型的演进节奏不同。也不为每种错误建立异常子类树；稳定错误代码和 `retryable` 已能支持调用方分支，过早的继承层级会增加兼容成本。

`ErrorDetail.context` 与 `TraceEvent.details` 仅接受标量值。两个模型复用一个内部敏感键检查器，按大小写和 `_`/`-` 差异识别 `password`、`token`、`api_key`、`authorization`、`credential` 和 `secret` 等键并拒绝创建模型。

### 2. TraceEvent v1 使用 Pydantic 严格模型

TraceEvent 字段固定为：`schema_version="v1"`、`trace_id`、`sequence`、`node`、`event`、`occurred_at` 和 `details`。`sequence` 非负，字符串字段非空，`occurred_at` 必须包含时区。`AgentState.trace` 改为 `list[TraceEvent]`，其余 AgentState 字段不变。

选择时区感知时间而不是自动填充当前时间，确保测试与重放由调用方显式控制且结果确定。暂不加入 Token、成本和延迟专用字段；后续可观察性 change 可在 v2 或兼容新增字段中处理。

### 3. 离线检索实现保持简单且显式确定

新增 `retrieval/mock.py`：

- Mock Retriever 按构造时的 Evidence 顺序返回结果，应用精确元数据过滤并稳定截断到 `top_k`；
- Mock Reranker 根据构造时提供的 `evidence_id -> score` 排序，分数相同则保持输入顺序；
- 两者记录不可变的调用参数快照，并对非正 `top_k` 抛出不可重试的 `KGCRAGError`。

这比隐式关键词启发式更适合基座测试：测试能够精确配置期望路径，且不会把临时排序规则误当成生产算法。

### 4. 内存 Store 是协议测试替身而非后端模拟器

新增 `vector_store/memory.py`，按 `chunk_id` 保存 Chunk 和向量，重复 upsert 覆盖同一记录。search 使用余弦相似度、稳定 ID 作为并列次序，并把命中转换为 Chunk Evidence。首次写入确定维度，之后所有写入和查询必须匹配该维度。

新增 `graph_store/memory.py`，按 `paper_id` 保存 Paper 与其 Chunk；retrieve 对论文标题和 Chunk 文本执行不区分大小写的全部实体词匹配，以稳定 Paper/Chunk ID 排序并转换为 Graph Evidence。该实现只支持测试多跳接口和生命周期，不声称复现图数据库语义。`max_hops` 必须非负，但不会改变这个简单替身的匹配算法。

过滤统一支持 `paper_id`、`section` 及 Chunk/Evidence 元数据中的精确标量匹配。未知过滤键返回空结果，而不是静默忽略。

### 5. 配置和健康响应保持向后兼容

`Settings` 为 `env`、`log_level`、`api_host`、模型名称等必要字符串增加非空约束，为 `api_port` 增加 1–65535 范围，并校验 Qdrant HTTP(S) URL 与 Neo4j `bolt://`/`neo4j://` 地址。现有环境变量名称和默认值不变。

`/health` 继续不探测外部依赖，但声明 `HealthResponse` 响应模型，固定 `status`、`version` 和 `environment` 三个字段。该选择保留进程存活检查语义；外部依赖就绪状态应由后续独立 readiness 能力承担。

### 6. Schema 快照由单一脚本确定性生成

新增 `scripts/export_model_schemas.py`，维护显式公共模型注册表，并把 `model_json_schema()` 以 UTF-8、排序键、固定缩进和末尾换行写入 `docs/schemas/*.schema.json`。`--check` 模式只比较内容，不写文件，供测试和 CI 使用。

快照覆盖 Paper、Chunk、Evidence、RouteDecision、RetrievalEvaluation、ErrorDetail、TraceEvent 和 HealthResponse。显式注册表比扫描全部 Pydantic 子类更可控，避免内部辅助模型意外成为公共 API。

### 7. 测试分层与兼容性

单元测试覆盖模型校验、敏感键拒绝、错误可重试语义、Mock 调用记录、内存 Store 幂等/过滤/删除、设置边界、健康响应和 schema `--check`。Protocol 通过 `isinstance(..., Protocol)` 或静态类型检查验证；真实数据库仍只属于后续集成测试。

公共导出只新增符号，不移除旧符号。TraceEvent 从 `workflow.state` 移至 `models` 后，`workflow.state` 继续导入并暴露同名符号，避免已有导入立即失效。

## Risks / Trade-offs

- [内存 Graph Store 的词法匹配可能被误用为生产图检索] → 在模块文档、README 和项目地图中明确其测试替身身份，不在生产配置中自动选择它。
- [敏感键检测可能误伤普通上下文键] → 只检查规范化后的明确凭据词，不扫描普通消息值；测试固定允许与拒绝集合。
- [Schema 快照造成有意字段变化时的维护成本] → 提供单命令重建和 `--check`，并要求变更评审同时查看快照差异。
- [余弦相似度实现与未来 Qdrant 细节不同] → 仅保证确定性 Protocol 行为，不以其结果作为真实检索指标。
- [收紧 Settings 可能暴露既有非法本地配置] → 保持默认值合法，在 README 记录约束；回滚时可单独撤销验证器，不影响环境变量名称。

## Migration Plan

1. 先加入新模型、异常和离线实现及其单元测试，不改变默认运行路径。
2. 将 AgentState 与健康接口切换到新公共模型，运行兼容性测试。
3. 生成并提交 schema 快照，更新 README 和 `PROJECT_MAP.md`。
4. 执行完整 pytest、Ruff、mypy、Compose 配置与 OpenSpec 严格校验。

若需回滚，可按相反顺序恢复 AgentState 和健康响应类型，再移除仅新增的离线模块、快照与导出；没有持久化数据迁移需要撤销。
