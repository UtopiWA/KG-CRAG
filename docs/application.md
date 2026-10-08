# 查询 API、演示界面与本地运行

迭代 08 把既有科研问答能力包装为严格、可诊断的应用边界。API 和 Gradio 界面不实现检索、充分性或纠错规则；真实工作流通过 `ApplicationService` 惰性注入，默认启动仍只加载三个脱敏回放案例，不连接 LLM、Web、Qdrant 或 Neo4j。

## 本地启动

安装演示依赖后分别启动 API 和 UI：

```powershell
python -m pip install -e ".[dev,demo]"
kg-crag-api
kg-crag-ui
```

浏览器访问 `http://127.0.0.1:7860`。默认 API 基址为 `http://127.0.0.1:8000`，可通过 `KG_CRAG_UI_API_BASE_URL` 修改。

容器按 profile 启动，未选中的服务不会常驻：

```powershell
# 只启动 API 和 UI；可完成三个固定回放案例
docker compose --profile replay up --build

# 只启动 Dense/Graph 所需的本地存储
docker compose --profile stores --profile graph up -d

# 启动全部组件；容器声明内存上限合计为 9 GiB
docker compose --profile full up --build
```

`replay` profile 不加载 BGE、重排器、数据库或在线模型。实时模式设置 `KG_CRAG_ENABLE_LIVE_QUERY=true` 后校验索引；启用 `KG_CRAG_LIVE_PRELOAD_MODELS=true` 时，API 会在后台提前加载本地模型，`/ready` 依次报告 `warming`、`ready` 或 `degraded`。预热期间实时请求快速返回可重试的 503，固定回放仍可使用；未启用或依赖不匹配时不会静默伪装成回放。预热失败不会自动重试，修复依赖后应重启 API。

本地实时启动示例：

```powershell
$env:KG_CRAG_ENABLE_LIVE_QUERY = "true"
$env:KG_CRAG_LIVE_PRELOAD_MODELS = "true"
$env:KG_CRAG_LIVE_ENABLE_RERANKER = "true"
$env:KG_CRAG_LIVE_ENABLE_GRAPH = "true"
$env:KG_CRAG_LIVE_ENABLE_ANSWER_CRITIC = "true"
$env:KG_CRAG_LIVE_PAPER_CONTEXT_LIMIT = "5"
$env:KG_CRAG_LLM_MAX_OUTPUT_TOKENS = "1600"
# glm-5.3-flash 不能关闭思考，使用 low 防止 reasoning_content 占满输出额度
$env:KG_CRAG_LLM_REASONING_EFFORT = "low"
$env:KG_CRAG_API_REQUEST_TIMEOUT_SECONDS = "120"
$env:KG_CRAG_ENABLE_WEB_FALLBACK = "true"
$env:KG_CRAG_WEB_SEARCH_PROVIDER = "tavily"
$env:KG_CRAG_WEB_SEARCH_API_KEY = "<Tavily API Key>"
# 多个 Sparse 版本并存时必须显式指定 64 位版本；只有一个时可省略
$env:KG_CRAG_LIVE_SPARSE_INDEX_VERSION = "<index_version>"
docker compose --profile stores up -d
kg-crag-api
kg-crag-ui
```

请求只公开 `mode`、`allow_web`、`include_trace` 与回放案例控制，不接受 Prompt、阈值或工具名。Web 必须同时满足全局 Provider/凭据配置、请求许可和内部证据不足门禁；Tavily Key 缺失时实时装配明确失败。Graph 使用当前冻结 pilot 构建轻量内存元数据图，不要求 Neo4j，但它不能替代尚未抽取的正文事实。`KG_CRAG_LLM_MAX_OUTPUT_TOKENS` 默认 1600，沿用现有 `KG_CRAG_LLM_PROVIDER`、模型、地址与密钥。`glm-5.3-flash` 建议使用 `KG_CRAG_LLM_REASONING_EFFORT=low`；`provider-default` 不改写其他兼容服务的默认行为。

复杂问题会先执行一次严格 Schema 约束的 Facet LLM，再运行 Hybrid 检索；简单问题仍直接使用规则 facet。模型响应非法、超时或失败时整批回退规则结果，不接纳部分候选。Facet 结果按问题、Prompt 和模型 revision 缓存，并把实际或保守估算 Token 计入纠错预算。当前多证据平衡值为：Dense/Sparse 各 20、融合 16、重排 12、最终 Evidence 12、同论文补充 5、回答上下文 16000 字符。

## 公共接口

- `GET /health`：只检查 API 进程存活，不探测外部依赖。
- `GET /ready`：报告 API、回放和可选实时/Graph 能力的就绪或降级状态。
- `POST /v1/queries`：执行单问题实时或显式回放查询。
- `GET /v1/documents/{paper_id}`：返回论文元数据、处理版本和最多 100 个 Chunk ID，不返回 Chunk 正文。
- `POST /v1/ingestion/runs`：默认 dry-run；实际摄取要求 `confirm=true`，单次最多 5 篇。
- `GET /v1/traces/{trace_id}`：返回最多 200 条脱敏 Trace 摘要。

请求模型拒绝未知字段。所有错误使用稳定 `code`、安全中文消息、`request_id`、`retryable` 和有界字段说明；响应不会包含完整 Prompt、原始模型响应、私有推理、连接串、密钥或整篇正文。查询响应中的 facet、冲突和纠错动作来自回答终止时的最终状态，预算的 `token_usage_source` 明确标记 `actual`、`estimated`、`mixed` 或 `none`。查询和摄取默认使用 120 秒应用层上限。

## 模型准备与快速启停

模型准备只下载/加载配置锁定的 revision，不调用 LLM，也不读取正式评测集。`--confirm-download` 显式允许模型下载；日常 API 默认使用 `KG_CRAG_MODEL_LOCAL_FILES_ONLY=true`，并同时启用 Hugging Face 与 Transformers 的进程级离线模式，避免缓存完整时仍进行远端探测：

```powershell
# 先查看模型、revision 和目标缓存目录
python scripts/prepare_demo_models.py

# 一次性准备 BGE-M3 与重排器；只要缓存未删除，后续无需重复下载
python scripts/prepare_demo_models.py --confirm-download

# 已准备完成后，可用离线检查确认缓存完整
python scripts/prepare_demo_models.py --local-files-only
```

日常启动：

```powershell
# 窗口 1：Qdrant 与 API
docker compose --profile stores up -d qdrant
kg-crag-api

# 窗口 2：Gradio
kg-crag-ui

# 任意窗口：等待 live_query 从 warming 变为 ready
Invoke-RestMethod http://127.0.0.1:8000/ready
```

日常关闭：

```powershell
# 先在运行 API 和 UI 的两个窗口中分别按 Ctrl+C
docker compose stop qdrant
```

`docker compose stop qdrant` 只停止容器，不删除 Qdrant 数据。无需使用 `down -v`、恢复出厂设置或删除 `.docker-data`。

`kg-crag-ui` 启动时只在当前进程内合并 `NO_PROXY`/`no_proxy`，确保 `localhost`、`127.0.0.1` 和 `::1` 绕过 HTTP/HTTPS 代理。这样 Gradio 的本地可达性检查和 UI 到本地 API 的请求不会被代理拦截；已有代理例外保持不变，`share` 始终显式关闭。

## 固定回放

版本 `demo-replay-v1` 包含：

1. `sufficient`：内部证据直接覆盖全部必需 facet。
2. `corrected-gap`：首次检索遗漏比较侧，执行一次 Sparse 补检后恢复。
3. `conservative-stop`：内部证据不足且外部依赖不可用，保守停止。

回放响应始终包含 `mode=replay`、fixture 版本和“非实时、不得作为正式实验结论”提示。实时模式失败时只返回结构化错误，不自动切换回放。回放和应用烟雾测试不读取统一评测 test split、不接受 `--confirm-test-run`，也不会创建正式测试锁。

## 持久化与资源

- Qdrant 默认写入 `.docker-data/qdrant`，可通过 `KG_CRAG_QDRANT_DATA_PATH` 移到空间充足的磁盘。
- Neo4j 数据和日志分别由 `KG_CRAG_NEO4J_DATA_PATH`、`KG_CRAG_NEO4J_LOG_PATH` 控制。
- BGE 与重排器统一由 `KG_CRAG_MODEL_CACHE_ROOT` 控制，应优先指向非系统盘；改变目录后需在新目录重新准备模型。
- API/UI/Qdrant/Neo4j 容器上限分别为 2/1/3/3 GiB；应用总体目标低于 10 GiB，硬边界为 12 GiB。
- 回放模式不加载 BGE 和重排器；实时模式可显式后台预热。增强配置开启轻量内存 Graph 和 Critic，资源紧张时可分别通过 `KG_CRAG_LIVE_ENABLE_GRAPH=false`、`KG_CRAG_LIVE_ENABLE_ANSWER_CRITIC=false` 关闭。
- Gradio 会话历史仅用于显示，每次 API 请求只携带当前问题；停止按钮取消当前客户端事件，服务端超时继续负责取消尚未开始的调用。

镜像构建上下文通过 `.dockerignore` 排除 `.env`、论文、权重、缓存、评测结果和数据库目录。真实密钥只放在本地 `.env`，不要提交或写入镜像。

## 故障排查

1. 先调用 `/health` 区分进程未启动与依赖问题。
2. 再调用 `/ready` 查看可用模式；`warming` 表示模型仍在后台预热，此时先使用固定回放；只有实时组件为 `ready` 后才提交实时问题。
3. 使用错误响应和响应头中的 `request_id` 对照本地日志，不要把密钥或原始 Provider 响应复制到报告。
4. 数据卷不可写或空间不足时，在启动索引/模型前更换上述持久化路径。
5. Compose 配置变更后先运行 `docker compose config --quiet`，再按最小 profile 启动。
6. GLM 没有 Token 记录而实时请求失败时，先检查 `/ready`；若仍是 `warming` 或 `degraded`，故障发生在本地模型/索引阶段，而不是在线生成阶段。
7. 若 Trace 出现 `answer_generation_failed`，查看 `failure_category`：`provider_request` 表示外部请求失败，`invalid_json`/`invalid_schema` 表示结构问题，`unknown_citation`/`unknown_facet`/`unsupported_facet_binding` 表示证据绑定未通过。Provider 失败还会给出安全的 `retryable`、`provider_status`、`provider_error_type`、`provider_finish_reason` 与 `provider_analysis_chars`；若 `finish_reason=length` 且正文为空，说明推理内容耗尽输出额度，可为 GLM-5.3-Flash 设置 `KG_CRAG_LLM_REASONING_EFFORT=low`。非超时可重试故障最多重试一次，超时和确定的长度耗尽不会盲目重试。Trace 不保存原始模型输出或私有推理。
8. facet 同时出现 Evidence 和 `missing` 时查看 `reason` 与 `conflict_ids`；`insufficient_internal_sources` 表示支撑阈值或独立来源不足，`blocking_conflict` 表示同一可比命题存在冲突。普通正文中的年份、编号、实验数字和无关否定词不会单独触发阻断冲突。
9. 若 Gradio 报错要求设置 `share=True`，不要开启公网分享。确认使用最新启动入口并重新运行 `kg-crag-ui`；入口会自动保留已有代理例外并补入本机地址。仅排查旧版本时，可临时在同一 PowerShell 设置 `$env:NO_PROXY="localhost,127.0.0.1,::1"` 后重启 UI。
10. 模型缓存已经通过 `prepare_demo_models.py --local-files-only`、但 `/ready` 仍在长时间 `warming` 后转为 `degraded` 时，依次检查 Qdrant、Dense/Sparse 索引身份和模型缓存。日常运行应保留默认的 `KG_CRAG_MODEL_LOCAL_FILES_ONLY=true`；只有明确需要在线解析尚未准备的模型时才关闭。

## 浏览器人工验收清单

1. 启动 API 和 `kg-crag-ui`，打开 `http://127.0.0.1:7860`，确认页面显示聊天气泡、发送、停止、清空、示例问题和有限控制项。
2. 选择“固定回放”，依次运行三个案例；确认每轮都显示“非正式实验结果”，折叠区含引用、facet、动作、预算、停止原因和 Trace。
3. 连续发送两个不同问题；确认保留两轮显示历史，同时浏览器网络面板中第二次 `/v1/queries` 请求只包含第二个问题，不含历史、Prompt、阈值或工具名。
4. 点击“清空”，确认输入和全部对话立即清除；发送慢请求后点击“停止”，确认页面不把取消请求显示为成功回答。
5. 在 Qdrant 与索引就绪后选择“实时工作流”，提交一个不在回放中的问题；确认响应标为实时并展示 `runtime_identity`、稳定引用和 Trace。
6. 停止 Qdrant 后再次实时提问；确认显示安全的 503 中文说明、`request_id` 和可重试状态，且不会出现回放答案、堆栈、连接串或密钥。
7. 默认关闭 Web；勾选 Web 后仍应只有在内部证据不足且全局 Web Provider 已启用时出现 Web 来源。
8. 实时提交“Voyager 在 Minecraft 中持续学习所依赖的三个核心组件是什么？”，预期回答包含自动课程、持续增长的可执行代码技能库和结合环境反馈/执行错误/自我验证的迭代提示机制；停止原因应为 `accepted`，引用至少包含 Voyager 摘要，facet 为 `covered`。若模型输出“暂无可信结论”等拒答，它不得被直接接受，Trace 应出现一次 `regenerate` 或最终保守停止。

## 本次真实链路验收记录

2026-10-07 使用 `glm-5.3-flash` 和 `reasoning_effort=low` 对 Voyager、AgentBench、CAMEL 三个用户问题执行受控烟雾测试，关闭 Web、Graph、Critic 和重排。三题均在一次模型调用后以 `accepted` 停止，分别返回 2、3、1 条内部引用；Provider 实际 usage 分别为输入/输出 1191/112、1898/225、822/192 Token。Voyager 回答列出自动课程、可执行代码技能库和迭代提示机制，AgentBench 回答包含推理与决策能力，CAMEL 回答说明角色扮演、inception prompting 与协作式对话。该记录只证明应用链路和结构化回答契约可运行，不属于正式效果评测，也未读取统一 test split 或创建正式测试锁。
