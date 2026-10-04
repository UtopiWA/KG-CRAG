# 查询 API、演示界面与本地运行

迭代 08 把既有科研问答能力包装为严格、可诊断的应用边界。API 和界面不实现检索、充分性或纠错规则；真实工作流通过 `ApplicationService` 注入，默认启动只加载三个脱敏回放案例，不连接 LLM、Web、Qdrant 或 Neo4j。

## 本地启动

安装演示依赖后分别启动 API 和 UI：

```powershell
python -m pip install -e ".[dev,demo]"
kg-crag-api
kg-crag-ui
```

浏览器访问 `http://127.0.0.1:8501`。默认 API 基址为 `http://127.0.0.1:8000`，可通过 `KG_CRAG_UI_API_BASE_URL` 修改。

容器按 profile 启动，未选中的服务不会常驻：

```powershell
# 只启动 API 和 UI；可完成三个固定回放案例
docker compose --profile replay up --build

# 只启动 Dense/Graph 所需的本地存储
docker compose --profile stores --profile graph up -d

# 启动全部组件；容器声明内存上限合计为 9 GiB
docker compose --profile full up --build
```

`replay` profile 不加载 BGE、重排器、数据库或在线模型。`full` 仍只负责启动应用和存储，真实查询工作流必须在进程内显式装配；未装配时 `/ready` 返回 `degraded`，实时查询返回 503，不会静默伪装成回放。

## 公共接口

- `GET /health`：只检查 API 进程存活，不探测外部依赖。
- `GET /ready`：报告 API、回放和可选实时/Graph 能力的就绪或降级状态。
- `POST /v1/queries`：执行单问题实时或显式回放查询。
- `GET /v1/documents/{paper_id}`：返回论文元数据、处理版本和最多 100 个 Chunk ID，不返回 Chunk 正文。
- `POST /v1/ingestion/runs`：默认 dry-run；实际摄取要求 `confirm=true`，单次最多 5 篇。
- `GET /v1/traces/{trace_id}`：返回最多 200 条脱敏 Trace 摘要。

请求模型拒绝未知字段。所有错误使用稳定 `code`、安全中文消息、`request_id`、`retryable` 和有界字段说明；响应不会包含完整 Prompt、原始模型响应、私有推理、连接串、密钥或整篇正文。查询和摄取默认 30 秒超时，用户配置最多放宽到 120 秒。

## 固定回放

版本 `demo-replay-v1` 包含：

1. `sufficient`：内部证据直接覆盖全部必需 facet。
2. `corrected-gap`：首次检索遗漏比较侧，执行一次 Sparse 补检后恢复。
3. `conservative-stop`：内部证据不足且外部依赖不可用，保守停止。

回放响应始终包含 `mode=replay`、fixture 版本和“非实时、不得作为正式实验结论”提示。实时模式失败时只返回结构化错误，不自动切换回放。回放和应用烟雾测试不读取统一评测 test split、不接受 `--confirm-test-run`，也不会创建正式测试锁。

## 持久化与资源

- Qdrant 默认写入 `.docker-data/qdrant`，可通过 `KG_CRAG_QDRANT_DATA_PATH` 移到空间充足的磁盘。
- Neo4j 数据和日志分别由 `KG_CRAG_NEO4J_DATA_PATH`、`KG_CRAG_NEO4J_LOG_PATH` 控制。
- 模型缓存继续由 `KG_CRAG_MODEL_CACHE_ROOT` 控制，应优先指向非系统盘。
- API/UI/Qdrant/Neo4j 容器上限分别为 2/1/3/3 GiB；应用总体目标低于 10 GiB，硬边界为 12 GiB。
- BGE 和重排器保持惰性加载，Graph 是可选工具；不要同时常驻不需要的模型和服务。

镜像构建上下文通过 `.dockerignore` 排除 `.env`、论文、权重、缓存、评测结果和数据库目录。真实密钥只放在本地 `.env`，不要提交或写入镜像。

## 故障排查

1. 先调用 `/health` 区分进程未启动与依赖问题。
2. 再调用 `/ready` 查看可用模式；只有 `replay` 时应在 UI 选择固定回放。
3. 使用错误响应和响应头中的 `request_id` 对照本地日志，不要把密钥或原始 Provider 响应复制到报告。
4. 数据卷不可写或空间不足时，在启动索引/模型前更换上述持久化路径。
5. Compose 配置变更后先运行 `docker compose config --quiet`，再按最小 profile 启动。
