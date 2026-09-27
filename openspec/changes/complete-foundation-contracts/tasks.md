# Tasks

## 1. 公共基座契约

- [x] 1.1 新增 `models/foundation.py` 中的 ErrorCode、ErrorDetail、TraceEvent v1、HealthResponse 及敏感键校验，更新公共导出，并用模型单元测试验证未知字段、错误可重试标记、时区时间、负序号和敏感字段拒绝行为
- [x] 1.2 新增携带 ErrorDetail 的 `KGCRAGError`，验证字符串表示不泄漏 context，且外部服务错误和输入错误可以由稳定代码与 `retryable` 区分
- [x] 1.3 将 AgentState 切换为公共 TraceEvent 模型，同时保留 `kg_crag.workflow.state.TraceEvent` 兼容导入，并通过类型检查和状态 fixture 验证现有字段未变化

## 2. 确定性离线实现

- [x] 2.1 实现 Mock Retriever 与 Mock Reranker、公共导出和调用记录，使用单元测试验证过滤、稳定排序、`top_k` 截断、溯源保留及非法参数错误
- [x] 2.2 实现内存 Vector Store 与公共导出，使用单元测试验证余弦排序、稳定并列次序、幂等 upsert、精确过滤、论文删除和向量维度错误
- [x] 2.3 实现内存 Graph Store 与公共导出，使用单元测试验证论文/Chunk 幂等写入、实体词匹配、稳定排序、论文删除以及非法 `top_k`/`max_hops`
- [x] 2.4 对四个离线实现执行运行时 Protocol 检查，并确认其测试在禁用网络和真实数据库时通过

## 3. 配置与健康接口

- [x] 3.1 收紧 Settings 的必要字符串、端口和服务地址校验，保持环境变量名称与默认值兼容，并通过合法默认值、端口越界和非法 URL 测试
- [x] 3.2 为 `/health` 声明 HealthResponse，扩展 API 测试以精确断言字段集合，并验证外部服务未启动时仍可成功且不返回密钥

## 4. Schema 与文档

- [x] 4.1 新增 `scripts/export_model_schemas.py` 的写入和 `--check` 模式，生成约定公共模型的 `docs/schemas/*.schema.json`，并以重复导出和检查模式测试字节稳定性
- [x] 4.2 更新 README 说明错误、Trace、离线实现、配置约束和 Schema 命令，并确认示例命令与实际 CLI 一致
- [x] 4.3 更新 `../PROJECT_MAP.md`，列出新增模块、测试、脚本和 schema 目录，并对照实际文件系统检查所有路径

## 5. 完整验证

- [x] 5.1 运行 `pytest -W error`、`ruff check .`、`ruff format --check .`、`mypy`、`docker compose config --quiet`，并确认默认测试未访问在线服务
- [x] 5.2 运行 `openspec validate complete-foundation-contracts --type change --strict --no-interactive`，复核 proposal/spec/design/tasks 与实现范围一致并记录任何环境警告
