# Raw paper corpus

此目录保存不可变的原始输入。PDF、运行时元数据、候选缓存和下载日志默认被 Git 忽略，避免提交大型二进制文件或机器相关路径；`.gitkeep` 和本文档用于保留目录与语料说明。

## 当前获取状态（2026-09-27）

`scripts/collect_arxiv_corpus.py` 已建立 140 篇候选池，并完成 110/110 篇 PDF 与元数据下载：其中 22 篇来自显式核心清单，88 篇由 OpenAlex 主题检索发现并通过 arXiv ID 归一化。PDF 总大小约为 392 MiB，均直接来自 arXiv 官方地址。

`scripts/verify_seed_corpus.py` 已独立逐项验证 110 组 PDF/JSON 的文件配对、PDF 文件头、文件大小和 SHA-256，失败数为 0。完整逐篇清单和最近运行结果分别见运行时目录 `papers/`、`metadata/` 与 `collection_report.json`；这些文件默认不提交 Git。

以下 22 篇是人工维护的核心子集，其余论文覆盖 Agentic RAG、多智能体、规划与反思、Graph/KG-RAG、科学问答和评测等主题：

| arXiv ID | 论文 | 主要用途 |
|---|---|---|
| 2005.11401 | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | Dense RAG 基线 |
| 2210.03629 | ReAct | 推理与行动 Agent |
| 2302.04761 | Toolformer | 工具使用 |
| 2303.11366 | Reflexion | Agent 反思 |
| 2303.17580 | HuggingGPT | 工具编排 |
| 2303.17760 | CAMEL | Multi-Agent |
| 2304.03442 | Generative Agents | 记忆、规划与反思 |
| 2305.06983 | Active Retrieval Augmented Generation | 主动检索 |
| 2305.10601 | Tree of Thoughts | 搜索式推理 |
| 2305.16291 | Voyager | 具身 Agent 与技能库 |
| 2307.16789 | ToolLLM | API 检索与工具学习 |
| 2308.00352 | MetaGPT | Multi-Agent 协作 |
| 2308.03688 | AgentBench | Agent 评测 |
| 2308.08155 | AutoGen | Multi-Agent 框架 |
| 2310.11511 | Self-RAG | 自反思 RAG |
| 2311.12983 | GAIA | 通用助手评测 |
| 2401.15884 | Corrective Retrieval Augmented Generation | 纠错检索 |
| 2401.18059 | RAPTOR | 分层检索 |
| 2403.14403 | Adaptive-RAG | 问题感知路由 |
| 2404.16130 | From Local to Global: A Graph RAG Approach | GraphRAG |
| 2405.14831 | HippoRAG | 图检索与多跳记忆 |
| 2405.15793 | SWE-agent | 软件工程 Agent |

## 获取与复核

```powershell
# 查看当前有效数量与默认目标，不联网
python scripts/collect_arxiv_corpus.py --dry-run

# 根据候选缓存/主题查询补齐到默认 110 篇
python scripts/collect_arxiv_corpus.py

# 独立复核所有本地 PDF 和元数据
python scripts/verify_seed_corpus.py
```

本次批量扩充期间 arXiv Atom Metadata API 不稳定，因此候选发现使用 OpenAlex Works API，并只接收能够解析出 arXiv ID 的记录；作者、摘要、DOI 和 OpenAlex ID 写入标准化元数据，PDF 仍直接来自 arXiv 并通过内容与哈希校验。显式核心清单继续作为无需发现服务即可复现的保底输入。
