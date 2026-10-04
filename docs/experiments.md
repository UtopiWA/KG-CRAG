# Experiments

Planned experiment families are: LLM-only, dense RAG, dense+sparse hybrid RAG, fixed vector+graph RAG, and full KG-CRAG. Ablations remove graph retrieval, sparse retrieval, adaptive routing, corrective retrieval, answer critique, or web fallback; RRF and weighted fusion are compared separately.

Every run must record the dataset split, random seed, configuration hash, prompt/model/index versions, latency, tool calls, and token or inference cost. Generated result files belong under `data/evaluation/results/` and are not committed by default.

## Dense RAG v1 基线

迭代 02 的冻结基线使用 `configs/pilot_corpus.json` 对应的 15 篇 processed 论文、`data/evaluation/dense_pilot_questions.json` 的 20 个真实 Chunk 标注、固定 BGE-M3 revision、版本化 Prompt 和 Qdrant 集合身份。指标为 Recall@5/10/20、MRR、nDCG@5/10/20、citation precision 与 citation recall；失败题保留在分母中。

问题清单中的 `corpus_snapshot_hash` 不匹配、目标 Chunk 缺失或题数少于 20 时，评测必须在任何模型或数据库调用前失败。相同语料、配置、Prompt、集合和 K 列表形成相同 baseline ID，但每次运行使用独立目录，禁止静默覆盖结果。探索集可以用于调试，冻结 pilot 结果不得被反向用于调参。

## Hybrid Retrieval v1 开发矩阵

`data/evaluation/hybrid_dev_questions.json` 绑定同一 pilot 快照，共 24 题；除原有语义问题外，加入大写缩写、连字符模型名、数据集名和低频实体。矩阵使用完全相同的问题、目标 Chunk 和 K 值比较 Dense、Sparse、RRF、Weighted 与 Fusion+Rerank，计算 Recall@K、MRR、nDCG@K、Evidence coverage、失败数、总时延与阶段时延。

矩阵默认零 LLM。配置、语料快照、Dense 集合、Sparse 索引、融合版本、Reranker 版本、策略和 K 值共同形成 `evaluation_version`；逐题检查点只有身份严格匹配时才复用。回答验证与检索报告分开保存，必须显式选择唯一 `fusion_rerank` 配置并给出与题数完全一致的调用预算。

2026-10-03 验收记录：pilot Sparse dry-run、970 Chunk 首次构建、全量幂等跳过、精确版本重建和带论文过滤查询均通过；真实五策略 smoke 因本机 Docker/Qdrant 守护进程未响应而未完成，未生成或伪造质量指标。SQLite + Mock Dense/Reranker 的零网络集成矩阵已覆盖建索引、重跑、单路降级、重排回退、报告和断点复用。

## Corrective Workflow v1 压力矩阵

`data/evaluation/corrective_dev_questions.json` 冻结 20 题，覆盖术语错配、实体别名、比较缺侧、多跳断链、关键指标缺失、冲突和内部知识缺失。默认矩阵在相同初始 Evidence 上比较 fixed Hybrid、type router 与 facet corrective，报告首次充分率、facet 诊断准确率、必需覆盖率、恢复率、纠错触发/无意义动作率、平均轮次、调用、Token、时延、失败和停止原因。失败题始终留在分母与逐题报告中。

该矩阵不生成回答，开发集可以调整规则与成本，最终测试集和回答质量不得用于本迭代调参。在线模式先展示最坏预算并要求确认：smoke 最多 3 题，扩大验证最多 20 题；离线与在线报告使用不同身份。

2026-10-04 验收记录：20 题 × 3 策略离线矩阵完成且零失败；facet 纠错的必需覆盖率为 0.8276、恢复率为 0.75、无意义动作率为 0.25，20 次纠错触发共计 40 轮内部检索。相同身份重复运行在约 1.7 秒内从逐题检查点重放。随后以 `glm-5.3-flash` 对 q07 执行一次在线 smoke：Provider 返回未通过严格结构校验，系统没有重试或切换 Provider，按规则整批回退并保守记录 1 次调用、4000 输入 Token、2000 输出 Token 和 5000 ms；最终工作流仍以 `sufficient` 停止。相同身份约 2.3 秒重放最终检查点，未再次调用模型。真实 LLM 成功生成结构化 facet 的路径仍标记为未验证，不据此扩大在线题数。

## Unified Evaluation v4

`data/evaluation/unified/manifest.json` 保存 60 题冻结集：dev 40 题与 test 20 题由 17 道具有具体真值的纠错题和 15 篇论文上的 43 道科研问答组成。原纠错 q07～q09、Graph 和 grounded fixture 因缺少可展开的语义真值而不进入统一集。`evidence.json` 冻结 36 条答案证据，`question-sources.json` 冻结 60 条题目来源；冲突综合 facet 使用全部 Evidence 共同覆盖语义。校验器在运行前检查 schema、规模、压力类型、稳定 ID、题目来源、facet/Evidence 解析、最小充分集、近重复及论文级跨划分泄漏。v4 已于 2026-10-04 完成 dev/test 全量人工复核并标记为 `reviewed=true`；正式测试仍不得用于调参且同版本最多执行一次。

统一 runner 以完整内容身份比较 `fixed_hybrid`、`type_router` 与 `facet_corrective`，逐题报告 Recall/MRR/nDCG、Evidence/facet 覆盖、充分性、恢复、无效动作、回答/引用、循环、Web 和资源成本。失败题留在固定分母，任一汇总指标保存组成 question ID；检查点、失败清单、运行清单和报告均原子发布。Trace v2 只保存运行/题目 ID、阶段、状态、有限标量和资源增量，旧 Trace v1 通过只读适配器读取。

2026-10-04 管线验收记录：v1 因答案要点占位、冲突证据丢失和论文级跨划分污染失效；v2 又在冲突 Evidence 绑定、问题来源定位和部分压力标签上发现歧义；v3 仍存在术语错配、多跳及必需 facet 判定过宽的问题。这些版本均未冻结为正式数据。v4 收紧了压力类型判定，并移除与问题无关的必需要点；全量人工复核已完成。随后在 v4 dev 上完成 40 题 × 3 策略 fixture 管线回放，生成 120 个逐题结果且零失败；同身份复跑约 2.0 秒并复用检查点。该报告只证明编排与恢复行为，不能作为检索质量结论。

2026-10-05 开发选择记录：以 `mixed-local-index-and-controlled-replay-v1` 生成 120 条零 LLM 观察且零失败，观察哈希为 `1e1849a8215f7485643a03dd0bc3702c1187dba8b0599245c6000caff6aa0922`。论文题实际使用本地 BGE-M3 Dense、SQLite Sparse 和处理后语料内存 Graph；17 道合成纠错题因不存在于真实语料，保留冻结受控回放并在 route 中显式区分。首次报告身份为 `unified-run-1588ece6728157236b44d18d7a4219aa`，同身份复跑约 2.09 秒并复用检查点。`facet_corrective` 的总体 Recall@10 为 0.8947、必需 facet 覆盖率为 0.875、充分性判断准确率为 0.925，优于 fixed Hybrid 的 0.5263/0.4875/0.575 和类型路由的 0.5526/0.5125/0.600，因此冻结为开发入选策略。该结论只适用于冻结 v4 dev 与上述混合观察契约；自然语言生成、Judge 和正式 test 尚未执行。

同日仅对入选的 `facet_corrective` 执行 Top-8 抽取式回答/引用验证，报告身份为 `unified-run-d900df44c9b643b1bd19e9065d9e6a2e`，120 个矩阵项零失败。在 38 道含答案要点的问题上，精确 Evidence 支持映射得到回答正确率/完整性 0.8684，Top-8 引用正确率与抽取式 faithfulness 均为 0.4539。该结果是检索证据可直接支撑答案要点的保守代理，不代表自然语言生成质量；Judge 未请求（`judge_status=not_requested`），模型调用和 token 均为 0。正式 test 仍未执行。
