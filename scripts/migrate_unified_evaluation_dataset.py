# ruff: noqa: RUF001
"""从可核验的现有 fixture 与冻结 Chunk 构建统一评测候选；默认只预览。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from kg_crag.evaluation.dataset import (
    canonical_digest,
    dataset_digest,
    normalize_question,
    validate_unified_dataset,
    write_json_atomic,
)
from kg_crag.models import (
    EvaluationAnswerPoint,
    EvaluationDatasetManifest,
    EvaluationEvidenceCatalog,
    EvaluationEvidenceRecord,
    EvaluationFacetTarget,
    EvaluationQuestionOrigin,
    EvaluationQuestionSourceCatalog,
    EvaluationQuestionSourceRecord,
    EvaluationSplit,
    EvidenceMatchMode,
    KnowledgeSufficiency,
    StressCategory,
    UnifiedEvaluationQuestion,
    UnifiedQuestionSet,
    UnifiedQuestionType,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "data" / "evaluation"
OUTPUT_ROOT = SOURCE_ROOT / "unified"
DATASET_VERSION = "unified-agent-qa-v4"
EVIDENCE_VERSION = "unified-evidence-catalog-v3"
QUESTION_SOURCES_PATH = "data/evaluation/unified/question-sources.json"

# 论文级 source_group_id 是跨划分门禁的最小单位，不能再按 fixture 题号生成。
PAPERS: dict[str, dict[str, str]] = {
    "camel": {
        "paper_id": "arxiv:2303.17760",
        "chunk_id": "chunk-ed0d660890edc141532bcf65e3a9c3a2a3533e218f5634534b850c791a332fce",
        "source_question": "camel-purpose",
    },
    "generative-agents": {
        "paper_id": "arxiv:2304.03442",
        "chunk_id": "chunk-c31d1b3ba91a88e9ab09166c23ac27d24a6b4af6b0f4e410b12063e10becad7b",
        "source_question": "generative-agents-definition",
    },
    "tree-of-thoughts": {
        "paper_id": "arxiv:2305.10601",
        "chunk_id": "chunk-ee356f4b22b0e48fd81d0bdd61201a0a5d78a74225e4c4ef66aedec48ed5a3d9",
        "source_question": "tree-of-thoughts-motivation",
    },
    "voyager": {
        "paper_id": "arxiv:2305.16291",
        "chunk_id": "chunk-7bb5df31611f489883655355b26af32847f8e2dce9db34fac7b04da18a731142",
        "source_question": "voyager-components",
    },
    "toolllm": {
        "paper_id": "arxiv:2307.16789",
        "chunk_id": "chunk-a69a8299fd42d8553b428fe529dcade7efe2c3907e1ec174943c243ef328a552",
        "source_question": "toolllm-problem",
    },
    "metagpt": {
        "paper_id": "arxiv:2308.00352",
        "chunk_id": "chunk-bb89570bbaacf958d60637ad907dccd529aa18f9fab9a283497ded3994a31b54",
        "source_question": "metagpt-sop",
    },
    "agentbench": {
        "paper_id": "arxiv:2308.03688",
        "chunk_id": "chunk-533216a1b2f716ddf3e30327e2a021b5c5e57d555ec3cbe7001266f08e4302a5",
        "source_question": "agentbench-scope",
    },
    "lats": {
        "paper_id": "arxiv:2310.04406",
        "chunk_id": "chunk-7a7f4de81adf2cd70f7dd20c59b6d22e5473ac95b273f08659951ccde1762754",
        "source_question": "lats-capabilities",
    },
    "gaia": {
        "paper_id": "arxiv:2311.12983",
        "chunk_id": "chunk-d5b20813e84b33e27284aaaa097e8afca2f54f68df1d27ecf596cb0581d5a177",
        "source_question": "gaia-abilities",
    },
    "paperqa": {
        "paper_id": "arxiv:2312.07559",
        "chunk_id": "chunk-b3f5e8f2514d97dec5d917e5740361e3f498cfcbad045c945addb5656fc1cdff",
        "source_question": "paperqa-motivation",
    },
    "graphrag": {
        "paper_id": "arxiv:2404.16130",
        "chunk_id": "chunk-7cd930ecb34b192fcba96c658559e204e0d73f7e0c729bd88e88e5a522f14455",
        "source_question": "graphrag-global-questions",
    },
    "hipporag": {
        "paper_id": "arxiv:2405.14831",
        "chunk_id": "chunk-9885b758a38e27148bf30ca9207630fc1995048aca2f27fc6eb7a2e1a5f12d38",
        "source_question": "hipporag-memory",
    },
    "medgraphrag": {
        "paper_id": "arxiv:2408.04187",
        "chunk_id": "chunk-3b6cd29389004ee34cde9a96b410df2a62690455621b7687c9608e2fbc39c6f2",
        "source_question": "medgraphrag-safety",
    },
    "lightrag": {
        "paper_id": "arxiv:2410.05779",
        "chunk_id": "chunk-456394fa3a9578ede80c2e422449a9eb163f9948a711f6fdf0c0398b4bf08d2f",
        "source_question": "lightrag-limitations",
    },
    "agentic-rag": {
        "paper_id": "arxiv:2501.09136",
        "chunk_id": "chunk-9494d8313ef6c0c24ee15e890e0d0ef2b148999cd0bb990c1350ee1ff8281294",
        "source_question": "agentic-rag-motivation",
    },
}


def _digest_id(prefix: str, value: str) -> str:
    return f"{prefix}-{hashlib.sha256(value.encode()).hexdigest()[:16]}"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _load_questions(name: str) -> list[dict[str, Any]]:
    payload = json.loads((SOURCE_ROOT / name).read_text(encoding="utf-8"))
    questions = payload.get("questions")
    if not isinstance(questions, list):
        raise ValueError(f"source fixture has no question list: {name}")
    return questions


def _annotations(
    question_id: str,
    evidence_ids: list[str],
    point_descriptions: list[str],
    facet_descriptions: list[str] | None = None,
    *,
    point_evidence_ids: list[list[str]] | None = None,
    facet_evidence_ids: list[list[str]] | None = None,
    facet_evidence_matches: list[EvidenceMatchMode] | None = None,
) -> tuple[list[EvaluationAnswerPoint], list[EvaluationFacetTarget]]:
    """用稳定序号生成答案要点和必需 facet，描述必须是可核验命题。"""

    facets = facet_descriptions if facet_descriptions is not None else point_descriptions
    point_targets = point_evidence_ids or [evidence_ids] * len(point_descriptions)
    facet_targets = facet_evidence_ids or [evidence_ids] * len(facets)
    facet_matches = facet_evidence_matches or [EvidenceMatchMode.ANY] * len(facets)
    if len(point_targets) != len(point_descriptions):
        raise ValueError("answer point Evidence bindings do not match point count")
    if len(facet_targets) != len(facets) or len(facet_matches) != len(facets):
        raise ValueError("facet Evidence bindings do not match facet count")
    return (
        [
            EvaluationAnswerPoint(
                point_id=_digest_id("point", f"{question_id}:{index}:{description}"),
                description=description,
                evidence_ids=point_targets[index - 1],
            )
            for index, description in enumerate(point_descriptions, start=1)
        ],
        [
            EvaluationFacetTarget(
                facet_id=_digest_id("facet", f"{question_id}:{index}:{description}"),
                description=description,
                target_evidence_ids=facet_targets[index - 1],
                evidence_match=facet_matches[index - 1],
            )
            for index, description in enumerate(facets, start=1)
        ],
    )


def _question(
    *,
    question_id: str,
    split: EvaluationSplit,
    question: str,
    question_types: list[UnifiedQuestionType],
    stress: StressCategory,
    knowledge: KnowledgeSufficiency,
    evidence_ids: list[str],
    points: list[str],
    facets: list[str] | None,
    source_group_ids: list[str],
    source_fixture: str,
    point_evidence_ids: list[list[str]] | None = None,
    facet_evidence_ids: list[list[str]] | None = None,
    facet_evidence_matches: list[EvidenceMatchMode] | None = None,
) -> UnifiedEvaluationQuestion:
    answer_points, facet_targets = _annotations(
        question_id,
        evidence_ids,
        points,
        facets,
        point_evidence_ids=point_evidence_ids,
        facet_evidence_ids=facet_evidence_ids,
        facet_evidence_matches=facet_evidence_matches,
    )
    return UnifiedEvaluationQuestion(
        question_id=question_id,
        split=split,
        question=question,
        normalized_question=normalize_question(question),
        question_types=question_types,
        stress_category=stress,
        knowledge_sufficiency=knowledge,
        answer_points=answer_points,
        facets=facet_targets,
        relevant_evidence={value: 3 for value in evidence_ids},
        minimum_sufficient_evidence_sets=(
            [evidence_ids] if knowledge is KnowledgeSufficiency.SUFFICIENT else []
        ),
        leakage_group_id=f"leak:{question_id}",
        source_group_ids=source_group_ids,
        source_fixture=source_fixture,
    )


def _corrective_questions() -> list[UnifiedEvaluationQuestion]:
    """保留具有具体语义真值的纠错题；丢弃 q07-q09 的占位比较题。"""

    source = "corrective_dev_questions.json"
    items = {str(item["question_id"]): item for item in _load_questions(source)}
    descriptions = {
        "q01": ["Agent System 1 使用 planner architecture。"],
        "q02": ["Agent System 2 使用 planner architecture。"],
        "q03": ["Agent System 3 使用 planner architecture。"],
        "q04": ["Paper 4 提出了 AlphaAgent 4。"],
        "q05": ["Paper 5 提出了 AlphaAgent 5。"],
        "q06": ["Paper 6 提出了 AlphaAgent 6。"],
        "q10": [
            "Agent 10 使用 intermediate planner。",
            "该 intermediate planner 与 Tool 10 相连。",
        ],
        "q11": [
            "Agent 11 使用 intermediate planner。",
            "该 intermediate planner 与 Tool 11 相连。",
        ],
        "q12": [
            "Agent 12 使用 intermediate planner。",
            "该 intermediate planner 与 Tool 12 相连。",
        ],
        "q13": ["Agent 13 报告的准确率为 83%。"],
        "q14": ["Agent 14 报告的准确率为 84%。"],
        "q15": ["Agent 15 报告的准确率为 85%。"],
    }
    category_types = {
        StressCategory.MULTI_HOP_GAP: [UnifiedQuestionType.MULTI_HOP],
        StressCategory.METRIC_MISSING: [UnifiedQuestionType.METRIC],
    }
    output: list[UnifiedEvaluationQuestion] = []
    for source_id, points in descriptions.items():
        item = items[source_id]
        evidence_ids = [str(value) for value in item["target_evidence_ids"]]
        # 源 fixture 未提供可核查的术语差异或别名映射，因此降为非压力对照题。
        stress = (
            StressCategory.CONTROL
            if source_id in {"q01", "q02", "q03", "q04", "q05", "q06"}
            else StressCategory(str(item["category"]))
        )
        output.append(
            _question(
                question_id=f"ueq-corrective-dev-questions-{source_id}",
                split=EvaluationSplit.DEV,
                question=str(item["question"]),
                question_types=category_types.get(stress, [UnifiedQuestionType.FACTOID]),
                stress=stress,
                knowledge=KnowledgeSufficiency.SUFFICIENT,
                evidence_ids=evidence_ids,
                points=points,
                facets=points,
                source_group_ids=[f"paper:paper-{source_id}"],
                source_fixture=f"data/evaluation/{source}#{source_id}",
            )
        )

    for source_id in ("q16", "q17", "q18"):
        item = items[source_id]
        evidence_ids = [
            *[str(value["evidence_id"]) for value in item["initial_evidence"]],
            *[str(value["evidence_id"]) for value in item["action_evidence"]["graph"]],
        ]
        number = source_id.removeprefix("q")
        points = [
            f"一个内部 Chunk 将 Agent {number} 的准确率报告为 80%。",
            f"另一个内部 Chunk 将 Agent {number} 的准确率报告为 90%。",
            "Graph Evidence 将该准确率记录为 85%。",
        ]
        point_bindings = [
            [evidence_ids[0]],
            [evidence_ids[1]],
            [evidence_ids[2]],
        ]
        facets = [
            *points,
            "应保留并归因报告冲突，不能把任一数值当作已消解的唯一金标准。",
        ]
        facet_bindings = [*point_bindings, evidence_ids]
        output.append(
            _question(
                question_id=f"ueq-corrective-dev-questions-{source_id}",
                split=EvaluationSplit.DEV,
                question=str(item["question"]),
                question_types=[UnifiedQuestionType.METRIC],
                stress=StressCategory.CONFLICT,
                knowledge=KnowledgeSufficiency.CONFLICTING,
                evidence_ids=evidence_ids,
                points=points,
                facets=facets,
                source_group_ids=[f"paper:paper-{source_id}"],
                source_fixture=f"data/evaluation/{source}#{source_id}",
                point_evidence_ids=point_bindings,
                facet_evidence_ids=facet_bindings,
                facet_evidence_matches=[
                    EvidenceMatchMode.ANY,
                    EvidenceMatchMode.ANY,
                    EvidenceMatchMode.ANY,
                    EvidenceMatchMode.ALL,
                ],
            )
        )

    for source_id in ("q19", "q20"):
        item = items[source_id]
        number = source_id.removeprefix("q")
        output.append(
            _question(
                question_id=f"ueq-corrective-dev-questions-{source_id}",
                split=EvaluationSplit.DEV,
                question=str(item["question"]),
                question_types=[UnifiedQuestionType.FACTOID],
                stress=StressCategory.INTERNAL_MISSING,
                knowledge=KnowledgeSufficiency.INSUFFICIENT,
                evidence_ids=[],
                points=[],
                facets=[
                    f"确定 Agent {number} 所称 finding 的具体内容与来源；"
                    "冻结内部语料未提供可接受 Evidence，回答应明确无法从内部知识确定。"
                ],
                source_group_ids=[f"missing:agent-{number}"],
                source_fixture=f"data/evaluation/{source}#{source_id}",
            )
        )
    return output


# 每个命题都能由对应冻结 Chunk 直接核对；同一论文的全部题目固定在同一划分。
LITERATURE_QUESTIONS: list[dict[str, Any]] = [
    {
        "id": "generative-agents-definition",
        "paper": "generative-agents",
        "split": "dev",
        "question": "Generative Agents 论文如何定义生成式智能体及其目标？",
        "types": ["factoid"],
        "stress": "control",
        "points": [
            "生成式智能体是模拟可信人类行为的计算软件智能体。",
            "其目标是产生可信的个体行为和涌现式社会行为。",
        ],
    },
    {
        "id": "generative-agents-applications",
        "paper": "generative-agents",
        "split": "dev",
        "question": "生成式智能体的可信人类行为代理可以支持哪些类型的交互应用？",
        "types": ["synthesis"],
        "stress": "control",
        "points": ["沉浸式环境。", "人际沟通排练空间。", "原型设计工具。"],
    },
    {
        "id": "generative-agents-architecture",
        "paper": "generative-agents",
        "split": "dev",
        "question": "Generative Agents 的记忆架构如何支持行为规划？",
        "types": ["multi_hop"],
        "stress": "multi_hop_gap",
        "points": [
            "用自然语言保存智能体经历的完整记录。",
            "把记忆随时间综合为更高层次的反思。",
            "动态检索记忆与反思以规划后续行为。",
        ],
    },
    {
        "id": "toolllm-problem",
        "paper": "toolllm",
        "split": "dev",
        "question": "ToolLLM 针对开源大语言模型的哪项工具使用缺陷提出方案？",
        "types": ["factoid"],
        "stress": "control",
        "points": [
            "开源 LLM 难以调用外部 API 来完成用户指令。",
            "原因是既有指令微调偏重基础语言任务而忽略工具使用领域。",
        ],
    },
    {
        "id": "toolllm-toolbench",
        "paper": "toolllm",
        "split": "dev",
        "question": "ToolBench 的 API 来源、规模和自动构建阶段是什么？",
        "types": ["metric", "synthesis"],
        "stress": "metric_missing",
        "points": [
            "从 RapidAPI Hub 收集 16,464 个真实 RESTful API，覆盖 49 个类别。",
            "用 ChatGPT 生成涉及单工具和多工具场景的指令。",
            "用 ChatGPT 为每条指令搜索有效的 API 调用链作为 solution path。",
        ],
    },
    {
        "id": "metagpt-sop",
        "paper": "metagpt",
        "split": "dev",
        "question": "MetaGPT 为什么在多智能体协作中引入标准作业程序 SOP？",
        "types": ["synthesis"],
        "stress": "control",
        "points": [
            "把 SOP 编码为提示序列以形成更顺畅的工作流。",
            "让具备领域角色的智能体核验中间结果并减少级联错误。",
        ],
    },
    {
        "id": "metagpt-roles",
        "paper": "metagpt",
        "split": "dev",
        "question": "MetaGPT 如何用装配线范式组织复杂任务？",
        "types": ["multi_hop"],
        "stress": "multi_hop_gap",
        "points": [
            "把不同角色分配给不同智能体。",
            "将复杂任务拆成由多个智能体协作完成的子任务。",
            "角色化智能体核验标准化的中间产物以提高一致性。",
        ],
    },
    {
        "id": "gaia-abilities",
        "paper": "gaia",
        "split": "dev",
        "question": "GAIA 基准的问题要求通用 AI 助手具备哪些基础能力？",
        "types": ["synthesis"],
        "stress": "control",
        "points": ["推理。", "多模态处理。", "网页浏览。", "工具使用能力。"],
    },
    {
        "id": "gaia-results",
        "paper": "gaia",
        "split": "dev",
        "question": "GAIA 报告的人类与带插件 GPT-4 的表现差距及数据规模是多少？",
        "types": ["metric", "comparison"],
        "stress": "comparison_side",
        "points": [
            "人类受试者得分为 92%，带插件的 GPT-4 为 15%。",
            "GAIA 构建了 466 个问题。",
            "其中 300 个答案被保留用于排行榜。",
        ],
    },
    {
        "id": "paperqa-motivation",
        "paper": "paperqa",
        "split": "dev",
        "question": "PaperQA 为什么需要用检索增强方法回答科学研究问题？",
        "types": ["synthesis"],
        "stress": "control",
        "points": [
            "普通 LLM 会产生幻觉且缺乏可解释性。",
            "RAG 可减少幻觉并为答案生成过程提供来源依据。",
            "科学文献数量巨大且发现过程仍高度依赖人工。",
        ],
    },
    {
        "id": "paperqa-workflow",
        "paper": "paperqa",
        "split": "dev",
        "question": "PaperQA 回答科学问题时依次执行哪些关键工作？",
        "types": ["multi_hop"],
        "stress": "multi_hop_gap",
        "points": [
            "在科学论文全文中进行信息检索。",
            "评估来源和段落的相关性。",
            "使用检索到的内容通过 RAG 生成答案。",
        ],
    },
    {
        "id": "graphrag-global-questions",
        "paper": "graphrag",
        "split": "dev",
        "question": "传统 RAG 为什么难以回答面向整个文档集合的全局问题？",
        "types": ["synthesis"],
        "stress": "control",
        "points": [
            "全局问题本质上是面向整个语料的查询聚焦摘要任务。",
            "传统 RAG 只检索少量局部相关记录，难以覆盖整个语料。",
            "既有查询聚焦摘要方法又难以扩展到典型 RAG 的文本规模。",
        ],
    },
    {
        "id": "graphrag-goal",
        "paper": "graphrag",
        "split": "dev",
        "question": "GraphRAG 的图方法旨在怎样改进面向语料库的摘要回答？",
        "types": ["synthesis"],
        "stress": "control",
        "points": [
            "同时适应更一般的用户问题和更大规模的来源文本。",
            "相较传统 RAG 提升全局回答的全面性和多样性。",
        ],
    },
    {
        "id": "graphrag-pipeline",
        "paper": "graphrag",
        "split": "dev",
        "question": "GraphRAG 如何从源文档构建索引并生成全局回答？",
        "types": ["multi_hop"],
        "stress": "multi_hop_gap",
        "points": [
            "先从源文档抽取实体知识图谱。",
            "为紧密相关的实体社区预生成社区摘要。",
            "查询时由各社区摘要生成局部回答，再汇总为最终回答。",
        ],
    },
    {
        "id": "hipporag-memory",
        "paper": "hipporag",
        "split": "dev",
        "question": "HippoRAG 从哺乳动物长期记忆中借鉴了什么核心能力？",
        "types": ["factoid"],
        "stress": "control",
        "points": ["存储大量世界知识。", "持续整合新经验而避免灾难性遗忘。"],
    },
    {
        "id": "hipporag-integration-gap",
        "paper": "hipporag",
        "split": "dev",
        "question": "当前 RAG 为何难以整合跨段知识，HippoRAG 如何应对？",
        "types": ["synthesis"],
        "stress": "control",
        "points": [
            "当前 RAG 往往孤立编码各段，难以整合跨段或跨文档的新知识。",
            "HippoRAG 借鉴海马索引理论建立关联图，以支持更深且更高效的知识整合。",
        ],
    },
    {
        "id": "hipporag-components",
        "paper": "hipporag",
        "split": "dev",
        "question": "HippoRAG 用哪些组件模拟新皮层与海马体的不同记忆角色？",
        "types": ["synthesis"],
        "stress": "control",
        "points": ["大语言模型。", "知识图谱。", "Personalized PageRank 算法。"],
    },
    {
        "id": "medgraphrag-safety",
        "paper": "medgraphrag",
        "split": "dev",
        "question": "MedGraphRAG 如何定位其在医疗问答安全性方面的作用？",
        "types": ["synthesis"],
        "stress": "control",
        "points": [
            "生成有证据支持的医疗回答。",
            "在处理私人医疗数据时提高安全性和可靠性。",
            "回答包含可信来源文档和定义。",
        ],
    },
    {
        "id": "medgraphrag-techniques",
        "paper": "medgraphrag",
        "split": "dev",
        "question": "MedGraphRAG 的 Triple Graph 与 U-Retrieval 分别如何工作？",
        "types": ["multi_hop", "comparison"],
        "stress": "multi_hop_gap",
        "points": [
            "Triple Graph 把用户文档、可信医疗来源和受控词表连接起来。",
            "U-Retrieval 结合自顶向下精确检索与自底向上回答细化。",
            "该组合在全局上下文感知与精确索引之间取得平衡。",
        ],
    },
    {
        "id": "lightrag-limitations",
        "paper": "lightrag",
        "split": "dev",
        "question": "LightRAG 认为已有 RAG 系统存在哪些结构或效率局限？",
        "types": ["synthesis"],
        "stress": "control",
        "points": [
            "依赖平面数据表示，难以表达实体间复杂关系。",
            "上下文感知不足，容易产生碎片化且不连贯的答案。",
        ],
    },
    {
        "id": "lightrag-design",
        "paper": "lightrag",
        "split": "dev",
        "question": "LightRAG 用哪些设计同时改善上下文检索和数据更新？",
        "types": ["synthesis"],
        "stress": "control",
        "points": [
            "把图结构纳入文本索引和检索。",
            "使用低层与高层知识发现的双层检索。",
            "结合图结构与向量表示检索实体及关系。",
            "用增量更新算法及时整合新数据。",
        ],
    },
    {
        "id": "agentic-rag-motivation",
        "paper": "agentic-rag",
        "split": "dev",
        "question": "Agentic RAG 综述认为静态训练数据给大语言模型带来什么限制？",
        "types": ["factoid"],
        "stress": "control",
        "points": ["难以响应动态、实时查询。", "会生成过时或不准确的输出。"],
    },
    {
        "id": "agentic-rag-design-patterns",
        "paper": "agentic-rag",
        "split": "dev",
        "question": "Agentic RAG 用哪些智能体设计模式克服传统 RAG 的静态工作流？",
        "types": ["synthesis"],
        "stress": "control",
        "points": ["反思。", "规划。", "工具使用。", "多智能体协作。"],
    },
    {
        "id": "camel-purpose",
        "paper": "camel",
        "split": "test",
        "question": "CAMEL 研究试图通过什么方式减少复杂任务求解中对人工输入的依赖？",
        "types": ["synthesis"],
        "stress": "control",
        "points": [
            "构建可扩展的交流式智能体自主协作。",
            "用角色扮演和 inception prompting 引导智能体在最少人工监督下完成任务。",
        ],
    },
    {
        "id": "camel-roleplay",
        "paper": "camel",
        "split": "test",
        "question": "CAMEL 如何利用角色扮演和交流式智能体完成任务？",
        "types": ["multi_hop"],
        "stress": "multi_hop_gap",
        "points": [
            "为交流式智能体设定角色扮演框架。",
            "使用 inception prompting 自主引导对话走向任务完成。",
            "在推进任务时保持与人类意图一致。",
        ],
    },
    {
        "id": "camel-acronym-exact",
        "paper": "camel",
        "split": "test",
        "question": "缩写 CAMEL 对应的智能体协作方法具有什么核心设计？",
        "types": ["factoid"],
        "stress": "control",
        "points": [
            "核心设计是交流式智能体的角色扮演框架。",
            "框架以 inception prompting 引导协作。",
        ],
    },
    {
        "id": "camel-research-use",
        "paper": "camel",
        "split": "test",
        "question": "CAMEL 的角色扮演框架除完成任务外还能生成什么研究资源？",
        "types": ["factoid"],
        "stress": "terminology_mismatch",
        "points": [
            "生成用于研究智能体社会行为与能力的对话数据。",
            "支持研究多智能体场景中的指令遵循式协作。",
        ],
    },
    {
        "id": "tree-of-thoughts-motivation",
        "paper": "tree-of-thoughts",
        "split": "test",
        "question": "Tree of Thoughts 主要解决语言模型逐 token 左到右决策的什么局限？",
        "types": ["synthesis"],
        "stress": "control",
        "points": [
            "难以探索多个候选路径。",
            "缺乏战略前瞻。",
            "早期错误决策会对后续过程产生关键影响。",
        ],
    },
    {
        "id": "tree-of-thoughts-search",
        "paper": "tree-of-thoughts",
        "split": "test",
        "question": "Tree of Thoughts 为什么要在多个推理思路之间进行探索？",
        "types": ["multi_hop"],
        "stress": "multi_hop_gap",
        "points": [
            "把连贯文本思路作为问题求解的中间步骤。",
            "比较多条推理路径并自我评估选择。",
            "必要时进行前瞻或回溯以作出全局选择。",
        ],
    },
    {
        "id": "tree-of-thoughts-hyphenated",
        "paper": "tree-of-thoughts",
        "split": "test",
        "question": "Tree-of-Thoughts 式搜索如何弥补逐 token 推理的局限？",
        "types": ["synthesis"],
        "stress": "entity_alias",
        "points": [
            "在连贯思路而非单个 token 的粒度上探索。",
            "允许多路径比较、自我评估、前瞻和回溯。",
        ],
    },
    {
        "id": "tree-of-thoughts-tasks",
        "paper": "tree-of-thoughts",
        "split": "test",
        "question": "Tree of Thoughts 在哪些需要非平凡规划或搜索的任务上进行了实验？",
        "types": ["factoid"],
        "stress": "control",
        "points": ["Game of 24。", "Creative Writing。", "Mini Crosswords。"],
    },
    {
        "id": "tree-of-thoughts-game24",
        "paper": "tree-of-thoughts",
        "split": "test",
        "question": (
            "Tree of Thoughts 与 GPT-4 Chain-of-Thought 在 Game of 24 上的成功率分别是多少？"
        ),
        "types": ["metric", "comparison"],
        "stress": "comparison_side",
        "points": ["Tree of Thoughts 的成功率为 74%。", "GPT-4 Chain-of-Thought 的成功率为 4%。"],
    },
    {
        "id": "voyager-components",
        "paper": "voyager",
        "split": "test",
        "question": "Voyager 在 Minecraft 中持续学习所依赖的三个核心组件是什么？",
        "types": ["synthesis"],
        "stress": "control",
        "points": [
            "最大化探索的自动课程。",
            "保存和检索复杂行为的可执行代码技能库。",
            "结合环境反馈、执行错误和自我验证的迭代提示机制。",
        ],
    },
    {
        "id": "voyager-lifelong-learning",
        "paper": "voyager",
        "split": "test",
        "question": "Voyager 为什么被称为具身终身学习智能体？",
        "types": ["synthesis"],
        "stress": "control",
        "points": [
            "在没有人工干预时持续探索 Minecraft 世界。",
            "不断获得多样技能并作出新发现。",
            "可在新世界复用技能库，从零完成新任务。",
        ],
    },
    {
        "id": "voyager-skill-library",
        "paper": "voyager",
        "split": "test",
        "question": "Voyager 的技能库以什么形式保存能力，并带来哪些性质？",
        "types": ["synthesis"],
        "stress": "multi_hop_gap",
        "points": [
            "以不断增长的可执行代码库保存和检索复杂行为。",
            "技能具有时间延展性、可解释性和可组合性。",
            "技能组合可快速累积能力并缓解灾难性遗忘。",
        ],
    },
    {
        "id": "voyager-results",
        "paper": "voyager",
        "split": "test",
        "question": "Voyager 相比既有最佳方法在 Minecraft 中报告了哪些量化提升？",
        "types": ["metric", "comparison"],
        "stress": "metric_missing",
        "points": [
            "获得 3.3 倍更多的独特物品。",
            "行进距离长 2.3 倍。",
            "关键科技树里程碑解锁快至 15.3 倍。",
        ],
    },
    {
        "id": "agentbench-scope",
        "paper": "agentbench",
        "split": "test",
        "question": "AgentBench 用什么类型的环境对 LLM 智能体进行多维评测？",
        "types": ["factoid"],
        "stress": "control",
        "points": ["由 8 个不同的交互环境组成。", "评估 LLM 智能体的推理和决策能力。"],
    },
    {
        "id": "agentbench-environment-focus",
        "paper": "agentbench",
        "split": "test",
        "question": "AgentBench 基准的交互环境重点评估智能体的哪些能力？",
        "types": ["factoid"],
        "stress": "control",
        "points": ["推理能力。", "决策能力。"],
    },
    {
        "id": "agentbench-model-results",
        "paper": "agentbench",
        "split": "test",
        "question": "AgentBench 评测了多少个模型，商业模型与不超过 70B 的开源模型表现有何差异？",
        "types": ["metric", "comparison"],
        "stress": "comparison_side",
        "points": [
            "评测覆盖 29 个 API 模型和开源模型。",
            "顶级商业模型明显强于许多参数不超过 70B 的开源模型。",
        ],
    },
    {
        "id": "agentbench-failures",
        "paper": "agentbench",
        "split": "test",
        "question": "AgentBench 识别出的可用 LLM 智能体主要失败障碍是什么？",
        "types": ["synthesis"],
        "stress": "control",
        "points": ["长期推理能力不足。", "决策能力不足。", "指令遵循能力不足。"],
    },
    {
        "id": "lats-capabilities",
        "paper": "lats",
        "split": "test",
        "question": "Language Agent Tree Search 统一了语言智能体的哪些关键能力？",
        "types": ["synthesis"],
        "stress": "control",
        "points": ["推理。", "行动。", "规划。"],
    },
    {
        "id": "lats-search-mechanism",
        "paper": "lats",
        "split": "test",
        "question": "LATS 如何把树搜索、模型自省和环境反馈结合起来改善决策？",
        "types": ["multi_hop"],
        "stress": "entity_alias",
        "points": [
            "把 Monte Carlo Tree Search 集成到语言智能体中。",
            "使用由语言模型驱动的价值函数和自我反思进行探索。",
            "引入环境的外部反馈以形成更审慎、适应性的求解过程。",
        ],
    },
    {
        "id": "lats-domains",
        "paper": "lats",
        "split": "test",
        "question": "LATS 在哪些不同领域验证了决策框架的通用性？",
        "types": ["factoid"],
        "stress": "control",
        "points": ["编程。", "交互式问答。", "网页导航。", "数学。"],
    },
]


def _literature_questions() -> list[UnifiedEvaluationQuestion]:
    output: list[UnifiedEvaluationQuestion] = []
    source_questions = {
        str(item["question_id"]): str(item["question"])
        for item in _load_questions("hybrid_dev_questions.json")
    }
    for item in LITERATURE_QUESTIONS:
        paper = PAPERS[str(item["paper"])]
        question_id = f"ueq-literature-{item['id']}"
        points = [str(value) for value in item["points"]]
        source_id = str(item["id"])
        question = str(item["question"])
        source_fixture = (
            f"data/evaluation/hybrid_dev_questions.json#{source_id}"
            if source_questions.get(source_id) == question
            else f"{QUESTION_SOURCES_PATH}#{question_id}"
        )
        output.append(
            _question(
                question_id=question_id,
                split=EvaluationSplit(str(item["split"])),
                question=question,
                question_types=[UnifiedQuestionType(value) for value in item["types"]],
                stress=StressCategory(str(item["stress"])),
                knowledge=KnowledgeSufficiency.SUFFICIENT,
                evidence_ids=[paper["chunk_id"]],
                points=points,
                facets=points,
                source_group_ids=[f"paper:{paper['paper_id']}"],
                source_fixture=source_fixture,
            )
        )
    return output


def _corrective_evidence(question_ids: set[str]) -> list[EvaluationEvidenceRecord]:
    records: list[EvaluationEvidenceRecord] = []
    for item in _load_questions("corrective_dev_questions.json"):
        source_id = str(item["question_id"])
        if source_id not in question_ids:
            continue
        candidates = [*item.get("initial_evidence", [])]
        for values in item.get("action_evidence", {}).values():
            candidates.extend(values)
        for evidence in candidates:
            evidence_id = str(evidence["evidence_id"])
            content = str(evidence["content"])
            source_type = str(evidence.get("source_type", "chunk"))
            records.append(
                EvaluationEvidenceRecord(
                    evidence_id=evidence_id,
                    source_group_id=f"paper:{evidence['paper_id']}",
                    source_kind="graph" if source_type == "graph" else "chunk",
                    source_locator=f"data/evaluation/corrective_dev_questions.json#{source_id}/{evidence_id}",
                    content=content,
                    content_hash=_sha256(content),
                )
            )
    return records


def _paper_evidence() -> list[EvaluationEvidenceRecord]:
    """从当前冻结 processed 语料解析目标 Chunk，并把正文固化到候选目录。"""

    targets = {value["chunk_id"]: value for value in PAPERS.values()}
    found: dict[str, EvaluationEvidenceRecord] = {}
    for path in sorted((PROJECT_ROOT / "data" / "processed").rglob("chunks.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            payload = json.loads(line)
            chunk_id = str(payload.get("chunk_id", ""))
            if chunk_id not in targets:
                continue
            paper = targets[chunk_id]
            actual_paper_id = str(payload.get("paper_id", ""))
            if actual_paper_id != paper["paper_id"]:
                raise ValueError(f"paper identity drifted for {chunk_id}: {actual_paper_id!r}")
            content = str(payload.get("text", "")).strip()
            if not content:
                raise ValueError(f"target Chunk has no text: {chunk_id}")
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            record = EvaluationEvidenceRecord(
                evidence_id=chunk_id,
                source_group_id=f"paper:{actual_paper_id}",
                source_kind="chunk",
                source_locator=f"{relative}:{line_number}",
                content=content,
                content_hash=_sha256(content),
            )
            if previous := found.get(chunk_id):
                if previous.content_hash != record.content_hash:
                    raise ValueError(f"target Chunk has conflicting copies: {chunk_id}")
            else:
                found[chunk_id] = record
    if missing := targets.keys() - found.keys():
        raise ValueError(f"processed corpus is missing target Chunks: {sorted(missing)}")
    return list(found.values())


def _question_sources(
    questions: list[UnifiedEvaluationQuestion],
) -> EvaluationQuestionSourceCatalog:
    """区分原题迁移和基于冻结 Chunk 新增的人工命题。"""

    records: list[EvaluationQuestionSourceRecord] = []
    derived_prefix = f"{QUESTION_SOURCES_PATH}#"
    for question in questions:
        evidence_ids = list(question.relevant_evidence)
        derived = question.source_fixture.startswith(derived_prefix)
        records.append(
            EvaluationQuestionSourceRecord(
                question_id=question.question_id,
                question=question.question,
                origin=(
                    EvaluationQuestionOrigin.CHUNK_DERIVED
                    if derived
                    else EvaluationQuestionOrigin.FIXTURE_MIGRATION
                ),
                origin_locator=(
                    f"evidence:{','.join(evidence_ids)}" if derived else question.source_fixture
                ),
                evidence_ids=evidence_ids,
            )
        )
    return EvaluationQuestionSourceCatalog(
        dataset_version=DATASET_VERSION,
        records=sorted(records, key=lambda item: item.question_id),
    )


def build_dataset() -> tuple[
    EvaluationDatasetManifest,
    EvaluationEvidenceCatalog,
    EvaluationQuestionSourceCatalog,
    UnifiedQuestionSet,
    UnifiedQuestionSet,
]:
    questions = [*_corrective_questions(), *_literature_questions()]
    dev = UnifiedQuestionSet(
        dataset_version=DATASET_VERSION,
        split=EvaluationSplit.DEV,
        questions=[item for item in questions if item.split is EvaluationSplit.DEV],
    )
    test = UnifiedQuestionSet(
        dataset_version=DATASET_VERSION,
        split=EvaluationSplit.TEST,
        questions=[item for item in questions if item.split is EvaluationSplit.TEST],
    )
    catalog = EvaluationEvidenceCatalog(
        dataset_version=DATASET_VERSION,
        evidence_version=EVIDENCE_VERSION,
        records=sorted(
            [
                *_corrective_evidence(
                    {f"q{number:02d}" for number in [*range(1, 7), *range(10, 19)]}
                ),
                *_paper_evidence(),
            ],
            key=lambda item: item.evidence_id,
        ),
    )
    question_sources = _question_sources(questions)
    dev_hash = canonical_digest(dev)
    test_hash = canonical_digest(test)
    evidence_hash = canonical_digest(catalog)
    question_sources_hash = canonical_digest(question_sources)
    corpus_snapshot = "corrective-dev-v1+processed-agent-papers-v1"
    manifest = EvaluationDatasetManifest(
        dataset_version=DATASET_VERSION,
        corpus_snapshot=corpus_snapshot,
        evidence_version=EVIDENCE_VERSION,
        evidence_path="data/evaluation/unified/evidence.json",
        evidence_hash=evidence_hash,
        question_sources_path=QUESTION_SOURCES_PATH,
        question_sources_hash=question_sources_hash,
        dev_path="data/evaluation/unified/dev.json",
        test_path="data/evaluation/unified/test.json",
        dev_count=len(dev.questions),
        test_count=len(test.questions),
        dev_hash=dev_hash,
        test_hash=test_hash,
        dataset_hash=dataset_digest(
            dataset_version=DATASET_VERSION,
            corpus_snapshot=corpus_snapshot,
            evidence_version=EVIDENCE_VERSION,
            evidence_hash=evidence_hash,
            question_sources_hash=question_sources_hash,
            dev_hash=dev_hash,
            test_hash=test_hash,
        ),
        annotation_guide_version="unified-annotation-guide-v4",
        reviewed=False,
        review_note=(
            "v4 已收紧术语错配和多跳标签，并移除非必需答案要点；"
            "自动校验通过不等于人工复核，逐题人工签署前不得用于真实开发选择或正式测试。"
        ),
    )
    validate_unified_dataset(
        manifest,
        dev,
        test,
        evidence_catalog=catalog,
        question_source_catalog=question_sources,
        workspace_root=PROJECT_ROOT,
    )
    return manifest, catalog, question_sources, dev, test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="写入候选数据；默认仅 dry-run")
    parser.add_argument("--force", action="store_true", help="允许覆盖现有候选数据")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest, catalog, question_sources, dev, test = build_dataset()
    print(
        f"dataset={manifest.dataset_version} dev={len(dev.questions)} "
        f"test={len(test.questions)} evidence={len(catalog.records)} "
        f"sources={len(question_sources.records)} "
        f"hash={manifest.dataset_hash}"
    )
    if not args.write:
        print("dry-run: no files written")
        return 0
    destinations = [
        OUTPUT_ROOT / "manifest.json",
        OUTPUT_ROOT / "evidence.json",
        OUTPUT_ROOT / "question-sources.json",
        OUTPUT_ROOT / "dev.json",
        OUTPUT_ROOT / "test.json",
    ]
    if not args.force and any(path.exists() for path in destinations):
        raise FileExistsError("unified dataset exists; use --force for an intentional rebuild")
    write_json_atomic(destinations[1], catalog)
    write_json_atomic(destinations[2], question_sources)
    write_json_atomic(destinations[3], dev)
    write_json_atomic(destinations[4], test)
    write_json_atomic(destinations[0], manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
