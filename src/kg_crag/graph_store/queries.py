"""Neo4j 固定、参数化且有界的只读查询模板。"""

# Cypher 长行保持模板结构可审计，禁止自动拆成易遗漏参数的字符串片段。
# ruff: noqa: E501

from kg_crag.models import GraphQueryTemplate

QUERY_TEMPLATES: dict[GraphQueryTemplate, str] = {
    GraphQueryTemplate.ENTITY_NEIGHBORS: """
MATCH p=(start)-[rels*1..1]-(target)
WHERE start.graph_version = $graph_version AND target.graph_version = $graph_version
  AND (start.entity_id IN $entity_ids OR start.normalized_name IN $entity_names)
  AND (size($node_types) = 0 OR target.node_type IN $node_types)
  AND (size($paper_ids) = 0 OR all(r IN rels WHERE r.source_paper_id IN $paper_ids))
  AND all(r IN rels WHERE r.graph_version = $graph_version AND r.confidence >= $min_confidence)
RETURN [n IN nodes(p) | n.entity_id] AS entity_ids, [r IN relationships(p) | r.fact_id] AS fact_ids,
       [r IN relationships(p) | r.provenance] AS provenances, [n IN nodes(p) | n.name] AS names,
       reduce(score = 1.0, r IN relationships(p) | CASE WHEN r.confidence < score THEN r.confidence ELSE score END) AS score
ORDER BY score DESC, fact_ids LIMIT $max_candidates
""",
    GraphQueryTemplate.RELATION_LOOKUP: """
MATCH p=(start)-[rels*1..1]-(target)
WHERE start.graph_version = $graph_version AND target.graph_version = $graph_version
  AND (start.entity_id IN $entity_ids OR start.normalized_name IN $entity_names)
  AND (size($node_types) = 0 OR target.node_type IN $node_types)
  AND (size($paper_ids) = 0 OR all(r IN rels WHERE r.source_paper_id IN $paper_ids))
  AND all(r IN rels WHERE r.graph_version = $graph_version AND type(r) IN $relation_types AND r.confidence >= $min_confidence)
RETURN [n IN nodes(p) | n.entity_id] AS entity_ids, [r IN relationships(p) | r.fact_id] AS fact_ids,
       [r IN relationships(p) | r.provenance] AS provenances, [n IN nodes(p) | n.name] AS names,
       reduce(score = 1.0, r IN relationships(p) | CASE WHEN r.confidence < score THEN r.confidence ELSE score END) AS score
ORDER BY score DESC, fact_ids LIMIT $max_candidates
""",
    GraphQueryTemplate.BOUNDED_PATH: """
MATCH p=(start)-[rels*1..3]-(target)
WHERE start.graph_version = $graph_version AND target.graph_version = $graph_version AND length(p) <= $max_hops
  AND (start.entity_id IN $entity_ids OR start.normalized_name IN $entity_names)
  AND (size($node_types) = 0 OR target.node_type IN $node_types)
  AND (size($paper_ids) = 0 OR all(r IN rels WHERE r.source_paper_id IN $paper_ids))
  AND all(r IN rels WHERE r.graph_version = $graph_version AND type(r) IN $relation_types AND r.confidence >= $min_confidence)
RETURN [n IN nodes(p) | n.entity_id] AS entity_ids, [r IN relationships(p) | r.fact_id] AS fact_ids,
       [r IN relationships(p) | r.provenance] AS provenances, [n IN nodes(p) | n.name] AS names,
       reduce(score = 1.0, r IN relationships(p) | CASE WHEN r.confidence < score THEN r.confidence ELSE score END) AS score
ORDER BY score DESC, length(p), fact_ids LIMIT $max_candidates
""",
    GraphQueryTemplate.METHOD_EVIDENCE: """
MATCH p=(start)-[rels*1..2]-(target)
WHERE start.graph_version = $graph_version AND target.graph_version = $graph_version
  AND (start.entity_id IN $entity_ids OR start.normalized_name IN $entity_names)
  AND (size($node_types) = 0 OR target.node_type IN $node_types)
  AND (size($paper_ids) = 0 OR all(r IN rels WHERE r.source_paper_id IN $paper_ids))
  AND all(r IN rels WHERE r.graph_version = $graph_version AND r.confidence >= $min_confidence)
RETURN [n IN nodes(p) | n.entity_id] AS entity_ids, [r IN relationships(p) | r.fact_id] AS fact_ids,
       [r IN relationships(p) | r.provenance] AS provenances, [n IN nodes(p) | n.name] AS names,
       reduce(score = 1.0, r IN relationships(p) | CASE WHEN r.confidence < score THEN r.confidence ELSE score END) AS score
ORDER BY score DESC, length(p), fact_ids LIMIT $max_candidates
""",
}


def query_text(template: GraphQueryTemplate) -> str:
    return QUERY_TEMPLATES[template].strip()
