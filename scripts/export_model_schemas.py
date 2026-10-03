"""确定性导出公共 Pydantic 模型的 JSON Schema 快照。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import BaseModel

from kg_crag.models import (
    Chunk,
    CorrectionRunResult,
    CorrectionState,
    CorrectiveEvaluationQuestionSet,
    CorrectiveEvaluationReport,
    DenseEvaluationReport,
    DenseRAGResult,
    ErrorDetail,
    Evidence,
    GraphBundle,
    GraphEvaluationQuestionSet,
    GraphPathHit,
    GraphQueryRequest,
    HealthResponse,
    HybridEvaluationReport,
    HybridQueryResult,
    HybridRetrievalResult,
    IndexRunManifest,
    IngestionRunManifest,
    Paper,
    ParsedDocument,
    PilotManifest,
    PilotQuestionSet,
    QualityReport,
    RetrievalEvaluation,
    RouteDecision,
    SparseIndexRunManifest,
    TraceEvent,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs" / "schemas"

SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "chunk": Chunk,
    "correction_run_result": CorrectionRunResult,
    "correction_state": CorrectionState,
    "corrective_evaluation_question_set": CorrectiveEvaluationQuestionSet,
    "corrective_evaluation_report": CorrectiveEvaluationReport,
    "dense_evaluation_report": DenseEvaluationReport,
    "dense_rag_result": DenseRAGResult,
    "error_detail": ErrorDetail,
    "evidence": Evidence,
    "graph_bundle": GraphBundle,
    "graph_evaluation_question_set": GraphEvaluationQuestionSet,
    "graph_path_hit": GraphPathHit,
    "graph_query_request": GraphQueryRequest,
    "health_response": HealthResponse,
    "hybrid_evaluation_report": HybridEvaluationReport,
    "hybrid_query_result": HybridQueryResult,
    "hybrid_retrieval_result": HybridRetrievalResult,
    "ingestion_run_manifest": IngestionRunManifest,
    "index_run_manifest": IndexRunManifest,
    "paper": Paper,
    "parsed_document": ParsedDocument,
    "pilot_manifest": PilotManifest,
    "pilot_question_set": PilotQuestionSet,
    "quality_report": QualityReport,
    "retrieval_evaluation": RetrievalEvaluation,
    "route_decision": RouteDecision,
    "sparse_index_run_manifest": SparseIndexRunManifest,
    "trace_event": TraceEvent,
}


def render_schema(model: type[BaseModel]) -> bytes:
    """以排序键、固定缩进和末尾换行生成稳定 UTF-8 字节。"""

    payload = json.dumps(
        model.model_json_schema(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return f"{payload}\n".encode()


def export_schemas(output_dir: Path, *, check: bool) -> int:
    """写入全部快照，或只检查磁盘内容是否与当前模型一致。"""

    mismatches: list[Path] = []
    for name, model in SCHEMA_MODELS.items():
        destination = output_dir / f"{name}.schema.json"
        expected = render_schema(model)
        if check:
            if not destination.is_file() or destination.read_bytes() != expected:
                mismatches.append(destination)
            continue
        if destination.is_file() and destination.read_bytes() == expected:
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(expected)

    if mismatches:
        for destination in mismatches:
            print(f"schema snapshot mismatch: {destination}")
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="schema 快照目录 (默认: docs/schemas)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="只检查快照是否最新，不写文件",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return export_schemas(args.output_dir, check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
