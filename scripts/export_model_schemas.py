"""确定性导出公共 Pydantic 模型的 JSON Schema 快照。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import BaseModel

from kg_crag.models import (
    Chunk,
    ErrorDetail,
    Evidence,
    HealthResponse,
    IngestionRunManifest,
    Paper,
    ParsedDocument,
    PilotManifest,
    QualityReport,
    RetrievalEvaluation,
    RouteDecision,
    TraceEvent,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs" / "schemas"

SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "chunk": Chunk,
    "error_detail": ErrorDetail,
    "evidence": Evidence,
    "health_response": HealthResponse,
    "ingestion_run_manifest": IngestionRunManifest,
    "paper": Paper,
    "parsed_document": ParsedDocument,
    "pilot_manifest": PilotManifest,
    "quality_report": QualityReport,
    "retrieval_evaluation": RetrievalEvaluation,
    "route_decision": RouteDecision,
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
