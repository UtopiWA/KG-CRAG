"""公共模型 Schema 快照的确定性测试。"""

import runpy
from collections.abc import Callable
from pathlib import Path
from typing import cast


def _load_exporter() -> Callable[[Path], int]:
    script = Path(__file__).parents[2] / "scripts" / "export_model_schemas.py"
    namespace = runpy.run_path(str(script))
    exporter = cast(Callable[..., int], namespace["export_schemas"])

    def run(output_dir: Path, *, check: bool = False) -> int:
        return exporter(output_dir, check=check)

    return run


def test_schema_export_is_byte_stable_and_check_detects_drift(tmp_path: Path) -> None:
    export = _load_exporter()

    assert export(tmp_path) == 0
    first = {path.name: path.read_bytes() for path in sorted(tmp_path.glob("*.json"))}
    assert len(first) == 16

    assert export(tmp_path) == 0
    second = {path.name: path.read_bytes() for path in sorted(tmp_path.glob("*.json"))}
    assert second == first
    assert export(tmp_path, check=True) == 0

    drifted = tmp_path / "paper.schema.json"
    drifted.write_bytes(b"{}\n")
    assert export(tmp_path, check=True) == 1
