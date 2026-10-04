"""校验统一评测集的 schema、哈希、规模、压力类型和跨划分泄漏。"""

from pathlib import Path

from kg_crag.evaluation.config import load_unified_evaluation_config
from kg_crag.evaluation.dataset import load_unified_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    config = load_unified_evaluation_config(PROJECT_ROOT / "configs" / "evaluation.yaml")
    manifest, dev, test = load_unified_dataset(
        PROJECT_ROOT / config.manifest_path,
        workspace_root=PROJECT_ROOT,
        near_duplicate_threshold=config.near_duplicate_threshold,
    )
    print(
        f"valid dataset={manifest.dataset_version} dev={len(dev.questions)} "
        f"test={len(test.questions)} hash={manifest.dataset_hash}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
