"""摄取配置与版本哈希测试。"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from kg_crag.ingestion.config import (
    IngestionConfig,
    config_hash,
    load_ingestion_config,
    processing_version,
)


def test_default_config_loads_expected_limits() -> None:
    config = load_ingestion_config(Path("configs/default.yaml"))
    assert config.limits.max_papers == 200
    assert config.chunking.tokenizer == "regex-v1"
    assert config.cleaning.repeated_margin_coverage == pytest.approx(0.60)


@pytest.mark.parametrize(
    "chunking",
    [
        {"overlap_tokens": 300},
        {"min_tokens": 501},
        {"target_tokens": 701},
    ],
)
def test_chunking_relationships_are_strict(chunking: dict[str, int]) -> None:
    with pytest.raises(ValidationError, match="overlap < min <= target <= max"):
        IngestionConfig.model_validate({"chunking": chunking})


def test_absolute_and_parent_paths_are_rejected() -> None:
    for value in ("C:/private/raw", "../raw"):
        with pytest.raises(ValidationError, match="relative"):
            IngestionConfig.model_validate({"paths": {"raw_root": value}})


def test_hash_is_stable_and_excludes_machine_paths() -> None:
    first = IngestionConfig()
    second = IngestionConfig.model_validate(
        {"paths": {"raw_root": "corpus/raw", "processed_root": "build/processed"}}
    )
    assert config_hash(first) == config_hash(second)
    assert processing_version("a" * 64, "pymupdf", "1", config_hash(first)) == (
        processing_version("a" * 64, "pymupdf", "1", config_hash(second))
    )
    assert processing_version("b" * 64, "pymupdf", "1", config_hash(first)) != (
        processing_version("a" * 64, "pymupdf", "1", config_hash(first))
    )
