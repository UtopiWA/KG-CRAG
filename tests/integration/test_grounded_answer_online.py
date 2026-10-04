"""显式启用后运行一次 5 题真实 LLM/Search 有界验收。"""

import os

import pytest

from kg_crag.evaluation.grounded_cli import main


@pytest.mark.online
@pytest.mark.skipif(
    os.getenv("KG_CRAG_RUN_GROUNDED_ONLINE") != "1",
    reason="set KG_CRAG_RUN_GROUNDED_ONLINE=1 to run bounded real LLM/Search acceptance",
)
def test_real_grounded_answer_providers_are_bounded() -> None:
    assert main(["--online", "--confirm", "--limit", "5"]) in {0, 1}
