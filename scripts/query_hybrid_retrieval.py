#!/usr/bin/env python3
"""执行一次有界 Hybrid 检索，默认不调用 LLM。"""

from kg_crag.retrieval.hybrid_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
