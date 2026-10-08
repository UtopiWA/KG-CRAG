"""把实时演示所需的固定本地模型提前准备到统一缓存目录。"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from kg_crag.errors import KGCRAGError
from kg_crag.retrieval import CrossEncoderReranker, load_hybrid_retrieval_config
from kg_crag.runtime import build_embedding_service
from kg_crag.settings import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="预先下载并加载实时演示的固定模型")
    parser.add_argument(
        "--confirm-download",
        action="store_true",
        help="允许缺少缓存时从模型仓库下载；省略时只显示计划",
    )
    parser.add_argument(
        "--reranker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="同时准备本地 Cross-Encoder 重排器",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="只校验现有缓存，禁止远端模型解析",
    )
    return parser.parse_args(argv)


def _cache_root(settings: Settings) -> Path:
    root = Path(settings.model_cache_root)
    return root.resolve() if root.is_absolute() else (PROJECT_ROOT / root).resolve()


async def _run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.local_files_only:
        # 部分 Sentence Transformers 模型即使收到 local_files_only，仍会探测 Hub API；
        # 离线校验必须同时启用上游库认可的进程级离线开关。
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    settings = Settings().model_copy(update={"model_local_files_only": bool(args.local_files_only)})
    config = load_hybrid_retrieval_config(PROJECT_ROOT / settings.live_retrieval_config_path)
    cache_root = _cache_root(settings)
    plan = {
        "cache_root": str(cache_root),
        "embedding": {
            "model": config.dense.embedding.model,
            "revision": config.dense.embedding.revision,
        },
        "reranker": (
            {
                "model": config.reranker.model,
                "revision": config.reranker.revision,
            }
            if args.reranker
            else None
        ),
        "local_files_only": bool(args.local_files_only),
        "will_prepare": bool(args.confirm_download or args.local_files_only),
    }
    if not plan["will_prepare"]:
        print(json.dumps(plan, ensure_ascii=False, sort_keys=True))
        return 0

    if args.confirm_download:
        cache_root.mkdir(parents=True, exist_ok=True)
        probe = cache_root / ".write-probe"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            raise ValueError("configured model cache root is not writable") from exc
    elif not cache_root.is_dir():
        raise ValueError("configured model cache root does not exist")

    # 最小推理同时验证权重、revision、维度和本机运行时是否可用。
    embedding = build_embedding_service(
        config.dense,
        workspace_root=PROJECT_ROOT,
        settings=settings,
    )
    await embedding.embed(
        ["scientific literature retrieval"],
        input_type="query",
        use_cache=False,
    )
    if args.reranker:
        reranker_config = config.reranker.model_copy(
            update={
                "cache_root": settings.model_cache_root,
                "device": settings.reranker_device,
            }
        )
        await CrossEncoderReranker(
            reranker_config,
            workspace_root=PROJECT_ROOT,
            local_files_only=settings.model_local_files_only,
        ).warmup()
    plan["prepared"] = True
    print(json.dumps(plan, ensure_ascii=False, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_run(argv))
    except (KGCRAGError, OSError, RuntimeError, ValueError) as exc:
        print(f"model preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
