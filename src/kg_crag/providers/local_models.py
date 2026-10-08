"""本地模型解析的进程级离线约束。"""

from __future__ import annotations

import os


def enforce_local_model_resolution(local_files_only: bool) -> None:
    """禁止上游库在本地模式下继续探测模型仓库。"""

    if not local_files_only:
        return
    # 部分 Transformers 路径不会完整传递 local_files_only，必须同步官方离线开关。
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
