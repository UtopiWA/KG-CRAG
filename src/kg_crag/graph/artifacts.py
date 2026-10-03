"""图流水线产物的工作区约束和原子写入。"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def workspace_path(workspace_root: Path, relative: str) -> Path:
    root = workspace_root.resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError("graph artifact path escapes the workspace")
    return target


def stable_json_bytes(value: BaseModel | dict[str, Any] | list[Any]) -> bytes:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def atomic_write_json(path: Path, value: BaseModel | dict[str, Any] | list[Any]) -> None:
    """写入、回读校验后替换；失败不留下可见成功文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    data = stable_json_bytes(value)
    try:
        temporary.write_bytes(data)
        if temporary.read_bytes() != data:
            raise OSError("graph artifact staging verification failed")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
