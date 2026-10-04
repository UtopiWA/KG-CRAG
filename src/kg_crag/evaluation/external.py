"""真实评测依赖的显式门禁与健康检查边界。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from kg_crag.evaluation.config import UnifiedEvaluationConfig


class ExternalServiceProbe(Protocol):
    async def healthy(self) -> bool: ...


async def validate_external_services(
    config: UnifiedEvaluationConfig,
    probes: Mapping[str, ExternalServiceProbe],
    *,
    confirmed_budget: bool,
) -> None:
    """先完成开关、预算和健康检查，再允许构造实际评测阶段。"""

    if not config.online:
        raise ValueError("real evaluation services require online=true")
    if not confirmed_budget:
        raise ValueError("real evaluation services require budget confirmation")
    if not probes:
        raise ValueError("online evaluation requires at least one declared service")
    failures: list[str] = []
    for name, probe in probes.items():
        try:
            if not await probe.healthy():
                failures.append(name)
        except Exception:
            failures.append(name)
    if failures:
        raise RuntimeError(f"external service health checks failed: {sorted(failures)}")
