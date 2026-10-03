"""多维预算的线程安全预留、释放与保守结算。"""

from __future__ import annotations

from threading import Lock
from uuid import uuid4

from kg_crag.models import ActionEstimate, BudgetLedger, BudgetUsage


def _usage_values(value: BudgetUsage) -> dict[str, int]:
    return {name: int(getattr(value, name)) for name in BudgetUsage.model_fields}


def add_usage(left: BudgetUsage, right: BudgetUsage) -> BudgetUsage:
    return BudgetUsage(
        **{
            name: _usage_values(left)[name] + _usage_values(right)[name]
            for name in BudgetUsage.model_fields
        }
    )


def subtract_usage(left: BudgetUsage, right: BudgetUsage) -> BudgetUsage:
    values = {
        name: _usage_values(left)[name] - _usage_values(right)[name]
        for name in BudgetUsage.model_fields
    }
    if any(item < 0 for item in values.values()):
        raise ValueError("cannot release more budget than reserved")
    return BudgetUsage(**values)


def estimate_usage(estimate: ActionEstimate) -> BudgetUsage:
    return BudgetUsage(**{name: getattr(estimate, name) for name in BudgetUsage.model_fields})


def can_reserve(ledger: BudgetLedger, estimate: ActionEstimate | BudgetUsage) -> bool:
    requested = estimate_usage(estimate) if isinstance(estimate, ActionEstimate) else estimate
    for name in BudgetUsage.model_fields:
        total = (
            getattr(ledger.used, name) + getattr(ledger.reserved, name) + getattr(requested, name)
        )
        if total > getattr(ledger.limit, name):
            return False
    return True


class BudgetManager:
    """预留令牌只能结算一次；失败调用默认按预留值计费。"""

    def __init__(self, ledger: BudgetLedger) -> None:
        self._ledger = ledger.model_copy(deep=True)
        self._reservations: dict[str, BudgetUsage] = {}
        self._lock = Lock()

    @property
    def ledger(self) -> BudgetLedger:
        with self._lock:
            return self._ledger.model_copy(deep=True)

    def reserve(self, estimate: ActionEstimate | BudgetUsage) -> str | None:
        requested = estimate_usage(estimate) if isinstance(estimate, ActionEstimate) else estimate
        with self._lock:
            if not can_reserve(self._ledger, requested):
                return None
            token = uuid4().hex
            self._reservations[token] = requested
            self._ledger = self._ledger.model_copy(
                update={"reserved": add_usage(self._ledger.reserved, requested)}
            )
            return token

    def release(self, token: str) -> None:
        with self._lock:
            requested = self._reservations.pop(token)
            self._ledger = self._ledger.model_copy(
                update={"reserved": subtract_usage(self._ledger.reserved, requested)}
            )

    def settle(
        self, token: str, actual: BudgetUsage | None = None, *, failed: bool = False
    ) -> None:
        with self._lock:
            reserved = self._reservations.pop(token)
            charged = reserved if failed or actual is None else actual
            # 实际 usage 超过预留时拒绝发布账本，避免静默超支。
            if any(
                getattr(charged, name) > getattr(reserved, name)
                for name in BudgetUsage.model_fields
            ):
                self._reservations[token] = reserved
                raise ValueError("actual usage exceeds the reserved worst case")
            remaining_reserved = subtract_usage(self._ledger.reserved, reserved)
            used = add_usage(self._ledger.used, charged)
            self._ledger = BudgetLedger(
                limit=self._ledger.limit,
                used=used,
                reserved=remaining_reserved,
            )
