"""回答阶段预算的线程安全预留与保守结算。"""

from __future__ import annotations

from threading import Lock
from uuid import uuid4

from kg_crag.models.answer import AnswerBudgetLedger, AnswerBudgetUsage


def _values(value: AnswerBudgetUsage) -> dict[str, int]:
    return {name: int(getattr(value, name)) for name in AnswerBudgetUsage.model_fields}


def add_answer_usage(left: AnswerBudgetUsage, right: AnswerBudgetUsage) -> AnswerBudgetUsage:
    return AnswerBudgetUsage(
        **{
            name: _values(left)[name] + _values(right)[name]
            for name in AnswerBudgetUsage.model_fields
        }
    )


def subtract_answer_usage(left: AnswerBudgetUsage, right: AnswerBudgetUsage) -> AnswerBudgetUsage:
    values = {
        name: _values(left)[name] - _values(right)[name] for name in AnswerBudgetUsage.model_fields
    }
    if any(item < 0 for item in values.values()):
        raise ValueError("cannot release more answer budget than reserved")
    return AnswerBudgetUsage(**values)


def can_reserve_answer(ledger: AnswerBudgetLedger, requested: AnswerBudgetUsage) -> bool:
    combined_reserved = add_answer_usage(ledger.reserved, requested)
    try:
        AnswerBudgetLedger(
            limit=ledger.limit,
            used=ledger.used,
            reserved=combined_reserved,
        )
    except ValueError:
        return False
    return True


class AnswerBudgetManager:
    """每个预留令牌只结算一次，失败调用按最坏预留成本计费。"""

    def __init__(self, ledger: AnswerBudgetLedger) -> None:
        self._ledger = ledger.model_copy(deep=True)
        self._reservations: dict[str, AnswerBudgetUsage] = {}
        self._lock = Lock()

    @property
    def ledger(self) -> AnswerBudgetLedger:
        with self._lock:
            return self._ledger.model_copy(deep=True)

    def reserve(self, requested: AnswerBudgetUsage) -> str | None:
        with self._lock:
            if not can_reserve_answer(self._ledger, requested):
                return None
            token = uuid4().hex
            self._reservations[token] = requested
            self._ledger = self._ledger.model_copy(
                update={"reserved": add_answer_usage(self._ledger.reserved, requested)}
            )
            return token

    def release(self, token: str) -> None:
        with self._lock:
            requested = self._reservations.pop(token)
            self._ledger = self._ledger.model_copy(
                update={"reserved": subtract_answer_usage(self._ledger.reserved, requested)}
            )

    def settle(
        self,
        token: str,
        actual: AnswerBudgetUsage | None = None,
        *,
        failed: bool = False,
    ) -> None:
        with self._lock:
            reserved = self._reservations.pop(token)
            charged = reserved if failed or actual is None else actual
            if any(
                getattr(charged, name) > getattr(reserved, name)
                for name in AnswerBudgetUsage.model_fields
            ):
                self._reservations[token] = reserved
                raise ValueError("actual answer usage exceeds the reserved worst case")
            self._ledger = AnswerBudgetLedger(
                limit=self._ledger.limit,
                used=add_answer_usage(self._ledger.used, charged),
                reserved=subtract_answer_usage(self._ledger.reserved, reserved),
            )
