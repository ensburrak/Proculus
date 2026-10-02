from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SetupPromotionDecision:
    setup_id: str
    promoted: bool
    trades: int
    profit_factor: float
    expectancy_r: float
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "setup_id": self.setup_id,
            "promoted": self.promoted,
            "trades": self.trades,
            "profit_factor": self.profit_factor,
            "expectancy_r": self.expectancy_r,
            "blockers": list(self.blockers),
        }


def evaluate_setup_promotions(
    by_setup: Mapping[str, Mapping[str, Any]] | None,
    *,
    min_trades: int = 100,
    min_profit_factor: float = 1.15,
    min_expectancy_r: float = 0.05,
) -> dict[str, SetupPromotionDecision]:
    decisions: dict[str, SetupPromotionDecision] = {}
    for setup_id, raw in sorted((by_setup or {}).items()):
        stats = raw if isinstance(raw, Mapping) else {}
        try:
            trades = int(stats.get("trades") or 0)
        except (TypeError, ValueError):
            trades = 0
        try:
            profit_factor = float(stats.get("profit_factor") or 0.0)
        except (TypeError, ValueError):
            profit_factor = 0.0
        try:
            expectancy_r = float(stats.get("expectancy_r") or 0.0)
        except (TypeError, ValueError):
            expectancy_r = 0.0

        blockers: list[str] = []
        if trades < int(min_trades):
            blockers.append("sample_lt_min")
        if profit_factor < float(min_profit_factor):
            blockers.append("profit_factor_lt_min")
        if expectancy_r < float(min_expectancy_r):
            blockers.append("expectancy_r_lt_min")

        decisions[str(setup_id)] = SetupPromotionDecision(
            setup_id=str(setup_id),
            promoted=not blockers,
            trades=trades,
            profit_factor=profit_factor,
            expectancy_r=expectancy_r,
            blockers=tuple(blockers),
        )
    return decisions


def promoted_setup_ids(decisions: Mapping[str, SetupPromotionDecision]) -> set[str]:
    return {
        setup_id
        for setup_id, decision in decisions.items()
        if bool(decision.promoted)
    }


__all__ = [
    "SetupPromotionDecision",
    "evaluate_setup_promotions",
    "promoted_setup_ids",
]
