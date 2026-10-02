from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PortfolioRiskDecision:
    allowed: bool
    risk_scale: float = 1.0
    reason: str = "ok"


def _load_portfolio_risk_limit_pct(*args: Any, **kwargs: Any) -> float:
    del args, kwargs
    return 0.02


def evaluate_portfolio_risk(current_risk_pct: float = 0.0, max_risk_pct: float = 0.02, **kwargs: Any) -> PortfolioRiskDecision:
    del kwargs
    current = max(0.0, float(current_risk_pct))
    maximum = max(1e-9, float(max_risk_pct))
    if current >= maximum:
        return PortfolioRiskDecision(False, 0.0, "portfolio_risk_limit")
    remaining = max(0.0, 1.0 - current / maximum)
    return PortfolioRiskDecision(True, min(1.0, remaining), "ok")


def check_portfolio_level_risk(*args: Any, **kwargs: Any) -> bool:
    return evaluate_portfolio_risk(*args, **kwargs).allowed
