from __future__ import annotations

from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"
MAX_PORTFOLIO_RISK_PERCENTAGE = 0.02
RISK_TIERS = {"low": 0.5, "normal": 1.0, "high": 0.5}
WALLET_CAP_TIERS = {"low": 0.10, "normal": 0.20, "high": 0.10}


def _get_risk_scale(profile: str = "normal", **kwargs: Any) -> float:
    del kwargs
    return float(RISK_TIERS.get(str(profile).lower(), 1.0))


def calculate_tiered_leverage_and_allocation(confidence: float, max_leverage: int = 2, **kwargs: Any) -> tuple[int, float]:
    del kwargs
    c = max(0.0, min(1.0, float(confidence)))
    leverage = 1 if c < 0.80 else min(2, int(max_leverage))
    allocation = min(0.20, 0.05 + c * 0.10)
    return leverage, allocation
