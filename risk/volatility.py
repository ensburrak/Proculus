from __future__ import annotations

from typing import Any

VOLATILITY_TIERS = {
    "low": {"max_risk_scale": 1.0},
    "normal": {"max_risk_scale": 1.0},
    "high": {"max_risk_scale": 0.75},
    "shock": {"max_risk_scale": 0.0},
}


def classify_volatility(atr: Any, price: Any = None) -> dict[str, Any]:
    try:
        atr_f = abs(float(atr))
        price_f = abs(float(price)) if price is not None else 1.0
        ratio = atr_f / price_f if price_f > 0 else atr_f
    except (TypeError, ValueError):
        ratio = 0.0
    if ratio >= 0.05:
        category = "shock"
    elif ratio >= 0.03:
        category = "high"
    elif ratio <= 0.005:
        category = "low"
    else:
        category = "normal"
    return {"category": category, "atr_ratio": ratio, "risk_factor": VOLATILITY_TIERS[category]["max_risk_scale"]}


def adjust_risk_for_volatility(base_risk: float, atr: Any, price: Any = None) -> float:
    info = classify_volatility(atr, price)
    return max(0.0, float(base_risk) * float(info["risk_factor"]))
