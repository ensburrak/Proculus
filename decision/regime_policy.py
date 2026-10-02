from __future__ import annotations

from typing import Any


def normalize_regime(value: Any) -> str:
    token = str(value or "unknown").strip().lower()
    aliases = {
        "strong_bull": "bull", "weak_bull": "bull", "uptrend": "bull",
        "strong_bear": "bear", "weak_bear": "bear", "downtrend": "bear",
        "sideways": "range", "ranging": "range",
        "squeeze": "compression",
        "volatile": "shock", "crisis": "shock", "high_vol": "shock",
    }
    return aliases.get(token, token)


def get_regime_policy(regime: Any) -> dict[str, Any]:
    r = normalize_regime(regime)
    policies = {
        "bull": {"regime": r, "action": "trade", "directions": {"long"}, "min_confidence": 0.62, "max_leverage": 2},
        "bear": {"regime": r, "action": "trade", "directions": {"short"}, "min_confidence": 0.62, "max_leverage": 2},
        "range": {"regime": r, "action": "trade", "directions": {"long", "short"}, "min_confidence": 0.66, "max_leverage": 1},
        "compression": {"regime": r, "action": "confirm", "directions": {"long", "short"}, "min_confidence": 0.70, "max_leverage": 1},
        "transition": {"regime": r, "action": "confirm", "directions": {"long", "short"}, "min_confidence": 0.74, "max_leverage": 1},
        "shock": {"regime": r, "action": "no_trade", "directions": set(), "min_confidence": 1.0, "max_leverage": 0},
        "conflict": {"regime": r, "action": "no_trade", "directions": set(), "min_confidence": 1.0, "max_leverage": 0},
        "unknown": {"regime": r, "action": "no_trade", "directions": set(), "min_confidence": 1.0, "max_leverage": 0},
    }
    return dict(policies.get(r, policies["unknown"]))
