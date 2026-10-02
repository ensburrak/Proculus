from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .mtf_trend_gate import evaluate_mtf_trend_gate
from .regime_policy import get_regime_policy


@dataclass(frozen=True)
class ExpertSignal:
    setup_id: str
    direction: str
    confidence: float
    strategy: str
    reasoning: list[str]
    max_leverage: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _f(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _range_signal(item: dict[str, Any], ta: dict[str, Any]) -> ExpertSignal | None:
    rsi = _f(ta.get("rsi"))
    stoch = _f(ta.get("stoch_rsi") or ta.get("stochrsi") or ta.get("stoch_k"))
    if stoch is not None and stoch > 1.0:
        stoch /= 100.0

    if (rsi is not None and rsi <= 35) or (stoch is not None and stoch <= 0.10):
        return ExpertSignal("range.mean_reversion.long.15m.v1", "long", 0.70, "mean_reversion", ["range lower extreme"], 1)
    if (rsi is not None and rsi >= 65) or (stoch is not None and stoch >= 0.90):
        return ExpertSignal("range.mean_reversion.short.15m.v1", "short", 0.70, "mean_reversion", ["range upper extreme"], 1)
    return None


def _breakout_signal(item: dict[str, Any]) -> ExpertSignal | None:
    mtf = item.get("mtf_features")
    features = mtf.get("15m", {}) if isinstance(mtf, dict) and isinstance(mtf.get("15m"), dict) else {}
    closes = features.get("recent_closes")
    highs = features.get("recent_highs")
    lows = features.get("recent_lows")
    if not all(isinstance(v, (list, tuple)) and len(v) >= 5 for v in (closes, highs, lows)):
        return None
    close = float(closes[-1])
    prev_high = max(float(x) for x in highs[-5:-1])
    prev_low = min(float(x) for x in lows[-5:-1])
    if close > prev_high:
        return ExpertSignal("compression.breakout.long.15m.v1", "long", 0.74, "breakout", ["confirmed close above compression range"], 1)
    if close < prev_low:
        return ExpertSignal("compression.breakout.short.15m.v1", "short", 0.74, "breakout", ["confirmed close below compression range"], 1)
    return None


def route_to_expert(*, regime: str, item: dict[str, Any], ta: dict[str, Any]) -> ExpertSignal | None:
    policy = get_regime_policy(regime)
    normalized = policy["regime"]
    if policy["action"] == "no_trade":
        return None

    if normalized in {"bull", "bear"}:
        side = "long" if normalized == "bull" else "short"
        gate = evaluate_mtf_trend_gate(side=side, item=item)
        if not gate.allowed:
            return None
        return ExpertSignal(
            f"{normalized}_trend.pullback.{side}.15m.v1",
            side,
            0.72,
            "trend_pullback_resumption",
            ["4h direction", "1h structure", f"{gate.audit.get('lower_tf')} pullback/resumption"],
            int(policy["max_leverage"]),
        )
    if normalized == "range":
        return _range_signal(item, ta)
    if normalized == "compression":
        return _breakout_signal(item)
    return None
