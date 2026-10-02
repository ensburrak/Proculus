from __future__ import annotations

import math
from typing import Any


def _f(value: Any) -> float | None:
    try:
        parsed = float(value)
        return parsed if parsed == parsed else None
    except (TypeError, ValueError):
        return None


def apply_llm_direction_override(*args: Any, **kwargs: Any) -> str:
    """Pipeline V2 forbids LLM direction authority."""
    if "base_decision" in kwargs:
        return str(kwargs.get("base_decision") or "neutral")
    if len(args) >= 2:
        return str(args[1] or "neutral")
    return "neutral"


def base_decision_from_ema(ema_fast: Any, ema_slow: Any) -> str:
    fast, slow = _f(ema_fast), _f(ema_slow)
    if fast is None or slow is None or fast == slow:
        return "neutral"
    return "long" if fast > slow else "short"


def safe_sigmoid(value: float) -> float:
    value = max(-60.0, min(60.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def compress_high_confidence(value: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    return value if value <= 0.85 else 0.85 + (value - 0.85) * 0.85


def horizon_multiplier(tf: str | None) -> float:
    tf = str(tf or "").lower()
    if tf in {"5m", "15m"}:
        return 0.90
    if tf in {"6h", "8h", "12h", "1d"}:
        return 1.10
    return 1.0


def is_horizon_allowed(tf: str | None, regime: str | None) -> bool:
    tf = str(tf or "").lower()
    regime = str(regime or "").upper()
    if regime in {"RANGE", "SIDEWAYS"} or "BEAR" in regime:
        return tf in {"", "5m", "15m", "30m", "1h"}
    return True


def tech_score(ta_pack: dict[str, Any], base_decision: str) -> dict[str, Any]:
    rsi = _f(ta_pack.get("rsi"))
    adx = _f(ta_pack.get("adx"))
    score = 0.55 if base_decision in {"long", "short"} else 0.45
    if adx is not None and adx >= 25:
        score += 0.10
    if rsi is not None:
        if base_decision == "long" and 45 <= rsi <= 70:
            score += 0.05
        elif base_decision == "short" and 30 <= rsi <= 55:
            score += 0.05
    return {"score": max(0.0, min(1.0, score)), "why": "deterministic technical score"}


def enhanced_tech_score(ta_pack: dict[str, Any], base_decision: str, regime: str | None = None) -> dict[str, Any]:
    result = tech_score(ta_pack, base_decision)
    result["regime"] = regime
    return result
