from __future__ import annotations

from typing import Any

from decision.mtf_trend_gate import evaluate_mtf_trend_gate
from . import ExpertSignal, f, tf


def evaluate_trend(*, side: str, item: dict[str, Any], ta: dict[str, Any], max_leverage: int) -> ExpertSignal | None:
    gate=evaluate_mtf_trend_gate(side=side,item=item)
    if not gate.allowed:
        return None

    adx=f(ta.get("adx"))
    rsi=f(ta.get("rsi"))
    atr=f(ta.get("atr_ratio") or ta.get("atr_pct"))
    vol_z=f(ta.get("vol_z"))
    ema=ta.get("ema") if isinstance(ta.get("ema"),dict) else {}
    fast=f(ema.get("fast")); slow=f(ema.get("slow"))
    price=f(ta.get("price"))
    h1=tf(item,"1h"); h4=tf(item,"4h")

    if adx is None or adx < 20.0 or adx > 48.0:
        return None
    if atr is not None and (atr < 0.001 or atr > 0.04):
        return None
    if vol_z is not None and abs(vol_z) >= 3.0:
        return None
    if None not in (fast,slow,price):
        if side=="long" and not (fast > slow and price >= slow):
            return None
        if side=="short" and not (fast < slow and price <= slow):
            return None
    if rsi is not None:
        if side=="long" and not (42.0 <= rsi <= 66.0):
            return None
        if side=="short" and not (34.0 <= rsi <= 58.0):
            return None

    score=0.50
    reasons=["4h direction aligned","1h structure aligned","15m pullback/resumption"]
    if 22.0 <= adx <= 38.0:
        score += 0.08; reasons.append(f"ADX quality {adx:.1f}")
    if rsi is not None:
        score += 0.06; reasons.append(f"RSI pullback zone {rsi:.1f}")
    h1_adx=f(h1.get("adx"))
    h4_adx=f(h4.get("adx"))
    if h1_adx is not None and h1_adx >= 18.0:
        score += 0.04; reasons.append("1h trend strength")
    if h4_adx is not None and h4_adx >= 18.0:
        score += 0.04; reasons.append("4h trend strength")
    if vol_z is not None and -1.5 <= vol_z <= 2.5:
        score += 0.03; reasons.append("volume non-shock")
    score=min(score,0.90)

    prefix="bull" if side=="long" else "bear"
    return ExpertSignal(
        setup_id=f"{prefix}_trend.pullback.{side}.15m.v2",
        direction=side,
        confidence=score,
        strategy="trend_pullback_resumption",
        reasoning=reasons,
        max_leverage=max(1,int(max_leverage)),
    )
