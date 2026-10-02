from __future__ import annotations

from typing import Any

from . import ExpertSignal, f, series, tf


def _aligned(item: dict[str, Any], side: str) -> bool:
    for name in ("1h","4h"):
        row=tf(item,name)
        fast=f(row.get("ema_fast")); slow=f(row.get("ema_slow")); close=f(row.get("close"))
        macd=f(row.get("macd_hist"))
        if None in (fast,slow):
            return False
        if side=="long":
            if not fast > slow or (close is not None and close < slow) or (name=="4h" and macd is not None and macd < 0):
                return False
        else:
            if not fast < slow or (close is not None and close > slow) or (name=="4h" and macd is not None and macd > 0):
                return False
    return True


def evaluate_compression(*, item: dict[str, Any], ta: dict[str, Any], max_leverage: int) -> ExpertSignal | None:
    row=tf(item,"15m")
    closes=series(row,"recent_closes"); highs=series(row,"recent_highs"); lows=series(row,"recent_lows")
    if not all(len(v)>=6 for v in (closes,highs,lows)):
        return None
    close=closes[-1]
    prev_high=max(highs[-6:-1]); prev_low=min(lows[-6:-1])
    vol_z=f(ta.get("vol_z"))
    adx=f(ta.get("adx"))
    atr=f(ta.get("atr_ratio") or ta.get("atr_pct"))
    if vol_z is None or vol_z < 1.2:
        return None
    if adx is not None and adx < 18.0:
        return None
    if atr is not None and atr > 0.04:
        return None

    side=None
    if close > prev_high * 1.0015:
        side="long"
    elif close < prev_low * 0.9985:
        side="short"
    if side is None or not _aligned(item,side):
        return None

    score=0.70
    reasons=["compression range broken","1h/4h direction aligned"]
    if vol_z >= 1.8:
        score += 0.08; reasons.append(f"volume expansion {vol_z:.2f}z")
    if adx is not None and adx >= 22.0:
        score += 0.04; reasons.append(f"ADX expansion {adx:.1f}")
    score=min(score,0.88)
    setup="compression_breakout.up_retest.long.15m.v2" if side=="long" else "compression_breakout.down_retest.short.15m.v2"
    return ExpertSignal(setup,side,score,"breakout",reasons,max(1,int(max_leverage)))
