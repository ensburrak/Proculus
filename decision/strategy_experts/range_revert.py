from __future__ import annotations

from typing import Any

from . import ExpertSignal, f, series, tf


def evaluate_range(*, item: dict[str, Any], ta: dict[str, Any], max_leverage: int) -> ExpertSignal | None:
    rsi=f(ta.get("rsi"))
    stoch=f(ta.get("stoch_rsi") or ta.get("stochrsi") or ta.get("stoch_k"))
    adx=f(ta.get("adx"))
    atr=f(ta.get("atr_ratio") or ta.get("atr_pct"))
    ema=ta.get("ema") if isinstance(ta.get("ema"),dict) else {}
    fast=f(ema.get("fast")); slow=f(ema.get("slow"))
    price=f(ta.get("price"))
    lower=tf(item,"15m")
    closes=series(lower,"recent_closes")
    if stoch is not None and stoch <= 1.0:
        stoch *= 100.0

    if None in (rsi,stoch,adx,price) or adx > 18.0:
        return None
    if atr is not None and (atr < 0.001 or atr > 0.025):
        return None
    if fast is not None and slow is not None and abs(fast-slow)/max(abs(price),1e-9) > 0.006:
        return None
    if len(closes) < 3:
        return None

    prev=closes[-2]
    recent_min=min(closes[-6:-1]) if len(closes)>=6 else min(closes[:-1])
    recent_max=max(closes[-6:-1]) if len(closes)>=6 else max(closes[:-1])

    long_extreme = rsi <= 30.0 and stoch <= 20.0
    short_extreme = rsi >= 70.0 and stoch >= 80.0
    long_reentry = closes[-1] > prev and closes[-1] > recent_min * 1.001
    short_reentry = closes[-1] < prev and closes[-1] < recent_max * 0.999

    if long_extreme and long_reentry:
        confidence=min(0.88,0.68 + min(0.12,(30.0-rsi)/100.0) + min(0.06,(20.0-stoch)/100.0))
        return ExpertSignal(
            "range_revert.low_band_rejection.long.15m.v2","long",confidence,
            "mean_reversion",["dual oversold extreme","price re-entry confirmation","low ADX range"],max(1,int(max_leverage)),
        )
    if short_extreme and short_reentry:
        confidence=min(0.88,0.68 + min(0.12,(rsi-70.0)/100.0) + min(0.06,(stoch-80.0)/100.0))
        return ExpertSignal(
            "range_revert.high_band_rejection.short.15m.v2","short",confidence,
            "mean_reversion",["dual overbought extreme","price rejection confirmation","low ADX range"],max(1,int(max_leverage)),
        )
    return None
