from __future__ import annotations

from typing import Any

from decision.mtf_trend_gate import evaluate_mtf_trend_gate
from . import ExpertSignal, f, series, tf


def _ema_separation(row: dict[str, Any]) -> float | None:
    fast=f(row.get("ema_fast")); slow=f(row.get("ema_slow")); close=f(row.get("close"))
    if close is None:
        closes=series(row,"recent_closes")
        close=closes[-1] if closes else None
    if None in (fast,slow,close) or abs(close) <= 1e-12:
        return None
    return abs(fast-slow)/abs(close)


def _microstructure_resumed(*, side: str, item: dict[str, Any], fast: float, atr: float | None) -> tuple[bool,list[str]]:
    row=tf(item,"15m")
    closes=series(row,"recent_closes")
    highs=series(row,"recent_highs")
    lows=series(row,"recent_lows")
    if len(closes)<4 or len(highs)<4 or len(lows)<4:
        return False,["15m microstructure missing"]

    close=closes[-1]
    prev_close=closes[-2]
    prior_high=max(highs[-4:-1])
    prior_low=min(lows[-4:-1])
    touch_tol=0.0025
    reasons: list[str]=[]

    if side=="long":
        touched=min(lows[-4:-1]) <= fast*(1.0+touch_tol)
        resumed=close > prev_close and close > prior_high
        if atr is not None and atr>0:
            not_chasing=(close-fast) <= 0.85*atr
        else:
            not_chasing=close <= fast*1.008
        if touched:
            reasons.append("recent EMA20 pullback touch")
        if resumed:
            reasons.append("15m bullish microstructure break")
        if not_chasing:
            reasons.append("entry not overextended from EMA20")
        return bool(touched and resumed and not_chasing),reasons

    touched=max(highs[-4:-1]) >= fast*(1.0-touch_tol)
    resumed=close < prev_close and close < prior_low
    if atr is not None and atr>0:
        not_chasing=(fast-close) <= 0.85*atr
    else:
        not_chasing=close >= fast*0.992
    if touched:
        reasons.append("recent EMA20 bounce touch")
    if resumed:
        reasons.append("15m bearish microstructure break")
    if not_chasing:
        reasons.append("entry not overextended from EMA20")
    return bool(touched and resumed and not_chasing),reasons


def evaluate_trend(*, side: str, item: dict[str, Any], ta: dict[str, Any], max_leverage: int) -> ExpertSignal | None:
    """High-selectivity trend pullback/resumption expert.

    v3 deliberately requires multi-timeframe trend strength, meaningful EMA
    separation and a true lower-timeframe structure break after an EMA20
    pullback. This reduces the v2 failure mode of repeatedly re-entering weak
    trends on a one-bar uptick/downtick.
    """
    gate=evaluate_mtf_trend_gate(side=side,item=item)
    if not gate.allowed:
        return None

    adx=f(ta.get("adx"))
    rsi=f(ta.get("rsi"))
    atr=f(ta.get("atr"))
    atr_ratio=f(ta.get("atr_ratio") or ta.get("atr_pct"))
    vol_z=f(ta.get("vol_z"))
    ema=ta.get("ema") if isinstance(ta.get("ema"),dict) else {}
    fast=f(ema.get("fast")); slow=f(ema.get("slow"))
    price=f(ta.get("price"))
    h1=tf(item,"1h"); h4=tf(item,"4h")

    if None in (adx,fast,slow,price):
        return None
    if not (22.0 <= adx <= 42.0):
        return None
    if atr_ratio is not None and not (0.0015 <= atr_ratio <= 0.032):
        return None
    if vol_z is not None and not (-1.25 <= vol_z <= 2.25):
        return None

    if side=="long":
        if not (fast > slow and price >= fast):
            return None
        if rsi is not None and not (45.0 <= rsi <= 60.0):
            return None
    else:
        if not (fast < slow and price <= fast):
            return None
        if rsi is not None and not (40.0 <= rsi <= 55.0):
            return None

    h1_adx=f(h1.get("adx")); h4_adx=f(h4.get("adx"))
    if h1_adx is None or h4_adx is None or h1_adx < 20.0 or h4_adx < 20.0:
        return None

    local_sep=abs(fast-slow)/max(abs(price),1e-12)
    h1_sep=_ema_separation(h1)
    h4_sep=_ema_separation(h4)
    if local_sep < 0.0007:
        return None
    if h1_sep is None or h1_sep < 0.0015:
        return None
    if h4_sep is None or h4_sep < 0.0020:
        return None

    micro_ok,micro_reasons=_microstructure_resumed(side=side,item=item,fast=fast,atr=atr)
    if not micro_ok:
        return None

    score=0.64
    reasons=[
        "4h trend direction and momentum aligned",
        "1h trend structure aligned",
        *micro_reasons,
    ]
    if 24.0 <= adx <= 36.0:
        score += 0.05; reasons.append(f"15m ADX quality {adx:.1f}")
    if h1_adx >= 23.0:
        score += 0.04; reasons.append(f"1h ADX {h1_adx:.1f}")
    if h4_adx >= 23.0:
        score += 0.04; reasons.append(f"4h ADX {h4_adx:.1f}")
    if rsi is not None:
        if side=="long" and 48.0 <= rsi <= 57.0:
            score += 0.04; reasons.append(f"long pullback RSI {rsi:.1f}")
        elif side=="short" and 43.0 <= rsi <= 52.0:
            score += 0.04; reasons.append(f"short bounce RSI {rsi:.1f}")
    if vol_z is not None and -0.5 <= vol_z <= 1.8:
        score += 0.03; reasons.append("volume quality")
    score=min(score,0.86)

    prefix="bull" if side=="long" else "bear"
    return ExpertSignal(
        setup_id=f"{prefix}_trend.pullback.{side}.15m.v3",
        direction=side,
        confidence=score,
        strategy="trend_pullback_resumption_v3",
        reasoning=reasons,
        max_leverage=max(1,int(max_leverage)),
    )
