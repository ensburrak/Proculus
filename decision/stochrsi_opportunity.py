from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class StochRsiDecision:
    action: str
    direction: str
    confidence: float
    setup_id: str
    stop_atr_mult: float
    tp_r_target: float
    max_hold_hours: float
    risk_scale: float
    reasons: list[str]
    blockers: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _f(value: Any) -> float | None:
    try:
        parsed=float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed else None


def _hold(reason: str) -> StochRsiDecision:
    return StochRsiDecision(
        action="hold",direction="neutral",confidence=0.0,
        setup_id="stochrsi_opportunity.blocked.v2",
        stop_atr_mult=0.0,tp_r_target=0.0,max_hold_hours=0.0,
        risk_scale=0.0,reasons=[],blockers=[reason],
    )


def evaluate_stochrsi_opportunity(*, item: dict[str, Any], ta: dict[str, Any] | None = None) -> StochRsiDecision:
    """Independent StochRSI-90 opportunity lane.

    This lane never changes the primary expert direction. It either emits its
    own candidate or stays flat. Live side effects remain disabled by config.
    """
    pack=ta if isinstance(ta,dict) else (item.get("ta_pack") if isinstance(item.get("ta_pack"),dict) else {})
    regime=str(item.get("regime") or pack.get("regime") or "unknown").lower()
    if regime in {"shock","compression","transition","conflict","unknown"}:
        return _hold(f"regime_block:{regime}")

    k=_f(pack.get("stoch_rsi_90_k")); d=_f(pack.get("stoch_rsi_90_d"))
    pk=_f(pack.get("stoch_rsi_90_prev_k")); pd=_f(pack.get("stoch_rsi_90_prev_d"))
    if None in (k,d,pk,pd):
        return _hold("stochrsi90_missing_or_not_warmed")

    vol_z=_f(pack.get("vol_z"))
    if vol_z is not None and abs(vol_z) >= 3.0:
        return _hold("volume_shock_veto")

    ema=pack.get("ema") if isinstance(pack.get("ema"),dict) else {}
    fast=_f(ema.get("fast")); slow=_f(ema.get("slow"))
    adx=_f(pack.get("adx")); rsi=_f(pack.get("rsi"))
    atr=_f(pack.get("atr_ratio") or pack.get("atr_pct"))
    if atr is not None and (atr < 0.001 or atr > 0.04):
        return _hold("atr_outside_tradable_band")

    cross_up=pk < pd and k >= d
    cross_down=pk > pd and k <= d
    side: str | None=None
    reasons: list[str]=[]

    if regime=="bull":
        if cross_up and k <= 30.0 and fast is not None and slow is not None and fast >= slow and (adx is None or adx >= 18.0):
            side="long"; reasons=["StochRSI90 true cross-up from bull pullback","EMA direction aligned"]
    elif regime=="bear":
        if cross_down and k >= 70.0 and fast is not None and slow is not None and fast <= slow and (adx is None or adx >= 18.0):
            side="short"; reasons=["StochRSI90 true cross-down from bear bounce","EMA direction aligned"]
    elif regime=="range":
        if adx is not None and adx <= 18.0 and rsi is not None:
            if cross_up and k <= 10.0 and rsi <= 35.0:
                side="long"; reasons=["StochRSI90 extreme cross-up","range RSI confirmation"]
            elif cross_down and k >= 90.0 and rsi >= 65.0:
                side="short"; reasons=["StochRSI90 extreme cross-down","range RSI confirmation"]

    if side is None:
        return _hold("no_stochrsi90_confirmed_setup")

    confidence=0.72
    if adx is not None:
        if regime in {"bull","bear"} and 20.0 <= adx <= 40.0:
            confidence += 0.05; reasons.append(f"ADX quality {adx:.1f}")
        elif regime=="range" and adx <= 15.0:
            confidence += 0.04; reasons.append(f"low ADX range {adx:.1f}")
    if rsi is not None and regime in {"bull","bear"}:
        confidence += 0.03
    confidence=min(confidence,0.86)

    profile="range" if regime=="range" else "trend"
    return StochRsiDecision(
        action="enter",direction=side,confidence=confidence,
        setup_id=f"stochrsi_opportunity.{regime}.{side}.15m.v2",
        stop_atr_mult=0.9 if profile=="range" else 1.2,
        tp_r_target=1.25 if profile=="range" else 1.6,
        max_hold_hours=6.0 if profile=="range" else 10.0,
        risk_scale=0.25,
        reasons=reasons,blockers=[],
    )


__all__=["StochRsiDecision","evaluate_stochrsi_opportunity"]
