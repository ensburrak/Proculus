from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResearchSignal:
    family: str
    setup_id: str
    side: str
    decision_idx: int
    stop_atr_mult: float
    tp_r_target: float | None
    trail_atr_mult: float | None
    max_hold_bars: int
    signal_timeframe: str
    atr_value: float | None = None


FAST_FAMILIES = (
    "trend_breakout_v3",
    "pullback_reclaim_v3",
    "range_failed_break_v3",
    "squeeze_expansion_v3",
    "vwap_reclaim_v3",
    "stoch_trend_reentry_v3",
)

SLOW_SPECS = (
    {"family": "donchian_4h_v3_20", "length": 20, "stop_atr_mult": 2.0, "trail_atr_mult": 2.5},
    {"family": "donchian_4h_v3_40", "length": 40, "stop_atr_mult": 2.0, "trail_atr_mult": 3.5},
    {"family": "donchian_4h_v3_10", "length": 10, "stop_atr_mult": 2.0, "trail_atr_mult": 2.5},
)


def _f(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _side_setup(family: str, side: str, timeframe: str) -> str:
    return f"{family}.entry.{side}.{timeframe}.v1"


def _recent(frame: pd.DataFrame, idx: int, field: str, bars: int, fn: str) -> float | None:
    if idx < 0 or field not in frame.columns:
        return None
    values = pd.to_numeric(frame.iloc[max(0, idx - bars + 1) : idx + 1][field], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.min() if fn == "min" else values.max())


def _fast_signal(frame: pd.DataFrame, idx: int, family: str) -> ResearchSignal | None:
    if idx < 220 or idx >= len(frame) - 1:
        return None
    row = frame.iloc[idx]
    prev = frame.iloc[idx - 1]

    close = _f(row.get("close"))
    prev_close = _f(prev.get("close"))
    ema20 = _f(row.get("ema_fast"))
    ema50 = _f(row.get("ema_slow"))
    ema200 = _f(row.get("ema200"))
    rsi = _f(row.get("rsi"))
    adx = _f(row.get("adx"))
    adx_slope = _f(row.get("adx_slope"))
    plus = _f(row.get("plus_di"))
    minus = _f(row.get("minus_di"))
    atr_pct = _f(row.get("atr_pct"))
    vol_ratio = _f(row.get("volume_spike_ratio"))
    bb_mid = _f(row.get("bb_middle"))
    bb_up = _f(row.get("bb_upper"))
    bb_low = _f(row.get("bb_lower"))
    bb_width = _f(row.get("boll_width"))
    prev_up = _f(prev.get("bb_upper"))
    prev_low = _f(prev.get("bb_lower"))
    vwap_delta = _f(row.get("vwap20_delta"))
    prev_vwap_delta = _f(prev.get("vwap20_delta"))
    h1_close = _f(row.get("close_1h"))
    h1_fast = _f(row.get("ema_fast_1h"))
    h1_slow = _f(row.get("ema_slow_1h"))
    h1_200 = _f(row.get("ema200_1h"))
    k90 = _f(row.get("stoch_rsi_90_k"))
    d90 = _f(row.get("stoch_rsi_90_d"))
    pk90 = _f(row.get("stoch_rsi_90_prev_k"))
    pd90 = _f(row.get("stoch_rsi_90_prev_d"))

    required = (close, prev_close, ema20, ema50, ema200, rsi, adx, plus, minus, atr_pct)
    if any(value is None for value in required):
        return None

    long_h1 = all(value is not None for value in (h1_close, h1_fast, h1_slow, h1_200)) and h1_fast > h1_slow > h1_200
    short_h1 = all(value is not None for value in (h1_close, h1_fast, h1_slow, h1_200)) and h1_fast < h1_slow < h1_200

    if family == "trend_breakout_v3":
        prior_high = _recent(frame.iloc[:idx], idx - 1, "high", 20, "max")
        prior_low = _recent(frame.iloc[:idx], idx - 1, "low", 20, "min")
        rising = adx_slope is not None and adx_slope > 0.0
        volume_ok = vol_ratio is not None and vol_ratio >= 1.20
        atr_ok = 0.002 <= atr_pct <= 0.03
        if prior_high is not None and ema50 > ema200 and long_h1 and close > prior_high * 1.001 and 20 <= adx <= 45 and rising and plus > minus and volume_ok and 52 <= rsi <= 72 and atr_ok:
            return ResearchSignal(family, _side_setup(family, "long", "15m"), "long", idx, 1.5, 2.5, None, 96, "15m", None)
        if prior_low is not None and ema50 < ema200 and short_h1 and close < prior_low * 0.999 and 20 <= adx <= 45 and rising and minus > plus and volume_ok and 28 <= rsi <= 48 and atr_ok:
            return ResearchSignal(family, _side_setup(family, "short", "15m"), "short", idx, 1.5, 2.5, None, 96, "15m")

    if family == "pullback_reclaim_v3":
        recent_low = _recent(frame, idx, "low", 5, "min")
        recent_high = _recent(frame, idx, "high", 5, "max")
        near = abs(close - ema20) / max(close, 1e-12) < 0.015
        volume_ok = vol_ratio is None or vol_ratio < 2.5
        if recent_low is not None and ema20 > ema50 > ema200 and long_h1 and recent_low <= ema20 * 1.003 and close >= ema20 and close > prev_close and near and 42 <= rsi <= 60 and 18 <= adx <= 40 and plus > minus and volume_ok:
            return ResearchSignal(family, _side_setup(family, "long", "15m"), "long", idx, 1.2, 2.0, None, 96, "15m")
        if recent_high is not None and ema20 < ema50 < ema200 and short_h1 and recent_high >= ema20 * 0.997 and close <= ema20 and close < prev_close and near and 40 <= rsi <= 58 and 18 <= adx <= 40 and minus > plus and volume_ok:
            return ResearchSignal(family, _side_setup(family, "short", "15m"), "short", idx, 1.2, 2.0, None, 96, "15m")

    if family == "range_failed_break_v3":
        if None not in (bb_mid, bb_up, bb_low, bb_width, prev_up, prev_low):
            if adx <= 18 and 0.008 <= bb_width <= 0.05:
                if prev_close < prev_low and close > bb_low and rsi < 45 and close < bb_mid:
                    return ResearchSignal(family, _side_setup(family, "long", "15m"), "long", idx, 1.0, 1.2, None, 48, "15m", None)
                if prev_close > prev_up and close < bb_up and rsi > 55 and close > bb_mid:
                    return ResearchSignal(family, _side_setup(family, "short", "15m"), "short", idx, 1.0, 1.2, None, 48, "15m")

    if family == "squeeze_expansion_v3":
        width = pd.to_numeric(frame["boll_width"], errors="coerce")
        q20 = _f(width.iloc[max(0, idx - 191) : idx + 1].quantile(0.20)) if idx >= 95 else None
        prior_high = _recent(frame.iloc[:idx], idx - 1, "high", 20, "max")
        prior_low = _recent(frame.iloc[:idx], idx - 1, "low", 20, "min")
        if None not in (q20, bb_width, vol_ratio, adx_slope) and idx > 0:
            prev_width = _f(prev.get("boll_width"))
            if prev_width is not None and prev_width <= q20 and bb_width > prev_width * 1.05 and adx >= 18 and adx_slope > 0 and vol_ratio >= 1.5:
                if prior_high is not None and close > prior_high * 1.001 and ema20 > ema50 and plus > minus:
                    return ResearchSignal(family, _side_setup(family, "long", "15m"), "long", idx, 1.2, 2.5, None, 96, "15m")
                if prior_low is not None and close < prior_low * 0.999 and ema20 < ema50 and minus > plus:
                    return ResearchSignal(family, _side_setup(family, "short", "15m"), "short", idx, 1.2, 2.5, None, 96, "15m")

    if family == "vwap_reclaim_v3":
        if None not in (vwap_delta, prev_vwap_delta):
            near = abs(close - ema20) / max(close, 1e-12) < 0.02
            if long_h1 and prev_vwap_delta < 0 <= vwap_delta and ema20 > ema50 and 18 <= adx <= 35 and 45 <= rsi <= 65 and plus > minus and near:
                return ResearchSignal(family, _side_setup(family, "long", "15m"), "long", idx, 1.0, 1.8, None, 64, "15m", None)
            if short_h1 and prev_vwap_delta > 0 >= vwap_delta and ema20 < ema50 and 18 <= adx <= 35 and 35 <= rsi <= 55 and minus > plus and near:
                return ResearchSignal(family, _side_setup(family, "short", "15m"), "short", idx, 1.0, 1.8, None, 64, "15m")

    if family == "stoch_trend_reentry_v3":
        if None not in (k90, d90, pk90, pd90):
            long_cross = pk90 < pd90 and k90 >= d90 and k90 <= 30
            short_cross = pk90 > pd90 and k90 <= d90 and k90 >= 70
            if long_cross and ema50 > ema200 and long_h1 and 20 <= adx <= 45 and plus > minus and 40 <= rsi <= 65:
                return ResearchSignal(family, _side_setup(family, "long", "15m"), "long", idx, 1.2, 2.0, None, 64, "15m")
            if short_cross and ema50 < ema200 and short_h1 and 20 <= adx <= 45 and minus > plus and 35 <= rsi <= 60:
                return ResearchSignal(family, _side_setup(family, "short", "15m"), "short", idx, 1.2, 2.0, None, 64, "15m")
    return None


def generate_fast_signals(frame: pd.DataFrame) -> list[ResearchSignal]:
    out: list[ResearchSignal] = []
    for idx in range(220, len(frame) - 1):
        for family in FAST_FAMILIES:
            signal = _fast_signal(frame, idx, family)
            if signal is not None:
                out.append(signal)
    return out


def _resample_4h(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    data = data.set_index(pd.DatetimeIndex(pd.to_datetime(data["timestamp"], utc=True)))
    bars = data.resample("4h", label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna()
    close = pd.to_numeric(bars["close"], errors="coerce")
    bars["ema50"] = close.ewm(span=50, adjust=False, min_periods=50).mean()
    bars["ema200"] = close.ewm(span=200, adjust=False, min_periods=200).mean()
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (bars["high"] - bars["low"]).abs(),
            (bars["high"] - prev_close).abs(),
            (bars["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    bars["atr"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    bars["close_time"] = bars.index + pd.Timedelta(hours=4)
    return bars.reset_index(drop=True)


def generate_slow_signals(frame: pd.DataFrame) -> list[ResearchSignal]:
    bars = _resample_4h(frame)
    out: list[ResearchSignal] = []
    times = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
    for spec in SLOW_SPECS:
        length = int(spec["length"])
        for i in range(max(210, length), len(bars)):
            row = bars.iloc[i]
            atr = _f(row.get("atr"))
            ema50 = _f(row.get("ema50"))
            ema200 = _f(row.get("ema200"))
            close = _f(row.get("close"))
            if None in (atr, ema50, ema200, close) or atr <= 0:
                continue
            prior = bars.iloc[i - length : i]
            hi = _f(pd.to_numeric(prior["high"], errors="coerce").max())
            lo = _f(pd.to_numeric(prior["low"], errors="coerce").min())
            if None in (hi, lo):
                continue
            side = None
            if ema50 > ema200 and close > hi * 1.001:
                side = "long"
            elif ema50 < ema200 and close < lo * 0.999:
                side = "short"
            if side is None:
                continue
            entry_time = pd.Timestamp(row["close_time"])
            decision_idx = int(times.searchsorted(entry_time, side="left")) - 1
            if decision_idx < 0 or decision_idx >= len(frame) - 1:
                continue
            family = str(spec["family"])
            out.append(
                ResearchSignal(
                    family=family,
                    setup_id=_side_setup(family, side, "4h"),
                    side=side,
                    decision_idx=decision_idx,
                    stop_atr_mult=float(spec["stop_atr_mult"]),
                    tp_r_target=None,
                    trail_atr_mult=float(spec["trail_atr_mult"]),
                    max_hold_bars=960,
                    signal_timeframe="4h",
                    atr_value=float(atr),
                )
            )
    return out


def generate_all_signals(frame: pd.DataFrame) -> list[ResearchSignal]:
    return sorted(
        [*generate_fast_signals(frame), *generate_slow_signals(frame)],
        key=lambda item: (item.decision_idx, item.family, item.side),
    )


__all__ = [
    "FAST_FAMILIES",
    "SLOW_SPECS",
    "ResearchSignal",
    "generate_all_signals",
    "generate_fast_signals",
    "generate_slow_signals",
]
