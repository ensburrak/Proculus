from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from analysis.indicator_math import wilder_dmi_adx
from research.edge_candidates_v3 import ResearchSignal

FAMILY = "breadth_donchian_10_v5"
PREREGISTERED_SYMBOLS = (
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "BNB/USDT:USDT",
    "ADA/USDT:USDT",
    "DOGE/USDT:USDT",
    "LINK/USDT:USDT",
    "AVAX/USDT:USDT",
)
BULL_BREADTH = 0.625
BEAR_BREADTH = 0.375
MIN_MEDIAN_ADX = 18.0
DONCHIAN_LENGTH = 10


def _ema(series: pd.Series, span: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").ewm(
        span=span,
        adjust=False,
        min_periods=span,
    ).mean()


def _prepare_4h(frame: pd.DataFrame) -> pd.DataFrame:
    raw = frame[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    indexed = raw.set_index("timestamp").sort_index()
    bars = indexed.resample("4h", label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["open", "high", "low", "close"])
    close = pd.to_numeric(bars["close"], errors="coerce")
    bars["ema50"] = _ema(close, 50)
    bars["ema200"] = _ema(close, 200)
    dmi = wilder_dmi_adx(
        pd.to_numeric(bars["high"], errors="coerce"),
        pd.to_numeric(bars["low"], errors="coerce"),
        close,
        period=14,
    )
    bars["atr"] = dmi["atr"]
    bars["adx"] = dmi["adx"]
    bars["close_time"] = bars.index + pd.Timedelta(hours=4)
    return bars


def _decision_index(frame: pd.DataFrame, entry_time: pd.Timestamp) -> int:
    times = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
    entry_idx = int(times.searchsorted(entry_time, side="left"))
    return entry_idx - 1


def generate_breadth_donchian_10_signals(
    frames: dict[str, pd.DataFrame],
) -> dict[str, list[ResearchSignal]]:
    if len(frames) < 3:
        return {symbol: [] for symbol in frames}

    prepared = {symbol: _prepare_4h(frame) for symbol, frame in frames.items()}
    common_index: pd.DatetimeIndex | None = None
    for bars in prepared.values():
        idx = pd.DatetimeIndex(bars.index)
        common_index = idx if common_index is None else common_index.intersection(idx)
    if common_index is None or common_index.empty:
        return {symbol: [] for symbol in frames}

    aligned = {symbol: bars.loc[common_index].copy() for symbol, bars in prepared.items()}
    out: dict[str, list[ResearchSignal]] = {symbol: [] for symbol in frames}

    for pos in range(max(210, DONCHIAN_LENGTH), len(common_index)):
        above: list[bool] = []
        adx_values: list[float] = []
        for bars in aligned.values():
            row = bars.iloc[pos]
            close = float(row["close"])
            ema200 = float(row["ema200"])
            adx = float(row["adx"])
            if np.isfinite(close) and np.isfinite(ema200):
                above.append(close > ema200)
            if np.isfinite(adx):
                adx_values.append(adx)

        if not above or not adx_values:
            continue
        breadth = float(sum(above)) / float(len(above))
        median_adx = float(np.median(adx_values))
        if median_adx < MIN_MEDIAN_ADX:
            continue

        bull_state = breadth >= BULL_BREADTH
        bear_state = breadth <= BEAR_BREADTH
        if not (bull_state or bear_state):
            continue

        for symbol, bars in aligned.items():
            row = bars.iloc[pos]
            ema50 = float(row["ema50"])
            ema200 = float(row["ema200"])
            close = float(row["close"])
            atr = float(row["atr"])
            if not all(np.isfinite(value) for value in (ema50, ema200, close, atr)) or atr <= 0:
                continue

            prior = bars.iloc[pos - DONCHIAN_LENGTH : pos]
            high = float(pd.to_numeric(prior["high"], errors="coerce").max())
            low = float(pd.to_numeric(prior["low"], errors="coerce").min())
            side: str | None = None
            if bull_state and ema50 > ema200 and close > high * 1.001:
                side = "long"
            elif bear_state and ema50 < ema200 and close < low * 0.999:
                side = "short"
            if side is None:
                continue

            close_time = pd.Timestamp(row["close_time"])
            decision_idx = _decision_index(frames[symbol], close_time)
            if decision_idx < 0 or decision_idx >= len(frames[symbol]) - 1:
                continue
            out[symbol].append(
                ResearchSignal(
                    family=FAMILY,
                    setup_id=f"{FAMILY}.entry.{side}.4h.v1",
                    side=side,
                    decision_idx=decision_idx,
                    stop_atr_mult=2.0,
                    tp_r_target=None,
                    trail_atr_mult=3.0,
                    max_hold_bars=960,
                    signal_timeframe="4h",
                    atr_value=atr,
                )
            )
    return out


def evaluate_fresh_holdout(
    primary: dict[str, Any],
    stress: dict[str, Any],
    by_symbol: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    evaluable = {
        symbol: stats
        for symbol, stats in by_symbol.items()
        if int(stats.get("trades") or 0) >= 10
    }
    positive = [
        symbol
        for symbol, stats in evaluable.items()
        if float(stats.get("profit_factor") or 0.0) >= 1.0
        and float(stats.get("expectancy_r") or 0.0) > 0.0
    ]
    checks = {
        "primary_min_trades": int(primary.get("trades") or 0) >= 30,
        "primary_profit_factor": float(primary.get("profit_factor") or 0.0) >= 1.15,
        "primary_expectancy": float(primary.get("expectancy_r") or -999.0) >= 0.05,
        "stress_min_trades": int(stress.get("trades") or 0) >= 20,
        "stress_profit_factor": float(stress.get("profit_factor") or 0.0) >= 1.05,
        "stress_expectancy": float(stress.get("expectancy_r") or -999.0) > 0.0,
        "breadth_min_symbols": len(evaluable) >= 3,
        "breadth_positive_symbols": len(positive) >= 2,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "evaluable_symbols": sorted(evaluable),
        "positive_symbols": sorted(positive),
        "status": "fresh_holdout_pass" if all(checks.values()) else "fresh_holdout_fail",
    }


__all__ = [
    "FAMILY",
    "PREREGISTERED_SYMBOLS",
    "generate_breadth_donchian_10_signals",
    "evaluate_fresh_holdout",
]
