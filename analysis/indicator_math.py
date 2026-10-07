"""Shared indicator math helpers used by non-TA-Lib paths.

These helpers standardize the fallback/runtime math so ML features,
inventory enrichment, and lightweight runtime analysis do not drift away
from the TA-Lib-backed reference path.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _to_float_series(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.astype(float)
    return pd.Series(values, dtype=float)


def wilder_rsi(series: Any, period: int = 14) -> pd.Series:
    close = _to_float_series(series)
    delta = close.diff()
    gain = delta.clip(lower=0.0).fillna(0.0)
    loss = (-delta.clip(upper=0.0)).fillna(0.0)

    alpha = 1.0 / max(int(period), 1)
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.astype(float)


def stochastic_rsi(
    series: Any,
    *,
    rsi_period: int = 90,
    stoch_period: int = 90,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Return smoothed Stochastic RSI %K/%D in the 0-100 range.

    The function intentionally emits NaN during warmup and when RSI has no
    range. That keeps downstream trading code fail-closed instead of creating
    a fake neutral or extreme signal.
    """
    close = _to_float_series(series)
    rsi = wilder_rsi(close, period=max(int(rsi_period), 1))
    lowest = rsi.rolling(window=max(int(stoch_period), 1), min_periods=max(int(stoch_period), 1)).min()
    highest = rsi.rolling(window=max(int(stoch_period), 1), min_periods=max(int(stoch_period), 1)).max()
    spread = highest - lowest
    raw = ((rsi - lowest) / spread.where(spread.abs() > 1e-12)) * 100.0
    k = raw.rolling(window=max(int(smooth_k), 1), min_periods=max(int(smooth_k), 1)).mean().clip(0.0, 100.0)
    d = k.rolling(window=max(int(smooth_d), 1), min_periods=max(int(smooth_d), 1)).mean().clip(0.0, 100.0)
    return k.astype(float), d.astype(float)


def true_range(high: Any, low: Any, close: Any) -> pd.Series:
    high_ser = _to_float_series(high)
    low_ser = _to_float_series(low)
    close_ser = _to_float_series(close)
    prev_close = close_ser.shift(1)
    return pd.concat(
        [
            high_ser - low_ser,
            (high_ser - prev_close).abs(),
            (low_ser - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1).astype(float)


def wilder_dmi_adx(high: Any, low: Any, close: Any, period: int = 14) -> dict[str, pd.Series]:
    high_ser = _to_float_series(high)
    low_ser = _to_float_series(low)
    close_ser = _to_float_series(close)

    up_move = high_ser.diff()
    down_move = -low_ser.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high_ser.index,
        dtype=float,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=low_ser.index,
        dtype=float,
    )

    tr = true_range(high_ser, low_ser, close_ser).fillna(0.0)
    alpha = 1.0 / max(int(period), 1)

    atr = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_dm_smoothed = plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    minus_dm_smoothed = minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    plus_di = 100.0 * plus_dm_smoothed / (atr + 1e-10)
    minus_di = 100.0 * minus_dm_smoothed / (atr + 1e-10)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    return {
        "tr": tr.astype(float),
        "atr": atr.astype(float),
        "plus_dm": plus_dm.astype(float),
        "minus_dm": minus_dm.astype(float),
        "plus_di": plus_di.astype(float),
        "minus_di": minus_di.astype(float),
        "dx": dx.astype(float),
        "adx": adx.astype(float),
    }


__all__ = ["stochastic_rsi", "true_range", "wilder_dmi_adx", "wilder_rsi"]
