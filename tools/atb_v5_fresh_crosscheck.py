from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

OKX_BASE = "https://www.okx.com"
SYMBOLS = (
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "BNB/USDT:USDT",
    "ADA/USDT:USDT",
    "DOGE/USDT:USDT",
    "LINK/USDT:USDT",
    "AVAX/USDT:USDT",
)
HOLDOUT_START = pd.Timestamp("2026-07-02T00:00:00Z")
FAMILY = "breadth_donchian_10_v5"
BULL_BREADTH = 0.625
BEAR_BREADTH = 0.375
MIN_MEDIAN_ADX = 18.0
DONCHIAN_LENGTH = 10
HISTORY_LIMIT = 300


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


def get_json(path: str, params: dict[str, Any], attempts: int = 7) -> dict[str, Any]:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            r = requests.get(
                OKX_BASE + path,
                params=params,
                timeout=30,
                headers={"User-Agent": "atb-v5-crosscheck/20261008"},
            )
            if r.status_code == 429:
                time.sleep(1.0 + attempt)
                continue
            r.raise_for_status()
            payload = r.json()
            if str(payload.get("code", "0")) != "0":
                raise RuntimeError(f"OKX code={payload.get('code')} msg={payload.get('msg')}")
            return payload
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last = exc
            time.sleep(min(8.0, 0.7 + attempt))
    raise RuntimeError(f"request failed {path} {params}: {last}")


def fetch_history(symbol: str, days: int = 220) -> pd.DataFrame:
    base = symbol.split("/", 1)[0]
    inst_id = f"{base}-USDT-SWAP"
    cutoff_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    rows: dict[int, tuple[float, float, float, float, float]] = {}
    after: str | None = None
    while True:
        params: dict[str, Any] = {"instId": inst_id, "bar": "15m", "limit": str(HISTORY_LIMIT)}
        if after is not None:
            params["after"] = after
        data = get_json("/api/v5/market/history-candles", params).get("data", [])
        if not data:
            break
        batch_min: int | None = None
        for raw in data:
            if not isinstance(raw, list) or len(raw) < 9:
                continue
            try:
                ts = int(raw[0])
                if str(raw[-1]) != "1":
                    continue
                rows[ts] = tuple(float(raw[i]) for i in range(1, 6))
                batch_min = ts if batch_min is None else min(batch_min, ts)
            except (TypeError, ValueError):
                continue
        if batch_min is None or batch_min <= cutoff_ms:
            break
        cursor = str(batch_min)
        if cursor == after:
            break
        after = cursor
        time.sleep(0.12)

    prepared = [
        (pd.to_datetime(ts, unit="ms", utc=True), *vals)
        for ts, vals in rows.items()
        if ts >= cutoff_ms
    ]
    if not prepared:
        raise RuntimeError(f"no confirmed bars for {symbol}")
    return (
        pd.DataFrame(prepared, columns=["timestamp", "open", "high", "low", "close", "volume"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )


def validate_locked_cadence(
    frames: dict[str, pd.DataFrame],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    expected = pd.Timedelta(minutes=15)
    details: dict[str, Any] = {}
    for symbol, frame in frames.items():
        times = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
        window = times[(times >= start) & (times <= end)]
        if len(window) < 2:
            raise RuntimeError(f"insufficient locked-window bars: {symbol}")
        if window.has_duplicates:
            raise RuntimeError(f"duplicate timestamps: {symbol}")
        deltas = window[1:] - window[:-1]
        if bool((deltas != expected).any()):
            bad = int((deltas != expected).argmax())
            raise RuntimeError(
                f"15m cadence gap {symbol}: {window[bad]} -> {window[bad+1]} "
                f"delta={deltas[bad]}"
            )
        details[symbol] = {
            "bars": len(window),
            "first": window[0].isoformat(),
            "last": window[-1].isoformat(),
        }
    return {
        "passed": True,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "symbols": details,
    }


def _to_float_series(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.astype(float)
    return pd.Series(values, dtype=float)


def true_range(high: Any, low: Any, close: Any) -> pd.Series:
    high_ser = _to_float_series(high)
    low_ser = _to_float_series(low)
    close_ser = _to_float_series(close)
    prev_close = close_ser.shift(1)
    return pd.concat(
        [high_ser - low_ser, (high_ser - prev_close).abs(), (low_ser - prev_close).abs()],
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
    plus_sm = plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    minus_sm = minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * plus_sm / (atr + 1e-10)
    minus_di = 100.0 * minus_sm / (atr + 1e-10)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    return {"atr": atr.astype(float), "adx": adx.astype(float)}


def _ema(series: pd.Series, span: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").ewm(
        span=span, adjust=False, min_periods=span
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
    dmi = wilder_dmi_adx(bars["high"], bars["low"], close, period=14)
    bars["atr"] = dmi["atr"]
    bars["adx"] = dmi["adx"]
    bars["close_time"] = bars.index + pd.Timedelta(hours=4)
    return bars


def _decision_index(frame: pd.DataFrame, entry_time: pd.Timestamp) -> int:
    times = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
    return int(times.searchsorted(entry_time, side="left")) - 1


def generate_signals(frames: dict[str, pd.DataFrame]) -> dict[str, list[ResearchSignal]]:
    prepared = {symbol: _prepare_4h(frame) for symbol, frame in frames.items()}
    common_index: pd.DatetimeIndex | None = None
    for bars in prepared.values():
        idx = pd.DatetimeIndex(bars.index)
        common_index = idx if common_index is None else common_index.intersection(idx)
    if common_index is None or common_index.empty:
        raise RuntimeError("no common 4h index")

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
            if not all(np.isfinite(v) for v in (ema50, ema200, close, atr)) or atr <= 0:
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


def _adverse_entry(raw: float, side: str, slip: float) -> float:
    return raw * (1.0 + slip) if side == "long" else raw * (1.0 - slip)


def _adverse_exit(raw: float, side: str, slip: float) -> float:
    return raw * (1.0 - slip) if side == "long" else raw * (1.0 + slip)


def _trade_r(frame: pd.DataFrame, signal: ResearchSignal, fee_bps: float, slip_bps: float):
    entry_idx = int(signal.decision_idx) + 1
    if entry_idx <= 0 or entry_idx >= len(frame):
        return None
    fee = float(fee_bps) / 10000.0
    slip = float(slip_bps) / 10000.0
    fill = _adverse_entry(float(frame.iloc[entry_idx]["open"]), signal.side, slip)
    atr = signal.atr_value
    if atr is None or not math.isfinite(float(atr)) or float(atr) <= 0:
        return None
    stop_distance = float(atr) * max(float(signal.stop_atr_mult), 0.1)
    risk_pct = stop_distance / max(fill, 1e-12)
    stop = fill - stop_distance if signal.side == "long" else fill + stop_distance
    max_end = min(len(frame) - 1, entry_idx + int(signal.max_hold_bars))
    bars = 0
    peak = fill
    current_stop = stop

    def close_r(raw_exit: float, exit_idx: int):
        exit_px = _adverse_exit(raw_exit, signal.side, slip)
        gross_pct = (
            (exit_px - fill) / fill if signal.side == "long" else (fill - exit_px) / fill
        )
        exit_fee_pct = fee * abs(exit_px / fill)
        funding_pct = 0.0001 * (bars // 32)
        net_pct = gross_pct - fee - exit_fee_pct - funding_pct
        return net_pct / risk_pct, exit_idx

    for idx in range(entry_idx, max_end + 1):
        bars += 1
        row = frame.iloc[idx]
        high, low, close = float(row["high"]), float(row["low"]), float(row["close"])
        if signal.side == "long":
            peak = max(peak, high)
            if low <= current_stop:
                return close_r(current_stop, idx)
        else:
            peak = min(peak, low)
            if high >= current_stop:
                return close_r(current_stop, idx)

        mfe_r = (
            (peak - fill) / stop_distance
            if signal.side == "long"
            else (fill - peak) / stop_distance
        )
        cost_buffer = fill * (2.0 * fee + 2.0 * slip)
        if mfe_r >= 0.5:
            breakeven = fill + cost_buffer if signal.side == "long" else fill - cost_buffer
            current_stop = (
                max(current_stop, breakeven)
                if signal.side == "long"
                else min(current_stop, breakeven)
            )
        if mfe_r >= 1.0:
            trail = float(signal.trail_atr_mult or 0.0) * float(atr)
            candidate = close - trail if signal.side == "long" else close + trail
            current_stop = (
                max(current_stop, candidate)
                if signal.side == "long"
                else min(current_stop, candidate)
            )
        if bars >= 192 and mfe_r < 0.5:
            return close_r(close, idx)
    return close_r(float(frame.iloc[max_end]["close"]), max_end)


def _stats(values: list[float]) -> dict[str, Any]:
    wins = [v for v in values if v > 0]
    losses = [v for v in values if v < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    return {
        "trades": len(values),
        "win_rate_pct": round(100.0 * len(wins) / max(1, len(values)), 4),
        "profit_factor": round(gross_win / gross_loss, 5) if gross_loss > 0 else (999.0 if wins else 0.0),
        "expectancy_r": round(sum(values) / max(1, len(values)), 5),
    }


def simulate(
    frames: dict[str, pd.DataFrame],
    signals: dict[str, list[ResearchSignal]],
    slip_bps: float,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    pooled: list[float] = []
    by_symbol_values: dict[str, list[float]] = defaultdict(list)
    for symbol, frame in frames.items():
        free_idx = -1
        last_entry_time: pd.Timestamp | None = None
        for signal in signals.get(symbol, []):
            entry_idx = int(signal.decision_idx) + 1
            if entry_idx <= 0 or entry_idx >= len(frame) or entry_idx < free_idx:
                continue
            entry_time = pd.Timestamp(frame.iloc[entry_idx]["timestamp"])
            if entry_time < HOLDOUT_START:
                continue
            if last_entry_time is not None and (entry_time - last_entry_time).total_seconds() < 39 * 60:
                continue
            result = _trade_r(frame, signal, fee_bps=5.0, slip_bps=slip_bps)
            if result is None:
                continue
            r_value, exit_idx = result
            pooled.append(float(r_value))
            by_symbol_values[symbol].append(float(r_value))
            free_idx = int(exit_idx) + 1
            last_entry_time = entry_time
    return _stats(pooled), {s: _stats(v) for s, v in sorted(by_symbol_values.items())}


def portfolio_simulate(
    frames: dict[str, pd.DataFrame],
    signals: dict[str, list[ResearchSignal]],
    *,
    slip_bps: float,
    outcome_complete_cutoff: pd.Timestamp | None = None,
    initial: float = 10_000.0,
) -> dict[str, Any]:
    fee = TAKER_FEE = 0.0005
    risk_per_trade = 0.0025
    wallet_cap = 0.20
    max_positions = 2
    cooldown_seconds = 39 * 60
    daily_limit = 0.005
    weekly_limit = 0.02
    reduce_threshold = 0.50
    hedge_threshold = 0.75

    candidates: list[tuple[pd.Timestamp, str, ResearchSignal]] = []
    for symbol, symbol_signals in signals.items():
        frame = frames[symbol]
        for signal in symbol_signals:
            entry_idx = int(signal.decision_idx) + 1
            if entry_idx <= 0 or entry_idx >= len(frame):
                continue
            entry_time = pd.Timestamp(frame.iloc[entry_idx]["timestamp"])
            if entry_time < HOLDOUT_START:
                continue
            if outcome_complete_cutoff is not None and entry_time > outcome_complete_cutoff:
                continue
            candidates.append((entry_time, symbol, signal))
    candidates.sort(key=lambda item: (item[0], item[1], item[2].setup_id))

    cash = float(initial)
    positions: list[dict[str, Any]] = []
    closed: list[dict[str, Any]] = []
    last_entry: dict[str, pd.Timestamp] = {}
    daily_realized: dict[str, float] = defaultdict(float)
    weekly_realized: dict[str, float] = defaultdict(float)
    blocked: dict[str, int] = defaultdict(int)
    equity_curve = [float(initial)]

    def local_keys(ts: pd.Timestamp) -> tuple[str, str]:
        local = ts.tz_convert("Europe/Istanbul")
        return local.strftime("%Y-%m-%d"), local.strftime("%Y-W%W")

    def current_close(symbol: str, ts: pd.Timestamp) -> float:
        frame = frames[symbol]
        times = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
        idx = int(times.searchsorted(ts, side="right")) - 1
        idx = max(0, min(idx, len(frame) - 1))
        return float(frame.iloc[idx]["close"])

    def mark_equity(ts: pd.Timestamp) -> float:
        equity = cash
        for pos in positions:
            price = current_close(str(pos["symbol"]), ts)
            sign = 1.0 if pos["side"] == "long" else -1.0
            qty = float(pos["notional"]) / max(float(pos["fill"]), 1e-12)
            equity += (price - float(pos["fill"])) * qty * sign
        return equity

    def realize(ts: pd.Timestamp) -> None:
        nonlocal cash, positions
        remaining = []
        for pos in positions:
            if pd.Timestamp(pos["exit_time"]) <= ts:
                # entry fee was already deducted when the position was opened.
                cash += float(pos["net_pnl"]) + float(pos["entry_fee"])
                closed.append(pos)
                day_key, week_key = local_keys(pd.Timestamp(pos["exit_time"]))
                daily_realized[day_key] += float(pos["net_pnl"])
                weekly_realized[week_key] += float(pos["net_pnl"])
            else:
                remaining.append(pos)
        positions = remaining
        equity_curve.append(mark_equity(ts))

    slip = float(slip_bps) / 10000.0
    for entry_time, symbol, signal in candidates:
        realize(entry_time)
        equity = max(0.0, mark_equity(entry_time))
        if equity <= 0.0:
            blocked["equity_depleted"] += 1
            continue

        day_key, week_key = local_keys(entry_time)
        daily_loss = abs(min(daily_realized.get(day_key, 0.0), 0.0))
        weekly_loss = abs(min(weekly_realized.get(week_key, 0.0), 0.0))
        daily_abs = equity * daily_limit
        weekly_abs = equity * weekly_limit
        daily_progress = daily_loss / daily_abs if daily_abs > 0 else 0.0
        weekly_progress = weekly_loss / weekly_abs if weekly_abs > 0 else 0.0

        if weekly_progress >= 1.0:
            blocked["weekly_loss_stop"] += 1
            continue
        if daily_progress >= 1.0:
            blocked["daily_loss_stop"] += 1
            continue
        if daily_progress >= hedge_threshold:
            blocked["daily_hedge_only"] += 1
            continue
        risk_multiplier = 0.5 if (
            daily_progress >= reduce_threshold or weekly_progress >= 0.75
        ) else 1.0

        prior = last_entry.get(symbol)
        if prior is not None and (entry_time - prior).total_seconds() < cooldown_seconds:
            blocked["trade_cooldown"] += 1
            continue
        if len(positions) >= max_positions:
            blocked["max_open_positions"] += 1
            continue
        if any(pos["symbol"] == symbol for pos in positions):
            blocked["symbol_already_open"] += 1
            continue

        frame = frames[symbol]
        result = _trade_r(frame, signal, fee_bps=5.0, slip_bps=slip_bps)
        if result is None:
            blocked["invalid_trade_path"] += 1
            continue
        r_value, exit_idx = result
        entry_idx = int(signal.decision_idx) + 1
        fill = _adverse_entry(float(frame.iloc[entry_idx]["open"]), signal.side, slip)
        atr = float(signal.atr_value or 0.0)
        stop_distance = atr * max(float(signal.stop_atr_mult), 0.1)
        stop_pct = stop_distance / max(fill, 1e-12)
        if stop_pct <= 0:
            blocked["invalid_stop"] += 1
            continue

        effective_scale = risk_multiplier
        risk_budget = equity * risk_per_trade * effective_scale
        notional = min(
            risk_budget / stop_pct,
            equity * wallet_cap * effective_scale,
        )
        used_margin = sum(float(pos["notional"]) for pos in positions)
        if notional <= 0 or used_margin + notional > equity * 0.95:
            blocked["margin_cap"] += 1
            continue

        risk_usd = notional * stop_pct
        net_pnl = float(r_value) * risk_usd
        entry_fee = notional * fee
        cash -= entry_fee
        positions.append(
            {
                "symbol": symbol,
                "side": signal.side,
                "entry_time": entry_time,
                "exit_time": pd.Timestamp(frame.iloc[int(exit_idx)]["timestamp"]),
                "fill": fill,
                "notional": notional,
                "risk_usd": risk_usd,
                "r_value": float(r_value),
                "net_pnl": net_pnl,
                "entry_fee": entry_fee,
            }
        )
        last_entry[symbol] = entry_time
        equity_curve.append(mark_equity(entry_time))

    if candidates:
        realize(pd.Timestamp.max.tz_localize("UTC"))

    pnls = [float(pos["net_pnl"]) for pos in closed]
    rvals = [float(pos["r_value"]) for pos in closed]
    wins = [value for value in pnls if value > 0]
    losses = [value for value in pnls if value < 0]
    peak = -float("inf")
    max_drawdown = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = min(max_drawdown, (equity - peak) / peak)

    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    return {
        "initial_balance": round(initial, 4),
        "final_balance": round(cash, 4),
        "return_pct": round(100.0 * (cash / initial - 1.0), 4),
        "max_drawdown_pct": round(abs(max_drawdown) * 100.0, 4),
        "trades": len(closed),
        "win_rate_pct": round(100.0 * len(wins) / max(1, len(closed)), 4),
        "profit_factor": round(gross_win / gross_loss, 5) if gross_loss > 0 else (999.0 if wins else 0.0),
        "expectancy_r": round(sum(rvals) / max(1, len(rvals)), 5),
        "blocked_entries": dict(sorted(blocked.items())),
        "risk_controls": {
            "max_open_positions": max_positions,
            "trade_cooldown_min": 39,
            "daily_loss_limit_pct": daily_limit,
            "weekly_loss_limit_pct": weekly_limit,
            "risk_per_trade_pct": risk_per_trade,
            "wallet_cap_pct": wallet_cap,
            "leverage": 1.0,
        },
    }


def evaluate(primary: dict[str, Any], stress: dict[str, Any], by_symbol: dict[str, dict[str, Any]]):
    evaluable = {s: st for s, st in by_symbol.items() if int(st.get("trades") or 0) >= 10}
    positive = [
        s for s, st in evaluable.items()
        if float(st.get("profit_factor") or 0.0) >= 1.0
        and float(st.get("expectancy_r") or 0.0) > 0.0
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
    }


def main() -> None:
    frames: dict[str, pd.DataFrame] = {}
    for symbol in SYMBOLS:
        print(f"FETCH {symbol}", flush=True)
        frames[symbol] = fetch_history(symbol)
        print(
            f"  bars={len(frames[symbol])} start={frames[symbol]['timestamp'].min()} end={frames[symbol]['timestamp'].max()}",
            flush=True,
        )

    data_end = min(pd.Timestamp(frame["timestamp"].max()) for frame in frames.values())
    holdout_days = (data_end - HOLDOUT_START).total_seconds() / 86400.0
    if holdout_days < 90:
        raise SystemExit(f"fresh holdout shorter than 90 days: {holdout_days:.2f}")
    data_integrity = validate_locked_cadence(
        frames,
        start=HOLDOUT_START - pd.Timedelta(days=40),
        end=data_end,
    )

    signals = generate_signals(frames)
    primary, by_symbol = simulate(frames, signals, 5.0)
    stress, _ = simulate(frames, signals, 15.0)
    decision = evaluate(primary, stress, by_symbol)

    outcome_complete_cutoff = data_end - pd.Timedelta(days=10)
    portfolio = {
        "5bps": portfolio_simulate(frames, signals, slip_bps=5.0),
        "15bps": portfolio_simulate(frames, signals, slip_bps=15.0),
        "50bps": portfolio_simulate(frames, signals, slip_bps=50.0),
        "outcome_complete_5bps": portfolio_simulate(
            frames,
            signals,
            slip_bps=5.0,
            outcome_complete_cutoff=outcome_complete_cutoff,
        ),
        "outcome_complete_15bps": portfolio_simulate(
            frames,
            signals,
            slip_bps=15.0,
            outcome_complete_cutoff=outcome_complete_cutoff,
        ),
    }

    result = {
        "schema": "atb-v5-independent-crosscheck-v1",
        "source_definition": "AutoTraderBot d8b8def/fde1da6 V5 logic copied verbatim where material",
        "research_only": True,
        "execution_authority": False,
        "symbols": list(SYMBOLS),
        "holdout_start": HOLDOUT_START.isoformat(),
        "data_end": data_end.isoformat(),
        "holdout_days": round(holdout_days, 3),
        "data_integrity": data_integrity,
        "5bps": primary,
        "15bps": stress,
        "by_symbol_5bps": by_symbol,
        "decision": decision,
        "portfolio_integrated": portfolio,
        "outcome_complete_cutoff": outcome_complete_cutoff.isoformat(),
        "notes": [
            "Uses confirmed OKX 15m public candles and completed 4h bars only.",
            "Entries occur on next 15m open after the completed 4h signal.",
            "This is an independent hosted cross-check, not the authoritative private-repo workflow.",
        ],
    }
    Path("atb_v5_crosscheck_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("RESULT=" + json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
