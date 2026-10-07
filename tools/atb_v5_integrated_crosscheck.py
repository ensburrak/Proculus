from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
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



@dataclass
class ScheduledTrade:
    symbol: str
    side: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    notional: float
    leverage: float
    entry_fee: float
    exit_fee: float
    funding: float
    risk_usd: float
    net_pnl: float
    bars: int
    exit_reason: str


def validate_locked_cadence(
    frames: dict[str, pd.DataFrame],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    expected = pd.Timedelta(minutes=15)
    audit: dict[str, Any] = {}
    for symbol, frame in frames.items():
        ts = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
        window = ts[(ts >= start) & (ts <= end)]
        if len(window) < 2:
            raise RuntimeError(f"insufficient locked-window bars: {symbol}")
        if window.has_duplicates or not window.is_monotonic_increasing:
            raise RuntimeError(f"invalid locked-window timestamps: {symbol}")
        deltas = window[1:] - window[:-1]
        bad = deltas != expected
        if bool(bad.any()):
            j = int(bad.argmax())
            raise RuntimeError(
                f"cadence gap {symbol} {window[j].isoformat()} -> "
                f"{window[j+1].isoformat()} delta={deltas[j]}"
            )
        if window[0] > start + expected:
            raise RuntimeError(f"locked window starts too late: {symbol} {window[0]}")
        if window[-1] < end - expected:
            raise RuntimeError(f"locked window ends too early: {symbol} {window[-1]}")
        audit[symbol] = {
            "bars": len(window),
            "first": window[0].isoformat(),
            "last": window[-1].isoformat(),
        }
    return {
        "passed": True,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "symbols": audit,
    }


def _trade_path(
    frame: pd.DataFrame,
    signal: ResearchSignal,
    *,
    fee: float,
    slip: float,
) -> dict[str, Any] | None:
    entry_idx = int(signal.decision_idx) + 1
    if entry_idx <= 0 or entry_idx >= len(frame):
        return None
    raw_open = float(frame.iloc[entry_idx]["open"])
    fill = _adverse_entry(raw_open, signal.side, slip)
    atr = signal.atr_value
    if atr is None or not math.isfinite(float(atr)) or float(atr) <= 0:
        return None
    atr = float(atr)
    stop_distance = atr * max(float(signal.stop_atr_mult), 0.1)
    if stop_distance <= 0:
        return None

    current_stop = fill - stop_distance if signal.side == "long" else fill + stop_distance
    initial_stop = current_stop
    peak = fill
    max_end = min(len(frame) - 1, entry_idx + int(signal.max_hold_bars))
    bars = 0

    def result(raw_exit: float, idx: int, reason: str) -> dict[str, Any]:
        exit_px = _adverse_exit(raw_exit, signal.side, slip)
        return {
            "entry_idx": entry_idx,
            "exit_idx": idx,
            "fill": fill,
            "exit_price": exit_px,
            "exit_time": pd.Timestamp(frame.iloc[idx]["timestamp"]),
            "bars": bars,
            "reason": reason,
            "stop_distance": stop_distance,
        }

    for idx in range(entry_idx, max_end + 1):
        bars += 1
        row = frame.iloc[idx]
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        # Exact AutoTraderBot integrated V5 leverage=1 liquidation guard.
        liq_buffer = max(0.0, 0.90)
        liq = fill * (1.0 - liq_buffer) if signal.side == "long" else fill * (1.0 + liq_buffer)
        if (low <= liq if signal.side == "long" else high >= liq):
            return result(liq, idx, "liquidation")

        peak = max(peak, high) if signal.side == "long" else min(peak, low)

        if (low <= current_stop if signal.side == "long" else high >= current_stop):
            moved = (
                current_stop > initial_stop + 1e-12
                if signal.side == "long"
                else current_stop < initial_stop - 1e-12
            )
            return result(current_stop, idx, "v5_trailing_stop" if moved else "stop_loss")

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
        if mfe_r >= 1.0 and signal.trail_atr_mult is not None:
            trail = float(signal.trail_atr_mult) * atr
            candidate_stop = close - trail if signal.side == "long" else close + trail
            current_stop = (
                max(current_stop, candidate_stop)
                if signal.side == "long"
                else min(current_stop, candidate_stop)
            )
        if bars >= 192 and mfe_r < 0.5:
            return result(close, idx, "v5_time_decay")

    return result(float(frame.iloc[max_end]["close"]), max_end, "time_stop")


def _mark_equity(
    cash: float,
    positions: list[ScheduledTrade],
    frames: dict[str, pd.DataFrame],
    ts: pd.Timestamp,
) -> float:
    equity = cash
    for pos in positions:
        frame = frames[pos.symbol]
        times = pd.Series(pd.to_datetime(frame["timestamp"], utc=True))
        idx = int(times.searchsorted(ts, side="right")) - 1
        if idx < 0:
            continue
        px = float(frame.iloc[min(idx, len(frame) - 1)]["close"])
        qty = pos.notional / max(pos.entry_price, 1e-12)
        sign = 1.0 if pos.side == "long" else -1.0
        equity += (px - pos.entry_price) * qty * sign
    return equity


def simulate_integrated(
    frames: dict[str, pd.DataFrame],
    signals: dict[str, list[ResearchSignal]],
    *,
    slip_bps: float,
    initial: float = 10000.0,
) -> dict[str, Any]:
    fee = 5.0 / 10000.0
    slip = float(slip_bps) / 10000.0
    cooldown_min = 39
    max_positions = 2
    daily_limit = 0.005
    weekly_limit = 0.02
    reduce_threshold = 0.50
    hedge_threshold = 0.75
    tz = ZoneInfo("Europe/Istanbul")

    candidates: list[tuple[pd.Timestamp, str, ResearchSignal]] = []
    for symbol, family_signals in signals.items():
        frame = frames[symbol]
        for signal in family_signals:
            entry_idx = int(signal.decision_idx) + 1
            if entry_idx <= 0 or entry_idx >= len(frame):
                continue
            entry_time = pd.Timestamp(frame.iloc[entry_idx]["timestamp"])
            if entry_time < HOLDOUT_START:
                continue
            candidates.append((entry_time, symbol, signal))
    candidates.sort(key=lambda item: (item[0], item[1], item[2].setup_id))

    cash = float(initial)
    positions: list[ScheduledTrade] = []
    closed: list[ScheduledTrade] = []
    blocked: dict[str, int] = defaultdict(int)
    last_entry: dict[str, pd.Timestamp] = {}
    daily_realized: dict[str, float] = defaultdict(float)
    weekly_realized: dict[str, float] = defaultdict(float)
    equity_curve = [float(initial)]

    def local_keys(ts: pd.Timestamp) -> tuple[str, str]:
        local = ts.tz_convert(tz)
        return local.strftime("%Y-%m-%d"), local.strftime("%Y-W%W")

    def realize(ts: pd.Timestamp) -> None:
        nonlocal cash, positions
        remaining: list[ScheduledTrade] = []
        for pos in positions:
            if pos.exit_time <= ts:
                qty = pos.notional / max(pos.entry_price, 1e-12)
                sign = 1.0 if pos.side == "long" else -1.0
                gross = (pos.exit_price - pos.entry_price) * qty * sign
                cash += gross - pos.exit_fee - pos.funding
                closed.append(pos)
                day, week = local_keys(pos.exit_time)
                daily_realized[day] += pos.net_pnl
                weekly_realized[week] += pos.net_pnl
            else:
                remaining.append(pos)
        positions = remaining
        equity_curve.append(_mark_equity(cash, positions, frames, ts))

    for entry_time, symbol, signal in candidates:
        realize(entry_time)
        eq = max(0.0, _mark_equity(cash, positions, frames, entry_time))
        if eq <= 0:
            blocked["equity_depleted"] += 1
            continue

        day, week = local_keys(entry_time)
        daily_loss = abs(min(daily_realized.get(day, 0.0), 0.0))
        weekly_loss = abs(min(weekly_realized.get(week, 0.0), 0.0))
        daily_abs = eq * daily_limit
        weekly_abs = eq * weekly_limit
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
        risk_multiplier = 0.5 if (daily_progress >= reduce_threshold or weekly_progress >= 0.75) else 1.0

        prior = last_entry.get(symbol)
        if prior is not None and (entry_time - prior).total_seconds() < cooldown_min * 60:
            blocked["trade_cooldown"] += 1
            continue
        if len(positions) >= max_positions:
            blocked["max_open_positions"] += 1
            continue
        if any(pos.symbol == symbol for pos in positions):
            blocked["symbol_already_open"] += 1
            continue

        frame = frames[symbol]
        path = _trade_path(frame, signal, fee=fee, slip=slip)
        if path is None:
            blocked["invalid_path"] += 1
            continue

        fill = float(path["fill"])
        stop_distance = float(path["stop_distance"])
        stop_pct = stop_distance / max(fill, 1e-12)
        risk_budget = eq * 0.0025 * risk_multiplier
        notional = min(
            risk_budget / max(stop_pct, 1e-9),
            eq * 0.20 * risk_multiplier,
        )
        margin = notional  # leverage=1
        used = sum(pos.notional / max(pos.leverage, 1.0) for pos in positions)
        if notional <= 0 or used + margin > eq * 0.95:
            blocked["margin_cap"] += 1
            continue

        exit_px = float(path["exit_price"])
        qty = notional / max(fill, 1e-12)
        sign = 1.0 if signal.side == "long" else -1.0
        gross = (exit_px - fill) * qty * sign
        entry_fee = notional * fee
        exit_fee = abs(exit_px * qty) * fee
        bars = int(path["bars"])
        funding = notional * 0.0001 * (bars // 32)
        net = gross - entry_fee - exit_fee - funding
        risk_usd = stop_distance * qty

        cash -= entry_fee
        positions.append(
            ScheduledTrade(
                symbol=symbol,
                side=signal.side,
                entry_time=entry_time,
                exit_time=pd.Timestamp(path["exit_time"]),
                entry_price=fill,
                exit_price=exit_px,
                notional=notional,
                leverage=1.0,
                entry_fee=entry_fee,
                exit_fee=exit_fee,
                funding=funding,
                risk_usd=risk_usd,
                net_pnl=net,
                bars=bars,
                exit_reason=str(path["reason"]),
            )
        )
        last_entry[symbol] = entry_time
        equity_curve.append(_mark_equity(cash, positions, frames, entry_time))

    if candidates:
        realize(pd.Timestamp.max.tz_localize("UTC"))

    pnls = [trade.net_pnl for trade in closed]
    wins = [value for value in pnls if value > 0]
    losses = [value for value in pnls if value < 0]
    rvals = [
        trade.net_pnl / trade.risk_usd
        for trade in closed
        if trade.risk_usd > 0
    ]
    peak = -float("inf")
    max_dd = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak > 0:
            max_dd = min(max_dd, (value - peak) / peak)

    by_symbol: dict[str, list[ScheduledTrade]] = defaultdict(list)
    by_reason: dict[str, int] = defaultdict(int)
    for trade in closed:
        by_symbol[trade.symbol].append(trade)
        by_reason[trade.exit_reason] += 1

    def group_stats(items: list[ScheduledTrade]) -> dict[str, Any]:
        vals = [item.net_pnl for item in items]
        w = [v for v in vals if v > 0]
        l = [v for v in vals if v < 0]
        rv = [item.net_pnl / item.risk_usd for item in items if item.risk_usd > 0]
        return {
            "trades": len(items),
            "net_pnl": round(sum(vals), 4),
            "profit_factor": round(sum(w) / abs(sum(l)), 5) if l else (999.0 if w else 0.0),
            "expectancy_r": round(sum(rv) / max(1, len(rv)), 5),
        }

    return {
        "initial_balance": round(initial, 4),
        "final_balance": round(cash, 4),
        "return_pct": round(100.0 * (cash / initial - 1.0), 4),
        "max_drawdown_pct": round(abs(max_dd) * 100.0, 4),
        "trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(100.0 * len(wins) / max(1, len(closed)), 4),
        "profit_factor": round(sum(wins) / abs(sum(losses)), 5) if losses else (999.0 if wins else 0.0),
        "expectancy_r": round(sum(rvals) / max(1, len(rvals)), 5),
        "blocked_entries": dict(sorted(blocked.items())),
        "exit_reasons": dict(sorted(by_reason.items())),
        "by_symbol": {
            symbol: group_stats(items)
            for symbol, items in sorted(by_symbol.items())
        },
    }


def integrated_gate(
    primary: dict[str, Any],
    stress: dict[str, Any],
    complete_primary: dict[str, Any],
    complete_stress: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "primary_min_trades": int(primary.get("trades", 0)) >= 30,
        "primary_profit_factor": float(primary.get("profit_factor", 0.0)) >= 1.15,
        "primary_expectancy": float(primary.get("expectancy_r", -999.0)) >= 0.05,
        "primary_drawdown": float(primary.get("max_drawdown_pct", 999.0)) <= 20.0,
        "stress_min_trades": int(stress.get("trades", 0)) >= 20,
        "stress_profit_factor": float(stress.get("profit_factor", 0.0)) >= 1.05,
        "stress_expectancy": float(stress.get("expectancy_r", -999.0)) > 0.0,
        "stress_drawdown": float(stress.get("max_drawdown_pct", 999.0)) <= 20.0,
        "outcome_complete_primary_profit_factor": float(complete_primary.get("profit_factor", 0.0)) >= 1.0,
        "outcome_complete_primary_expectancy": float(complete_primary.get("expectancy_r", -999.0)) > 0.0,
        "outcome_complete_stress_profit_factor": float(complete_stress.get("profit_factor", 0.0)) >= 1.0,
        "outcome_complete_stress_expectancy": float(complete_stress.get("expectancy_r", -999.0)) > 0.0,
    }
    return {"passed": all(checks.values()), "checks": checks}


def main() -> None:
    frames: dict[str, pd.DataFrame] = {}
    for symbol in SYMBOLS:
        print(f"FETCH730 {symbol}", flush=True)
        frames[symbol] = fetch_history(symbol, days=730)
        print(
            f"  bars={len(frames[symbol])} "
            f"start={frames[symbol]['timestamp'].min()} "
            f"end={frames[symbol]['timestamp'].max()}",
            flush=True,
        )

    common_end = min(pd.Timestamp(frame["timestamp"].max()) for frame in frames.values())
    holdout_days = (common_end - HOLDOUT_START).total_seconds() / 86400.0
    if holdout_days < 90:
        raise SystemExit(f"integrated holdout shorter than 90 days: {holdout_days:.3f}")

    integrity = validate_locked_cadence(
        frames,
        start=HOLDOUT_START - pd.Timedelta(days=40),
        end=common_end,
    )
    all_signals = generate_signals(frames)
    holdout_signals = {
        symbol: [
            signal for signal in values
            if pd.Timestamp(frames[symbol].iloc[signal.decision_idx + 1]["timestamp"]) >= HOLDOUT_START
            and pd.Timestamp(frames[symbol].iloc[signal.decision_idx + 1]["timestamp"]) <= common_end
        ]
        for symbol, values in all_signals.items()
    }
    complete_cutoff = common_end - pd.Timedelta(minutes=15 * 960)
    complete_signals = {
        symbol: [
            signal for signal in values
            if pd.Timestamp(frames[symbol].iloc[signal.decision_idx + 1]["timestamp"]) <= complete_cutoff
        ]
        for symbol, values in holdout_signals.items()
    }

    scenarios = {}
    for bps in (5.0, 15.0, 50.0):
        scenarios[f"slippage_{int(bps)}bps"] = {
            "v5_only": simulate_integrated(frames, holdout_signals, slip_bps=bps),
            "v5_outcome_complete_sensitivity": simulate_integrated(
                frames, complete_signals, slip_bps=bps
            ),
        }

    decision = integrated_gate(
        scenarios["slippage_5bps"]["v5_only"],
        scenarios["slippage_15bps"]["v5_only"],
        scenarios["slippage_5bps"]["v5_outcome_complete_sensitivity"],
        scenarios["slippage_15bps"]["v5_outcome_complete_sensitivity"],
    )
    result = {
        "schema": "atb-v5-independent-integrated-crosscheck-v1",
        "source_definition": "AutoTraderBot V5 integrated portfolio/risk semantics copied from default branch on 2026-10-08",
        "research_only": True,
        "implementation_eligible": True,
        "execution_authority": False,
        "prospective_shadow_required": True,
        "symbols": list(SYMBOLS),
        "fetch_days": 730,
        "holdout_start": HOLDOUT_START.isoformat(),
        "common_data_end": common_end.isoformat(),
        "holdout_days": round(holdout_days, 3),
        "data_integrity": integrity,
        "candidate_counts": {
            "v5_only": sum(len(v) for v in holdout_signals.values()),
            "v5_outcome_complete_sensitivity": sum(len(v) for v in complete_signals.values()),
        },
        "outcome_complete_cutoff": complete_cutoff.isoformat(),
        "scenarios": scenarios,
        "integrated_acceptance": decision,
        "next_gate_on_pass": "default_off_runtime_feature_flag_plus_prospective_paper_shadow",
        "notes": [
            "Uses confirmed OKX 15m public candles and completed 4h bars only.",
            "Uses 730-day fetch to match authoritative integrated replay indicator seeding.",
            "Portfolio constraints: leverage 1x, max 2 positions, 39m per-symbol cooldown, 0.25% risk/trade, 20% wallet cap.",
            "Daily/weekly loss controls and Europe/Istanbul risk-day boundaries match AutoTraderBot integrated replay.",
            "Independent hosted cross-check; authoritative AutoTraderBot self-hosted workflow remains a separate gate.",
        ],
    }
    Path("atb_v5_integrated_crosscheck_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("INTEGRATED_RESULT=" + json.dumps({
        "acceptance": decision,
        "holdout_days": result["holdout_days"],
        "candidate_counts": result["candidate_counts"],
        "primary": scenarios["slippage_5bps"]["v5_only"],
        "stress": scenarios["slippage_15bps"]["v5_only"],
        "complete_primary": scenarios["slippage_5bps"]["v5_outcome_complete_sensitivity"],
        "complete_stress": scenarios["slippage_15bps"]["v5_outcome_complete_sensitivity"],
    }, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
