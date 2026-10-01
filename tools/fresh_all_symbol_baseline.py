from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

OKX_BASE = "https://www.okx.com"
EXCLUDED_BASES = {
    "AAPL","AMZN","COIN","CRCL","GOOGL","MSFT","NVDA","PLTR","SPY","TSLA","XAG","XAU","ZEC"
}
MIN_VOLUME_USD = 50_000_000.0
MAX_SPREAD_FRAC = 0.0005
MIN_LISTING_DAYS = 90
MAX_FUNDING_ABS = 0.001
TAKER_FEE = 0.0005
FUNDING_8H_STRESS = 0.0002
RISK_PER_TRADE = 0.0025
DAILY_LOSS_LIMIT = 0.005
MAX_WALLET_PCT = 0.20
SL_ATR = 1.5
TP_ATR = 2.5
MAX_HOLD_BARS = 96
BAR_MINUTES = 15
BARS_PER_8H = 32
ISTANBUL_OFFSET = timedelta(hours=3)

COST_SCENARIOS = {
    "liquid_5bps": 0.0005,
    "realistic_15bps": 0.0015,
    "config_stress_50bps": 0.0050,
}
PROFILES = {
    "autotraderbot_probe": {"max_positions": 2, "cooldown_min": 39, "size_scale": 0.25},
    "autotraderbot_promoted_canary": {"max_positions": 2, "cooldown_min": 39, "size_scale": 1.0},
    "proculus_probe_research": {"max_positions": 1, "cooldown_min": 60, "size_scale": 0.25},
    "proculus_promoted_canary": {"max_positions": 1, "cooldown_min": 60, "size_scale": 1.0},
}


def get_json(path: str, params: dict[str, Any] | None = None, attempts: int = 5) -> dict[str, Any]:
    url = OKX_BASE + path
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            r = requests.get(
                url,
                params=params,
                timeout=20,
                headers={"User-Agent": "proculus-fresh-audit/1.0"},
            )
            if r.status_code == 429:
                time.sleep(1.0 + attempt * 0.75)
                continue
            r.raise_for_status()
            payload = r.json()
            if str(payload.get("code", "0")) != "0":
                raise RuntimeError(
                    f"OKX error code={payload.get('code')} msg={payload.get('msg')}"
                )
            return payload
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last = exc
            time.sleep(0.5 + attempt * 0.75)
    raise RuntimeError(f"request failed: {path} {params}: {last}")


def discover_universe() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inst = get_json("/api/v5/public/instruments", {"instType": "SWAP"}).get("data", [])
    tickers = get_json("/api/v5/market/tickers", {"instType": "SWAP"}).get("data", [])
    ticker_by_id = {str(x.get("instId")): x for x in tickers}
    now_ms = int(time.time() * 1000)
    rejection = Counter()
    scanned = 0
    prelim: list[dict[str, Any]] = []

    for item in inst:
        inst_id = str(item.get("instId") or "")
        if not inst_id:
            continue
        if str(item.get("state") or "").lower() != "live":
            continue
        if str(item.get("settleCcy") or "").upper() != "USDT":
            continue
        if str(item.get("ctType") or "").lower() != "linear":
            continue
        scanned += 1
        base = inst_id.removesuffix("-USDT-SWAP")
        if base in EXCLUDED_BASES or base.startswith("TEST"):
            rejection["excluded_base"] += 1
            continue
        try:
            list_ms = int(item.get("listTime") or 0)
        except (TypeError, ValueError):
            list_ms = 0
        listing_days = (now_ms - list_ms) / 86_400_000 if list_ms > 0 else 0
        if listing_days < MIN_LISTING_DAYS:
            rejection["listing_lt_90d"] += 1
            continue
        t = ticker_by_id.get(inst_id)
        if not t:
            rejection["ticker_missing"] += 1
            continue
        try:
            bid = float(t.get("bidPx") or 0)
            ask = float(t.get("askPx") or 0)
            last = float(t.get("last") or 0)
            vol_ccy = float(t.get("volCcy24h") or 0)
        except (TypeError, ValueError):
            rejection["ticker_invalid"] += 1
            continue
        if min(bid, ask, last) <= 0:
            rejection["ticker_nonpositive"] += 1
            continue
        mid = (bid + ask) / 2
        spread = (ask - bid) / mid if mid > 0 else 1
        if spread > MAX_SPREAD_FRAC:
            rejection["spread_gt_5bps"] += 1
            continue
        approx_quote_volume = vol_ccy * last
        if approx_quote_volume < MIN_VOLUME_USD:
            rejection["volume_lt_50m"] += 1
            continue
        prelim.append(
            {
                "instId": inst_id,
                "base": base,
                "listing_days": round(listing_days, 1),
                "spread_bps": round(spread * 10_000, 3),
                "approx_quote_volume_usd": round(approx_quote_volume, 2),
            }
        )

    eligible: list[dict[str, Any]] = []
    for idx, row in enumerate(prelim, 1):
        try:
            fr = get_json(
                "/api/v5/public/funding-rate", {"instId": row["instId"]}
            ).get("data", [])
            rate = float(fr[0].get("fundingRate") or 0) if fr else 0.0
        except Exception:
            rejection["funding_unavailable"] += 1
            continue
        if abs(rate) > MAX_FUNDING_ABS:
            rejection["funding_abs_gt_0_1pct"] += 1
            continue
        row["current_funding_rate"] = rate
        eligible.append(row)
        if idx % 20 == 0:
            time.sleep(0.25)

    meta = {
        "scanned_live_linear_usdt_swaps": scanned,
        "pre_funding_filter_count": len(prelim),
        "eligible_count": len(eligible),
        "rejection_reasons": dict(rejection),
        "filters": {
            "excluded_bases": sorted(EXCLUDED_BASES),
            "min_24h_approx_quote_volume_usd": MIN_VOLUME_USD,
            "max_spread_bps": MAX_SPREAD_FRAC * 10_000,
            "min_listing_days": MIN_LISTING_DAYS,
            "max_abs_current_funding_rate": MAX_FUNDING_ABS,
        },
        "volume_note": (
            "OKX derivative volCcy24h is converted to approximate quote USD "
            "via current last price."
        ),
    }
    return eligible, meta


def fetch_history(inst_id: str, days: int) -> pd.DataFrame:
    cutoff_ms = int(
        (datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000
    )
    rows: dict[int, list[Any]] = {}
    after: str | None = None
    stagnant = 0

    while True:
        params: dict[str, Any] = {
            "instId": inst_id,
            "bar": "15m",
            "limit": "300",
        }
        if after is not None:
            params["after"] = after
        data = get_json("/api/v5/market/history-candles", params).get("data", [])
        if not data:
            break
        batch_min = None
        for raw in data:
            if len(raw) < 9:
                continue
            try:
                ts = int(raw[0])
                confirm = str(raw[8])
                if confirm != "1":
                    continue
                rows[ts] = raw
                batch_min = ts if batch_min is None else min(batch_min, ts)
            except (TypeError, ValueError):
                continue
        if batch_min is None:
            break
        if batch_min <= cutoff_ms:
            break
        if after == str(batch_min):
            stagnant += 1
            if stagnant >= 2:
                break
        else:
            stagnant = 0
        after = str(batch_min)
        time.sleep(0.40)

    prepared = []
    for ts, raw in rows.items():
        if ts < cutoff_ms:
            continue
        try:
            prepared.append(
                (
                    pd.to_datetime(ts, unit="ms", utc=True),
                    float(raw[1]),
                    float(raw[2]),
                    float(raw[3]),
                    float(raw[4]),
                    float(raw[5]),
                )
            )
        except (TypeError, ValueError):
            continue
    if not prepared:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(
        prepared,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    return (
        df.drop_duplicates("timestamp")
        .sort_values("timestamp")
        .set_index("timestamp")
    )


def wilder_rsi(close: pd.Series, period: int = 90) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0).fillna(0.0)
    loss = (-delta.clip(upper=0.0)).fillna(0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100.0 - (100.0 / (1.0 + rs))


def stochastic_rsi(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    rsi = wilder_rsi(close, 90)
    lo = rsi.rolling(90, min_periods=90).min()
    hi = rsi.rolling(90, min_periods=90).max()
    spread = hi - lo
    raw = ((rsi - lo) / spread.where(spread.abs() > 1e-12)) * 100.0
    k = raw.rolling(3, min_periods=3).mean().clip(0.0, 100.0)
    d = k.rolling(3, min_periods=3).mean().clip(0.0, 100.0)
    return k, d


def add_signals(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    close, high, low = x["close"], x["high"], x["low"]
    x["ema_fast"] = close.ewm(span=20, adjust=False).mean()
    x["ema_slow"] = close.ewm(span=50, adjust=False).mean()
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    x["atr"] = tr.rolling(14).mean()

    plus_orig = high.diff().clip(lower=0)
    minus_orig = (-low.diff()).clip(lower=0)
    plus_dm = plus_orig.where(plus_orig >= minus_orig, 0)
    minus_dm = minus_orig.where(minus_orig >= plus_orig, 0)
    plus_di = 100 * (plus_dm.rolling(14).mean() / x["atr"].replace(0, 1e-10))
    minus_di = 100 * (minus_dm.rolling(14).mean() / x["atr"].replace(0, 1e-10))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    x["adx"] = dx.rolling(14).mean()

    k, d = stochastic_rsi(close)
    x["stochrsi_k90"] = k
    x["stochrsi_d90"] = d
    regime = pd.Series("range", index=x.index, dtype=object)
    bull = (x["adx"] > 25.0) & (x["ema_fast"] >= x["ema_slow"])
    bear = (x["adx"] > 25.0) & (x["ema_fast"] < x["ema_slow"])
    regime.loc[bull] = "bull"
    regime.loc[bear] = "bear"
    x["regime"] = regime
    x["signal"] = 0
    x.loc[
        ((regime == "bull") | (regime == "range")) & (k <= 15.0) & (k >= d),
        "signal",
    ] = 1
    x.loc[
        ((regime == "bear") | (regime == "range")) & (k >= 85.0) & (k <= d),
        "signal",
    ] = -1

    x["entry_signal"] = x["signal"].shift(1).fillna(0).astype(int)
    x["entry_atr"] = x["atr"].shift(1)
    return x.dropna(subset=["entry_atr", "stochrsi_k90", "stochrsi_d90"])


@dataclass
class Position:
    symbol: str
    direction: int
    regime: str
    qty: float
    entry_price: float
    entry_notional: float
    stop: float
    target: float
    entry_time: pd.Timestamp
    risk_dollars: float
    bars: int = 0


@dataclass
class Trade:
    symbol: str
    direction: str
    regime: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    pnl_usd: float
    pnl_pct_notional: float
    r_multiple: float
    bars_held: int
    exit_reason: str
    fees_usd: float
    funding_usd: float


def local_day(ts: pd.Timestamp):
    return (
        ts.to_pydatetime().astimezone(timezone.utc) + ISTANBUL_OFFSET
    ).date()


def simulate(
    frames: dict[str, pd.DataFrame],
    *,
    profile_name: str,
    slippage: float,
    oos_start: pd.Timestamp,
    initial: float = 10_000.0,
) -> dict[str, Any]:
    profile = PROFILES[profile_name]
    max_positions = int(profile["max_positions"])
    cooldown_bars = max(
        1, math.ceil(int(profile["cooldown_min"]) / BAR_MINUTES)
    )
    size_scale = float(profile["size_scale"])

    balance = initial
    peak = initial
    max_dd = 0.0
    positions: dict[str, Position] = {}
    last_entry_idx: dict[str, int] = defaultdict(lambda: -10**9)
    trades: list[Trade] = []
    day_start_balance: dict[Any, float] = {}
    day_halted: set[Any] = set()

    combined = []
    for symbol, df in frames.items():
        d = df.copy()
        d["symbol"] = symbol
        d["bar_number"] = np.arange(len(d))
        combined.append(d.reset_index())
    if not combined:
        return {}
    allbars = pd.concat(combined, ignore_index=True).sort_values(
        ["timestamp", "symbol"]
    )

    for ts, chunk in allbars.groupby("timestamp", sort=True):
        if ts < oos_start:
            continue
        day = local_day(ts)
        if day not in day_start_balance:
            day_start_balance[day] = balance
        if balance <= day_start_balance[day] * (1 - DAILY_LOSS_LIMIT):
            day_halted.add(day)

        by_symbol = {
            str(row["symbol"]): row for _, row in chunk.iterrows()
        }

        to_close: list[tuple[str, float, str]] = []
        for sym, pos in list(positions.items()):
            row = by_symbol.get(sym)
            if row is None:
                continue
            pos.bars += 1
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])
            if pos.direction == 1:
                if low <= pos.stop:
                    to_close.append((sym, pos.stop, "sl"))
                elif high >= pos.target:
                    to_close.append((sym, pos.target, "tp"))
                elif pos.bars >= MAX_HOLD_BARS:
                    to_close.append((sym, close, "timeout"))
            else:
                if high >= pos.stop:
                    to_close.append((sym, pos.stop, "sl"))
                elif low <= pos.target:
                    to_close.append((sym, pos.target, "tp"))
                elif pos.bars >= MAX_HOLD_BARS:
                    to_close.append((sym, close, "timeout"))

        for sym, raw_exit, reason in to_close:
            pos = positions.pop(sym)
            exit_px = raw_exit * (
                1 - slippage if pos.direction == 1 else 1 + slippage
            )
            gross = (exit_px - pos.entry_price) * pos.qty * pos.direction
            exit_notional = abs(exit_px * pos.qty)
            fees = TAKER_FEE * (pos.entry_notional + exit_notional)
            funding = (
                pos.entry_notional
                * FUNDING_8H_STRESS
                * (pos.bars / BARS_PER_8H)
            )
            pnl = gross - fees - funding
            balance += pnl
            trades.append(
                Trade(
                    symbol=sym,
                    direction="long" if pos.direction == 1 else "short",
                    regime=pos.regime,
                    entry_time=str(pos.entry_time),
                    exit_time=str(ts),
                    entry_price=pos.entry_price,
                    exit_price=exit_px,
                    pnl_usd=pnl,
                    pnl_pct_notional=(
                        pnl / pos.entry_notional * 100
                        if pos.entry_notional
                        else 0.0
                    ),
                    r_multiple=(
                        pnl / pos.risk_dollars if pos.risk_dollars else 0.0
                    ),
                    bars_held=pos.bars,
                    exit_reason=reason,
                    fees_usd=fees,
                    funding_usd=funding,
                )
            )

        if balance <= day_start_balance[day] * (1 - DAILY_LOSS_LIMIT):
            day_halted.add(day)

        if day not in day_halted and len(positions) < max_positions:
            for _, row in chunk.sort_values("symbol").iterrows():
                if len(positions) >= max_positions:
                    break
                sym = str(row["symbol"])
                if sym in positions:
                    continue
                signal = int(row["entry_signal"])
                if signal == 0:
                    continue
                bar_no = int(row["bar_number"])
                if bar_no - last_entry_idx[sym] < cooldown_bars:
                    continue
                raw_open = float(row["open"])
                atr = float(row["entry_atr"])
                if raw_open <= 0 or atr <= 0:
                    continue
                entry_px = raw_open * (
                    1 + slippage if signal == 1 else 1 - slippage
                )
                stop_distance = SL_ATR * atr
                stop_pct = stop_distance / entry_px
                if stop_pct <= 0:
                    continue

                risk_dollars = (
                    balance * RISK_PER_TRADE * size_scale
                )
                target_notional_by_risk = risk_dollars / stop_pct
                max_notional = balance * MAX_WALLET_PCT * size_scale
                notional = min(target_notional_by_risk, max_notional)
                if notional <= 1:
                    continue
                qty = notional / entry_px
                if signal == 1:
                    stop = entry_px - stop_distance
                    target = entry_px + TP_ATR * atr
                else:
                    stop = entry_px + stop_distance
                    target = entry_px - TP_ATR * atr
                if stop <= 0 or target <= 0:
                    continue
                positions[sym] = Position(
                    symbol=sym,
                    direction=signal,
                    regime=str(row.get("regime") or "unknown"),
                    qty=qty,
                    entry_price=entry_px,
                    entry_notional=notional,
                    stop=stop,
                    target=target,
                    entry_time=ts,
                    risk_dollars=risk_dollars,
                )
                last_entry_idx[sym] = bar_no

        unreal = 0.0
        for sym, pos in positions.items():
            row = by_symbol.get(sym)
            if row is None:
                continue
            cp = float(row["close"])
            unreal += (
                (cp - pos.entry_price)
                * pos.qty
                * pos.direction
            )
        equity = balance + unreal
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    for sym, pos in list(positions.items()):
        tail = frames[sym][frames[sym].index >= oos_start]
        if tail.empty:
            continue
        raw_exit = float(tail.iloc[-1]["close"])
        ts = tail.index[-1]
        exit_px = raw_exit * (
            1 - slippage if pos.direction == 1 else 1 + slippage
        )
        gross = (exit_px - pos.entry_price) * pos.qty * pos.direction
        exit_notional = abs(exit_px * pos.qty)
        fees = TAKER_FEE * (pos.entry_notional + exit_notional)
        funding = (
            pos.entry_notional
            * FUNDING_8H_STRESS
            * (pos.bars / BARS_PER_8H)
        )
        pnl = gross - fees - funding
        balance += pnl
        trades.append(
            Trade(
                symbol=sym,
                direction="long" if pos.direction == 1 else "short",
                regime=pos.regime,
                entry_time=str(pos.entry_time),
                exit_time=str(ts),
                entry_price=pos.entry_price,
                exit_price=exit_px,
                pnl_usd=pnl,
                pnl_pct_notional=(
                    pnl / pos.entry_notional * 100
                    if pos.entry_notional
                    else 0.0
                ),
                r_multiple=(
                    pnl / pos.risk_dollars if pos.risk_dollars else 0.0
                ),
                bars_held=pos.bars,
                exit_reason="end_of_data",
                fees_usd=fees,
                funding_usd=funding,
            )
        )

    pnls = [t.pnl_usd for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    rvals = [t.r_multiple for t in trades]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    by_symbol: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        by_symbol[t.symbol].append(t.pnl_usd)
    symbol_stats = [
        {"symbol": s, "trades": len(v), "pnl_usd": round(sum(v), 2)}
        for s, v in by_symbol.items()
    ]
    symbol_stats.sort(key=lambda x: x["pnl_usd"], reverse=True)

    def grouped_stats(key_fn):
        grouped: dict[str, list[Trade]] = defaultdict(list)
        for trade in trades:
            grouped[str(key_fn(trade))].append(trade)
        out = []
        for key, group in grouped.items():
            gp = [t.pnl_usd for t in group]
            gw = [p for p in gp if p > 0]
            gl = [p for p in gp if p <= 0]
            gr = [t.r_multiple for t in group]
            gross_p = sum(gw)
            gross_l = abs(sum(gl))
            out.append({
                "key": key,
                "trades": len(group),
                "wins": len(gw),
                "win_rate_pct": round(len(gw) / len(group) * 100, 2) if group else 0.0,
                "pnl_usd": round(sum(gp), 2),
                "profit_factor": round(gross_p / gross_l, 3) if gross_l > 0 else None,
                "expectancy_r": round(sum(gr) / len(gr), 4) if gr else 0.0,
                "avg_r": round(sum(gr) / len(gr), 4) if gr else 0.0,
            })
        out.sort(key=lambda row: row["pnl_usd"], reverse=True)
        return out

    monthly_groups: dict[str, list[Trade]] = defaultdict(list)
    for trade in trades:
        try:
            month = str(pd.Timestamp(trade.exit_time).to_period("M"))
        except Exception:
            month = "unknown"
        monthly_groups[month].append(trade)
    monthly_stats = []
    for month, group in sorted(monthly_groups.items()):
        vals = [t.pnl_usd for t in group]
        rvals_m = [t.r_multiple for t in group]
        wins_m = [v for v in vals if v > 0]
        losses_m = [v for v in vals if v <= 0]
        monthly_stats.append({
            "month": month,
            "trades": len(group),
            "pnl_usd": round(sum(vals), 2),
            "win_rate_pct": round(len(wins_m) / len(group) * 100, 2) if group else 0.0,
            "profit_factor": round(sum(wins_m) / abs(sum(losses_m)), 3) if losses_m and abs(sum(losses_m)) > 0 else None,
            "expectancy_r": round(sum(rvals_m) / len(rvals_m), 4) if rvals_m else 0.0,
        })

    common_last = min(df.index.max() for df in frames.values())
    oos_days = max(
        1.0,
        (
            common_last.to_pydatetime()
            - oos_start.to_pydatetime()
        ).total_seconds()
        / 86400,
    )
    ret = balance / initial - 1
    annualized = (
        (balance / initial) ** (365.0 / oos_days) - 1
        if balance > 0
        else -1.0
    )

    return {
        "profile": profile_name,
        "cost_scenario": next(
            (k for k, v in COST_SCENARIOS.items() if v == slippage),
            str(slippage),
        ),
        "slippage_per_side_bps": slippage * 10_000,
        "taker_fee_per_side_bps": TAKER_FEE * 10_000,
        "funding_stress_per_8h_bps": FUNDING_8H_STRESS * 10_000,
        "initial_balance": initial,
        "final_balance": round(balance, 2),
        "return_pct": round(ret * 100, 3),
        "annualized_return_pct": round(annualized * 100, 3),
        "max_drawdown_pct": round(max_dd * 100, 3),
        "trades": len(trades),
        "wins": len(wins),
        "win_rate_pct": (
            round(len(wins) / len(trades) * 100, 2) if trades else 0.0
        ),
        "profit_factor": (
            round(gross_profit / gross_loss, 3)
            if gross_loss > 0
            else None
        ),
        "expectancy_r": (
            round(sum(rvals) / len(rvals), 4) if rvals else 0.0
        ),
        "avg_trade_pnl_usd": (
            round(sum(pnls) / len(pnls), 3) if pnls else 0.0
        ),
        "total_fees_usd": round(sum(t.fees_usd for t in trades), 2),
        "total_funding_usd": round(
            sum(t.funding_usd for t in trades), 2
        ),
        "daily_halt_days": len(day_halted),
        "best_symbols": symbol_stats[:10],
        "worst_symbols": symbol_stats[-10:],
        "all_symbols": symbol_stats,
        "breakdown": {
            "direction": grouped_stats(lambda t: t.direction),
            "regime": grouped_stats(lambda t: t.regime),
            "exit_reason": grouped_stats(lambda t: t.exit_reason),
            "month": monthly_stats,
        },
        "profile_parameters": dict(profile),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--oos-days", type=int, default=60)
    ap.add_argument(
        "--output", default="fresh_all_symbol_results.json"
    )
    args = ap.parse_args()
    if (
        args.days < 120
        or args.oos_days < 30
        or args.oos_days >= args.days
    ):
        raise SystemExit(
            "Use days>=120, oos-days>=30, oos-days<days"
        )

    started = datetime.now(timezone.utc)
    eligible, universe_meta = discover_universe()
    print(json.dumps({"universe": universe_meta}, ensure_ascii=False))
    frames: dict[str, pd.DataFrame] = {}
    data_failures: dict[str, str] = {}

    def load_one(item: dict[str, Any]) -> tuple[str, pd.DataFrame]:
        inst = str(item["instId"])
        return inst, fetch_history(inst, args.days)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(load_one, item): item for item in eligible}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            item = futures[future]
            inst = str(item["instId"])
            print(
                f"[{completed}/{len(eligible)}] history complete {inst}",
                flush=True,
            )
            try:
                _, df = future.result()
                if len(df) < max(500, args.oos_days * 24 * 4):
                    data_failures[inst] = f"insufficient_bars:{len(df)}"
                    continue
                frames[inst] = add_signals(df)
            except Exception as exc:
                data_failures[inst] = f"{type(exc).__name__}:{exc}"

    if not frames:
        raise SystemExit(
            "No eligible historical frames downloaded"
        )

    common_last = min(df.index.max() for df in frames.values())
    oos_start = common_last - pd.Timedelta(days=args.oos_days)
    results = []
    for profile in PROFILES:
        for scenario, slip in COST_SCENARIOS.items():
            print(
                f"simulate profile={profile} cost={scenario}",
                flush=True,
            )
            results.append(
                simulate(
                    frames,
                    profile_name=profile,
                    slippage=slip,
                    oos_start=oos_start,
                )
            )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(
            (
                datetime.now(timezone.utc) - started
            ).total_seconds(),
            2,
        ),
        "methodology": {
            "fresh_run": True,
            "old_backtest_results_used": False,
            "market_data_source": (
                "OKX public REST history-candles"
            ),
            "universe_source": (
                "all current live linear USDT swaps, then current "
                "project liquidity/listing/funding filters"
            ),
            "bar": "15m",
            "history_days": args.days,
            "oos_days": args.oos_days,
            "signal": (
                "current canonical technical-core StochRSI90 + "
                "EMA20/50 + ADX regime; AI/LLM/ML disabled to "
                "prevent historical leakage"
            ),
            "entry": "next bar open after signal",
            "intrabar_ambiguity": (
                "stop assumed before target if both touched"
            ),
            "risk": (
                "1x canary, 0.25% equity risk/trade before "
                "profile size scale, 20% wallet cap, 0.5% "
                "realized daily loss halt"
            ),
            "costs": (
                "5bps taker fee/side + constant conservative "
                "2bps/8h funding + scenario slippage/side"
            ),
            "survivorship_bias": (
                "current active universe is used; delisted "
                "historical symbols are not included"
            ),
            "not_full_stack": True,
            "not_full_stack_reason": (
                "AI/LLM/ML point-in-time artifacts are not "
                "replayed; Proculus current PR lacks canonical "
                "backtesting/runtime.py"
            ),
        },
        "universe": universe_meta,
        "eligible_instruments": eligible,
        "downloaded_frame_count": len(frames),
        "data_failures": data_failures,
        "oos_start": str(oos_start),
        "common_last": str(common_last),
        "results": results,
    }
    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== FRESH OOS SUMMARY ===")
    for item in results:
        print(
            f"{item['profile']:34s} "
            f"{item['cost_scenario']:22s} "
            f"ret={item['return_pct']:8.3f}% "
            f"dd={item['max_drawdown_pct']:7.3f}% "
            f"trades={item['trades']:4d} "
            f"wr={item['win_rate_pct']:6.2f}% "
            f"pf={item['profit_factor']} "
            f"expR={item['expectancy_r']:7.4f}"
        )


if __name__ == "__main__":
    main()

# fresh-run-trigger: 2026-10-01-two-year