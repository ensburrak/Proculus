#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import decision.official_pipeline as v2_pipeline
from risk.stop_manager import calculate_dynamic_tp_sl

OUT_DIR = ROOT / "scratch" / "fresh_proculus_two_year"


@dataclass
class Candidate:
    symbol: str
    decision_idx: int
    entry_idx: int
    entry_time: pd.Timestamp
    side: str
    setup_id: str
    strategy: str
    regime: str
    confidence: float
    risk_scale: float
    leverage: float
    atr: float
    decision_price: float


@dataclass
class OpenTrade:
    symbol: str
    side: str
    setup_id: str
    strategy: str
    regime: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    notional: float
    leverage: float
    entry_fee: float
    exit_fee: float
    funding: float
    gross_pnl: float
    net_pnl: float
    risk_usd: float
    exit_reason: str


def f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def macd_hist(close: pd.Series) -> pd.Series:
    fast = ema(close, 12)
    slow = ema(close, 26)
    macd = fast - slow
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    return macd - signal


def wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)


def dmi_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr.replace(0.0, np.nan)
    minus = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr.replace(0.0, np.nan)
    dx = 100.0 * (plus - minus).abs() / (plus + minus).replace(0.0, np.nan)
    adx = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return pd.DataFrame({"atr": atr, "plus_di": plus, "minus_di": minus, "adx": adx})


def classic_stoch(df: pd.DataFrame, period: int = 14) -> pd.Series:
    lo = df["low"].rolling(period, min_periods=period).min()
    hi = df["high"].rolling(period, min_periods=period).max()
    raw = 100.0 * (df["close"] - lo) / (hi - lo).replace(0.0, np.nan)
    return raw.rolling(3, min_periods=3).mean().clip(0, 100)


def symbol_from_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_15m"):
        stem = stem[:-4]
    parts = stem.split("_")
    if len(parts) >= 3 and parts[-1] == "USDT":
        return f"{parts[0]}/USDT:USDT"
    return stem.replace("_", "/")


def load_frame(path: Path, days: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)
    if not df.empty:
        cutoff = df["timestamp"].max() - pd.Timedelta(days=days)
        df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
    return df


def merge_closed(base: pd.DataFrame, higher: pd.DataFrame, col: str, minutes: int) -> pd.Series:
    left = pd.DataFrame({"decision_close": base["timestamp"] + pd.Timedelta(minutes=15), "_idx": base.index})
    right = higher[["timestamp", col]].copy()
    right["closed_at"] = right["timestamp"] + pd.Timedelta(minutes=minutes)
    merged = pd.merge_asof(
        left.sort_values("decision_close"),
        right.sort_values("closed_at")[["closed_at", col]],
        left_on="decision_close",
        right_on="closed_at",
        direction="backward",
    )
    return merged.sort_values("_idx")[col].reset_index(drop=True)


def prepare_frame(symbol: str, data_dir: Path, days: int) -> pd.DataFrame:
    key = symbol.replace("/", "_").replace(":", "_")
    df = load_frame(data_dir / f"{key}_15m.parquet", days)
    close = df["close"]
    df["ema_fast"] = ema(close, 20)
    df["ema_slow"] = ema(close, 50)
    df["ema200"] = ema(close, 200)
    df["rsi"] = wilder_rsi(close)
    dmi = dmi_adx(df)
    for col in dmi.columns:
        df[col] = dmi[col]
    df["atr_ratio"] = df["atr"] / df["close"].replace(0.0, np.nan)
    df["stoch_k"] = classic_stoch(df)
    vol_mean = df["volume"].rolling(20, min_periods=20).mean().shift(1)
    vol_std = df["volume"].rolling(20, min_periods=20).std(ddof=0).shift(1)
    df["volume_spike_ratio"] = df["volume"] / vol_mean.replace(0.0, np.nan)
    df["vol_z"] = (df["volume"] - vol_mean) / vol_std.replace(0.0, np.nan)
    bb_mid = close.rolling(20, min_periods=20).mean()
    bb_std = close.rolling(20, min_periods=20).std(ddof=0)
    df["boll_width"] = (4.0 * bb_std) / bb_mid.replace(0.0, np.nan)
    df["boll_q20"] = df["boll_width"].rolling(192, min_periods=96).quantile(0.20)

    for tf, minutes in (("1h", 60), ("4h", 240)):
        p = data_dir / f"{key}_{tf}.parquet"
        higher = load_frame(p, days + 45)
        higher[f"close_{tf}"] = higher["close"]
        higher[f"ema_fast_{tf}"] = ema(higher["close"], 20)
        higher[f"ema_slow_{tf}"] = ema(higher["close"], 50)
        higher[f"ema200_{tf}"] = ema(higher["close"], 200)
        higher[f"macd_{tf}"] = macd_hist(higher["close"])
        hdmi = dmi_adx(higher)
        higher[f"adx_{tf}"] = hdmi["adx"]
        for col in (
            f"close_{tf}",
            f"ema_fast_{tf}",
            f"ema_slow_{tf}",
            f"ema200_{tf}",
            f"macd_{tf}",
            f"adx_{tf}",
        ):
            df[col] = merge_closed(df, higher, col, minutes)

    def regime(row: pd.Series) -> str:
        atrp = f(row.get("atr_ratio"))
        vol = f(row.get("volume_spike_ratio"))
        adx = f(row.get("adx"))
        ef, es = f(row.get("ema_fast")), f(row.get("ema_slow"))
        plus, minus = f(row.get("plus_di")), f(row.get("minus_di"))
        bw, q20 = f(row.get("boll_width")), f(row.get("boll_q20"))
        if vol >= 5.0 or atrp >= 0.06 or abs(f(row.get("vol_z"))) >= 4.0:
            return "shock"
        if bw > 0 and q20 > 0 and bw <= q20 and adx <= 22.0:
            return "compression"
        if adx <= 18.0:
            return "range"
        if adx >= 18.0 and ef > es and plus >= minus:
            return "bull"
        if adx >= 18.0 and ef < es and minus >= plus:
            return "bear"
        return "transition"

    df["regime"] = df.apply(regime, axis=1)
    return df


def prefilter(row: pd.Series, recent: pd.DataFrame) -> bool:
    regime = str(row.get("regime") or "unknown")
    if regime in {"shock", "transition", "unknown", "conflict"}:
        return False
    if regime == "range":
        rsi, stoch = f(row.get("rsi"), 50.0), f(row.get("stoch_k"), 50.0)
        return rsi <= 38.0 or rsi >= 62.0 or stoch <= 15.0 or stoch >= 85.0
    if regime == "compression":
        if len(recent) < 5:
            return False
        prev_high = float(recent["high"].iloc[-5:-1].max())
        prev_low = float(recent["low"].iloc[-5:-1].min())
        close = f(row.get("close"))
        return close > prev_high or close < prev_low
    return True


def build_item(symbol: str, row: pd.Series, recent: pd.DataFrame) -> dict[str, Any]:
    mtf = {
        "15m": {
            "close": f(row.get("close")),
            "prev_close": f(recent["close"].iloc[-2]) if len(recent) >= 2 else None,
            "recent_closes": [float(v) for v in recent["close"].tail(8).tolist()],
            "recent_lows": [float(v) for v in recent["low"].tail(8).tolist()],
            "recent_highs": [float(v) for v in recent["high"].tail(8).tolist()],
            "ema_fast": f(row.get("ema_fast")),
            "ema_slow": f(row.get("ema_slow")),
            "ema200": f(row.get("ema200")),
            "atr_ratio": f(row.get("atr_ratio")),
            "vol_z": f(row.get("vol_z")),
        },
        "1h": {
            "close": f(row.get("close_1h")),
            "recent_closes": [f(row.get("close_1h"))],
            "ema_fast": f(row.get("ema_fast_1h")),
            "ema_slow": f(row.get("ema_slow_1h")),
            "ema200": f(row.get("ema200_1h")),
            "macd_hist": f(row.get("macd_1h")),
            "adx": f(row.get("adx_1h")),
        },
        "4h": {
            "close": f(row.get("close_4h")),
            "recent_closes": [f(row.get("close_4h"))],
            "ema_fast": f(row.get("ema_fast_4h")),
            "ema_slow": f(row.get("ema_slow_4h")),
            "ema200": f(row.get("ema200_4h")),
            "macd_hist": f(row.get("macd_4h")),
            "adx": f(row.get("adx_4h")),
        },
    }
    ta = {
        "price": f(row.get("close")),
        "close": f(row.get("close")),
        "rsi": f(row.get("rsi")),
        "stoch_k": f(row.get("stoch_k")),
        "adx": f(row.get("adx")),
        "atr": f(row.get("atr")),
        "atr_ratio": f(row.get("atr_ratio")),
        "atr_pct": f(row.get("atr_ratio")),
        "vol_z": f(row.get("vol_z")),
        "regime": str(row.get("regime")),
        "ema": {"fast": f(row.get("ema_fast")), "slow": f(row.get("ema_slow"))},
    }
    return {
        "symbol": symbol,
        "runtime_mode": "paper",
        "regime": str(row.get("regime")),
        "edge_validated": True,
        "meta_model_calibrated": False,
        "ta_pack": ta,
        "mtf_features": mtf,
        "price": f(row.get("close")),
    }


def generate_candidates(symbol: str, frame: pd.DataFrame) -> list[Candidate]:
    candidates: list[Candidate] = []
    for idx in range(800, len(frame) - 1):
        row = frame.iloc[idx]
        recent = frame.iloc[max(0, idx - 7): idx + 1]
        if not prefilter(row, recent):
            continue
        item = build_item(symbol, row, recent)
        decision = v2_pipeline.process_symbol_decision(item=item, ai_part={})
        if str(decision.get("action") or "").lower() != "enter":
            continue
        atr = f(row.get("atr"))
        if atr <= 0:
            continue
        candidates.append(
            Candidate(
                symbol=symbol,
                decision_idx=idx,
                entry_idx=idx + 1,
                entry_time=pd.Timestamp(frame.iloc[idx + 1]["timestamp"]),
                side=str(decision.get("direction")),
                setup_id=str(decision.get("setup_id") or "unknown"),
                strategy=str(decision.get("strategy") or "unknown"),
                regime=str(decision.get("regime") or row.get("regime") or "unknown"),
                confidence=f(decision.get("master_confidence")),
                risk_scale=f(decision.get("risk_scale"), 1.0),
                leverage=max(1.0, f(decision.get("lev"), 1.0)),
                atr=atr,
                decision_price=f(row.get("close")),
            )
        )
    return candidates


def adverse_entry(raw: float, side: str, slip: float) -> float:
    return raw * (1.0 + slip) if side == "long" else raw * (1.0 - slip)


def adverse_exit(raw: float, side: str, slip: float) -> float:
    return raw * (1.0 - slip) if side == "long" else raw * (1.0 + slip)


def find_exit(frame: pd.DataFrame, candidate: Candidate, entry_fill: float, slippage: float) -> tuple[pd.Timestamp, float, str, int, float, float]:
    protection = calculate_dynamic_tp_sl(candidate.side, entry_fill, atr=candidate.atr)
    stop = f(protection.get("stop_loss"))
    target = f(protection.get("take_profit"))
    max_end = min(len(frame) - 1, candidate.entry_idx + 192)
    for idx in range(candidate.entry_idx, max_end + 1):
        row = frame.iloc[idx]
        hi, lo = f(row["high"]), f(row["low"])
        stop_hit = lo <= stop if candidate.side == "long" else hi >= stop
        target_hit = hi >= target if candidate.side == "long" else lo <= target
        if stop_hit:
            return pd.Timestamp(row["timestamp"]), adverse_exit(stop, candidate.side, slippage), "stop_loss", idx, stop, target
        if target_hit:
            return pd.Timestamp(row["timestamp"]), adverse_exit(target, candidate.side, slippage), "take_profit", idx, stop, target
    row = frame.iloc[max_end]
    return pd.Timestamp(row["timestamp"]), adverse_exit(f(row["close"]), candidate.side, slippage), "max_hold", max_end, stop, target


def group_stats(trades: list[OpenTrade], field: str) -> dict[str, Any]:
    buckets: dict[str, list[OpenTrade]] = defaultdict(list)
    for t in trades:
        buckets[str(getattr(t, field))].append(t)
    out = {}
    for key, items in sorted(buckets.items()):
        pnls = [t.net_pnl for t in items]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        rvals = [t.net_pnl / t.risk_usd for t in items if t.risk_usd > 0]
        out[key] = {
            "trades": len(items),
            "net_pnl": round(sum(pnls), 4),
            "win_rate_pct": round(100.0 * len(wins) / max(1, len(items)), 3),
            "profit_factor": round(sum(wins) / abs(sum(losses)), 4) if losses else (999.0 if wins else 0.0),
            "expectancy_r": round(sum(rvals) / max(1, len(rvals)), 4),
        }
    return out


def simulate(
    *,
    candidates: list[Candidate],
    frames: dict[str, pd.DataFrame],
    initial_balance: float,
    fee_bps: float,
    slippage_bps: float,
) -> dict[str, Any]:
    fee = fee_bps / 10_000.0
    slip = slippage_bps / 10_000.0
    cash = float(initial_balance)
    positions: list[OpenTrade] = []
    closed: list[OpenTrade] = []
    skipped = defaultdict(int)
    equity_points = [cash]

    def realize_until(ts: pd.Timestamp) -> None:
        nonlocal cash, positions
        remaining: list[OpenTrade] = []
        for pos in positions:
            if pos.exit_time <= ts:
                cash += pos.gross_pnl - pos.exit_fee - pos.funding
                closed.append(pos)
                equity_points.append(cash)
            else:
                remaining.append(pos)
        positions = remaining

    for cand in sorted(candidates, key=lambda x: (x.entry_time, x.symbol, x.setup_id)):
        realize_until(cand.entry_time)
        if len(positions) >= 2:
            skipped["max_open_positions"] += 1
            continue
        if any(p.symbol == cand.symbol for p in positions):
            skipped["symbol_already_open"] += 1
            continue
        if cand.risk_scale <= 0:
            skipped["zero_risk_scale"] += 1
            continue

        frame = frames[cand.symbol]
        raw_open = f(frame.iloc[cand.entry_idx]["open"])
        if raw_open <= 0:
            skipped["invalid_open"] += 1
            continue
        entry = adverse_entry(raw_open, cand.side, slip)
        stop_distance = max(cand.atr * 1.5, entry * 0.001)
        stop_pct = stop_distance / entry
        risk_budget = cash * 0.0025 * cand.risk_scale
        notional = min(
            risk_budget / max(stop_pct, 1e-9),
            cash * 0.20 * cand.leverage,
        )
        margin = notional / max(cand.leverage, 1.0)
        used_margin = sum(p.notional / max(p.leverage, 1.0) for p in positions)
        if notional <= 0 or used_margin + margin > cash * 0.95:
            skipped["margin_cap"] += 1
            continue

        exit_time, exit_price, reason, exit_idx, stop, target = find_exit(frame, cand, entry, slip)
        qty = notional / entry
        sign = 1.0 if cand.side == "long" else -1.0
        gross = (exit_price - entry) * qty * sign
        entry_fee = notional * fee
        exit_fee = abs(exit_price * qty) * fee
        bars_held = max(1, exit_idx - cand.entry_idx + 1)
        funding = notional * 0.0001 * (bars_held // 32)
        risk_usd = abs(entry - stop) * qty
        net = gross - entry_fee - exit_fee - funding

        cash -= entry_fee
        positions.append(
            OpenTrade(
                symbol=cand.symbol,
                side=cand.side,
                setup_id=cand.setup_id,
                strategy=cand.strategy,
                regime=cand.regime,
                entry_time=cand.entry_time,
                exit_time=exit_time,
                entry_price=entry,
                exit_price=exit_price,
                notional=notional,
                leverage=cand.leverage,
                entry_fee=entry_fee,
                exit_fee=exit_fee,
                funding=funding,
                gross_pnl=gross,
                net_pnl=net,
                risk_usd=risk_usd,
                exit_reason=reason,
            )
        )
        equity_points.append(cash)

    realize_until(pd.Timestamp.max.tz_localize("UTC"))
    pnls = [t.net_pnl for t in closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    peak, max_dd = initial_balance, 0.0
    for value in equity_points:
        peak = max(peak, value)
        if peak > 0:
            max_dd = min(max_dd, (value - peak) / peak)
    rvals = [t.net_pnl / t.risk_usd for t in closed if t.risk_usd > 0]

    return {
        "initial_balance": round(initial_balance, 4),
        "final_balance": round(cash, 4),
        "return_pct": round(100.0 * (cash / initial_balance - 1.0), 4),
        "max_drawdown_pct_realized": round(abs(max_dd) * 100.0, 4),
        "trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(100.0 * len(wins) / max(1, len(closed)), 4),
        "profit_factor": round(sum(wins) / abs(sum(losses)), 5) if losses else (999.0 if wins else 0.0),
        "expectancy_r": round(sum(rvals) / max(1, len(rvals)), 5),
        "fees_paid": round(sum(t.entry_fee + t.exit_fee for t in closed), 4),
        "funding_paid": round(sum(t.funding for t in closed), 4),
        "skipped_entries": dict(sorted(skipped.items())),
        "by_setup": group_stats(closed, "setup_id"),
        "by_regime": group_stats(closed, "regime"),
        "by_symbol": group_stats(closed, "symbol"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--initial-balance", type=float, default=10000.0)
    parser.add_argument("--fee-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", default="5,15,50")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    v2_pipeline._load_config = lambda: cfg  # research replay cache, same config content

    frames: dict[str, pd.DataFrame] = {}
    candidates: list[Candidate] = []
    paths = sorted(args.data_dir.glob("*_15m.parquet"))
    for idx, path in enumerate(paths, 1):
        symbol = symbol_from_path(path)
        try:
            frame = prepare_frame(symbol, args.data_dir, args.days)
            if len(frame) < 1200:
                continue
            frames[symbol] = frame
            symbol_candidates = generate_candidates(symbol, frame)
            candidates.extend(symbol_candidates)
            print(f"[CANDIDATES {idx}/{len(paths)}] {symbol} bars={len(frame)} candidates={len(symbol_candidates)}", flush=True)
        except Exception as exc:
            print(f"[SKIP] {symbol}: {type(exc).__name__}: {exc}", flush=True)

    scenarios = {}
    for raw in str(args.slippage_bps).split(","):
        bps = float(raw.strip())
        name = f"slippage_{int(bps)}bps"
        scenarios[name] = simulate(
            candidates=candidates,
            frames=frames,
            initial_balance=args.initial_balance,
            fee_bps=args.fee_bps,
            slippage_bps=bps,
        )
        print("FRESH_BACKTEST_SCENARIO=" + json.dumps({"scenario": name, **scenarios[name]}, ensure_ascii=False), flush=True)

    report = {
        "schema": "proculus-fresh-v2-two-year-v1",
        "previous_backtests_used": False,
        "history_days": args.days,
        "bar": "15m",
        "symbols": sorted(frames),
        "symbol_count": len(frames),
        "candidate_count": len(candidates),
        "methodology": {
            "decision_engine": "decision.official_pipeline.process_symbol_decision",
            "edge_gate": "research override edge_validated=True to measure setup edge; production strict gate unchanged",
            "ml_direction_authority": "disabled",
            "meta_quality": "not enforced because no calibrated historical meta model is available",
            "entry": "next 15m open after closed-candle decision",
            "max_open_positions": 2,
            "risk_per_trade_pct": 0.25,
            "wallet_cap_pct": 20.0,
            "funding_assumption": "1 bp per 8h, conservative cost",
            "same_bar_priority": "stop before target",
            "lookahead": "1h/4h values become available only after higher-timeframe candle close",
        },
        "scenarios": scenarios,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("FRESH_BACKTEST_SUMMARY=" + json.dumps({
        "symbol_count": report["symbol_count"],
        "candidate_count": report["candidate_count"],
        "scenarios": {k: {m: v[m] for m in ("return_pct","max_drawdown_pct_realized","trades","win_rate_pct","profit_factor","expectancy_r")} for k, v in scenarios.items()},
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
