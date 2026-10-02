#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OKX_BASE = "https://www.okx.com"
HISTORY_PATH = "/api/v5/market/history-candles"
HISTORY_LIMIT = 300


class RateLimiter:
    def __init__(self, requests_per_second: float = 8.5) -> None:
        self.interval = 1.0 / max(float(requests_per_second), 0.1)
        self._lock = threading.Lock()
        self._next_at = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next_at:
                time.sleep(self._next_at - now)
                now = time.monotonic()
            self._next_at = max(now, self._next_at) + self.interval


RATE_LIMITER = RateLimiter()


def get_json(path: str, params: dict[str, Any] | None = None, attempts: int = 7) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        RATE_LIMITER.wait()
        try:
            response = requests.get(
                OKX_BASE + path,
                params=params,
                timeout=30,
                headers={"User-Agent": "autotraderbot-v2-expert-replay/1.0"},
            )
            if response.status_code == 429:
                time.sleep(1.0 + attempt * 0.8)
                continue
            response.raise_for_status()
            payload = response.json()
            if str(payload.get("code", "0")) != "0":
                raise RuntimeError(f"OKX code={payload.get('code')} msg={payload.get('msg')}")
            return payload
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            time.sleep(min(8.0, 0.6 + attempt * 0.9))
    raise RuntimeError(f"request failed path={path} params={params}: {last_error}")


def runtime_filters() -> dict[str, Any]:
    config_path = ROOT / "config" / "trading.json"
    if not config_path.exists():
        config_path = ROOT / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    trade = payload.get("trade_parameters", {})
    filters = payload.get("symbol_filters", {})
    return {
        "quote": str(trade.get("symbol_quote") or "USDT").upper(),
        "excluded_bases": {str(x).upper() for x in trade.get("symbol_exclude_bases", [])},
        "excluded_prefixes": tuple(str(x).upper() for x in trade.get("symbol_exclude_prefixes", [])),
        "min_volume_usd": float(filters.get("min_24h_volume") or 50_000_000.0),
        "max_spread_fraction": float(filters.get("max_spread_pct") or 0.05) / 100.0,
        "min_listing_days": float(filters.get("min_listing_days") or 90.0),
        "max_funding_abs": float(filters.get("max_funding_rate_abs") or 0.001),
    }


def discover_universe() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = runtime_filters()
    instruments = get_json("/api/v5/public/instruments", {"instType": "SWAP"}).get("data", [])
    tickers = get_json("/api/v5/market/tickers", {"instType": "SWAP"}).get("data", [])
    ticker_by_id = {str(item.get("instId") or ""): item for item in tickers}
    now_ms = int(time.time() * 1000)
    rejection: Counter[str] = Counter()
    prelim: list[dict[str, Any]] = []
    scanned = 0

    for item in instruments:
        inst_id = str(item.get("instId") or "")
        if str(item.get("state") or "").lower() != "live":
            continue
        if str(item.get("settleCcy") or "").upper() != cfg["quote"]:
            continue
        if str(item.get("ctType") or "").lower() != "linear":
            continue
        scanned += 1
        base = inst_id.removesuffix(f"-{cfg['quote']}-SWAP").upper()
        if base in cfg["excluded_bases"] or any(base.startswith(p) for p in cfg["excluded_prefixes"]):
            rejection["excluded_base"] += 1
            continue
        try:
            list_ms = int(item.get("listTime") or 0)
        except (TypeError, ValueError):
            list_ms = 0
        listing_days = (now_ms - list_ms) / 86_400_000.0 if list_ms > 0 else 0.0
        if listing_days < cfg["min_listing_days"]:
            rejection["listing_lt_min"] += 1
            continue

        ticker = ticker_by_id.get(inst_id)
        if not ticker:
            rejection["ticker_missing"] += 1
            continue
        try:
            bid = float(ticker.get("bidPx") or 0)
            ask = float(ticker.get("askPx") or 0)
            last = float(ticker.get("last") or 0)
            vol_ccy = float(ticker.get("volCcy24h") or 0)
        except (TypeError, ValueError):
            rejection["ticker_invalid"] += 1
            continue
        if min(bid, ask, last) <= 0:
            rejection["ticker_nonpositive"] += 1
            continue
        mid = (bid + ask) / 2.0
        spread = (ask - bid) / mid if mid > 0 else math.inf
        if spread > cfg["max_spread_fraction"]:
            rejection["spread_gt_max"] += 1
            continue
        approx_volume = vol_ccy * last
        if approx_volume < cfg["min_volume_usd"]:
            rejection["volume_lt_min"] += 1
            continue
        prelim.append({
            "instId": inst_id,
            "base": base,
            "listTime": list_ms,
            "listing_days": round(listing_days, 3),
            "spread_bps": round(spread * 10_000.0, 4),
            "approx_quote_volume_usd": round(approx_volume, 2),
        })

    eligible: list[dict[str, Any]] = []
    for row in prelim:
        try:
            funding_rows = get_json("/api/v5/public/funding-rate", {"instId": row["instId"]}).get("data", [])
            funding = float(funding_rows[0].get("fundingRate") or 0.0) if funding_rows else 0.0
        except (RuntimeError, TypeError, ValueError):
            rejection["funding_unavailable"] += 1
            continue
        if abs(funding) > cfg["max_funding_abs"]:
            rejection["funding_abs_gt_max"] += 1
            continue
        item = dict(row)
        item["current_funding_rate"] = funding
        eligible.append(item)

    eligible.sort(key=lambda x: float(x["approx_quote_volume_usd"]), reverse=True)
    meta = {
        "scanned_live_linear_usdt_swaps": scanned,
        "pre_funding_filter_count": len(prelim),
        "eligible_count": len(eligible),
        "rejection_reasons": dict(rejection),
        "filters": {
            "quote": cfg["quote"],
            "excluded_bases": sorted(cfg["excluded_bases"]),
            "excluded_prefixes": list(cfg["excluded_prefixes"]),
            "min_24h_approx_quote_volume_usd": cfg["min_volume_usd"],
            "max_spread_bps": cfg["max_spread_fraction"] * 10_000.0,
            "min_listing_days": cfg["min_listing_days"],
            "max_abs_current_funding_rate": cfg["max_funding_abs"],
        },
        "survivorship_bias": "Current-live universe only; historical delisted swaps are absent.",
        "volume_note": "volCcy24h is converted to approximate quote-USDT notional with current last price.",
    }
    return eligible, meta


def fetch_history_15m(inst_id: str, cutoff_ms: int, listing_ms: int) -> pd.DataFrame:
    effective_cutoff = max(int(cutoff_ms), int(listing_ms or 0))
    rows: dict[int, tuple[float, float, float, float, float]] = {}
    after: str | None = None
    stagnant = 0

    while True:
        params: dict[str, Any] = {"instId": inst_id, "bar": "15m", "limit": str(HISTORY_LIMIT)}
        if after is not None:
            params["after"] = after
        data = get_json(HISTORY_PATH, params).get("data", [])
        if not data:
            break
        batch_min: int | None = None
        for raw in data:
            if not isinstance(raw, list) or len(raw) < 6:
                continue
            try:
                ts = int(raw[0])
                if str(raw[-1]) != "1":
                    continue
                rows[ts] = tuple(float(raw[i]) for i in range(1, 6))
                batch_min = ts if batch_min is None else min(batch_min, ts)
            except (TypeError, ValueError):
                continue
        if batch_min is None or batch_min <= effective_cutoff:
            break
        cursor = str(batch_min)
        if cursor == after:
            stagnant += 1
            if stagnant >= 2:
                break
        else:
            stagnant = 0
        after = cursor

    prepared = [(pd.to_datetime(ts, unit="ms", utc=True), *values) for ts, values in rows.items() if ts >= effective_cutoff]
    if not prepared:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    return (
        pd.DataFrame(prepared, columns=["timestamp", "open", "high", "low", "close", "volume"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )


def resample(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    indexed = frame.set_index("timestamp").sort_index()
    out = indexed.resample(rule, label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    })
    return out.dropna(subset=["open", "high", "low", "close"]).reset_index()


def write_symbol(output_dir: Path, item: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    key = f"{item['base']}_USDT_USDT"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_dir / f"{key}_15m.parquet", index=False)
    resample(frame, "1h").to_parquet(output_dir / f"{key}_1h.parquet", index=False)
    resample(frame, "4h").to_parquet(output_dir / f"{key}_4h.parquet", index=False)
    return {
        "instId": item["instId"],
        "symbol": f"{item['base']}/USDT:USDT",
        "bars_15m": len(frame),
        "start": frame["timestamp"].min().isoformat(),
        "end": frame["timestamp"].max().isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, default=None)
    args = parser.parse_args()
    if args.days < 120:
        raise SystemExit("--days must be >=120")

    output_dir = args.output_dir.expanduser().resolve()
    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
    cutoff_ms = int(cutoff.timestamp() * 1000)
    started = time.monotonic()
    eligible, universe = discover_universe()
    print(json.dumps({"universe": universe}, ensure_ascii=False), flush=True)

    successes: list[dict[str, Any]] = []
    failures: dict[str, str] = {}

    def fetch_one(item: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
        return item, fetch_history_15m(str(item["instId"]), cutoff_ms, int(item.get("listTime") or 0))

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(fetch_one, item): item for item in eligible}
        for completed, future in enumerate(as_completed(futures), 1):
            item = futures[future]
            inst_id = str(item["instId"])
            try:
                resolved, frame = future.result()
                if len(frame) < 500:
                    failures[inst_id] = f"insufficient_bars:{len(frame)}"
                else:
                    successes.append(write_symbol(output_dir, resolved, frame))
            except Exception as exc:
                failures[inst_id] = f"{type(exc).__name__}:{exc}"
            print(f"[{completed}/{len(eligible)}] {inst_id} {'ok' if inst_id not in failures else failures[inst_id]}", flush=True)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "history_days_requested": args.days,
        "cutoff_utc": cutoff.isoformat(),
        "source": "OKX public REST /api/v5/market/history-candles",
        "source_bar": "15m",
        "history_limit_per_request": HISTORY_LIMIT,
        "derived_timeframes": ["1h", "4h"],
        "universe": universe,
        "eligible_instruments": eligible,
        "downloaded": sorted(successes, key=lambda x: x["symbol"]),
        "downloaded_count": len(successes),
        "failures": failures,
        "runtime_seconds": round(time.monotonic() - started, 3),
        "methodology_limits": [
            "Current-live universe creates survivorship bias because delisted historical swaps are absent.",
            "Current liquidity/spread/funding filters are applied now, not point-in-time historically.",
            "1h and 4h candles are deterministically resampled from completed 15m candles.",
        ],
    }
    metadata_path = args.metadata_output or (output_dir / "fetch_metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"ok": bool(successes), "downloaded_count": len(successes), "metadata": str(metadata_path)}, indent=2))
    return 0 if successes else 2


if __name__ == "__main__":
    raise SystemExit(main())
