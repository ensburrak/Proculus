# -*- coding: utf-8 -*-
"""
generate_ohlc_bulk.py (PROFESSIONAL ASYNC VERSION)
==================================================

Features:
- **AsyncIO & CCXT Pro:** Fetches data for multiple symbols in parallel (10x speedup).
- **Pagination (Unlimited History):** Loops 'since' timestamp to fetch 100,000+ candles.
- **Robust Rate Limiting:** Uses semaphores and smart retries to avoid 429 errors.
- **Atomic Saving:** Prevents file corruption during write.
- **Binance Fallback:** Falls back to Binance when OKX has no data for a symbol/timeframe.
- **Live Mode for Data:** Always fetches from live OKX (not testnet) for maximum data depth.
- **Partial Fetch:** If requested candles aren't available, fetches whatever is available.

Usage:
    python generate_ohlc_bulk.py --limit 5000
"""

from core.exceptions import BEST_EFFORT_EXCEPTIONS
import logging
import asyncio
import json
import pathlib
import sys
import argparse
from datetime import datetime, timezone
import time

# Try importing ccxt or ccxt.async_support
try:
    import ccxt.async_support as ccxt
except ImportError:
    try:
        import ccxt
    except ImportError:
        print("[ERR] CCXT libraries missing. Install with: pip install ccxt")
        sys.exit(1)

from settings import OKX_LIVE_API_KEY, OKX_LIVE_API_SECRET, OKX_LIVE_API_PASSPHRASE
from atomic_io import atomic_write_json
from data.ohlc_quality import apply_ohlc_quality_gate as _apply_ohlc_quality_gate
from runtime_paths import LOGS_DIR, METRICS_DIR, get_symbols_okx_path

# Paths
ROOT = pathlib.Path(__file__).resolve().parent
PROJECT_METRICS_DIR = ROOT / "metrics"
OHLC_FILE = METRICS_DIR / "ohlc_history.json"
BACKUP_DIR = METRICS_DIR / "backups"
SYMBOLS_FILE = ROOT / "symbols_okx.json"

# Logging setup to allow real-time user monitoring in workspace logs
LOG_FILE = LOGS_DIR / "generate_ohlc_bulk.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
if LOG_FILE.exists():
    try:
        LOG_FILE.unlink()
    except Exception:
        pass

_original_print = print
def print(*args, **kwargs):
    _original_print(*args, **kwargs)
    try:
        msg = " ".join(str(arg) for arg in args)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass

METRICS_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# Config
DEFAULT_TIMEFRAMES = ['5m', '15m', '1h', '4h']
MAX_CONCURRENCY = 8   # Highly optimized safe concurrency limit for Binance-primary fetching
BATCH_LIMIT = 100     # OKX max per request is typically 100 for candles


def _looks_like_okx_auth_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "api key doesn't exist" in text
        or '"code":"50119"' in text
        or "'code':'50119'" in text
        or "code\":\"50119" in text
    )

def load_symbols():
    try:
        symbols_file = get_symbols_okx_path()
    except BEST_EFFORT_EXCEPTIONS:
        symbols_file = SYMBOLS_FILE

    if symbols_file.exists():
        try:
            content = json.loads(symbols_file.read_text(encoding="utf-8"))
            if isinstance(content, dict) and "symbols" in content:
                return content["symbols"]
            elif isinstance(content, list):
                return content
        except BEST_EFFORT_EXCEPTIONS:
            pass
    # Fallback
    return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"]

def backup_file():
    if OHLC_FILE.exists():
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        bkp = BACKUP_DIR / f"ohlc_history_{ts}.json"
        try:
            payload = json.loads(OHLC_FILE.read_text(encoding="utf-8"))
            if atomic_write_json(bkp, payload):
                print(f"[B] Backup created: {bkp.name}")
        except BEST_EFFORT_EXCEPTIONS:
            pass


def normalize_symbol_for_storage(symbol: str | None) -> str:
    raw = str(symbol or "").upper()
    return "".join(ch for ch in raw if ch.isalnum())


def normalize_ohlc_rows(rows):
    normalized = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        item["symbol"] = normalize_symbol_for_storage(item.get("symbol"))
        item["timeframe"] = str(item.get("timeframe") or "").strip().lower()
        normalized.append(item)
    return normalized


def _parse_ts_for_sort(value):
    try:
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value or "")
        if not text:
            return 0.0
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).timestamp()
    except BEST_EFFORT_EXCEPTIONS:
        return 0.0


def dedupe_sort_ohlc_rows(rows):
    """Normalize, dedupe and sort OHLC rows by symbol/timeframe/timestamp."""
    keyed = {}
    for row in normalize_ohlc_rows(rows):
        symbol = str(row.get("symbol") or "")
        timeframe = str(row.get("timeframe") or "")
        ts = str(row.get("ts") or "")
        if not symbol or not timeframe or not ts:
            continue
        keyed[(symbol, timeframe, ts)] = row
    return sorted(
        keyed.values(),
        key=lambda item: (
            str(item.get("symbol") or ""),
            str(item.get("timeframe") or ""),
            _parse_ts_for_sort(item.get("ts")),
        ),
    )


def apply_ohlc_quality_gate(rows, *, requested_symbols, requested_timeframes):
    return _apply_ohlc_quality_gate(
        normalize_ohlc_rows(rows),
        requested_symbols=requested_symbols,
        requested_timeframes=requested_timeframes,
    )


def build_ohlc_coverage_report(
    rows,
    *,
    requested_symbols,
    requested_timeframes,
    min_rows_per_symbol_timeframe: int = 1,
):
    normalized_rows = normalize_ohlc_rows(rows)
    requested = [normalize_symbol_for_storage(symbol) for symbol in requested_symbols]
    timeframes = [str(tf).strip().lower() for tf in requested_timeframes]
    counts: dict[str, dict[str, int]] = {
        symbol: {tf: 0 for tf in timeframes}
        for symbol in requested
        if symbol
    }
    slash_symbol_rows = 0
    for original, row in zip(rows or [], normalized_rows):
        if isinstance(original, dict) and "/" in str(original.get("symbol", "")):
            slash_symbol_rows += 1
        symbol = str(row.get("symbol", ""))
        timeframe = str(row.get("timeframe", ""))
        if symbol in counts and timeframe in counts[symbol]:
            counts[symbol][timeframe] += 1

    missing_symbols = [
        symbol
        for symbol, tf_counts in counts.items()
        if all(count <= 0 for count in tf_counts.values())
    ]
    partial_symbols = [
        symbol
        for symbol, tf_counts in counts.items()
        if symbol not in missing_symbols
        and any(count < int(min_rows_per_symbol_timeframe) for count in tf_counts.values())
    ]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": len(normalized_rows),
        "symbol_count": len({row.get("symbol") for row in normalized_rows if row.get("symbol")}),
        "timeframes": sorted({str(row.get("timeframe")) for row in normalized_rows if row.get("timeframe")}),
        "requested_symbols": requested,
        "requested_timeframes": timeframes,
        "missing_symbols": missing_symbols,
        "partial_symbols": partial_symbols,
        "slash_symbol_rows": slash_symbol_rows,
        "counts": counts,
    }


def _read_existing_ohlc_payload():
    source_file = OHLC_FILE
    if not source_file.exists():
        legacy_file = PROJECT_METRICS_DIR / "ohlc_history.json"
        source_file = legacy_file if legacy_file.exists() else source_file
    if not source_file.exists():
        return None, []
    try:
        payload = json.loads(source_file.read_text(encoding="utf-8"))
        rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
        return payload, rows if isinstance(rows, list) else []
    except BEST_EFFORT_EXCEPTIONS:
        return None, []


def _backup_existing_payload(existing_payload):
    if existing_payload is None:
        return None
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = BACKUP_DIR / f"ohlc_history_{ts}.json"
    if atomic_write_json(backup_path, existing_payload):
        print(f"[B] Backup created: {backup_path.name}")
        return backup_path
    print("[WARN] Backup write failed after canonical save")
    return None


def persist_ohlc_results(results, *, force: bool, requested_symbols, requested_timeframes):
    normalized = dedupe_sort_ohlc_rows(results)
    clean_rows, quality_report = apply_ohlc_quality_gate(
        normalized,
        requested_symbols=requested_symbols,
        requested_timeframes=requested_timeframes,
    )
    coverage = build_ohlc_coverage_report(
        clean_rows,
        requested_symbols=requested_symbols,
        requested_timeframes=requested_timeframes,
    )
    coverage.update(quality_report)
    existing_payload, existing_rows = _read_existing_ohlc_payload()
    existing_count = len(existing_rows)

    if not clean_rows:
        print("[ERR] Quality gate rejected all fetched OHLC rows; canonical file will not be overwritten.")
        atomic_write_json(METRICS_DIR / "ohlc_coverage_report.json", coverage)
        return {"status": "quality_rejected", "coverage": coverage}

    if existing_count > len(clean_rows) * 2 and len(clean_rows) < 50000 and not force:
        print(f"[WARN] Mevcut dosya {existing_count} kayit iceriyor, yeni temiz veri sadece {len(clean_rows)}.")
        print("[WARN] Kucuk veriyle buyuk dosya ezilmeyecek. --force ile zorlayabilirsiniz.")
        partial_path = METRICS_DIR / "ohlc_history_partial.json"
        atomic_write_json(partial_path, {"rows": clean_rows, "coverage": coverage})
        atomic_write_json(METRICS_DIR / "ohlc_coverage_report.json", coverage)
        print(f"[OK] Partial data saved to {partial_path}")
        return {"status": "partial_saved", "path": str(partial_path), "coverage": coverage}

    if not atomic_write_json(OHLC_FILE, {"rows": clean_rows, "coverage": coverage}):
        print("[ERR] Save Error: atomic write failed")
        return {"status": "save_failed", "coverage": coverage}
    backup_path = _backup_existing_payload(existing_payload)
    atomic_write_json(METRICS_DIR / "ohlc_coverage_report.json", coverage)
    print(f"[OK] DONE. Saved to {OHLC_FILE}")
    print(f"[OK] Coverage report saved to {METRICS_DIR / 'ohlc_coverage_report.json'}")
    return {
        "status": "saved",
        "path": str(OHLC_FILE),
        "backup_path": str(backup_path) if backup_path else None,
        "coverage": coverage,
    }

async def fetch_symbol_history(exchange, symbol, timeframe, max_candles, *, auth_guard=None):
    """
    Pagination loop to fetch deep history for a single symbol.
    Returns whatever data is available, even if less than max_candles.
    """
    duration_seconds = exchange.parse_timeframe(timeframe)
    now = exchange.milliseconds()
    start_ts = now - (max_candles * duration_seconds * 1000)

    fetched = []
    current_since = start_ts

    retries = 4

    start_time = time.time()
    MAX_DURATION = 300  # 5 min per symbol/tf - conservative for rate-limited fetching

    while True:
        if auth_guard and auth_guard.get("auth_failed"):
            break

        # Global Timeout Guard
        if time.time() - start_time > MAX_DURATION:
            break

        try:
            if len(fetched) >= max_candles:
                break

            # Small jitter to avoid connection bursts
            await asyncio.sleep(0.02)

            # Dynamic batch limit depending on exchange capabilities
            batch_limit = 1000 if 'binance' in getattr(exchange, 'id', '').lower() else 100

            # Using 'limit' is good, but safely wrap with wait_for
            candles = await asyncio.wait_for(
                exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=batch_limit),
                timeout=15.0
            )

            if not candles:
                break

            fetched.extend(candles)

            # Update 'since' to the last timestamp + 1ms
            last_ts = candles[-1][0]
            if last_ts == current_since:
                # Stuck loop check
                break

            # Prevent infinite loop if timestamp isn't moving
            if last_ts <= current_since:
                 current_since += 1
            else:
                 current_since = last_ts + 1

            # If we reached current time, stop
            if last_ts >= now - duration_seconds*1000:
                break

            # Rate limit sleep between pagination requests
            await asyncio.sleep(0.05)

        except ccxt.RateLimitExceeded:
            if 'okx' in getattr(exchange, 'id', '').lower():
                print(f"[...] Rate limit hit on OKX for {symbol}, switching directly to fallback provider (Binance) to save time...")
                break
            wait = min(30.0, 2.0 ** max(0, 4 - retries))
            print(f"[...] Rate limit on {symbol}, backing off {wait:.1f}s...")
            retries -= 1
            if retries <= 0:
                break
            await asyncio.sleep(wait)
            continue
        except asyncio.TimeoutError:
             break
        except BEST_EFFORT_EXCEPTIONS as e:
            if auth_guard is not None and _looks_like_okx_auth_error(e):
                auth_guard["auth_failed"] = True
                auth_guard["reason"] = str(e)
                if not auth_guard.get("logged"):
                    auth_guard["logged"] = True
                    print(
                        "[WARN] OKX LIVE auth rejected (50119). "
                        "Remaining history fetches will skip OKX and use Binance fallback."
                    )
                break
            retries -= 1
            if str(e) and ("does not exist" in str(e).lower() or "does not have market symbol" in str(e).lower()):
                 break
            if retries <= 0:
                print(f"[ERR] Error fetching {symbol} {timeframe}: {e}")
                break
            await asyncio.sleep(1)

    # Convert to list of dicts
    # CCXT Structure: [timestamp, open, high, low, close, volume]
    data = []
    provider = str(getattr(exchange, "id", "") or "").lower()
    for c in fetched:
        data.append({
            "symbol": symbol.replace("/", ""),
            "timeframe": str(timeframe).lower(),
            "ts": datetime.fromtimestamp(c[0]/1000, tz=timezone.utc).isoformat(),
            "open": c[1],
            "high": c[2],
            "low": c[3],
            "close": c[4],
            "volume": c[5],
            "source": provider,
            "provider": provider,
        })

    # Normalize length
    if len(data) > max_candles:
        data = data[-max_candles:]

    return data

async def worker(sem, okx_exchange, binance_exchange, symbol, timeframes, max_candles, results, okx_auth_guard):
    """Fetch data for a symbol across all timeframes, with OKX fallback.

    Logic:
    1. Fetch from Binance (primary).
    2. Fallback: If Binance fails or has fewer candles, fetch from OKX (fallback).
    3. [E2] If both have data → merge (median price, sum volume).
    4. Fallback: keep whichever exchange returned MORE data.
    5. If neither has data → log error.
    """
    async with sem:
        print(f"[~] Fetching {symbol}...")
        for tf in timeframes:
            binance_data = None
            binance_count = 0
            binance_failed = False
            try:
                binance_data = await fetch_symbol_history(binance_exchange, symbol, tf, max_candles)
                binance_count = len(binance_data) if binance_data else 0
            except (ccxt.RateLimitExceeded, ccxt.NetworkError, asyncio.TimeoutError) as e:
                binance_failed = True
                print(f"  [WARN] {symbol} {tf}: Binance unavailable/rate-limited, switching provider: {e}")
            except BEST_EFFORT_EXCEPTIONS as e:
                binance_failed = True
                print(f"  [WARN] {symbol} {tf}: Binance failed, switching provider: {e}")

            okx_data = None
            okx_count = 0
            if binance_failed or binance_count < int(max_candles * 0.8):
                try:
                    okx_data = await fetch_symbol_history(
                        okx_exchange, symbol, tf, max_candles, auth_guard=okx_auth_guard
                    )
                    okx_count = len(okx_data) if okx_data else 0
                except (ccxt.RateLimitExceeded, ccxt.AuthenticationError, ccxt.NetworkError, asyncio.TimeoutError) as e:
                    print(f"  [WARN] {symbol} {tf}: OKX unavailable/rate-limited, switching provider unavailable: {e}")
                except BEST_EFFORT_EXCEPTIONS as e:
                    print(f"  [WARN] {symbol} {tf}: OKX failed: {e}")

            # --- Step 3: [E2] Merge if both have data ---
            if okx_count > 0 and binance_count > 0:
                try:
                    from data.multi_exchange import merge_ohlcv_multi_exchange
                    # Convert dicts to OHLCV list format for merge
                    okx_raw = [[d["ts"], d["open"], d["high"], d["low"], d["close"], d["volume"]] for d in okx_data]
                    bin_raw = [[d["ts"], d["open"], d["high"], d["low"], d["close"], d["volume"]] for d in binance_data]
                    merged_raw = merge_ohlcv_multi_exchange({"okx": okx_raw, "binance": bin_raw})
                    if merged_raw:
                        best_data = []
                        for row in merged_raw:
                            best_data.append({
                                "symbol": normalize_symbol_for_storage(symbol), "timeframe": str(tf).lower(),
                                "ts": row[0], "open": row[1], "high": row[2],
                                "low": row[3], "close": row[4], "volume": row[5],
                                "source": "merged",
                                "provider": "okx+binance",
                            })
                        source = "MERGED(OKX+Binance)"
                    else:
                        best_data = binance_data
                        source = "Binance"
                except BEST_EFFORT_EXCEPTIONS as e:
                    print(f"  [WARN] {symbol} {tf}: E2 merge failed, fallback: {e}")
                    best_data = binance_data if binance_count >= okx_count else okx_data
                    source = "Binance" if binance_count >= okx_count else "OKX"
            elif binance_count > 0:
                best_data = binance_data
                source = "Binance"
            elif okx_count > 0:
                best_data = okx_data
                source = "OKX"
            else:
                print(f"  [ERR] {symbol} {tf}: No data from OKX or Binance")
                continue

            tag = f"(partial, {len(best_data)})" if len(best_data) < max_candles else f"{len(best_data)}"
            provider_tag = str(source).lower().replace("(", "").replace(")", "")
            for row in best_data or []:
                if isinstance(row, dict):
                    row.setdefault("source", provider_tag)
                    row.setdefault("provider", provider_tag)
                    row["timeframe"] = str(row.get("timeframe") or tf).lower()
            results.extend(best_data)
            print(f"  [OK] {symbol} {tf}: {tag} candles [{source}]")

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=15000, help="Max candles per timeframe (default: 15000)")
    parser.add_argument("--force", action="store_true", help="Force overwrite even if existing data is larger")
    args = parser.parse_args()

    symbols = load_symbols()
    print(f"[*] Starting Bulk Async Fetch for {len(symbols)} symbols. Limit: {args.limit}")

    # ── OKX Exchange (ALWAYS LIVE for data fetching) ──
    okx_config = {
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'}
    }
    # Live API keyler varsa ekle (yoksa public endpoint ile çalışır)
    if OKX_LIVE_API_KEY:
        okx_config['apiKey'] = OKX_LIVE_API_KEY
        okx_config['secret'] = OKX_LIVE_API_SECRET
        okx_config['password'] = OKX_LIVE_API_PASSPHRASE
        print("[>] Mode: LIVE + AUTH (Canlı OKX API keyleri ile veri çekiliyor)")
    else:
        print("[>] Mode: LIVE PUBLIC (API key olmadan public veri çekiliyor)")
    okx_exchange = ccxt.okx(okx_config)
    # NOTE: Sandbox/testnet mode is intentionally NOT set here.
    # We always use live OKX for data fetching to get maximum historical depth.

    # ── Binance Exchange (Fallback, no API key needed for public data) ──
    binance_exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}  # Binance futures for swap/margin symbols
    })
    print("[>] Binance fallback: READY")

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    results = []
    tasks = []
    okx_auth_guard = {"auth_failed": False, "reason": "", "logged": False}

    for sym in symbols:
        tasks.append(
            worker(
                semaphore,
                okx_exchange,
                binance_exchange,
                sym,
                DEFAULT_TIMEFRAMES,
                args.limit,
                results,
                okx_auth_guard,
            )
        )

    try:
        await asyncio.gather(*tasks)
    finally:
        # Proper cleanup to avoid Unclosed Connector warning
        await okx_exchange.close()
        await binance_exchange.close()

    print(f"\n[S] Saving {len(results)} total records...")

    persist_ohlc_results(
        results,
        force=bool(args.force),
        requested_symbols=symbols,
        requested_timeframes=DEFAULT_TIMEFRAMES,
    )
    print("[>] Next: Run 'python -m ml.build_dataset' to create training data")

if __name__ == "__main__":
    try:
        # On Windows, SelectorEventLoop must be used in non-interactive background environments to prevent startup hangs.
        # We enforce SelectorEventLoop but keep concurrency low to avoid the 64-handle limit.
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[ERR] Interrupted.")
