"""
onchain_data_updater.py
-----------------------

This module generates pseudo "on-chain" metrics using only free
exchange data rather than paid on-chain providers.  The purpose of
these metrics is to derive sentiment and flow information from market
data such as funding rates, open interest trends and taker volume
imbalances.  These indicators can serve as proxies for bullish or
bearish on-chain conditions without relying on third‑party services
that require API keys or subscriptions.

The script defines a single public function ``update_onchain_metrics``
which iterates over a list of symbols, queries the exchange for
funding rates, open interest and taker imbalances (placeholders are
provided for integration with your existing client), normalises the
results to the range ``[-1, 1]`` and writes them to
``data/onchain_metrics.json``.  The resulting JSON file is consumed by
``onchain_analytics.py`` to calculate on‑chain sentiment scores.

Environment variables:

* ``EXCHANGE`` – Optional identifier for the exchange to query (e.g.
  ``OKX`` or ``BINANCE``).  It does not affect logic directly but
  allows conditional behaviour if needed.
* ``EXCHANGE_API`` – Optional base URL or API identifier for a custom
  exchange client.  Not currently used but reserved for future use.

Usage as a script:

    python -m onchain_data_updater

This will compute the metrics for the default symbol list and report
how many assets were processed.
"""

from __future__ import annotations


from core.exceptions import BEST_EFFORT_EXCEPTIONS
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from atomic_io import atomic_write_json
from typing import Dict, List

# Load environment from a .env file if available.  Use find_dotenv to
# locate the nearest .env file up the directory tree.  If dotenv is not
# installed, silently continue.
try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore
    load_dotenv(find_dotenv())
except BEST_EFFORT_EXCEPTIONS:
    pass

import requests  # Added for HTTP requests to external APIs (e.g. Etherscan)

try:
    from retry_utils import retry  # type: ignore
except BEST_EFFORT_EXCEPTIONS:  # pragma: no cover
    def retry(*args, **kwargs):  # type: ignore
        def decorator(func): return func
        return decorator

try:
    from cache_manager import file_cache  # type: ignore
except BEST_EFFORT_EXCEPTIONS:  # pragma: no cover
    def file_cache(*args, **kwargs):  # type: ignore
        def decorator(func): return func
        return decorator

logger = logging.getLogger(__name__)

# Exchange configuration – currently unused but kept for future
EXCHANGE = os.getenv("EXCHANGE", "OKX").upper()

# Default list of symbols to compute metrics for.  These should match
# the symbols your trading bot uses.  You can override this list by
# passing a ``symbols`` argument to ``update_onchain_metrics``.
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT"]


def _symbol_to_base(sym: str) -> str:
    """Convert a symbol like ``BTC/USDT`` to its base coin (``BTC``)."""
    return sym.split("/")[0].upper()


def _normalize_metric(value: float, scale: float) -> float:
    """Normalise a raw metric to the range [-1, 1].

    Parameters
    ----------
    value : float
        The raw metric value.
    scale : float
        A scale factor representing the expected maximum absolute
        magnitude of the value.  Values with absolute magnitude
        greater than ``scale`` saturate at +/-1.

    Returns
    -------
    float
        A number in the range ``[-1, 1]`` proportional to
        ``value/scale``.
    """
    if scale <= 0:
        scale = 1.0
    x = value / scale
    if x < -1.0:
        return -1.0
    if x > 1.0:
        return 1.0
    return x


def _fetch_metrics_for_symbol(symbol: str) -> Dict[str, float]:
    """Compute pseudo or real on‑chain metrics for a single symbol.

    This function can be customised to integrate real on‑chain data from
    external services.  By default it returns zero values for all
    metrics.  When the symbol is ETH and an Etherscan API key is
    available (see ``ETHERSCAN_API_KEY`` in .env), the current gas
    price is fetched from Etherscan's ``gasoracle`` endpoint and
    converted into a simple sentiment metric.  For other symbols you
    can integrate funding rate, open interest and taker ratio data
    using your exchange client or additional APIs.

    Metrics computed:

    * ``funding_bias`` – Derived from the funding rate of perpetual
      futures.  Normalised using a scale of 0.0005 (0.05% funding).
    * ``oi_trend`` – Change in open interest over a lookback window.
      Normalised using a scale of 0.5 (50%).
    * ``taker_imbalance`` – Taker buy/sell volume imbalance.
      Normalised using a scale of 1.0 (100% imbalance).
    * ``eth_gas_sentiment`` – For ETH, a sentiment score in [-1, 1]
      derived from the current gas price.  Lower gas implies higher
      bullish sentiment (cheaper transactions), while high gas costs
      are mapped to bearish sentiment.  Absent data yields 0.

    Returns
    -------
    dict
        A dictionary keyed by metric name with values in the range
        ``[-1, 1]``.  If the underlying data is unavailable or an
        error occurs, metrics default to zero.
    """
    # Initialise metrics with neutral defaults.  These values will be
    # updated using real exchange data where possible.
    funding_rate: float = 0.0
    oi_change: float = 0.0
    taker_ratio: float = 1.0  # Neutral (no bias) when no data is available

    # Normalise the symbol to the format expected by Binance futures API (e.g. BTCUSDT).
    # Symbols may be provided as "BTC/USDT" or "BTCUSDT".  Remove separators and
    # convert to uppercase.
    symbol_normalized = symbol.replace("/", "").upper()
    # Allow overriding the API base URL via environment; fallback to Binance
    base_url = os.getenv("BINANCE_FAPI_BASE_URL", "https://fapi.binance.com")
    # Attempt to fetch the latest funding rate
    try:
        fr_resp = requests.get(
            f"{base_url}/fapi/v1/fundingRate",
            params={"symbol": symbol_normalized, "limit": 1},
            timeout=5,
        )
        fr_resp.raise_for_status()
        fr_data = fr_resp.json()
        # Binance returns a list of funding records; use the most recent
        if isinstance(fr_data, list) and fr_data:
            last = fr_data[-1]
            # fundingRate field contains the rate per eight hours as string
            rate_str = last.get("fundingRate")
            if rate_str is not None:
                funding_rate = float(rate_str)
    except BEST_EFFORT_EXCEPTIONS as exc:
        logger.debug(f"Funding rate fetch failed for {symbol}: {exc}")

    # Attempt to compute open interest change over a short window
    try:
        # Current open interest
        oi_resp = requests.get(
            f"{base_url}/fapi/v1/openInterest",
            params={"symbol": symbol_normalized},
            timeout=5,
        )
        oi_resp.raise_for_status()
        oi_data = oi_resp.json()
        current_oi = float(oi_data.get("openInterest") or 0.0)
        # Historical open interest (two points) to estimate change
        hist_resp = requests.get(
            f"{base_url}/futures/data/openInterestHist",
            params={"symbol": symbol_normalized, "period": "5m", "limit": 2},
            timeout=5,
        )
        hist_resp.raise_for_status()
        hist_data = hist_resp.json()
        if isinstance(hist_data, list) and len(hist_data) >= 2:
            try:
                # Use sumOpenInterest if available; fallback to openInterest
                prev_entry = hist_data[-2]
                last_entry = hist_data[-1]
                oi0 = float(prev_entry.get("sumOpenInterest") or prev_entry.get("openInterest") or 0.0)
                oi1 = float(last_entry.get("sumOpenInterest") or last_entry.get("openInterest") or 0.0)
                if oi0 > 0.0:
                    oi_change = (oi1 - oi0) / oi0
                else:
                    oi_change = 0.0
            except BEST_EFFORT_EXCEPTIONS:
                oi_change = 0.0
        else:
            # If insufficient history, treat change as zero
            oi_change = 0.0
    except BEST_EFFORT_EXCEPTIONS as exc:
        logger.debug(f"Open interest fetch failed for {symbol}: {exc}")

    # Attempt to fetch taker long/short ratio to derive a taker imbalance
    try:
        ratio_resp = requests.get(
            f"{base_url}/futures/data/takerlongshortRatio",
            params={"symbol": symbol_normalized, "period": "5m", "limit": 1},
            timeout=5,
        )
        ratio_resp.raise_for_status()
        ratio_data = ratio_resp.json()
        if isinstance(ratio_data, list) and ratio_data:
            entry = ratio_data[-1]
            r = entry.get("longShortRatio")
            if r is not None:
                taker_ratio = float(r)
    except BEST_EFFORT_EXCEPTIONS as exc:
        logger.debug(f"Taker long/short ratio fetch failed for {symbol}: {exc}")

    # Compute taker imbalance from the ratio.  A ratio of 1.0 means neutral (no bias).
    if taker_ratio <= 0:
        taker_imbalance = 0.0
    else:
        taker_imbalance = taker_ratio - 1.0
        # Clamp to [-1, 1]
        if taker_imbalance < -1.0:
            taker_imbalance = -1.0
        if taker_imbalance > 1.0:
            taker_imbalance = 1.0

    # [B4] Fetch global long/short account ratio (Binance Futures — free, no key)
    long_short_ratio: float = 1.0  # neutral
    try:
        ls_resp = requests.get(
            f"{base_url}/futures/data/globalLongShortAccountRatio",
            params={"symbol": symbol_normalized, "period": "5m", "limit": 1},
            timeout=5,
        )
        ls_resp.raise_for_status()
        ls_data = ls_resp.json()
        if isinstance(ls_data, list) and ls_data:
            _ls = ls_data[-1].get("longShortRatio")
            if _ls is not None:
                long_short_ratio = float(_ls)
    except BEST_EFFORT_EXCEPTIONS as exc:
        logger.debug(f"Long/short account ratio fetch failed for {symbol}: {exc}")

    # [B4] Fetch top trader long/short ratio (whale positioning proxy)
    top_trader_ratio: float = 1.0  # neutral
    try:
        tt_resp = requests.get(
            f"{base_url}/futures/data/topLongShortAccountRatio",
            params={"symbol": symbol_normalized, "period": "5m", "limit": 1},
            timeout=5,
        )
        tt_resp.raise_for_status()
        tt_data = tt_resp.json()
        if isinstance(tt_data, list) and tt_data:
            _tt = tt_data[-1].get("longShortRatio")
            if _tt is not None:
                top_trader_ratio = float(_tt)
    except BEST_EFFORT_EXCEPTIONS as exc:
        logger.debug(f"Top trader ratio fetch failed for {symbol}: {exc}")

    # Compute long/short imbalance: >1 = more longs (bullish), <1 = more shorts (bearish)
    ls_imbalance = max(-1.0, min(1.0, long_short_ratio - 1.0))
    tt_imbalance = max(-1.0, min(1.0, top_trader_ratio - 1.0))

    metrics: Dict[str, float] = {
        "funding_bias": _normalize_metric(funding_rate, scale=0.0005),
        "oi_trend": _normalize_metric(oi_change, scale=0.5),
        "taker_imbalance": taker_imbalance,
        "long_short_ratio": round(long_short_ratio, 4),
        "long_short_imbalance": round(ls_imbalance, 4),
        "top_trader_ratio": round(top_trader_ratio, 4),
        "top_trader_imbalance": round(tt_imbalance, 4),
    }

    # Integrate Etherscan gas price for ETH if key is present
    base = _symbol_to_base(symbol)
    if base == "ETH":
        api_key = os.environ.get("ETHERSCAN_API_KEY")
        if api_key:
            try:
                url = "https://api.etherscan.io/api"
                params = {
                    "module": "gastracker",
                    "action": "gasoracle",
                    "apikey": api_key,
                }
                # Simple retry wrapper; no external dependency to avoid
                # coupling with retry_utils here
                resp = requests.get(url, params=params, timeout=5)
                resp.raise_for_status()
                # Etherscan sometimes returns a bare string on errors.  Safely
                # parse the JSON and verify expected types before indexing.
                gas_price_gwei = None
                try:
                    raw_json = resp.json()
                except BEST_EFFORT_EXCEPTIONS:
                    raw_json = None
                # Only proceed if the JSON is a dict
                if isinstance(raw_json, dict):
                    result = raw_json.get("result")
                    if isinstance(result, dict):
                        gas_str = result.get("ProposeGasPrice") or result.get("SafeGasPrice") or result.get("FastGasPrice")
                        try:
                            gas_price_gwei = float(gas_str)
                        except BEST_EFFORT_EXCEPTIONS:
                            gas_price_gwei = None
                if gas_price_gwei is not None:
                    # Map gas price to sentiment: assume typical gas range 10–100 Gwei
                    # Map low gas (<=20) to +1 (bullish), high gas (>=100) to -1 (bearish)
                    low, high = 20.0, 100.0
                    if gas_price_gwei <= low:
                        gas_sent = 1.0
                    elif gas_price_gwei >= high:
                        gas_sent = -1.0
                    else:
                        # Linear interpolation between low (+1) and high (-1)
                        gas_sent = 1.0 - 2.0 * ((gas_price_gwei - low) / (high - low))
                    metrics["eth_gas_price_gwei"] = gas_price_gwei
                    metrics["eth_gas_sentiment"] = gas_sent
                else:
                    # If parsing fails, fall back to default values
                    metrics["eth_gas_price_gwei"] = 0.0
                    metrics["eth_gas_sentiment"] = 0.0
            except BEST_EFFORT_EXCEPTIONS as e:
                logger.warning(f"Failed to fetch ETH gas data: {e}")
                metrics["eth_gas_price_gwei"] = 0.0
                metrics["eth_gas_sentiment"] = 0.0
        else:
            # If no API key, skip gas metric
            metrics["eth_gas_price_gwei"] = 0.0
            metrics["eth_gas_sentiment"] = 0.0

    return metrics


# ---------------------------------------------------------------------------
# DeFiLlama integration

def _fetch_defillama_global_tvl(timeout: float = 10.0) -> float:
    """
    Fetch the latest total value locked (TVL) across all chains from
    DeFiLlama's public API.

    DeFiLlama offers a free endpoint ``https://api.llama.fi/v2/historicalChainTvl``
    that returns a list of historical TVL records.  The final element in
    the list corresponds to the most recent TVL (in USD).  This helper
    attempts to retrieve that value and convert it to a float.  It
    returns ``0.0`` if the request fails or the response does not match
    the expected structure.

    Parameters
    ----------
    timeout : float, optional
        Maximum time to wait for the HTTP response, in seconds.  Defaults
        to 10 seconds.

    Returns
    -------
    float
        The latest global DeFi TVL in USD, or ``0.0`` on error.
    """
    url = "https://api.llama.fi/v2/historicalChainTvl"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data:
            last = data[-1]
            tvl = last.get("tvl")
            try:
                return float(tvl)
            except BEST_EFFORT_EXCEPTIONS:
                return 0.0
        return 0.0
    except BEST_EFFORT_EXCEPTIONS as exc:
        logger.debug("Failed to fetch DeFiLlama TVL: %s", exc)
        return 0.0


@retry()
@file_cache("onchain_metrics_cache.json", ttl=300)
def update_onchain_metrics(
    symbols: List[str] | None = None,
    interval: str = "1h",
) -> Dict[str, object]:
    """Fetch pseudo on‑chain metrics for multiple symbols and write them to disk.

    Parameters
    ----------
    symbols : list of str, optional
        List of symbols to process.  Defaults to ``DEFAULT_SYMBOLS``.
    interval : str, optional
        Metadata describing the time granularity of the metrics.

    Returns
    -------
    dict
        A dictionary with keys ``asset``, ``interval``, ``metrics`` and
        ``updated_at``.  The ``metrics`` field contains a mapping of
        base symbols (e.g. ``BTC``) to metric dictionaries.
    """
    try:
        cfg_path = Path("config.json")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        if not cfg.get("enable_pseudo_onchain", False):
            logger.info("Pseudo on-chain metrics disabled via config (enable_pseudo_onchain=False).")
            return {"metrics": {}}
    except BEST_EFFORT_EXCEPTIONS:
        pass

    symbols = symbols or DEFAULT_SYMBOLS
    results: Dict[str, Dict[str, float]] = {}
    for sym in symbols:
        base = _symbol_to_base(sym)
        try:
            metrics = _fetch_metrics_for_symbol(sym)
            results[base] = metrics
        except BEST_EFFORT_EXCEPTIONS as e:
            logger.warning(f"Failed to compute pseudo on‑chain metrics for {sym}: {e}")
    if not results:
        return {}
    # Incorporate global DeFi TVL sentiment into each symbol's metrics.  We
    # retrieve the latest TVL from DeFiLlama and normalise it to [-1, 1].
    try:
        tvl_usd = _fetch_defillama_global_tvl()
        # [FIX] Statik $100B yerine Rate-of-Change hesapla.
        # Son 30 TVL ölçümünün ortalamasını referans al.
        import json as _json
        import pathlib
        tvl_cache_file = pathlib.Path(__file__).resolve().parent / "metrics" / "tvl_history.json"
        tvl_history = []
        try:
            if tvl_cache_file.exists():
                tvl_history = _json.loads(tvl_cache_file.read_text(encoding="utf-8"))
        except BEST_EFFORT_EXCEPTIONS:
            tvl_history = []
        
        # Yeni TVL'yi geçmişe ekle
        tvl_history.append(tvl_usd)
        # Son 30 ölçümü tut
        tvl_history = tvl_history[-30:]
        try:
            tvl_cache_file.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(tvl_cache_file, tvl_history)
        except BEST_EFFORT_EXCEPTIONS:
            pass
        
        # Rate of Change: Mevcut TVL vs son 30 ölçüm ortalaması
        if len(tvl_history) >= 5:
            tvl_ma = sum(tvl_history) / len(tvl_history)
            diff = (tvl_usd - tvl_ma) / (tvl_ma + 1e-9)
            # Clamp [-1, 1]
            defi_sent = max(-1.0, min(1.0, diff * 5.0))  # 5x amplify for sensitivity
        else:
            # Yeterli geçmiş yok → nötr
            defi_sent = 0.0
    except BEST_EFFORT_EXCEPTIONS as tvl_exc:
        logger.debug("DeFiLlama TVL sentiment computation failed: %s", tvl_exc)
        defi_sent = 0.0
    # Append to all results
    for b in results:
        results[b]["defi_tvl_sentiment"] = float(defi_sent)
    out = {
        "asset": "MIXED",
        "interval": interval,
        "metrics": results,
        "updated_at": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
    }
    out_path = Path("data/onchain_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        logger.info(f"On‑chain pseudo metrics written to {out_path} ({len(results)} assets).")
    except BEST_EFFORT_EXCEPTIONS as e:
        logger.warning(f"Failed to write on‑chain metrics: {e}")
    return out

import asyncio
async def update_onchain_metrics_async(*args, **kwargs) -> Dict[str, object]:
    """Async wrapper that runs the synchronous fetching in a separate thread to prevent Event Loop blocking."""
    return await asyncio.to_thread(update_onchain_metrics, *args, **kwargs)

if __name__ == "__main__":  # pragma: no cover
    data = update_onchain_metrics()
    print(f"Updated pseudo on‑chain metrics for {len(data.get('metrics', {}))} assets.")
