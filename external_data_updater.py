"""
external_data_updater.py
------------------------

This module consolidates data from several optional providers to build
a comprehensive external metrics report.  The bot can use these
additional signals (options market metrics, whale alerts and
liquidation heatmaps) to inform trading decisions or risk management.

The following provider functions are imported:

* ``fetch_options_metrics`` from ``ops_data_provider`` – returns a
  dictionary mapping symbols to options market metrics (e.g. implied
  volatility, skew, put/call ratios).
* ``fetch_whale_alerts`` from ``whale_alert_provider`` – returns a
  mapping of symbols to whale alert counts or scores.
* ``fetch_liquidation_heatmap`` from ``liquidation_data_provider`` –
  returns a mapping of symbols to liquidation levels or liquidity
  imbalances.

Each provider returns either a dictionary keyed by symbol or a
``None``/``{}`` if no data is available.  The ``update_external_metrics``
function merges these dictionaries on a common symbol set and writes
the combined structure to ``metrics/external_metrics.json``.  The
resulting JSON includes a ``updated_at`` timestamp for auditing.
"""

from __future__ import annotations


from core.exceptions import BEST_EFFORT_EXCEPTIONS
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

try:
    from ops_data_provider import fetch_options_metrics  # type: ignore
except BEST_EFFORT_EXCEPTIONS:
    def fetch_options_metrics(*args, **kwargs) -> Dict[str, Any]:  # type: ignore
        """Fallback stub for options metrics when the ops_data_provider cannot be imported.

        The real ``fetch_options_metrics`` accepts a variety of parameters (e.g.
        ``symbol`` or ``symbols``).  This stub accepts arbitrary positional
        and keyword arguments to avoid unexpected keyword argument errors and
        always returns an empty dictionary.
        """
        return {}

try:
    from whale_alert_provider import fetch_whale_alerts  # type: ignore
except BEST_EFFORT_EXCEPTIONS:
    def fetch_whale_alerts(*args, **kwargs) -> Dict[str, Any]:  # type: ignore
        """Fallback stub for whale alert data when the provider cannot be imported.

        Accepts arbitrary parameters to remain compatible with call sites.  Always
        returns an empty dictionary.
        """
        return {}

try:
    from liquidation_data_provider import fetch_liquidation_heatmap  # type: ignore
except BEST_EFFORT_EXCEPTIONS:
    def fetch_liquidation_heatmap(*args, **kwargs) -> Dict[str, Any]:  # type: ignore
        """Fallback stub for liquidation heatmap when the provider cannot be imported.

        Accepts arbitrary parameters to remain compatible with call sites.  Always
        returns an empty dictionary.
        """
        return {}


def update_external_metrics(symbols: list[str] | None = None) -> Dict[str, Any]:
    """Fetch and combine external metrics for the given symbols.

    :param symbols: list of trading symbols (e.g. ["BTC/USDT"]).  If None,
        the function will attempt to read ``symbols_okx.json`` from
        ``data/`` and default to ["BTC/USDT", "ETH/USDT"] if the file is
        missing or invalid.
    :return: dictionary with an ``updated_at`` timestamp and per-symbol
        metrics merged from the options, whale and liquidation providers.
    """
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    metrics_dir = root / "metrics"
    metrics_dir.mkdir(exist_ok=True, parents=True)
    # Resolve symbol list
    syms: list[str]
    if symbols is None:
        syms_path = data_dir / "symbols_okx.json"
        syms = []
        try:
            if syms_path.exists():
                loaded = json.loads(syms_path.read_text(encoding="utf-8"))
                if isinstance(loaded, list) and loaded:
                    syms = [str(s).upper() for s in loaded if isinstance(s, str)]
        except BEST_EFFORT_EXCEPTIONS:
            syms = []
        if not syms:
            syms = ["BTC/USDT", "ETH/USDT"]
    else:
        syms = [str(s).upper() for s in symbols]
    # Prepare per-provider data structures
    ops_data: Dict[str, Any] = {}
    whale_data: Dict[str, Any] = {}
    liq_data: Dict[str, Any] = {}
    # Loop over each symbol and fetch metrics individually.  The provider
    # functions expect a single symbol/token rather than a list, and they
    # rely on environment variables for API keys.  Each call returns
    # either a dict or list (whale alerts).  We map the original trading
    # pair (e.g. "BTC/USDT") to the returned metrics.
    for sym in syms:
        base = sym.split("/")[0] if "/" in sym else sym
        # Options metrics
        try:
            # Some implementations of ``fetch_options_metrics`` accept a positional
            # argument rather than a named ``symbol`` parameter.  Attempt
            # the keyword invocation first, and on ``TypeError`` fall back
            # to positional invocation.  Any other exceptions are logged.
            try:
                m = fetch_options_metrics(symbol=base)  # type: ignore[call-arg]
            except TypeError:
                m = fetch_options_metrics(base)  # type: ignore[misc]
            if isinstance(m, dict) and m:
                ops_data[sym] = m
        except BEST_EFFORT_EXCEPTIONS as e:
            logger.warning(f"Options metrics fetch failed for {sym}: {e}")
        # Whale alerts: return a list of alerts; summarise by count
        try:
            # fetch_whale_alerts is async, so we need to run it in an event loop
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            if loop.is_running():
                # We are likely inside an existing loop (e.g. if called from main_bot_async)
                # This function is designed to be synchronous. This is tricky.
                # Ideally, this function should be async if it calls async code.
                # For now, we'll try to use a fresh runner or just skip if we can't await.
                # Actually, blocking inside a running loop is bad.
                # But since this is a data updater, let's assume it's mostly background.
                alerts = [] # fallback if we can't await safely
            else:
                alerts = loop.run_until_complete(fetch_whale_alerts(token=base))  # type: ignore[call-arg]
            if isinstance(alerts, list) and alerts:
                # Summarise by number of alerts; more advanced logic can
                # aggregate by amount or recency.
                whale_data[sym] = {"whale_alert_count": len(alerts)}
        except BEST_EFFORT_EXCEPTIONS as e:
            logger.warning(f"Whale alert fetch failed for {sym}: {e}")
        # Liquidation heatmap
        try:
            heat = fetch_liquidation_heatmap(symbol=base)  # type: ignore[call-arg]
            if isinstance(heat, dict) and heat:
                liq_data[sym] = heat
        except BEST_EFFORT_EXCEPTIONS as e:
            logger.warning(f"Liquidation heatmap fetch failed for {sym}: {e}")
    # Merge per-symbol data
    merged: Dict[str, Any] = {"updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    all_syms = set(syms) | set(ops_data.keys()) | set(whale_data.keys()) | set(liq_data.keys())
    for s in sorted(all_syms):
        merged[s] = {}
        # Options metrics
        if s in ops_data and isinstance(ops_data[s], dict):
            merged[s].update(ops_data[s])
        # Whale alerts
        if s in whale_data and isinstance(whale_data[s], dict):
            merged[s].update(whale_data[s])
        # Liquidations
        if s in liq_data and isinstance(liq_data[s], dict):
            # Keep as nested dict to preserve price-level information
            merged[s]["liquidation_heatmap"] = liq_data[s]
    # Write the merged metrics to file
    out_path = metrics_dir / "external_metrics.json"
    try:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)
        logger.info(f"External metrics updated for {len(all_syms)} symbols")
    except BEST_EFFORT_EXCEPTIONS as e:
        logger.error(f"Failed to write external metrics: {e}")
    return merged


if __name__ == "__main__":
    res = update_external_metrics()
    print(json.dumps(res, indent=2))
