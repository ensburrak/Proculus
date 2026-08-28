"""
liquidation_data_provider.py
---------------------------

Liquidation events occur when leveraged positions are forcibly closed and
often precede heightened volatility.  Monitoring large liquidation
clusters can provide early warning signals for rapid price movements.

This module offers a simple interface to retrieve recent liquidation data
from derivatives exchanges or analytics services.  The primary
function, ``fetch_liquidation_heatmap``, returns a dictionary keyed by
price levels with values representing the cumulative notional size of
liquidations at each level.

Example usage::

    from liquidation_data_provider import fetch_liquidation_heatmap
    heatmap = fetch_liquidation_heatmap("BTC", interval="1h")
    print(heatmap)

Custom implementations can use exchanges' liquidation feeds or
third-party services like Coinalyze.  API keys can be supplied via
environment variables ``LIQUIDATION_API_KEY``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict

from core.adaptive_backoff import AdaptiveBackoffState
from core.exceptions import BEST_EFFORT_EXCEPTIONS

try:
    import aiohttp
except BEST_EFFORT_EXCEPTIONS:
    aiohttp = None

logger = logging.getLogger(__name__)
_HTTP_BACKOFF_UNTIL = 0.0
_ADAPTIVE_CACHE_TTL_SECONDS = 600
_HTTP_BACKOFF_STATE = AdaptiveBackoffState(base_ttl=600, max_delay=30.0)
_HEATMAP_CACHE: Dict[tuple[str, str], tuple[float, Dict[str, Any]]] = {}


@dataclass
class _HTTPResult:
    status_code: int
    headers: Dict[str, str]
    payload: Any

    def json(self) -> Any:
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _sync_legacy_backoff_globals() -> None:
    global _HTTP_BACKOFF_UNTIL, _ADAPTIVE_CACHE_TTL_SECONDS
    snapshot = _HTTP_BACKOFF_STATE.snapshot()
    _HTTP_BACKOFF_UNTIL = float(snapshot["backoff_until"])
    _ADAPTIVE_CACHE_TTL_SECONDS = int(snapshot["adaptive_cache_ttl"])


async def _http_get_with_backoff(url: str, **kwargs: Any) -> _HTTPResult | None:
    if aiohttp is None:
        return None
    timeout = kwargs.pop("timeout", 10)
    if _HTTP_BACKOFF_STATE.should_skip():
        _sync_legacy_backoff_globals()
        return None
    for attempt in range(3):
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=timeout, **kwargs) as response:
                status = int(getattr(response, "status", getattr(response, "status_code", 0)) or 0)
                headers = dict(getattr(response, "headers", {}) or {})
                payload = await response.json()
        decision = _HTTP_BACKOFF_STATE.record_response(status, headers, attempt=attempt)
        _sync_legacy_backoff_globals()
        result = _HTTPResult(status_code=status, headers=headers, payload=payload)
        if decision.should_backoff:
            if attempt < 2:
                await asyncio.sleep(decision.delay_seconds)
                continue
        return result
    return result


def _run_coro_blocking(coro) -> Dict[str, Any]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    box: dict[str, Any] = {}

    def _runner() -> None:
        try:
            box["result"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 - propagate from sync compatibility boundary
            box["error"] = exc

    thread = threading.Thread(target=_runner, name="LiquidationAsyncFetch", daemon=True)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return dict(box.get("result") or {})


def fetch_liquidation_heatmap(symbol: str = "BTC", interval: str = "1h") -> Dict[str, Any]:
    """Retrieve liquidation heatmap with adaptive cache TTL after 401/429/5xx."""
    return _run_coro_blocking(fetch_liquidation_heatmap_async(symbol=symbol, interval=interval))


async def fetch_liquidation_heatmap_async(symbol: str = "BTC", interval: str = "1h") -> Dict[str, Any]:
    """Async liquidation heatmap fetch that avoids blocking the runtime event loop."""
    key = (str(symbol or "BTC").upper(), str(interval or "1h"))
    cached = _HEATMAP_CACHE.get(key)
    now = time.time()
    if cached is not None and now - cached[0] < _ADAPTIVE_CACHE_TTL_SECONDS:
        return dict(cached[1])
    payload = await _fetch_liquidation_heatmap_uncached(symbol=symbol, interval=interval)
    _HEATMAP_CACHE[key] = (time.time(), dict(payload or {}))
    return payload


async def _fetch_liquidation_heatmap_uncached(symbol: str = "BTC", interval: str = "1h") -> Dict[str, Any]:
    """Retrieve a price-level liquidation heatmap.

    :param symbol: the underlying asset symbol (e.g. ``BTC``)
    :param interval: the time interval over which to aggregate liquidations (e.g. ``1h``)
    :return: mapping from price levels (as strings) to notional size of liquidations
      at that level.  Returns an empty dict if no data is available or if
      the API key is missing.

    Adaptive in-memory caching minimises API calls and handles transient
    rate-limit failures without blocking the runtime event loop.
    """
    global _ADAPTIVE_CACHE_TTL_SECONDS
    # Read configuration from environment: base URL and API key (optional)
    api_url = os.getenv("LIQUIDATION_API_URL", "").strip()
    api_key = os.getenv("LIQUIDATION_API_KEY", "").strip()
    # If aiohttp is not available or no API URL is provided, attempt a
    # public Binance Futures fallback.  Use the market-data
    # ``/fapi/v1/allForceOrders`` endpoint rather than the signed
    # account-specific ``/fapi/v1/forceOrders`` endpoint so analysis
    # never depends on private Binance credentials in paper/runtime
    # loops.
    if (not api_url or not api_url.strip()) or aiohttp is None:
        # Skip fallback entirely if the async HTTP client is unavailable.
        if aiohttp is None:
            logger.debug("Liquidation data API and fallback unavailable; returning empty heatmap")
            return {}
        try:
            import urllib.parse

            # Prepare request parameters.
            # Binance Futures expects symbols like BTCUSDT (no separator).
            # Upstream callers sometimes pass base assets only (e.g. "BTC").
            # Normalise to a USDT-quoted symbol for allForceOrders.
            sym = (symbol or "").replace("/", "").strip().upper()
            if sym and not sym.endswith("USDT"):
                sym = f"{sym}USDT"
            params = {
                "symbol": sym,
                "limit": 50,
            }
            query = urllib.parse.urlencode(params)
            base_url = os.getenv("BINANCE_FAPI_BASE_URL", "https://fapi.binance.com").rstrip("/")
            url = f"{base_url}/fapi/v1/allForceOrders?{query}"
            resp = await _http_get_with_backoff(url, timeout=10)
            if resp is None:
                return {}
            if getattr(resp, "status_code", None) == 400:
                _ADAPTIVE_CACHE_TTL_SECONDS = max(_ADAPTIVE_CACHE_TTL_SECONDS, 3600)
                logger.debug("Public Binance liquidation feed unavailable for %s; returning empty heatmap", symbol)
                return {}
            resp.raise_for_status()
            data = resp.json() or []
            total_notional = 0.0
            # Each forced order may include origQty and avgPrice (or shorthand keys)
            for item in data:
                try:
                    qty = float(item.get("executedQty") or item.get("origQty") or item.get("o") or 0.0)
                    price = float(item.get("averagePrice") or item.get("avgPrice") or item.get("p") or 0.0)
                    total_notional += abs(qty) * price
                except (AttributeError, TypeError, ValueError):
                    continue
            # Return simplified heatmap: downstream logic sums the values
            return {"TOTAL": total_notional}
        except (ValueError, TypeError, OSError) as exc:
            logger.warning("Failed to fetch public Binance liquidation feed for %s: %s", symbol, exc)
            return {}
    # Prepare query parameters.  Symbols for many APIs use uppercase without
    # separators (e.g. BTCUSDT).  Consumers can override the API to handle
    # other formats if necessary.
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
    }
    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = api_key
    try:
        resp = await _http_get_with_backoff(api_url, params=params, headers=headers, timeout=10)
        if resp is None:
            return {}
        resp.raise_for_status()
        data = resp.json()
        heatmap: Dict[str, Any] = {}
        # If the response is a list, assume each item contains a price and notional
        if isinstance(data, list):
            for item in data:
                try:
                    price_val = item.get("price") or item.get("p") or item.get("PX")
                    notional_val = item.get("notional") or item.get("qty") or item.get("volume") or item.get("size")
                    if price_val is None or notional_val is None:
                        continue
                    price_str = str(price_val)
                    notional_num = float(notional_val)
                    heatmap[price_str] = heatmap.get(price_str, 0.0) + notional_num
                except BEST_EFFORT_EXCEPTIONS:
                    continue
            return heatmap
        # If the response is a dict, it may already map price levels to notional sizes
        if isinstance(data, dict):
            for k, v in data.items():
                try:
                    heatmap[str(k)] = float(v)
                except BEST_EFFORT_EXCEPTIONS:
                    continue
            return heatmap
        logger.debug("Unexpected liquidation API response format: %s", type(data))
        return {}
    except BEST_EFFORT_EXCEPTIONS as exc:
        logger.warning("Failed to fetch liquidation heatmap for %s: %s", symbol, exc)
        return {}


# ---------------------------------------------------------------------------
# WebSocket streaming for liquidation data


async def start_liquidation_ws(
    exchange: str = "binance",
    symbols: list[str] | None = None,
    update_callback=None,
    reconnect_delay: int = 5,
) -> None:
    """
    Connect to a derivatives exchange's liquidation order stream via
    WebSocket and forward real‑time liquidation events to a callback.

    This helper supports **Binance** and **Bybit** public streams without
    requiring any API keys.  For Binance, you can subscribe to either a
    single symbol (``<symbol>@forceOrder``) or the global liquidation
    stream (``!forceOrder@arr``).  For Bybit, the ``allLiquidation.<symbol>``
    topic is used.  The function runs indefinitely until cancelled and
    automatically reconnects on errors using exponential backoff.

    Parameters
    ----------
    exchange : str, optional
        The exchange name: ``"binance"`` or ``"bybit"``.  Defaults to
        ``"binance"``.
    symbols : list of str, optional
        List of symbols to subscribe to (e.g. ["BTCUSDT", "ETHUSDT"]).
        If omitted or empty for Binance, the global stream is used.  For
        Bybit a non‑empty list is required.
    update_callback : callable, optional
        A callback function or coroutine that accepts a single event
        dictionary.  Each event contains ``exchange``, ``symbol``,
        ``side`` (``Buy``/``Sell``), ``quantity``, ``price``, ``notional``
        and ``timestamp``.  If the callback is a coroutine it will be
        awaited; otherwise it will be called synchronously.
    reconnect_delay : int, optional
        Initial delay in seconds before attempting to reconnect after an
        error.  The delay doubles after each failure up to 60 seconds.

    Notes
    -----
    The ``websockets`` library must be installed.  If it is missing,
    the function logs an error and returns immediately.  Real‑time
    liquidation feeds can generate high event rates; consider
    aggregation, throttling or filtering in the callback to avoid
    overwhelming your application.
    """
    try:
        import asyncio
        import json

        import websockets  # type: ignore
    except BEST_EFFORT_EXCEPTIONS:
        logger.error("websockets library is not available; cannot start liquidation stream")
        return
    ex = (exchange or "").lower()
    if ex not in {"binance", "bybit"}:
        logger.error("Unsupported exchange for liquidation WS: %s", exchange)
        return
    # Normalise symbols list
    symbol_list = symbols or []

    async def _handle_event(event: Dict[str, Any]) -> None:
        if update_callback is None:
            return
        try:
            if asyncio.iscoroutinefunction(update_callback):
                await update_callback(event)
            else:
                update_callback(event)
        except BEST_EFFORT_EXCEPTIONS as exc:
            logger.debug("Liquidation callback error: %s", exc)

    async def _connect_and_listen_binance():
        # Determine the appropriate WebSocket URL for Binance
        if symbol_list:
            streams = "/".join([f"{s.lower()}@forceOrder" for s in symbol_list])
            url = f"wss://fstream.binance.com/stream?streams={streams}"
        else:
            # All market liquidation snapshot
            url = "wss://fstream.binance.com/ws/!forceOrder@arr"
        # Disable ping keepalive tasks to prevent lingering asyncio tasks on shutdown
        async with websockets.connect(url, ping_interval=None) as ws:
            async for msg in ws:
                try:
                    data = json.loads(msg)
                except BEST_EFFORT_EXCEPTIONS:
                    continue
                # Multi‑stream messages have 'data' field
                payload = data.get("data") if isinstance(data, dict) else data
                if not isinstance(payload, dict):
                    continue
                ev = payload.get("o")
                if not isinstance(ev, dict):
                    continue
                try:
                    symbol = ev.get("s")
                    side = ev.get("S")
                    qty = float(ev.get("q") or ev.get("l") or 0.0)
                    price = float(ev.get("ap") or ev.get("p") or 0.0)
                    notional = qty * price
                    ts = int(ev.get("T") or payload.get("E") or 0)
                    event = {
                        "exchange": "binance",
                        "symbol": symbol,
                        "side": side,
                        "quantity": qty,
                        "price": price,
                        "notional": notional,
                        "timestamp": ts,
                    }
                    await _handle_event(event)
                except BEST_EFFORT_EXCEPTIONS:
                    continue

    async def _connect_and_listen_bybit():
        if not symbol_list:
            logger.error("Bybit liquidation stream requires a non‑empty symbol list")
            return
        url = "wss://stream.bybit.com/v5/public/linear"
        # Disable ping keepalive tasks to prevent lingering asyncio tasks on shutdown
        async with websockets.connect(url, ping_interval=None) as ws:
            # Subscribe to each topic
            topics = [f"allLiquidation.{s.upper()}" for s in symbol_list]
            sub_msg = {"op": "subscribe", "args": topics}
            await ws.send(json.dumps(sub_msg))
            async for msg in ws:
                try:
                    data = json.loads(msg)
                except BEST_EFFORT_EXCEPTIONS:
                    continue
                if not isinstance(data, dict):
                    continue
                if data.get("type") != "snapshot":
                    continue
                ts_ms = int(data.get("ts") or 0)
                for item in data.get("data", []) or []:
                    try:
                        symbol = item.get("s")
                        side = item.get("S")
                        qty = float(item.get("v") or 0.0)
                        price = float(item.get("p") or 0.0)
                        notional = qty * price
                        ts_event = int(item.get("T") or ts_ms)
                        event = {
                            "exchange": "bybit",
                            "symbol": symbol,
                            "side": side,
                            "quantity": qty,
                            "price": price,
                            "notional": notional,
                            "timestamp": ts_event,
                        }
                        await _handle_event(event)
                    except BEST_EFFORT_EXCEPTIONS:
                        continue

    # Main reconnect loop
    delay = reconnect_delay
    while True:
        try:
            if ex == "binance":
                await _connect_and_listen_binance()
            else:
                await _connect_and_listen_bybit()
            return  # exit if websocket closes normally
        except BEST_EFFORT_EXCEPTIONS as exc:
            logger.warning("Liquidation WS error (%s): %s; reconnecting in %ds", ex, exc, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)
