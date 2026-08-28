"""
ops_data_provider.py
--------------------

This module acts as a data interface to cryptocurrency options markets.
Options data such as implied volatility (IV), put/call ratios, and open
interest can provide valuable information about market sentiment and
potential price movements.  The functions in this file define a
standardised structure for fetching such data from external APIs.  They
return Python dictionaries to be consumed by the trading bot's signal
generation logic.

At the moment this module contains simple placeholders.  To integrate
live options data, you can implement ``fetch_options_metrics`` using
services like Deribit, Paradigm or Delta Exchange.  Make sure to set
appropriate API keys via environment variables and handle pagination
and rate limits responsibly.

Example usage::

    from ops_data_provider import fetch_options_metrics
    data = fetch_options_metrics(symbol="BTC")
    print(data["implied_volatility"])

"""

from __future__ import annotations


from core.exceptions import BEST_EFFORT_EXCEPTIONS
import logging
import os
import random
import time
from datetime import datetime, timezone
from typing import Any, Dict
import json
from pathlib import Path
try:
    import requests  # optional; placeholder endpoints may use it
except BEST_EFFORT_EXCEPTIONS:  # pragma: no cover
    requests = None

# Ensure environment variables from .env are loaded once.  Use
# find_dotenv to locate the file in parent directories.  If
# python-dotenv is unavailable, ignore the error.
try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore
    load_dotenv(find_dotenv())
except BEST_EFFORT_EXCEPTIONS:
    pass

# Import retry and caching decorators.  If unavailable they fall back
# to no‑ops so that the function still works without extra dependencies.
try:
    from retry_utils import retry  # type: ignore
except BEST_EFFORT_EXCEPTIONS:
    def retry(*args, **kwargs):  # type: ignore
        def decorator(func):  # type: ignore
            return func
        return decorator
try:
    from cache_manager import file_cache  # type: ignore
except BEST_EFFORT_EXCEPTIONS:
    def file_cache(cache_name: str, ttl: int = 3600):  # type: ignore
        def decorator(func):  # type: ignore
            return func
        return decorator

logger = logging.getLogger(__name__)

# Deribit circuit breaker to avoid tight failure loops (e.g. ConnectionResetError)
_DERIBIT_CB = {"fail_count": 0, "open_until": 0.0}
try:
    _DERIBIT_CB_FAIL_THRESHOLD = int(os.getenv("DERIBIT_CB_FAIL_THRESHOLD", "3"))
except BEST_EFFORT_EXCEPTIONS:
    _DERIBIT_CB_FAIL_THRESHOLD = 3
try:
    _DERIBIT_CB_COOLDOWN_SEC = float(os.getenv("DERIBIT_CB_COOLDOWN_SEC", "300"))
except BEST_EFFORT_EXCEPTIONS:
    _DERIBIT_CB_COOLDOWN_SEC = 300.0

_DERIBIT_SUPPORTED = {"BTC", "ETH"}

# ---------------------------------------------------------------------------
# Deribit helpers: circuit breaker + last-known-good cache
# ---------------------------------------------------------------------------
_DERIBIT_LKG_PATH = Path("data/cache/deribit_options_metrics_last_ok.json")

def _deribit_cb_status(now: datetime) -> str:
    """Return 'open' if circuit breaker is open, else 'closed'."""
    try:
        open_until = float(_DERIBIT_CB.get("open_until", 0.0))
        return "open" if open_until and now.timestamp() < open_until else "closed"
    except BEST_EFFORT_EXCEPTIONS:
        return "closed"

def _deribit_cb_success() -> None:
    try:
        _DERIBIT_CB["fail_count"] = 0
        _DERIBIT_CB["open_until"] = 0.0
    except BEST_EFFORT_EXCEPTIONS:
        pass

def _deribit_cb_failure() -> None:
    """Increment failure count and open the circuit if threshold is exceeded."""
    try:
        _DERIBIT_CB["fail_count"] = int(_DERIBIT_CB.get("fail_count", 0)) + 1
        if int(_DERIBIT_CB["fail_count"]) >= int(_DERIBIT_CB_FAIL_THRESHOLD):
            _DERIBIT_CB["open_until"] = time.time() + float(_DERIBIT_CB_COOLDOWN_SEC)
    except BEST_EFFORT_EXCEPTIONS:
        pass

def get_cached_options_metrics(currency: str) -> Dict[str, Any] | None:
    """Load last-known-good (LKG) options metrics from disk."""
    try:
        if not _DERIBIT_LKG_PATH.exists():
            return None
        with _DERIBIT_LKG_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        cur = (currency or "").upper()
        item = data.get(cur)
        return item if isinstance(item, dict) else None
    except BEST_EFFORT_EXCEPTIONS:
        return None

def set_cached_options_metrics(currency: str, payload: Dict[str, Any]) -> None:
    """Persist last-known-good (LKG) options metrics for the given currency."""
    try:
        cur = (currency or "").upper()
        if cur not in _DERIBIT_SUPPORTED:
            return
        _DERIBIT_LKG_PATH.parent.mkdir(parents=True, exist_ok=True)
        data: Dict[str, Any] = {}
        if _DERIBIT_LKG_PATH.exists():
            try:
                with _DERIBIT_LKG_PATH.open("r", encoding="utf-8") as f:
                    existing = json.load(f)
                if isinstance(existing, dict):
                    data.update(existing)
            except BEST_EFFORT_EXCEPTIONS:
                pass
        # store a compact schema
        data[cur] = payload
        with _DERIBIT_LKG_PATH.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except BEST_EFFORT_EXCEPTIONS:
        pass

@retry()
def fetch_options_metrics(symbol: str, timeout: int = 6, max_retries: int = 4) -> Dict[str, Any]:
    """Fetch Deribit options-implied volatility index (IV) for BTC/ETH.

    This provider is best-effort: it MUST return a schema-stable dict and never raise.
    It uses:
      - exception-aware retry/backoff (+ jitter)
      - cache fallback
      - circuit breaker for repeated transport failures (e.g., ConnectionResetError 10054)

    Returns schema:
      {
        "supported": bool,
        "status": "ok" | "stale" | "down" | "unsupported",
        "iv_index": float | None,
        "currency": "BTC"|"ETH"|None,
        "updated_at": iso8601 str,
        "error": str|None
      }
    """
    now = datetime.now(timezone.utc)
    currency = symbol.split("/")[0].upper() if symbol else ""
    # Hard dependency missing → return soft-failure payload
    if requests is None:
        return {
            "supported": True if currency in ("BTC", "ETH") else False,
            "status": "down",
            "implied_volatility": None,
            "iv_index": None,
            "currency": currency or None,
            "updated_at": now.isoformat(),
            "error": "requests_missing",
            "source": "deribit",
        }
    base_schema = {
        "supported": False,
        "status": "unsupported",
        "implied_volatility": None,
        "iv_index": None,
        "currency": None,
        "source": "deribit",
        "updated_at": now.isoformat(),
        "error": None,
    }

    if currency not in ("BTC", "ETH"):
        base_schema["currency"] = currency or None
        return base_schema

    base_schema["supported"] = True
    base_schema["currency"] = currency

    # Circuit breaker check
    cb_status = _deribit_cb_status(now)
    if cb_status == "open":
        cached = get_cached_options_metrics(currency)
        if cached:
            cached["status"] = "stale"
            cached.setdefault("supported", True)
            cached.setdefault("currency", currency)
            cached.setdefault("updated_at", now.isoformat())
            cached.setdefault("implied_volatility", cached.get("iv_index"))
            cached.setdefault("source", "deribit")
            cached.setdefault("error", "deribit_circuit_open")
            return cached
        base_schema["status"] = "down"
        base_schema["error"] = "deribit_circuit_open"
        return base_schema

    url = f"https://www.deribit.com/api/v2/public/get_iv_index?currency={currency}"
    session = requests.Session()

    last_err = None
    # Exception-aware retry loop (covers ConnectionResetError / ProtocolError / timeouts)
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, timeout=(timeout, timeout))
            if resp.status_code == 200:
                try:
                    data = resp.json()
                except BEST_EFFORT_EXCEPTIONS as e:
                    last_err = f"json_decode:{type(e).__name__}"
                else:
                    iv = None
                    if isinstance(data, dict):
                        # Deribit payload: {"jsonrpc":"2.0","result":{"currency":"BTC","timestamp":...,"iv":...}}
                        res = data.get("result") or {}
                        iv = res.get("iv")
                    if isinstance(iv, (int, float)):
                        out = {
                            "supported": True,
                            "status": "ok",
                            "iv_index": float(iv),
                            "implied_volatility": float(iv),
                            "currency": currency,
                            "updated_at": now.isoformat(),
                            "error": None,
                        }
                        set_cached_options_metrics(currency, out)
                        _deribit_cb_success()
                        return out
                    last_err = "missing_iv"
            else:
                # HTTP-level failures: retry only on rate/5xx
                last_err = f"http_{resp.status_code}"
                if resp.status_code in (429, 500, 502, 503, 504):
                    pass
                else:
                    break
        except requests.exceptions.RequestException as e:
            last_err = f"req:{type(e).__name__}"
        # backoff with jitter
        sleep_s = min(2 ** (attempt - 1), 8)
        sleep_s = sleep_s + random.random() * 0.25
        time.sleep(sleep_s)

    # failure path: update circuit breaker + return cached or down schema
    _deribit_cb_failure()
    cached = get_cached_options_metrics(currency)
    if cached:
        cached["status"] = "stale"
        cached.setdefault("supported", True)
        cached.setdefault("currency", currency)
        cached.setdefault("updated_at", now.isoformat())
        cached.setdefault("implied_volatility", cached.get("iv_index"))
        cached.setdefault("source", "deribit")
        cached["error"] = last_err or "deribit_fetch_failed"
        return cached

    base_schema["status"] = "down"
    base_schema["error"] = last_err or "deribit_fetch_failed"
    return base_schema

async def start_deribit_ws(symbols: list[str], update_callback, ws_url: str | None = None,
                           testnet: bool = False, reconnect_delay: int = 5) -> None:
    """
    Connect to the Deribit WebSocket API and subscribe to option‑related channels.

    This helper allows you to receive real‑time option metrics such as implied
    volatility, open interest and book summaries.  It subscribes to the
    appropriate channels for each currency and forwards every received
    notification to the provided ``update_callback`` coroutine for further
    processing.  The function runs until cancelled and will attempt to
    reconnect on errors using an exponential backoff.

    Parameters
    ----------
    symbols : list of str
        List of underlying asset symbols (e.g. ["BTC", "ETH"]).  For each
        symbol the function subscribes to channels ``iv_index.SYMBOL``,
        ``book_summary.SYMBOL.option`` and ``open_interest.SYMBOL``.
    update_callback : callable
        An async callback taking a parsed JSON message as its only argument.
        It will be awaited for each notification received from the socket.
    ws_url : str, optional
        WebSocket endpoint to use.  Defaults to mainnet ``wss://www.deribit.com/ws/api/v2``.
        To connect to testnet you can pass
        ``wss://test.deribit.com/ws/api/v2`` or set ``testnet=True``.
    testnet : bool, optional
        If True and ``ws_url`` is not provided, the function uses Deribit's
        testnet endpoint.  Default is False.
    reconnect_delay : int, optional
        Base delay in seconds before attempting to reconnect after an error.

    Notes
    -----
    Deribit may change channel naming conventions over time.  Consult the
    official docs for up‑to‑date channel names.  See
    https://docs.deribit.com for details.
    """
    try:
        import asyncio
        import json
        import math
        try:
            import websockets  # type: ignore
        except BEST_EFFORT_EXCEPTIONS:
            # WebSocket kütüphanesi yoksa veya kullanılamıyorsa REST tabanlı bir
            # fallback döngüsü ile gerçek zamanlıya yakın güncellemeler sağla.
            logger.warning("websockets kütüphanesi bulunamadı; Deribit WS yerine REST polling kullanılacak")
            # Fallback interval süresini ortamdan oku veya varsayılan 10 saniye kullan
            try:
                interval = int(float(os.getenv("DERIBIT_REST_INTERVAL", "10")))
            except BEST_EFFORT_EXCEPTIONS:
                interval = 10
            async def _rest_loop():
                while True:
                    for sym in symbols:
                        try:
                            # fetch_options_metrics kullanılabilir; callback'e ham mesaj olarak geçir
                            data = fetch_options_metrics(sym)  # type: ignore
                            if data:
                                # Callback coroutine ise await et, değilse doğrudan çağır
                                if asyncio.iscoroutinefunction(update_callback):
                                    await update_callback(data)
                                else:
                                    update_callback(data)
                        except BEST_EFFORT_EXCEPTIONS:
                            continue
                    await asyncio.sleep(interval)
            # Sonsuz döngü: hatalarda tekrar dene
            while True:
                try:
                    await _rest_loop()
                except BEST_EFFORT_EXCEPTIONS:
                    await asyncio.sleep(interval)
            return

        # Determine the WebSocket URL
        if ws_url:
            url = ws_url
        else:
            url = "wss://www.deribit.com/ws/api/v2" if not testnet else "wss://test.deribit.com/ws/api/v2"

        # Build channel list for subscription
        channels: list[str] = []
        for sym in symbols:
            cur = sym.upper()
            channels.append(f"iv_index.{cur}")
            channels.append(f"book_summary.{cur}.option")
            channels.append(f"open_interest.{cur}")

        async def _subscribe_and_listen():
            """Subscribe to Deribit channels and forward messages to the callback.

            This coroutine manages a single WebSocket connection.  We disable
            the built‑in ping mechanism by passing ``ping_interval=None`` so
            that websockets does not create background ping tasks that linger
            after cancellation (avoiding ``Task was destroyed but it is pending!``
            warnings).  All received messages are parsed and forwarded to the
            provided callback.
            """
            # Disable WebSocket ping intervals to avoid creating orphaned tasks.
            async with websockets.connect(url, ping_interval=None) as ws:
                # Subscribe to the requested channels
                sub_msg = {
                    "jsonrpc": "2.0",
                    "method": "public/subscribe",
                    "params": {"channels": channels},
                    "id": 1,
                }
                await ws.send(json.dumps(sub_msg))
                while True:
                    msg = await ws.recv()
                    try:
                        data = json.loads(msg)
                    except BEST_EFFORT_EXCEPTIONS:
                        continue
                    # Pass data to callback
                    try:
                        if asyncio.iscoroutinefunction(update_callback):
                            await update_callback(data)
                        else:
                            update_callback(data)
                    except BEST_EFFORT_EXCEPTIONS as cb_exc:
                        logger.warning("Deribit WS callback error: %s", cb_exc)

        # Outer reconnect loop.  Keep trying to (re)connect indefinitely with exponential backoff.
        backoff = reconnect_delay
        while True:
            try:
                await _subscribe_and_listen()
            except BEST_EFFORT_EXCEPTIONS as e:
                logger.warning("Deribit WS connection error: %s; reconnecting in %ds", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
    except BEST_EFFORT_EXCEPTIONS as e:
        logger.warning("Deribit WS general error: %s", e)
        return
