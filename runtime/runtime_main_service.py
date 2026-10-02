from __future__ import annotations

import inspect
import os
from typing import Any

from .runtime_loop_services import trading_loop_async_service


async def initialize_exchange_async() -> Any:
    try:
        import ccxt.async_support as ccxt
    except ImportError:
        import ccxt  # type: ignore

    options: dict[str, Any] = {
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    }
    api_key = os.getenv("OKX_API_KEY")
    secret = os.getenv("OKX_API_SECRET")
    password = os.getenv("OKX_API_PASSPHRASE")
    if api_key and secret and password:
        options.update({"apiKey": api_key, "secret": secret, "password": password})
    return ccxt.okx(options)


async def main_service(*, symbols: list[str] | None = None, runtime_mode: str = "paper", once: bool = False) -> dict[str, dict[str, Any]]:
    exchange = await initialize_exchange_async()
    try:
        return await trading_loop_async_service(
            exchange,
            symbols or ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
            runtime_mode=runtime_mode,
            once=once,
        )
    finally:
        close = getattr(exchange, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result


def _noop(*args: Any, **kwargs: Any) -> None:
    return None


_bootstrap_startup_phase = _noop
_dashboard_close_position_best_effort = _noop
_dashboard_symbol_key = lambda value: str(value or "").replace("/", "")
_extract_dashboard_payload = lambda value: value
_load_dynamic_strategy_modules = _noop
_normalize_dashboard_symbol = lambda value: str(value or "").upper()
_register_dashboard_event_handlers = _noop
_start_microservice_runner = _noop
