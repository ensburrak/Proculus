from __future__ import annotations

from bot.order_processing import (
    monitor_pending_orders as _monitor_pending_orders_impl,
    orders_worker as _orders_worker_impl,
    process_single_order as _process_single_order_impl,
)
from core.service_container import resolve_execution_dry_run
from core.config_loader import load_config
from execution.order_service_bridge import (
    cancel_open_order,
    fetch_runtime_open_orders,
    fetch_runtime_ticker,
    set_runtime_leverage,
    submit_entry_plan,
    submit_native_trailing_stop,
    submit_slippage_protected_close,
    submit_vwap_slice,
    update_managed_stop_loss,
)
from runtime.runtime_service_deps import _bind_main_runtime

_SLIPPAGE_TOLERANCE = 0.005


async def _slippage_protected_close(
    exchange,
    okx_sym: str,
    side_str: str,
    size: float,
    slippage_pct: float = _SLIPPAGE_TOLERANCE,
):
    return await submit_slippage_protected_close(
        exchange,
        okx_sym,
        side_str,
        size,
        slippage_pct=slippage_pct,
        dry_run=resolve_execution_dry_run(load_config(), exchange),
    )


async def process_single_order_service(exchange, sym, decision):
    # Keep runtime globals synchronized before delegating to the canonical worker.
    _bind_main_runtime()
    return await _process_single_order_impl(exchange, sym, decision)


async def orders_worker(exchange, order_queue=None):
    return await _orders_worker_impl(exchange, order_queue=order_queue)


async def monitor_pending_orders(exchange, check_interval=None, stale_seconds=None):
    return await _monitor_pending_orders_impl(
        exchange,
        check_interval=check_interval,
        stale_seconds=stale_seconds,
    )


__all__ = [
    "_slippage_protected_close",
    "cancel_open_order",
    "fetch_runtime_open_orders",
    "fetch_runtime_ticker",
    "monitor_pending_orders",
    "orders_worker",
    "process_single_order_service",
    "set_runtime_leverage",
    "submit_entry_plan",
    "submit_native_trailing_stop",
    "submit_slippage_protected_close",
    "submit_vwap_slice",
    "update_managed_stop_loss",
]
