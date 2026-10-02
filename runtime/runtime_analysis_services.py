from __future__ import annotations

import asyncio
from typing import Any

from .runtime_ta_pack import build_mtf_features, build_ta_pack_from_multidata, safe_last as _safe_last


_TICKER_SEM: asyncio.Semaphore | None = None


def _get_ticker_sem() -> asyncio.Semaphore:
    global _TICKER_SEM
    if _TICKER_SEM is None:
        _TICKER_SEM = asyncio.Semaphore(4)
    return _TICKER_SEM


def _build_ta_pack_from_multidata(multi_data: dict[str, Any], tf_main: str = "15m", **kwargs: Any) -> dict[str, Any]:
    del kwargs
    return build_ta_pack_from_multidata(multi_data, tf_main)


def _populate_imbalance(item: dict[str, Any], orderbook: Any = None) -> dict[str, Any]:
    if isinstance(orderbook, dict):
        bids = orderbook.get("bids") or []
        asks = orderbook.get("asks") or []
        try:
            bid_qty = sum(float(row[1]) for row in bids[:10])
            ask_qty = sum(float(row[1]) for row in asks[:10])
            total = bid_qty + ask_qty
            if total > 0:
                item["imbalance"] = (bid_qty - ask_qty) / total
        except (TypeError, ValueError, IndexError):
            pass
    return item


async def _analyze_one(exchange: Any, symbol: str, tf_main: str = "15m", runtime_mode: str = "paper") -> dict[str, Any] | None:
    from analyzer import get_multi_timeframe_analysis_async

    multi_data = await get_multi_timeframe_analysis_async(
        exchange,
        symbol,
        timeframes=["5m", "15m", "1h", "4h"],
        limit=300,
    )
    if not isinstance(multi_data, dict) or not multi_data:
        return None

    mtf = build_mtf_features(multi_data)
    ta_pack = build_ta_pack_from_multidata(multi_data, tf_main)
    price = ta_pack.get("price")
    if price is None:
        return None

    return {
        "symbol": symbol,
        "tf": tf_main,
        "price": float(price),
        "runtime_mode": runtime_mode,
        "ta_pack": ta_pack,
        "mtf_features": mtf,
        "mtf_data": multi_data,
    }
