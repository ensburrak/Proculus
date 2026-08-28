"""
trade_metrics.py
-----------------

This module provides helper functions to compute more accurate trade exit
metrics such as the volume‑weighted average price (VWAP) of a set of fills,
the total fees paid across those fills, and the realised slippage relative
to a reference mid‑price.  It is intended to be used during trade closure
to replace simplistic exit price estimates with a true aggregated value.

Functions
~~~~~~~~~

``async compute_vwap_and_fee(exchange, symbol, since_ts)``
    Fetches recent fills for ``symbol`` via ``fetch_my_trades`` and
    returns the VWAP of all fills executed since ``since_ts`` along with
    the total fee paid.  If no fills are found, ``None`` is returned for
    both values.

``calculate_slippage(entry_mid_price, exit_price, side)``
    Computes the realised slippage as a positive fraction based on the
    difference between the entry mid‑price and the executed exit price.
    For long positions, only downside slippage is considered; for
    shorts, only upside slippage is counted.  If a favourable move
    occurred, the slippage is zero.
"""

from __future__ import annotations


from core.exceptions import BEST_EFFORT_EXCEPTIONS
from typing import Optional, Tuple
import asyncio

async def compute_vwap_and_fee(
    exchange: object,
    symbol: str,
    since_ts: Optional[int] = None,
    max_trades: int = 20,
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Compute VWAP and total fee for fills since a given timestamp.

    This helper calls ``fetch_my_trades`` via the provided ``exchange`` to
    retrieve up to ``max_trades`` recent trades for the specified OKX
    symbol (e.g. ``BTC‑USDT‑SWAP``).  It then filters those trades by
    timestamp (if ``since_ts`` is provided) and calculates the VWAP and
    total fee paid.  The side of the most recent trade is also returned
    to aid downstream slippage calculations.

    Args:
        exchange: The ccxt exchange instance.
        symbol: The OKX symbol (dash separated) to query, e.g. ``BTC‑USDT‑SWAP``.
        since_ts: Unix timestamp in milliseconds; trades with a ``timestamp``
            greater than or equal to this value will be included.  If ``None``
            all returned trades are used.
        max_trades: Maximum number of trades to fetch.  Defaults to 20.

    Returns:
        Tuple of ``(vwap, total_fee, side)``.  ``vwap`` and ``total_fee``
        are floats (or ``None`` if insufficient data) and ``side`` is the
        ``"buy"``/``"sell"`` side of the most recent fill if available.
    """
    try:
        # Attempt to fetch trades.  ccxt's ``fetch_my_trades`` accepts
        # ``since`` and ``limit`` parameters.  Use the provided since_ts
        # to filter trades on the server if supported; otherwise filter
        # locally.
        trades = None
        kwargs = {}
        if since_ts is not None:
            kwargs["since"] = since_ts
        kwargs["limit"] = max_trades
        # Note: we intentionally avoid using the custom ``call`` wrapper here
        # to keep this module independent of rate‑limit logic.  The caller
        # should wrap this function using their own retry mechanism.
        if asyncio.iscoroutinefunction(exchange.fetch_my_trades):
            trades = await exchange.fetch_my_trades(symbol, **kwargs)
        else:
            # fetch trades may be synchronous; run in thread
            trades = await asyncio.to_thread(exchange.fetch_my_trades, symbol, **kwargs)
        if not trades or not isinstance(trades, list):
            return None, None, None
        # Filter by timestamp if needed
        filtered = []
        for t in trades:
            ts = t.get("timestamp")
            if since_ts is None or (ts is not None and ts >= since_ts):
                # Ensure price and amount are present
                try:
                    price = float(t.get("price"))
                    amount = float(t.get("amount"))
                    # skip trades with zero amount
                    if amount == 0:
                        continue
                    filtered.append(t)
                except BEST_EFFORT_EXCEPTIONS:
                    continue
        if not filtered:
            return None, None, None
        # Compute vwap and total fee
        total_qty = 0.0
        total_cost = 0.0
        total_fee = 0.0
        last_side = None
        for tr in filtered:
            try:
                p = float(tr.get("price"))
                amt = float(tr.get("amount"))
            except BEST_EFFORT_EXCEPTIONS:
                continue
            total_qty += amt
            total_cost += p * amt
            # side of last trade (for slippage logic)
            side = tr.get("side")
            if side:
                last_side = str(side)
            # [FIX-15.1+15.1B] Fee deduplikasyonu — currency+amount bazlı seen set
            # NOT: seen_fees her trade için ayrıdır, farklı trade'lerdeki aynı ücretler
            # bağımsız sayılır. Sadece AYNı trade içindeki duplikasyonu engeller.
            seen_fees = set()
            try:
                fee_dict = tr.get("fee")
                if fee_dict:
                    cost = fee_dict.get("cost")
                    if cost is not None:
                        fee_key = (fee_dict.get("currency", ""), round(abs(float(cost)), 8))
                        if fee_key not in seen_fees:
                            total_fee += abs(float(cost))
                            seen_fees.add(fee_key)
            except BEST_EFFORT_EXCEPTIONS:
                pass
            try:
                fees_list = tr.get("fees") or []
                for f in fees_list:
                    try:
                        cost = f.get("cost")
                        if cost is not None:
                            fee_key = (f.get("currency", ""), round(abs(float(cost)), 8))
                            if fee_key not in seen_fees:
                                total_fee += abs(float(cost))
                                seen_fees.add(fee_key)
                    except BEST_EFFORT_EXCEPTIONS:
                        continue
            except BEST_EFFORT_EXCEPTIONS:
                pass
        if total_qty == 0:
            return None, None, last_side
        vwap = total_cost / total_qty
        return vwap, total_fee if total_fee > 0 else None, last_side
    except BEST_EFFORT_EXCEPTIONS:
        return None, None, None


def calculate_slippage(
    entry_mid_price: Optional[float],
    exit_price: Optional[float],
    side: Optional[str],
) -> Optional[float]:
    """Calculate realised slippage as a positive fraction.

    Given the mid‑price observed when the order was placed (``entry_mid_price``),
    the actual volume‑weighted exit price (``exit_price``) and the trade side
    (``"buy"``, ``"sell"``, ``"long"`` or ``"short"``), compute a
    fractional slippage.  For long/"buy" trades we only consider a worse
    realised price (i.e. exit below the entry mid); for shorts we consider
    cases where the exit price is higher than the entry mid.  If the move
    was favourable, the slippage returned is zero.  Unknown sides or
    missing data return ``None``.

    Args:
        entry_mid_price: The mid price at order placement.
        exit_price: The realised VWAP exit price.
        side: The trade side (string).  Accepts ``"long"``, ``"buy"``
            for long positions and ``"short"``, ``"sell"`` for shorts.

    Returns:
        A non‑negative slippage fraction (0.0–1.0) or ``None`` if inputs
        are invalid.
    """
    try:
        if entry_mid_price is None or exit_price is None:
            return None
        entry_mid = float(entry_mid_price)
        exit_p = float(exit_price)
        if entry_mid <= 0:
            return None
    except BEST_EFFORT_EXCEPTIONS:
        return None
    if side is None:
        return None
    s = side.lower()
    # Determine whether the trade is long/buy or short/sell
    is_long = s.startswith("long") or s.startswith("buy")
    is_short = s.startswith("short") or s.startswith("sell")
    if not (is_long or is_short):
        return None
    # Compute relative difference
    diff = entry_mid - exit_p
    frac = diff / entry_mid
    # For longs, slippage is positive when exit price < entry mid (bad)
    if is_long:
        slip = max(0.0, frac)
    else:
        # For shorts, a worse exit means exit price > entry mid
        diff_short = exit_p - entry_mid
        slip = max(0.0, diff_short / entry_mid)
    # Bound slip to [0, 1]
    if slip < 0:
        slip = 0.0
    elif slip > 1.0:
        slip = 1.0
    return slip
