from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Iterable
import math

# Research-only verification utility. This is NOT wired to a live V5 order path.
# A V5 setup must not be promoted until a real executor invokes this check
# immediately before every prospective order and fails closed on rejection.
DEFAULT_MAX_MARKET_IMPACT_BPS = 15.0
DEFAULT_MAX_SPREAD_BPS = 10.0
DEFAULT_MAX_QUOTE_AGE_SECONDS = 3.0


@dataclass(frozen=True)
class V5ExecutionCostDecision:
    allowed: bool
    reason: str
    spread_bps: float | None = None
    market_impact_bps: float | None = None
    quote_age_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "spread_bps": self.spread_bps,
            "market_impact_bps": self.market_impact_bps,
            "quote_age_seconds": self.quote_age_seconds,
        }


def _deny(reason: str, **metrics: Any) -> V5ExecutionCostDecision:
    return V5ExecutionCostDecision(False, reason, **metrics)


def _normalize_levels(raw: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for raw_price, raw_amount in raw:
        price = float(raw_price)
        amount = float(raw_amount)
        if not (math.isfinite(price) and math.isfinite(amount)):
            raise ValueError("nonfinite_orderbook_level")
        if price <= 0 or amount <= 0:
            raise ValueError("invalid_orderbook_level")
        out.append((price, amount))
    return out


def assess_v5_market_execution_cost(
    *,
    side: str,
    notional_usdt: float,
    bids: Iterable[tuple[float, float]] | None,
    asks: Iterable[tuple[float, float]] | None,
    quote_time: datetime | None,
    now: datetime | None = None,
    max_market_impact_bps: float = DEFAULT_MAX_MARKET_IMPACT_BPS,
    max_spread_bps: float = DEFAULT_MAX_SPREAD_BPS,
    max_quote_age_seconds: float = DEFAULT_MAX_QUOTE_AGE_SECONDS,
) -> V5ExecutionCostDecision:
    """Assess conservative taker fill impact using a live orderbook snapshot.

    Levels contain (price, quantity in BASE units), not quote currency.
    The market impact metric includes the half-spread and depth slippage from
    the mid-price, so its bound is per side. Re-check the book immediately
    before any order; offline results are never proof of executable quotes.
    """
    if side not in ("long", "short"):
        return _deny("invalid_side")
    if notional_usdt <= 0 or not math.isfinite(float(notional_usdt)):
        return _deny("invalid_notional")
    if (max_market_impact_bps <= 0 or max_spread_bps <= 0
            or max_quote_age_seconds <= 0):
        return _deny("invalid_limits")
    if bids is None or asks is None or quote_time is None:
        return _deny("missing_live_orderbook")

    try:
        bids_list = _normalize_levels(bids)
        asks_list = _normalize_levels(asks)
    except (ValueError, TypeError, OverflowError):
        return _deny("malformed_orderbook")

    if not bids_list or not asks_list:
        return _deny("empty_orderbook")
    bids_list.sort(key=lambda row: row[0], reverse=True)
    asks_list.sort(key=lambda row: row[0])
    bid = bids_list[0][0]
    ask = asks_list[0][0]
    if bid >= ask:
        return _deny("crossed_orderbook")
    mid = (bid + ask) / 2.0
    spread_bps = (ask - bid) / mid * 10000.0

    now = now or datetime.now(UTC)
    if now.tzinfo is None or quote_time.tzinfo is None:
        return _deny("missing_timezone")
    age = (now.astimezone(UTC) - quote_time.astimezone(UTC)).total_seconds()
    metrics = {"spread_bps": spread_bps, "quote_age_seconds": age}
    if age < 0 or age > max_quote_age_seconds:
        return _deny("stale_quote", **metrics)
    if spread_bps > max_spread_bps:
        return _deny("spread_limit_exceeded", **metrics)

    levels = asks_list if side == "long" else bids_list
    remaining = float(notional_usdt)
    filled_qty = 0.0
    spent_quote = 0.0
    # Sized in QUOTE notional. At each level, quantity of base units that can
    # be purchased/sold until the requested notional is exhausted.
    for price, quantity in levels:
        available_quote = price * quantity
        take_quote = min(remaining, available_quote)
        spent_quote += take_quote
        filled_qty += take_quote / price
        remaining -= take_quote
        if remaining <= max(1e-8, float(notional_usdt) * 1e-10):
            remaining = 0.0
            break
    if remaining > 0 or filled_qty <= 0:
        return _deny("insufficient_depth", **metrics)
    average_px = spent_quote / filled_qty
    impact_bps = (average_px - mid) / mid * 10000 if side == "long" else (mid - average_px) / mid * 10000
    if not math.isfinite(impact_bps):
        return _deny("invalid_impact", **metrics)
    if impact_bps > max_market_impact_bps:
        return _deny("market_impact_limit_exceeded", market_impact_bps=impact_bps, **metrics)
    return V5ExecutionCostDecision(True, "live_book_costs_within_limits", spread_bps, impact_bps, age)


__all__ = ["V5ExecutionCostDecision", "assess_v5_market_execution_cost"]
