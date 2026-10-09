from __future__ import annotations

from datetime import UTC, datetime, timedelta

from decision.v5_cost_safety import assess_v5_market_execution_cost


NOW = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)


def _check(**kwargs):
    default = {
        "side": "long",
        "notional_usdt": 1000.0,
        "bids": [(99.99, 100.0)],
        "asks": [(100.01, 100.0)],
        "quote_time": NOW,
        "now": NOW,
    }
    default.update(kwargs)
    return assess_v5_market_execution_cost(**default)


def test_small_impact_long_and_short_pass() -> None:
    for side in ("long", "short"):
        result = _check(side=side)
        assert result.allowed is True
        assert result.market_impact_bps is not None
        assert 0 < result.market_impact_bps < 15
        assert result.spread_bps is not None
        assert result.spread_bps < 10


def test_missing_book_is_never_eligible() -> None:
    assert _check(asks=None).reason == "missing_live_orderbook"


def test_stale_quote_is_rejected() -> None:
    result = _check(quote_time=NOW - timedelta(seconds=4))
    assert result.allowed is False
    assert result.reason == "stale_quote"


def test_wide_spread_is_rejected() -> None:
    result = _check(bids=[(99.9, 100.0)], asks=[(100.1, 100.0)])
    assert result.allowed is False
    assert result.reason == "spread_limit_exceeded"


def test_shallow_book_is_rejected() -> None:
    result = _check(asks=[(100.01, 0.1)])
    assert result.allowed is False
    assert result.reason == "insufficient_depth"


def test_deep_book_with_high_impact_is_rejected() -> None:
    result = _check(
        asks=[(100.01, 1.0), (110.0, 100.0)],
        notional_usdt=1000.0,
    )
    assert result.allowed is False
    assert result.reason == "market_impact_limit_exceeded"
    assert result.market_impact_bps > 15


def test_crossed_book_is_rejected() -> None:
    result = _check(bids=[(100.02, 100.0)])
    assert result.allowed is False
    assert result.reason == "crossed_orderbook"


def test_nonfinite_book_is_rejected() -> None:
    result = _check(asks=[(float("nan"), 1.0)])
    assert result.allowed is False
    assert result.reason == "malformed_orderbook"
