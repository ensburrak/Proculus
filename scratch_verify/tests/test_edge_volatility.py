from decimal import Decimal

from hizlitrade.bus import BroadcastBus
from hizlitrade.engine.edge import EdgeEngine
from hizlitrade.models import (
    BinaryMarketSpec,
    ConsensusPrice,
    DecisionTrace,
    OraclePrice,
    Venue,
)


def _spec() -> BinaryMarketSpec:
    return BinaryMarketSpec(
        market_id="m-vol",
        canonical_id="poly:m-vol",
        venue=Venue.POLYMARKET,
        symbol="BTCUSD",
        yes_instrument="yes",
        no_instrument="no",
        strike=Decimal("100"),
        starts_at_ms=1_000_000,
        expires_at_ms=1_200_000,
        oracle="chainlink",
        oracle_window_seconds=60,
        strike_mode="fixed",
        settlement_rule="gte_strike",
        taker_fee_rate=Decimal("0.07"),
        enabled=True,
    )


def _consensus(price: str, ts_ns: int) -> ConsensusPrice:
    return ConsensusPrice(
        symbol="BTCUSD",
        price=Decimal(price),
        dispersion_bps=Decimal("1"),
        venues=(Venue.BINANCE, Venue.OKX),
        recv_ts_ns=ts_ns,
    )


def test_edge_fails_closed_until_volatility_warmup_completes() -> None:
    now = [1_100_000_000_000]
    bus = BroadcastBus()
    decisions = bus.subscribe("decision-test", event_types=(DecisionTrace,))
    engine = EdgeEngine(
        bus,
        (_spec(),),
        min_net_edge=Decimal("0.025"),
        slippage=Decimal("0.003"),
        safety_margin=Decimal("0.005"),
        order_notional=Decimal("2.50"),
        max_age_ms=1_500,
        clock_ns=lambda: now[0],
    )

    engine.process_event(_consensus("100", now[0]))

    trace = decisions.get_nowait()
    assert isinstance(trace, DecisionTrace)
    assert trace.reason_code == "volatility_warmup_incomplete"
    assert trace.metadata["volatility_samples"] == 1
    assert trace.metadata["required_return_samples"] == 8


def test_edge_advances_after_volatility_warmup() -> None:
    base = 1_100_000_000_000
    now = [base]
    bus = BroadcastBus()
    decisions = bus.subscribe("decision-test", event_types=(DecisionTrace,))
    engine = EdgeEngine(
        bus,
        (_spec(),),
        min_net_edge=Decimal("0.025"),
        slippage=Decimal("0.003"),
        safety_margin=Decimal("0.005"),
        order_notional=Decimal("2.50"),
        max_age_ms=1_500,
        clock_ns=lambda: now[0],
    )

    for second in range(9):
        now[0] = base + second * 1_000_000_000
        engine.process_event(
            _consensus(str(Decimal("100") + Decimal(second) / Decimal("100")), now[0])
        )

    traces = []
    while not decisions.empty():
        traces.append(decisions.get_nowait())

    assert engine.volatility.is_ready("BTCUSD")
    assert any(
        isinstance(item, DecisionTrace)
        and item.reason_code == "prediction_quote_missing"
        for item in traces
    )


def test_chainlink_opening_market_requires_fresh_matching_oracle_reference() -> None:
    base = 1_100_000_000_000
    now = [base]
    spec = BinaryMarketSpec(
        market_id="m-oracle",
        canonical_id="poly:m-oracle",
        venue=Venue.POLYMARKET,
        symbol="BTCUSD",
        yes_instrument="yes-oracle",
        no_instrument="no-oracle",
        strike=Decimal("0"),
        starts_at_ms=1_000_000,
        expires_at_ms=1_200_000,
        oracle="chainlink",
        oracle_window_seconds=60,
        strike_mode="opening_oracle",
        settlement_rule="gte_strike",
        taker_fee_rate=Decimal("0.07"),
        enabled=True,
    )
    bus = BroadcastBus()
    decisions = bus.subscribe("decision-oracle", event_types=(DecisionTrace,))
    engine = EdgeEngine(
        bus,
        (spec,),
        min_net_edge=Decimal("0.025"),
        slippage=Decimal("0.003"),
        safety_margin=Decimal("0.005"),
        order_notional=Decimal("2.50"),
        max_age_ms=1_500,
        oracle_reference_max_age_ms=5_000,
        clock_ns=lambda: now[0],
    )
    engine.opening_strikes[spec.market_id] = Decimal("100")

    for second in range(9):
        now[0] = base + second * 1_000_000_000
        engine.process_event(
            _consensus(str(Decimal("100") + Decimal(second) / Decimal("100")), now[0])
        )
    while not decisions.empty():
        decisions.get_nowait()

    engine.process_event(_consensus("100.08", now[0]))
    missing = decisions.get_nowait()
    assert missing.reason_code == "oracle_reference_missing"

    wrong_window = OraclePrice(
        venue=Venue.CHAINLINK,
        feed_id="btc/usd:30",
        symbol="BTCUSD",
        price=Decimal("100.05"),
        confidence=None,
        publish_ts_ms=now[0] // 1_000_000,
        recv_ts_ns=now[0],
        window_seconds=30,
    )
    engine.process_event(wrong_window)
    assert decisions.empty()

    stale = OraclePrice(
        venue=Venue.CHAINLINK,
        feed_id="btc/usd:60",
        symbol="BTCUSD",
        price=Decimal("100.05"),
        confidence=None,
        publish_ts_ms=now[0] // 1_000_000 - 6_000,
        recv_ts_ns=now[0] - 6_000_000_000,
        window_seconds=60,
    )
    engine.process_event(stale)
    stale_trace = decisions.get_nowait()
    assert stale_trace.reason_code == "oracle_reference_stale"

    fresh = OraclePrice(
        venue=Venue.CHAINLINK,
        feed_id="btc/usd:60",
        symbol="BTCUSD",
        price=Decimal("100.05"),
        confidence=None,
        publish_ts_ms=now[0] // 1_000_000,
        recv_ts_ns=now[0],
        window_seconds=60,
    )
    engine.process_event(fresh)
    fresh_trace = decisions.get_nowait()
    assert fresh_trace.reason_code == "prediction_quote_missing"
