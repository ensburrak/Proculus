import time
from decimal import Decimal

from hizlitrade.engine.fair_value import BinaryFairValueModel, RollingVolatility
from hizlitrade.models import BinaryMarketSpec, ConsensusPrice, Venue


def spec(strike: str) -> BinaryMarketSpec:
    return BinaryMarketSpec(
        market_id="m",
        canonical_id="c",
        venue=Venue.POLYMARKET,
        symbol="BTCUSD",
        yes_instrument="yes",
        no_instrument="no",
        strike=Decimal(strike),
        expires_at_ms=int(time.time() * 1000) + 60_000,
        oracle="chainlink",
        taker_fee_rate=Decimal("0.07"),
    )


def consensus(price: str, ts_ns: int) -> ConsensusPrice:
    return ConsensusPrice(
        "BTCUSD",
        Decimal(price),
        Decimal(0),
        (Venue.BINANCE, Venue.OKX),
        ts_ns,
    )


def test_price_far_above_strike_has_high_yes_probability() -> None:
    vol = RollingVolatility(floor=1e-5)
    model = BinaryFairValueModel(vol)
    spot = ConsensusPrice(
        "BTCUSD",
        Decimal("110000"),
        Decimal(0),
        (Venue.BINANCE, Venue.OKX),
        time.time_ns(),
    )
    p = model.probability_yes(spec("100000"), spot, int(time.time() * 1000))
    assert p > Decimal("0.95")


def test_explicit_oracle_reference_controls_fair_value_state() -> None:
    vol = RollingVolatility(floor=1e-5)
    model = BinaryFairValueModel(vol)
    spot = ConsensusPrice(
        "BTCUSD",
        Decimal("110000"),
        Decimal(0),
        (Venue.BINANCE, Venue.OKX),
        time.time_ns(),
    )
    now_ms = int(time.time() * 1000)

    spot_based = model.probability_yes(spec("100000"), spot, now_ms)
    oracle_based = model.probability_yes(
        spec("100000"),
        spot,
        now_ms,
        reference_price=Decimal("90000"),
    )

    assert spot_based > Decimal("0.95")
    assert oracle_based < Decimal("0.05")


def test_volatility_uses_fixed_time_buckets_instead_of_event_rate() -> None:
    vol = RollingVolatility(
        window_seconds=120,
        bucket_ms=1_000,
        min_samples=8,
        return_clip_bps=100,
    )
    base = 1_000_000_000_000

    # Hundreds of asynchronous sub-second updates must collapse into one
    # one-second close rather than each micro-tick becoming a volatility sample.
    for second in range(12):
        second_start = base + second * 1_000_000_000
        for tick in range(50):
            price = Decimal("100") + Decimal(second) / Decimal("100")
            price += Decimal(tick % 2) / Decimal("10000")
            vol.update(consensus(str(price), second_start + tick * 10_000_000))

    assert vol.is_ready("BTCUSD")
    assert vol.sample_count("BTCUSD") == 12
    assert vol.sigma_per_sqrt_second("BTCUSD") < 0.001


def test_volatility_ignores_out_of_order_observations() -> None:
    vol = RollingVolatility(
        window_seconds=120,
        bucket_ms=1_000,
        min_samples=2,
        return_clip_bps=100,
    )
    base = 2_000_000_000_000
    vol.update(consensus("100", base))
    vol.update(consensus("101", base + 1_000_000_000))
    vol.update(consensus("102", base + 2_000_000_000))
    sigma_before = vol.sigma_per_sqrt_second("BTCUSD")
    count_before = vol.sample_count("BTCUSD")

    vol.update(consensus("500", base + 500_000_000))

    assert vol.sample_count("BTCUSD") == count_before
    assert vol.sigma_per_sqrt_second("BTCUSD") == sigma_before


def test_volatility_warmup_is_explicit() -> None:
    vol = RollingVolatility(
        window_seconds=120,
        bucket_ms=1_000,
        min_samples=4,
        return_clip_bps=100,
    )
    base = 3_000_000_000_000
    for second in range(4):
        vol.update(consensus(str(100 + second), base + second * 1_000_000_000))

    assert vol.sample_count("BTCUSD") == 4
    assert not vol.is_ready("BTCUSD")
    assert vol.sigma_per_sqrt_second("BTCUSD") == vol.floor

    vol.update(consensus("104", base + 4_000_000_000))
    assert vol.sample_count("BTCUSD") == 5
    assert vol.is_ready("BTCUSD")
