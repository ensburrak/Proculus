from decimal import Decimal

import pytest

from hizlitrade.config import Settings
from hizlitrade.models import Venue


def test_live_guard_is_off_by_default() -> None:
    with pytest.raises(RuntimeError):
        Settings(_env_file=None).assert_live_guard()


def test_live_guard_requires_exact_ack() -> None:
    settings = Settings(
        _env_file=None,
        live_trading=True,
        live_ack="YES_I_ACCEPT_REAL_LOSS",
        order_notional_usd=Decimal("2.5"),
        live_max_notional_usd=Decimal("5"),
    )
    settings.assert_live_guard()


def test_auto_research_universe_requires_multiple_reference_venues() -> None:
    settings = Settings(
        _env_file=None,
        min_spot_venues=2,
        binance_symbols="btcusdt,xrpusdt,dogeusdt",
        okx_instruments="BTC-USDT,XRP-USDT",
        coinbase_products="BTC-USD",
        hyperliquid_coins="BTC,XRP",
    )
    assert settings.research_symbols_list == ("BTCUSD", "XRPUSD")


def test_explicit_research_symbols_can_narrow_but_not_expand() -> None:
    settings = Settings(
        _env_file=None,
        research_symbols="XRPUSD,DOGEUSD",
        min_spot_venues=2,
        binance_symbols="xrpusdt,dogeusdt",
        okx_instruments="XRP-USDT",
        coinbase_products="",
        hyperliquid_coins="XRP",
    )
    assert settings.research_symbols_list == ("XRPUSD",)


def test_bare_hyperliquid_symbol_counts_as_usd_reference() -> None:
    settings = Settings(
        _env_file=None,
        research_symbols="XRP",
        min_spot_venues=2,
        binance_symbols="xrpusdt",
        okx_instruments="",
        coinbase_products="",
        hyperliquid_coins="XRP",
    )
    assert settings.research_symbols_list == ("XRPUSD",)


def test_polymarket_only_paper_execution_profile() -> None:
    settings = Settings(_env_file=None, paper_execution_venues="polymarket")
    assert settings.paper_execution_venues_list == (Venue.POLYMARKET,)


def test_spot_venue_cannot_be_selected_for_paper_prediction_execution() -> None:
    settings = Settings(_env_file=None, paper_execution_venues="binance")
    with pytest.raises(ValueError, match="not a prediction venue"):
        _ = settings.paper_execution_venues_list


def test_volatility_warmup_requires_at_least_two_return_samples() -> None:
    with pytest.raises(ValueError, match="volatility_min_samples"):
        Settings(_env_file=None, volatility_min_samples=1)



def test_default_paper_notional_is_one_percent_worst_case_risk() -> None:
    settings = Settings(_env_file=None)

    assert settings.paper_balance_usd == Decimal("250")
    assert settings.order_notional_usd == Decimal("2.50")
    assert settings.order_notional_usd / settings.paper_balance_usd == Decimal("0.01")


def test_oracle_reference_freshness_must_be_positive() -> None:
    with pytest.raises(ValueError, match="oracle_reference_max_age_ms"):
        Settings(_env_file=None, oracle_reference_max_age_ms=0)
