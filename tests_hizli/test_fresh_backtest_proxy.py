from pathlib import Path


def _load_backtest_module():
    import importlib.util
    import sys

    script = Path(__file__).resolve().parents[1] / "tools" / "fresh_hizlitrade_backtest.py"
    spec = importlib.util.spec_from_file_location("fresh_hizlitrade_backtest_tested", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_MODULE = _load_backtest_module()
Candidate = _MODULE.Candidate
Market = _MODULE.Market
SpotSeries = _MODULE.SpotSeries
market_candidates = _MODULE.market_candidates
simulate = _MODULE.simulate


def _candidate(
    *,
    ts: int,
    end: int,
    market_id: str,
    winner: str = "NO",
) -> Candidate:
    return Candidate(
        ts=ts,
        market_slug=f"btc-updown-5m-{market_id}",
        market_id=market_id,
        symbol="BTCUSD",
        end=end,
        outcome="YES",
        quote_price=0.40,
        fair=0.90,
        net_edge=0.40,
        winner=winner,
        history=[(ts, 0.40)],
        spot_price=100.0,
        strike=100.0,
        basis_bps=0.0,
        momentum_1m_bps=None,
        momentum_3m_bps=None,
        market_duration_ms=300_000.0,
    )


def test_market_candidates_never_forward_fill_opposite_outcome_history() -> None:
    market = Market(
        slug="btc-updown-5m-100",
        market_id="m1",
        symbol="BTCUSD",
        start=100,
        end=400,
        winner="YES",
        yes_instrument="yes",
        no_instrument="no",
        yes_history=[(100, 0.20)],
        no_history=[(110, 0.80)],
        volume=None,
    )
    spot = SpotSeries(
        times=[100, 110],
        prices=[100.0, 110.0],
        sigmas=[0.01, 0.01],
    )

    rows = market_candidates(market, spot)

    assert ("YES", 100) in {(row.outcome, row.ts) for row in rows}
    assert ("YES", 110) not in {(row.outcome, row.ts) for row in rows}
    assert ("NO", 100) not in {(row.outcome, row.ts) for row in rows}


def test_simulate_resets_daily_loss_baseline_on_utc_day_change() -> None:
    first = _candidate(ts=100, end=101, market_id="m1")
    second = _candidate(ts=200, end=201, market_id="m2")
    next_day = _candidate(ts=86_500, end=86_501, market_id="m3")

    result = simulate(
        [first, second, next_day],
        execution_slippage=0.0,
    )

    assert result["trades"] == 3
    assert result["final_cash_usd"] < 100.0


def test_bulk_gamma_discovery_walks_keyset_cursor(monkeypatch) -> None:
    calls: list[str] = []
    pages = [
        {
            "markets": [
                {"slug": "btc-updown-5m-100", "closed": True},
                {"slug": "other-market", "closed": True},
            ],
            "next_cursor": "cursor-1",
        },
        {
            "markets": [
                {"slug": "eth-updown-15m-200", "closed": True},
            ],
        },
    ]

    def fake_fetch(url: str, attempts: int = 5):
        del attempts
        calls.append(url)
        return pages.pop(0)

    monkeypatch.setattr(_MODULE, "fetch_json", fake_fetch)

    found = _MODULE.gamma_markets_for_slugs(
        {"btc-updown-5m-100", "eth-updown-15m-200"},
        cutoff_epoch=0,
        max_pages=5,
    )

    assert set(found) == {"btc-updown-5m-100", "eth-updown-15m-200"}
    assert len(calls) == 2
    assert "after_cursor=cursor-1" in calls[1]
