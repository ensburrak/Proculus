import urllib.parse
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
        moneyness_bps=0.0,
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


def test_bulk_gamma_discovery_batches_repeated_slug_filters(monkeypatch) -> None:
    calls: list[str] = []

    def fake_fetch(url: str, attempts: int = 5):
        del attempts
        calls.append(url)
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        slugs = query.get("slug", [])
        return [
            {
                "markets": [
                    {"slug": slug, "closed": True}
                    for slug in slugs
                ]
            }
        ]

    monkeypatch.setattr(_MODULE, "fetch_json", fake_fetch)

    found = _MODULE.gamma_markets_for_slugs(
        {
            "btc-updown-5m-100",
            "eth-updown-15m-200",
            "sol-updown-5m-300",
        },
        batch_size=2,
    )

    assert set(found) == {
        "btc-updown-5m-100",
        "eth-updown-15m-200",
        "sol-updown-5m-300",
    }
    assert len(calls) == 2
    first_query = urllib.parse.parse_qs(urllib.parse.urlsplit(calls[0]).query)
    second_query = urllib.parse.parse_qs(urllib.parse.urlsplit(calls[1]).query)
    assert len(first_query["slug"]) == 2
    assert len(second_query["slug"]) == 1


def test_batch_price_history_groups_tokens_without_fabricating_rows(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_post(url: str, payload: dict, attempts: int = 5):
        del attempts
        assert url.endswith("/batch-prices-history")
        calls.append(payload)
        return {
            "history": {
                token: [
                    {"t": 100, "p": "0.40"},
                    {"t": 110, "p": "0.60"},
                ]
                for token in payload["markets"]
            }
        }

    monkeypatch.setattr(_MODULE, "post_json", fake_post)

    found = _MODULE.batch_instrument_histories(
        {"token-a", "token-b", "token-c"},
        start_ts=90,
        end_ts=120,
        batch_size=2,
    )

    assert len(calls) == 2
    assert calls[0]["start_ts"] == 90
    assert calls[0]["end_ts"] == 120
    assert calls[0]["fidelity"] == 1
    assert set(found) == {"token-a", "token-b", "token-c"}
    assert found["token-a"] == [(100, 0.4), (110, 0.6)]


def test_gamma_discovery_reuses_settled_market_cache(tmp_path: Path, monkeypatch) -> None:
    slug = "btc-updown-5m-100"
    network_calls = 0

    def fake_fetch(url: str, attempts: int = 5):
        del attempts
        nonlocal network_calls
        network_calls += 1
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        requested = query.get("slug", [])
        return [
            {
                "markets": [
                    {
                        "slug": item,
                        "closed": True,
                        "outcomes": '["Up", "Down"]',
                        "outcomePrices": '["1", "0"]',
                    }
                    for item in requested
                ]
            }
        ]

    monkeypatch.setattr(_MODULE, "fetch_json", fake_fetch)
    first = _MODULE.gamma_markets_for_slugs(
        {slug},
        cache_dir=tmp_path,
        workers=2,
    )
    assert slug in first
    assert network_calls == 1

    def fail_fetch(*_args, **_kwargs):
        raise AssertionError("settled Gamma cache should avoid network")

    monkeypatch.setattr(_MODULE, "fetch_json", fail_fetch)
    second = _MODULE.gamma_markets_for_slugs(
        {slug},
        cache_dir=tmp_path,
        workers=2,
    )
    assert second[slug]["slug"] == slug


def test_batch_price_history_reuses_immutable_token_cache(tmp_path: Path, monkeypatch) -> None:
    network_calls = 0

    def fake_post(url: str, payload: dict, attempts: int = 5):
        del attempts
        nonlocal network_calls
        network_calls += 1
        assert url.endswith("/batch-prices-history")
        return {
            "history": {
                token: [
                    {"t": 100, "p": "0.40"},
                    {"t": 110, "p": "0.60"},
                ]
                for token in payload["markets"]
            }
        }

    monkeypatch.setattr(_MODULE, "post_json", fake_post)
    first = _MODULE.batch_instrument_histories(
        {"token-a", "token-b"},
        start_ts=90,
        end_ts=120,
        cache_dir=tmp_path,
        workers=2,
    )
    assert set(first) == {"token-a", "token-b"}
    assert network_calls == 1

    def fail_post(*_args, **_kwargs):
        raise AssertionError("cached closed-market token history should avoid network")

    monkeypatch.setattr(_MODULE, "post_json", fail_post)
    second = _MODULE.batch_instrument_histories(
        {"token-a", "token-b"},
        start_ts=90,
        end_ts=120,
        cache_dir=tmp_path,
        workers=2,
    )
    assert second == first


def test_gamma_successful_batch_does_not_refetch_absent_slugs_one_by_one(monkeypatch) -> None:
    present = "btc-updown-5m-100"
    absent = "btc-updown-5m-200"
    fallback_calls = 0

    def fake_fetch(url: str, attempts: int = 5):
        del attempts
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        requested = query.get("slug", [])
        assert present in requested
        assert absent in requested
        return [
            {
                "markets": [
                    {
                        "slug": present,
                        "closed": True,
                        "outcomes": '["Up", "Down"]',
                        "outcomePrices": '["1", "0"]',
                    }
                ]
            }
        ]

    def fallback(_slug: str):
        nonlocal fallback_calls
        fallback_calls += 1
        return None

    monkeypatch.setattr(_MODULE, "fetch_json", fake_fetch)
    monkeypatch.setattr(_MODULE, "gamma_market_for_slug", fallback)

    result = _MODULE.gamma_markets_for_slugs(
        {present, absent},
        workers=2,
    )

    assert set(result) == {present}
    assert fallback_calls == 0


def test_nested_validation_split_is_chronological_and_purges_late_labels() -> None:
    from hizlitrade.oos import MarketOutcome

    markets = tuple(
        MarketOutcome(
            market_id=f"m{index}",
            symbol="ZECUSD",
            signal_ts_ns=index * 100,
            settled_ts_ns=(index * 100) + (450 if index == 5 else 50),
            trade_count=1,
            cost_basis_usd=2.5,
            realized_pnl_usd=1.0,
            realized_return=0.4,
            mean_abs_momentum_1s_bps=None,
            mean_abs_oracle_basis_bps=None,
        )
        for index in range(12)
    )

    split = _MODULE.nested_train_validation_split(
        markets,
        min_fit_markets=5,
        min_validation_markets=4,
    )

    assert split is not None
    fit_ids, validation_ids, validation_start_ts_ns, purged = split
    assert len(validation_ids) >= 4
    assert set(fit_ids).isdisjoint(validation_ids)
    assert max(
        row.settled_ts_ns for row in markets if row.market_id in set(fit_ids)
    ) < validation_start_ts_ns
    assert "m5" in purged


def test_nested_validation_requires_positive_market_ci() -> None:
    def settled(market_id: str, index: int, pnl: float):
        return _MODULE.SettledTrade(
            market_id=market_id,
            instrument=f"{market_id}:YES",
            symbol="ZECUSD",
            signal_ts_ns=index,
            settled_ts_ns=index + 1,
            cost_basis_usd=1.0,
            realized_pnl_usd=pnl,
            shares=1.0,
            fair_probability=None,
            outcome="YES",
            signal_net_edge=None,
            time_to_expiry_ms=None,
            spot_dispersion_bps=None,
            volatility_sigma_per_sqrt_second=None,
            won=pnl > 0,
            momentum_1s_bps=None,
            oracle_basis_bps=None,
            prediction_spread=None,
        )

    positive = [
        settled(f"p{index}", index, 0.4)
        for index in range(10)
    ]
    mixed = [
        settled(f"m{index}", index, 0.4 if index < 5 else -0.4)
        for index in range(10)
    ]

    assert _MODULE.nested_validation_passes(positive, min_markets=8)
    assert not _MODULE.nested_validation_passes(mixed, min_markets=8)
