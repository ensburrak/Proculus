from dataclasses import replace
from pathlib import Path

from hizlitrade.alpha_validation import (
    AdaptivePolicyConfig,
    PortfolioReplayConfig,
    adaptive_walk_forward_report,
    evaluate_policy_stages,
    fit_policy,
    replay_portfolio,
    technical_consensus_score,
)
from hizlitrade.oos import SettledTrade, load_research_trades


def test_adaptive_walk_forward_empty_dataset_is_explicit(tmp_path: Path) -> None:
    report = adaptive_walk_forward_report(
        tmp_path,
        min_train_markets=4,
        test_markets=2,
        step_markets=2,
        min_symbol_train_markets=1,
        min_ml_train_trades=2,
    )
    assert report["unique_markets"] == 0
    assert report["folds"] == []
    assert not report["qualified"]
    assert "insufficient_markets_for_first_walk_forward_fold" in report["blockers"]


def _write_rows(root: Path, name: str, rows: list[dict]) -> None:
    import json

    day = root / "2026-09-11"
    day.mkdir(parents=True, exist_ok=True)
    (day / name).write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _feature(
    market: int,
    ts: int,
    *,
    symbol: str = "BTCUSD",
    outcome: str = "YES",
    momentum: float = 8.0,
    basis: float = 10.0,
    spread: float = 0.01,
    imbalance: float = 0.25,
) -> dict:
    return {
        "event_type": "LeadLagFeature",
        "market_id": f"m{market}",
        "instrument": f"{outcome.lower()}-{market}",
        "outcome": outcome,
        "symbol": symbol,
        "created_ts_ns": ts,
        "spot_momentum_250ms_bps": str(momentum * 0.5),
        "spot_momentum_1s_bps": str(momentum),
        "spot_momentum_3s_bps": str(momentum * 1.25),
        "oracle_basis_bps": str(basis),
        "prediction_spread": str(spread),
        "book_imbalance": str(imbalance),
        "time_to_expiry_ms": 120_000,
    }


def _settlement(
    market: int,
    ts: int,
    pnl: float,
    *,
    symbol: str = "BTCUSD",
    outcome: str = "YES",
    net_edge: float = 0.04,
    market_duration_ms: int = 300_000,
) -> dict:
    return {
        "event_type": "ResearchSettlement",
        "market_id": f"m{market}",
        "instrument": f"{outcome.lower()}-{market}",
        "outcome": outcome,
        "winning_outcome": outcome if pnl > 0 else ("NO" if outcome == "YES" else "YES"),
        "shares": "10",
        "cost_basis_usd": "5",
        "realized_pnl_usd": str(pnl),
        "settled_ts_ns": ts + 5_000_000_000,
        "metadata": {
            "symbol": symbol,
            "signal_created_ts_ns": ts,
            "fair_probability": "0.60",
            "net_edge": str(net_edge),
            "signal_executable_price": "0.50",
            "market_duration_ms": market_duration_ms,
        },
    }


def test_load_settled_trades_preserves_rich_point_in_time_features(tmp_path: Path) -> None:
    ts = 2_000_000_000
    _write_rows(tmp_path, "leadlagfeature.ndjson", [_feature(1, ts - 100_000_000)])
    _write_rows(tmp_path, "papersettlement.ndjson", [_settlement(1, ts, 1.0)])

    trade = load_research_trades(tmp_path)[0]

    assert trade.outcome == "YES"
    assert trade.momentum_250ms_bps == 4.0
    assert trade.momentum_1s_bps == 8.0
    assert trade.momentum_3s_bps == 10.0
    assert trade.oracle_basis_bps == 10.0
    assert trade.prediction_spread == 0.01
    assert trade.book_imbalance == 0.25
    assert trade.time_to_expiry_ms == 120_000
    assert trade.net_edge == 0.04


def test_technical_consensus_score_respects_outcome_direction(tmp_path: Path) -> None:
    ts = 2_000_000_000
    features = [
        _feature(1, ts - 100_000_000, outcome="YES", momentum=8, basis=10, imbalance=0.3),
        _feature(2, ts + 900_000_000, outcome="NO", momentum=-8, basis=-10, imbalance=-0.3),
    ]
    settlements = [
        _settlement(1, ts, 1.0, outcome="YES"),
        _settlement(2, ts + 1_000_000_000, 1.0, outcome="NO"),
    ]
    _write_rows(tmp_path, "leadlagfeature.ndjson", features)
    _write_rows(tmp_path, "papersettlement.ndjson", settlements)

    yes_trade, no_trade = load_research_trades(tmp_path)

    assert technical_consensus_score(yes_trade) == 1.0
    assert technical_consensus_score(no_trade) == 1.0


def test_adaptive_walk_forward_fits_symbol_selection_only_on_train(tmp_path: Path) -> None:
    features: list[dict] = []
    settlements: list[dict] = []
    for market in range(18):
        ts = (market + 1) * 10_000_000_000
        symbol = "BTCUSD" if market % 2 == 0 else "ETHUSD"
        in_future_test = market >= 12
        pnl = 1.0 if symbol == "BTCUSD" or in_future_test else -1.0
        momentum = 9.0 if pnl > 0 else -4.0
        basis = 11.0 if pnl > 0 else -3.0
        features.append(
            _feature(
                market,
                ts - 100_000_000,
                symbol=symbol,
                momentum=momentum,
                basis=basis,
            )
        )
        settlements.append(_settlement(market, ts, pnl, symbol=symbol))

    _write_rows(tmp_path, "leadlagfeature.ndjson", features)
    _write_rows(tmp_path, "papersettlement.ndjson", settlements)

    report = adaptive_walk_forward_report(
        tmp_path,
        # Keep at least five per-symbol training markets so the bootstrap
        # confidence-bound selector has enough evidence to choose a subset.
        min_train_markets=10,
        test_markets=4,
        step_markets=4,
        min_symbol_train_markets=3,
        min_ml_train_trades=6,
    )

    assert len(report["folds"]) == 2
    first = report["folds"][0]
    assert first["train_end_ts_ns"] < first["test_start_ts_ns"]
    assert first["policy"]["selected_symbols"] == ["BTCUSD"]
    assert "ETHUSD" not in first["policy"]["selected_symbols"]
    assert set(first["ablation"]) == {
        "baseline",
        "dynamic_symbol",
        "setup_alignment",
        "regime_veto",
        "technical_consensus",
        "execution_quality",
        "point_in_time_ml",
        "strict_edge_evidence",
    }


def test_adaptive_walk_forward_never_overlaps_train_and_test_markets(tmp_path: Path) -> None:
    features: list[dict] = []
    settlements: list[dict] = []
    for market in range(20):
        ts = (market + 1) * 10_000_000_000
        pnl = 1.0 if market % 3 else -0.5
        features.append(_feature(market, ts - 100_000_000))
        settlements.append(_settlement(market, ts, pnl))

    _write_rows(tmp_path, "leadlagfeature.ndjson", features)
    _write_rows(tmp_path, "papersettlement.ndjson", settlements)

    report = adaptive_walk_forward_report(
        tmp_path,
        min_train_markets=8,
        test_markets=4,
        step_markets=4,
        min_symbol_train_markets=2,
        min_ml_train_trades=6,
    )

    for fold in report["folds"]:
        assert set(fold["train_market_ids"]).isdisjoint(fold["test_market_ids"])


def test_setup_family_selection_is_train_only(tmp_path: Path) -> None:
    features: list[dict] = []
    settlements: list[dict] = []
    for market in range(20):
        ts = (market + 1) * 10_000_000_000
        is_five_minute = market % 2 == 0
        duration = 300_000 if is_five_minute else 900_000
        in_future_test = market >= 12
        pnl = 1.0 if is_five_minute or in_future_test else -1.0
        features.append(
            _feature(
                market,
                ts - 100_000_000,
                momentum=8.0 if pnl > 0 else -8.0,
                basis=10.0 if pnl > 0 else -10.0,
            )
        )
        settlements.append(
            _settlement(
                market,
                ts,
                pnl,
                market_duration_ms=duration,
            )
        )

    _write_rows(tmp_path, "leadlagfeature.ndjson", features)
    _write_rows(tmp_path, "researchsettlement.ndjson", settlements)

    report = adaptive_walk_forward_report(
        tmp_path,
        min_train_markets=12,
        test_markets=4,
        step_markets=4,
        min_symbol_train_markets=3,
        min_setup_train_markets=3,
        min_ml_train_trades=100,
    )

    first = report["folds"][0]
    assert first["policy"]["selected_setup_families"] == ["5m"]
    assert "15m" not in first["policy"]["selected_setup_families"]


def _trade(
    market_id: str,
    signal_ts_ns: int,
    settled_ts_ns: int,
    pnl: float,
    *,
    symbol: str = "BTCUSD",
    cost: float = 5.0,
    net_edge: float = 0.04,
) -> SettledTrade:
    return SettledTrade(
        market_id=market_id,
        instrument=f"yes-{market_id}",
        symbol=symbol,
        signal_ts_ns=signal_ts_ns,
        settled_ts_ns=settled_ts_ns,
        cost_basis_usd=cost,
        realized_pnl_usd=pnl,
        shares=10.0,
        fair_probability=0.60,
        outcome="YES",
        signal_net_edge=net_edge,
        time_to_expiry_ms=120_000.0,
        spot_dispersion_bps=None,
        volatility_sigma_per_sqrt_second=None,
        won=pnl > 0,
        momentum_1s_bps=None,
        oracle_basis_bps=None,
        prediction_spread=None,
    )


def test_portfolio_replay_applies_correlated_symbol_and_open_exposure_limits() -> None:
    trades = [
        _trade("m1", 10, 100, 1.0),
        _trade("m2", 10, 100, 1.0),
        _trade("m3", 10, 100, 1.0),
    ]

    result = replay_portfolio(
        trades,
        PortfolioReplayConfig(
            initial_balance=100.0,
            max_open_exposure_pct=0.10,
            max_market_exposure_pct=0.05,
            max_symbol_exposure_pct=0.10,
            max_concurrent_positions=8,
            max_daily_loss_pct=0.20,
            max_drawdown_pct=0.20,
        ),
    )

    assert len(result.accepted_trades) == 2
    assert result.realized_pnl_usd == 2.0
    assert result.rejection_counts["open_exposure_limit"] == 1


def test_portfolio_replay_uses_settlement_time_before_daily_loss_reentry() -> None:
    trades = [
        _trade("m1", 10, 20, -5.0),
        _trade("m2", 30, 40, 1.0),
    ]

    result = replay_portfolio(
        trades,
        PortfolioReplayConfig(
            initial_balance=100.0,
            max_open_exposure_pct=0.20,
            max_market_exposure_pct=0.10,
            max_symbol_exposure_pct=0.20,
            max_concurrent_positions=8,
            max_daily_loss_pct=0.05,
            max_drawdown_pct=0.20,
        ),
    )

    assert [trade.market_id for trade in result.accepted_trades] == ["m1"]
    assert result.final_cash == 95.0
    assert result.rejection_counts["daily_loss_kill_switch"] == 1


def test_portfolio_replay_prioritizes_higher_edge_when_signals_are_simultaneous() -> None:
    low = _trade("low", 10, 100, -5.0, net_edge=0.03)
    high = _trade("high", 10, 100, 1.0, net_edge=0.08)

    result = replay_portfolio(
        [low, high],
        PortfolioReplayConfig(
            initial_balance=100.0,
            max_open_exposure_pct=0.05,
            max_market_exposure_pct=0.05,
            max_symbol_exposure_pct=0.05,
            max_concurrent_positions=1,
            max_daily_loss_pct=0.20,
            max_drawdown_pct=0.20,
        ),
    )

    assert [trade.market_id for trade in result.accepted_trades] == ["high"]
    assert result.realized_pnl_usd == 1.0


def test_negative_symbol_evidence_ends_in_strict_no_trade() -> None:
    trades = [
        _trade(
            f"btc-{index}",
            10 + index,
            100 + index,
            -1.0,
            symbol="BTCUSD",
        )
        for index in range(6)
    ] + [
        _trade(
            f"eth-{index}",
            30 + index,
            130 + index,
            -1.0,
            symbol="ETHUSD",
        )
        for index in range(6)
    ]
    config = AdaptivePolicyConfig(
        min_symbol_train_markets=3,
        min_setup_train_markets=100,
        min_ml_train_trades=100,
    )

    policy = fit_policy(trades, config)
    stages = evaluate_policy_stages(trades, policy, config)

    assert policy.symbol_gate_available is True
    assert len(stages["dynamic_symbol"]) == len(trades)
    assert policy.strict_edge_evidence_passed is False
    assert stages["strict_edge_evidence"] == []


def test_insufficient_symbol_evidence_keeps_symbol_gate_unavailable() -> None:
    trades = [
        _trade("btc-1", 10, 20, -1.0, symbol="BTCUSD"),
        _trade("eth-1", 30, 40, -1.0, symbol="ETHUSD"),
    ]
    config = AdaptivePolicyConfig(
        min_symbol_train_markets=3,
        min_setup_train_markets=100,
        min_ml_train_trades=100,
    )

    policy = fit_policy(trades, config)
    stages = evaluate_policy_stages(trades, policy, config)

    assert policy.symbol_gate_available is False
    assert len(stages["dynamic_symbol"]) == len(trades)


def test_negative_setup_evidence_ends_in_strict_no_trade() -> None:
    trades = [
        replace(
            _trade(f"setup-{index}", 10 + index, 100 + index, -1.0),
            market_duration_ms=300_000.0,
        )
        for index in range(6)
    ]
    config = AdaptivePolicyConfig(
        min_symbol_train_markets=100,
        min_setup_train_markets=3,
        min_ml_train_trades=100,
    )

    policy = fit_policy(trades, config)
    stages = evaluate_policy_stages(trades, policy, config)

    assert policy.symbol_gate_available is False
    assert policy.setup_gate_available is True
    assert len(stages["setup_alignment"]) == len(trades)
    assert policy.strict_edge_evidence_passed is False
    assert stages["strict_edge_evidence"] == []


def test_negative_regime_evidence_ends_in_strict_no_trade() -> None:
    trades = [
        replace(
            _trade(f"regime-{index}", 10 + index, 100 + index, -1.0),
            momentum_1s_bps=float(index + 1),
        )
        for index in range(12)
    ]
    config = AdaptivePolicyConfig(
        min_symbol_train_markets=100,
        min_setup_train_markets=100,
        min_ml_train_trades=100,
    )

    policy = fit_policy(trades, config)
    stages = evaluate_policy_stages(trades, policy, config)

    assert policy.regime_gate_available is True
    assert len(stages["regime_veto"]) == len(trades)
    assert policy.strict_edge_evidence_passed is False
    assert stages["strict_edge_evidence"] == []


def test_negative_consensus_evidence_ends_in_strict_no_trade() -> None:
    trades = [
        replace(
            _trade(f"consensus-{index}", 10 + index, 100 + index, -1.0),
            momentum_1s_bps=5.0,
            oracle_basis_bps=5.0,
            book_imbalance=0.25,
        )
        for index in range(6)
    ]
    config = AdaptivePolicyConfig(
        min_symbol_train_markets=100,
        min_setup_train_markets=100,
        min_ml_train_trades=100,
    )

    policy = fit_policy(trades, config)
    stages = evaluate_policy_stages(trades, policy, config)

    assert policy.regime_gate_available is False
    assert policy.consensus_gate_available is True
    assert len(stages["technical_consensus"]) == len(trades)
    assert policy.strict_edge_evidence_passed is False
    assert stages["strict_edge_evidence"] == []


def test_negative_execution_evidence_ends_in_strict_no_trade() -> None:
    trades = [
        _trade(f"execution-{index}", 10 + index, 100 + index, -1.0)
        for index in range(6)
    ]
    config = AdaptivePolicyConfig(
        min_symbol_train_markets=100,
        min_setup_train_markets=100,
        min_ml_train_trades=100,
    )

    policy = fit_policy(trades, config)
    stages = evaluate_policy_stages(trades, policy, config)

    assert policy.consensus_gate_available is False
    assert policy.execution_gate_available is True
    assert len(stages["execution_quality"]) == len(trades)
    assert policy.strict_edge_evidence_passed is False
    assert stages["strict_edge_evidence"] == []


def test_negative_ml_evidence_ends_in_strict_no_trade() -> None:
    trades = [
        replace(
            _trade(
                f"ml-{index}",
                10 + index,
                100 + index,
                0.10 if index == 0 else -1.0,
            ),
            signal_net_edge=None,
        )
        for index in range(6)
    ]
    config = AdaptivePolicyConfig(
        min_symbol_train_markets=100,
        min_setup_train_markets=100,
        min_ml_train_trades=6,
    )

    policy = fit_policy(trades, config)
    stages = evaluate_policy_stages(trades, policy, config)

    assert policy.execution_gate_available is False
    assert policy.ml_gate_available is True
    assert len(stages["point_in_time_ml"]) == len(trades)
    assert policy.strict_edge_evidence_passed is False
    assert stages["strict_edge_evidence"] == []
