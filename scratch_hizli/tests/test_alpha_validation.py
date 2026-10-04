from pathlib import Path

import pytest

import hizlitrade.alpha_validation as alpha_validation
from hizlitrade.alpha_validation import (
    AdaptivePolicyConfig,
    adaptive_walk_forward_report,
    evaluate_policy_stages,
    fit_policy,
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
        _feature(2, ts + 900_000_000, outcome="NO", momentum=-8, basis=-10, imbalance=0.3),
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
        "session_filter",
        "regime_veto",
        "technical_consensus",
        "execution_quality",
        "realized_calibration",
        "point_in_time_ml",
        "alpha_evidence_gate",
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


def test_adaptive_walk_forward_excludes_other_config_epochs(tmp_path: Path) -> None:
    features: list[dict] = []
    settlements: list[dict] = []
    for market in range(12):
        ts = (market + 1) * 10_000_000_000
        feature = _feature(market, ts - 100_000_000)
        feature["evidence_config_sha256"] = "alpha-a"
        feature["run_id"] = "run-a"
        features.append(feature)
        settlement = _settlement(market, ts, 1.0)
        settlement["evidence_config_sha256"] = "alpha-a" if market < 10 else "alpha-b"
        settlement["run_id"] = "run-a" if market < 10 else "run-b"
        settlements.append(settlement)

    _write_rows(tmp_path, "leadlagfeature.ndjson", features)
    _write_rows(tmp_path, "researchsettlement.ndjson", settlements)

    report = adaptive_walk_forward_report(
        tmp_path,
        min_train_markets=4,
        test_markets=2,
        step_markets=2,
        min_symbol_train_markets=1,
        min_setup_train_markets=1,
        min_ml_train_trades=100,
        required_evidence_config_sha256="alpha-a",
    )

    assert report["settled_trades_all_configs"] == 12
    assert report["settled_trades"] == 10
    assert report["excluded_config_mismatch_or_legacy_trades"] == 2
    assert report["required_evidence_config_sha256"] == "alpha-a"


def test_adaptive_ablation_confidence_uses_market_grain(tmp_path: Path) -> None:
    features: list[dict] = []
    settlements: list[dict] = []
    for market in range(14):
        ts = (market + 1) * 10_000_000_000
        features.append(_feature(market, ts - 100_000_000))
        first = _settlement(market, ts, 1.0)
        second = _settlement(market, ts + 1, -0.25)
        second["instrument"] = f"yes-extra-{market}"
        settlements.extend((first, second))

    _write_rows(tmp_path, "leadlagfeature.ndjson", features)
    _write_rows(tmp_path, "researchsettlement.ndjson", settlements)

    report = adaptive_walk_forward_report(
        tmp_path,
        min_train_markets=8,
        test_markets=2,
        step_markets=2,
        min_symbol_train_markets=2,
        min_setup_train_markets=2,
        min_ml_train_trades=100,
    )

    assert report["folds"]
    baseline = report["folds"][0]["ablation"]["baseline"]
    assert baseline["unique_markets"] == 2
    assert baseline["return_stats"]["count"] == 2
    assert baseline["trade_return_stats"]["count"] == 4


def test_train_only_calibration_removes_overconfident_marginal_edge() -> None:
    from hizlitrade.oos import SettledTrade

    def row(index: int, *, pnl: float) -> SettledTrade:
        return SettledTrade(
            market_id=f"m{index}",
            instrument=f"yes-{index}",
            symbol="BTCUSD",
            signal_ts_ns=(index + 1) * 10_000_000_000,
            settled_ts_ns=(index + 1) * 10_000_000_000 + 5_000_000_000,
            cost_basis_usd=1.0,
            realized_pnl_usd=pnl,
            shares=1.0,
            fair_probability=0.90,
            won=pnl > 0,
            momentum_1s_bps=2.0,
            oracle_basis_bps=2.0,
            prediction_spread=0.01,
            outcome="YES",
            momentum_250ms_bps=1.0,
            momentum_3s_bps=3.0,
            book_imbalance=0.2,
            time_to_expiry_ms=120_000,
            net_edge=0.03,
            signal_executable_price=0.50,
            market_duration_ms=300_000,
        )

    train = [row(index, pnl=1.0 if index >= 8 else -0.2) for index in range(10)]
    candidate = row(99, pnl=1.0)
    config = AdaptivePolicyConfig(
        min_symbol_train_markets=100,
        min_setup_train_markets=100,
        min_ml_train_trades=100,
        min_calibration_bin_samples=10,
        min_calibrated_net_edge=0.025,
    )

    policy = fit_policy(train, config)
    assert policy.probability_calibration is not None

    stages = evaluate_policy_stages([candidate], policy, config)

    assert stages["execution_quality"] == [candidate]
    assert stages["realized_calibration"] == []
    assert stages["point_in_time_ml"] == []



def test_adaptive_policy_abstains_without_positive_train_edge() -> None:
    from hizlitrade.oos import SettledTrade

    def row(index: int, pnl: float) -> SettledTrade:
        return SettledTrade(
            market_id=f"neg-{index}",
            instrument=f"yes-neg-{index}",
            symbol="BTCUSD",
            signal_ts_ns=(index + 1) * 10_000_000_000,
            settled_ts_ns=(index + 1) * 10_000_000_000 + 5_000_000_000,
            cost_basis_usd=1.0,
            realized_pnl_usd=pnl,
            shares=1.0,
            fair_probability=0.60,
            won=pnl > 0,
            momentum_1s_bps=2.0,
            oracle_basis_bps=2.0,
            prediction_spread=0.01,
            outcome="YES",
            momentum_250ms_bps=1.0,
            momentum_3s_bps=3.0,
            book_imbalance=0.2,
            time_to_expiry_ms=120_000,
            net_edge=0.03,
            signal_executable_price=0.50,
            market_duration_ms=300_000,
        )

    train = [row(index, -0.20) for index in range(24)]
    candidate = row(99, 1.0)
    config = AdaptivePolicyConfig(
        min_symbol_train_markets=100,
        min_setup_train_markets=100,
        min_ml_train_trades=100,
        min_calibration_bin_samples=100,
        min_final_train_markets=15,
        stability_windows=3,
        min_positive_stability_windows=2,
    )

    policy = fit_policy(train, config)
    stages = evaluate_policy_stages([candidate], policy, config)

    assert not policy.alpha_evidence_qualified
    assert policy.alpha_evidence_reason == "non_positive_train_edge"
    assert stages["point_in_time_ml"] == [candidate]
    assert stages["alpha_evidence_gate"] == []


def test_adaptive_policy_requires_minimum_point_one_r_expectancy() -> None:
    from hizlitrade.oos import SettledTrade

    def row(index: int, pnl: float) -> SettledTrade:
        return SettledTrade(
            market_id=f"weak-{index}",
            instrument=f"yes-weak-{index}",
            symbol="BTCUSD",
            signal_ts_ns=(index + 1) * 10_000_000_000,
            settled_ts_ns=(index + 1) * 10_000_000_000 + 5_000_000_000,
            cost_basis_usd=1.0,
            realized_pnl_usd=pnl,
            shares=1.0,
            fair_probability=0.60,
            won=pnl > 0,
            momentum_1s_bps=2.0,
            oracle_basis_bps=2.0,
            prediction_spread=0.01,
            outcome="YES",
            momentum_250ms_bps=1.0,
            momentum_3s_bps=3.0,
            book_imbalance=0.2,
            time_to_expiry_ms=120_000,
            net_edge=0.03,
            signal_executable_price=0.50,
            market_duration_ms=300_000,
        )

    train = [row(index, 0.05) for index in range(24)]
    candidate = row(99, 1.0)
    config = AdaptivePolicyConfig(
        min_symbol_train_markets=100,
        min_setup_train_markets=100,
        min_ml_train_trades=100,
        min_calibration_bin_samples=100,
        min_final_train_markets=15,
        min_alpha_mean_return=0.10,
        stability_windows=3,
        min_positive_stability_windows=2,
    )

    policy = fit_policy(train, config)
    stages = evaluate_policy_stages([candidate], policy, config)

    assert not policy.alpha_evidence_qualified
    assert policy.alpha_evidence_reason == "train_expectancy_below_minimum"
    assert policy.alpha_train_mean_return == pytest.approx(0.05)
    assert stages["alpha_evidence_gate"] == []


def test_adaptive_policy_requires_time_stable_positive_train_edge() -> None:
    from hizlitrade.oos import SettledTrade

    def row(index: int, pnl: float) -> SettledTrade:
        return SettledTrade(
            market_id=f"stable-{index}",
            instrument=f"yes-stable-{index}",
            symbol="BTCUSD",
            signal_ts_ns=(index + 1) * 10_000_000_000,
            settled_ts_ns=(index + 1) * 10_000_000_000 + 5_000_000_000,
            cost_basis_usd=1.0,
            realized_pnl_usd=pnl,
            shares=1.0,
            fair_probability=0.60,
            won=pnl > 0,
            momentum_1s_bps=2.0,
            oracle_basis_bps=2.0,
            prediction_spread=0.01,
            outcome="YES",
            momentum_250ms_bps=1.0,
            momentum_3s_bps=3.0,
            book_imbalance=0.2,
            time_to_expiry_ms=120_000,
            net_edge=0.03,
            signal_executable_price=0.50,
            market_duration_ms=300_000,
        )

    train = [row(index, 0.40) for index in range(24)]
    candidate = row(99, 1.0)
    config = AdaptivePolicyConfig(
        min_symbol_train_markets=100,
        min_setup_train_markets=100,
        min_ml_train_trades=100,
        min_calibration_bin_samples=100,
        min_final_train_markets=15,
        stability_windows=3,
        min_positive_stability_windows=2,
    )

    policy = fit_policy(train, config)
    stages = evaluate_policy_stages([candidate], policy, config)

    assert policy.alpha_evidence_qualified
    assert policy.alpha_evidence_reason == "qualified"
    assert policy.alpha_stability_positive_windows == 3
    assert policy.alpha_stability_total_windows == 3
    assert stages["alpha_evidence_gate"] == [candidate]



def test_joint_consensus_edge_challenger_finds_stable_interaction() -> None:
    from hizlitrade.oos import SettledTrade

    def row(
        index: int,
        *,
        pnl: float,
        aligned: bool,
        edge: float,
    ) -> SettledTrade:
        signed = 3.0 if aligned else -3.0
        return SettledTrade(
            market_id=f"joint-{index}",
            instrument=f"yes-joint-{index}",
            symbol="BTCUSD",
            signal_ts_ns=(index + 1) * 10_000_000_000,
            settled_ts_ns=(index + 1) * 10_000_000_000 + 5_000_000_000,
            cost_basis_usd=1.0,
            realized_pnl_usd=pnl,
            shares=1.0,
            fair_probability=0.60,
            won=pnl > 0,
            momentum_1s_bps=signed,
            oracle_basis_bps=signed,
            prediction_spread=None,
            outcome="YES",
            momentum_250ms_bps=signed,
            momentum_3s_bps=signed,
            book_imbalance=0.2 if aligned else -0.2,
            time_to_expiry_ms=120_000,
            net_edge=edge,
            signal_executable_price=0.50,
            market_duration_ms=300_000,
        )

    # Neither consensus alone nor edge alone has positive expectancy. The
    # profitable intersection is interleaved through time so it must survive
    # both the older fit partition and the newer inner-validation partition.
    train = []
    for group in range(30):
        base = group * 3
        train.extend(
            (
                row(base, pnl=0.60, aligned=True, edge=0.06),
                row(base + 1, pnl=-0.80, aligned=True, edge=0.03),
                row(base + 2, pnl=-0.80, aligned=False, edge=0.06),
            )
        )
    config = AdaptivePolicyConfig(
        min_symbol_train_markets=100,
        min_setup_train_markets=100,
        min_ml_train_trades=100,
        min_calibration_bin_samples=100,
        min_final_train_markets=15,
        stability_windows=3,
        min_positive_stability_windows=2,
        challenger_validation_fraction=0.33,
        min_challenger_validation_markets=10,
    )

    policy = fit_policy(train, config)

    assert policy.selection_mode == "joint_consensus_edge"
    assert policy.alpha_evidence_qualified
    assert policy.min_consensus >= 0.50
    assert policy.min_net_edge is not None
    assert policy.min_net_edge >= 0.06

    good = row(100, pnl=1.0, aligned=True, edge=0.06)
    low_edge = row(101, pnl=1.0, aligned=True, edge=0.03)
    misaligned = row(102, pnl=1.0, aligned=False, edge=0.06)
    stages = evaluate_policy_stages([good, low_edge, misaligned], policy, config)

    assert stages["alpha_evidence_gate"] == [good]


def test_joint_challenger_rejects_edge_that_fails_inner_validation() -> None:
    from hizlitrade.oos import SettledTrade

    def row(
        index: int,
        *,
        pnl: float,
        aligned: bool,
        edge: float,
    ) -> SettledTrade:
        signed = 3.0 if aligned else -3.0
        return SettledTrade(
            market_id=f"joint-fail-{index}",
            instrument=f"yes-joint-fail-{index}",
            symbol="BTCUSD",
            signal_ts_ns=(index + 1) * 10_000_000_000,
            settled_ts_ns=(index + 1) * 10_000_000_000 + 5_000_000_000,
            cost_basis_usd=1.0,
            realized_pnl_usd=pnl,
            shares=1.0,
            fair_probability=0.60,
            won=pnl > 0,
            momentum_1s_bps=signed,
            oracle_basis_bps=signed,
            prediction_spread=None,
            outcome="YES",
            momentum_250ms_bps=signed,
            momentum_3s_bps=signed,
            book_imbalance=0.2 if aligned else -0.2,
            time_to_expiry_ms=120_000,
            net_edge=edge,
            signal_executable_price=0.50,
            market_duration_ms=300_000,
        )

    train = []
    # Older fit partition contains a genuine-looking joint interaction.
    for group in range(20):
        base = group * 3
        train.extend(
            (
                row(base, pnl=0.60, aligned=True, edge=0.06),
                row(base + 1, pnl=-0.80, aligned=True, edge=0.03),
                row(base + 2, pnl=-0.80, aligned=False, edge=0.06),
            )
        )
    # Newer inner-validation partition reverses the candidate's edge.
    for group in range(10):
        base = 60 + group * 3
        train.extend(
            (
                row(base, pnl=-1.00, aligned=True, edge=0.06),
                row(base + 1, pnl=-0.10, aligned=True, edge=0.03),
                row(base + 2, pnl=-0.10, aligned=False, edge=0.06),
            )
        )

    config = AdaptivePolicyConfig(
        min_symbol_train_markets=100,
        min_setup_train_markets=100,
        min_session_train_markets=100,
        min_ml_train_trades=100,
        min_calibration_bin_samples=100,
        min_final_train_markets=15,
        stability_windows=3,
        min_positive_stability_windows=2,
        challenger_validation_fraction=0.33,
        min_challenger_validation_markets=10,
    )

    policy = fit_policy(train, config)

    assert policy.selection_mode != "joint_consensus_edge"
    assert policy.challenger_validation_reason in {
        "candidate_validation_ci_not_positive",
        "candidate_validation_mean_not_positive",
    }



def test_session_filter_learns_time_of_day_only_from_train_markets() -> None:
    from hizlitrade.oos import SettledTrade

    def row(index: int, *, hour_utc: int, pnl: float) -> SettledTrade:
        day_start_s = (20_000 + index) * 86_400
        signal_s = day_start_s + hour_utc * 3_600
        return SettledTrade(
            market_id=f"session-{hour_utc}-{index}",
            instrument=f"yes-session-{hour_utc}-{index}",
            symbol="BTCUSD",
            signal_ts_ns=signal_s * 1_000_000_000,
            settled_ts_ns=(signal_s + 300) * 1_000_000_000,
            cost_basis_usd=1.0,
            realized_pnl_usd=pnl,
            shares=1.0,
            fair_probability=0.60,
            won=pnl > 0,
            momentum_1s_bps=3.0,
            oracle_basis_bps=3.0,
            prediction_spread=None,
            outcome="YES",
            momentum_250ms_bps=2.0,
            momentum_3s_bps=4.0,
            book_imbalance=0.2,
            time_to_expiry_ms=120_000,
            net_edge=0.03,
            signal_executable_price=0.50,
            market_duration_ms=300_000,
        )

    train = [
        *[row(index, hour_utc=21, pnl=0.50) for index in range(30)],
        *[row(index + 30, hour_utc=18, pnl=-0.50) for index in range(30)],
    ]
    config = AdaptivePolicyConfig(
        min_symbol_train_markets=100,
        min_setup_train_markets=100,
        min_session_train_markets=20,
        min_ml_train_trades=100,
        min_calibration_bin_samples=100,
        min_final_train_markets=15,
        stability_windows=3,
        min_positive_stability_windows=2,
    )

    policy = fit_policy(train, config)

    assert policy.selected_sessions == ("utc_21_24",)
    assert policy.alpha_evidence_qualified

    good = row(100, hour_utc=21, pnl=1.0)
    bad_session = row(101, hour_utc=18, pnl=1.0)
    stages = evaluate_policy_stages([good, bad_session], policy, config)

    assert stages["session_filter"] == [good]
    assert stages["alpha_evidence_gate"] == [good]


def test_rejected_joint_challenger_does_not_fallback_to_unvalidated_sequential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trade = SettledTrade(
        market_id="m1",
        instrument="yes-m1",
        symbol="BTCUSD",
        signal_ts_ns=1_000_000_000,
        settled_ts_ns=2_000_000_000,
        cost_basis_usd=1.0,
        realized_pnl_usd=0.1,
        shares=1.0,
        fair_probability=0.6,
        won=True,
        momentum_1s_bps=None,
        oracle_basis_bps=None,
        prediction_spread=None,
        outcome="YES",
        market_duration_ms=300_000.0,
        net_edge=0.05,
    )
    config = AdaptivePolicyConfig(
        min_symbol_train_markets=1,
        min_setup_train_markets=1,
        min_session_train_markets=1,
        min_ml_train_trades=2,
        min_final_train_markets=1,
        stability_windows=1,
        min_positive_stability_windows=1,
        min_challenger_validation_markets=5,
    )

    monkeypatch.setattr(alpha_validation, "_fit_symbols", lambda *_args: ("BTCUSD",))
    monkeypatch.setattr(alpha_validation, "_fit_setup_families", lambda *_args: ("5m",))
    monkeypatch.setattr(alpha_validation, "_fit_sessions", lambda *_args: ("utc_00_03",))
    monkeypatch.setattr(
        alpha_validation,
        "_fit_regimes",
        lambda *_args: (None, None, ("unknown",)),
    )
    monkeypatch.setattr(
        alpha_validation,
        "_fit_joint_consensus_edge_gate",
        lambda *_args: (
            None,
            None,
            "candidate_validation_ci_not_positive",
            10,
            0.05,
            -0.01,
        ),
    )

    def unvalidated_fallback_must_not_run(*_args):
        raise AssertionError("same-sample sequential fallback must not run")

    monkeypatch.setattr(
        alpha_validation,
        "_fit_consensus_threshold",
        unvalidated_fallback_must_not_run,
    )
    monkeypatch.setattr(
        alpha_validation,
        "_fit_execution_quality",
        unvalidated_fallback_must_not_run,
    )
    monkeypatch.setattr(alpha_validation, "_fit_probability_calibration", lambda *_args: None)
    monkeypatch.setattr(alpha_validation, "_fit_logistic", lambda *_args: None)

    policy = fit_policy([trade], config)

    assert policy.selection_mode == "no_validated_challenger"
    assert policy.min_consensus == 0.0
    assert policy.min_net_edge is None
    assert policy.max_prediction_spread is None
    assert policy.challenger_validation_reason == "candidate_validation_ci_not_positive"
