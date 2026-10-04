from __future__ import annotations

from datetime import UTC, datetime, timedelta

from tools.core_trend_shadow_report import build_report


def _events(days: int = 35, symbols: int = 8, up: bool = True):
    rows = []
    start = datetime(2026, 1, 1, tzinfo=UTC)
    names = [f"S{i}/USDT:USDT" for i in range(symbols)]
    price = {name: 100.0 for name in names}
    for bar in range(days * 6 + 1):
        ts = start + timedelta(hours=4 * bar)
        for name in names:
            current = price[name]
            rows.append(
                {
                    "strategy_id": "core_trend_4h_ema50_200_voltarget.v1",
                    "policy_fingerprint_sha256": "a" * 64,
                    "symbol": name,
                    "status": "shadow_signal",
                    "closed_bar_timestamp": ts.isoformat(),
                    "entry_reference_price": current,
                    "target_exposure": 0.5 if up else -0.5,
                }
            )
            price[name] = current * (1.002 if up else 1.002)
    return rows


def _policy():
    return {
        "min_shadow_days": 30,
        "min_observations": 500,
        "min_symbols": 8,
        "min_profit_factor": 1.2,
        "min_avg_observation_bps": 0.0,
        "max_drawdown_pct": 15.0,
        "require_positive_ci95_low": True,
        "require_real_tca": True,
        "allow_live": False,
    }


def test_forward_report_can_promote_statistics_but_not_live_without_tca() -> None:
    report = build_report(
        events=_events(),
        policy=_policy(),
        tca=None,
        one_way_cost_bps=1.0,
        funding_bp_per_4h=0.0,
    )

    assert report["statistical_passed"] is True
    assert report["paper_candidate"] is True
    assert report["live_candidate"] is False
    assert "real_tca" in report["blockers"]
    assert "manual_live_release" in report["blockers"]


def test_forward_report_fails_negative_direction_edge() -> None:
    report = build_report(
        events=_events(up=False),
        policy=_policy(),
        tca={"scope": "real_paper_execution", "summary": {"fills": 100}},
        one_way_cost_bps=1.0,
        funding_bp_per_4h=0.0,
    )

    assert report["statistical_passed"] is False
    assert report["paper_candidate"] is False
    assert report["live_candidate"] is False
    assert report["metrics"]["avg_observation_bps"] < 0


def test_hypothetical_tca_never_satisfies_live_gate() -> None:
    policy = _policy()
    policy["allow_live"] = True
    report = build_report(
        events=_events(),
        policy=policy,
        tca={"scope": "historical_backtest_hypothetical", "summary": {"fills": 999}},
        one_way_cost_bps=1.0,
        funding_bp_per_4h=0.0,
    )

    assert report["real_tca"]["passed"] is False
    assert report["live_candidate"] is False
    assert "real_tca" in report["blockers"]


def test_mixed_policy_fingerprints_fail_closed() -> None:
    rows = _events()
    rows[-1]["policy_fingerprint_sha256"] = "b" * 64
    report = build_report(
        events=rows,
        policy=_policy(),
        tca={"scope": "real_paper_execution", "summary": {"fills": 100}},
        one_way_cost_bps=1.0,
        funding_bp_per_4h=0.0,
    )

    assert report["coverage"]["policy_consistent"] is False
    assert report["statistical_checks"]["policy_consistent"] is False
    assert report["statistical_passed"] is False
    assert report["live_candidate"] is False
    assert "policy_consistent" in report["blockers"]
