from __future__ import annotations

import json
from pathlib import Path

from tools.v5_shadow_evidence_report import build_report


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_v5_shadow_report_remains_blocked_without_trade_outcomes(tmp_path) -> None:
    events = tmp_path / "events.jsonl"
    trades = tmp_path / "trades.jsonl"
    config = tmp_path / "config.json"
    _write_jsonl(
        events,
        [
            {
                "family": "breadth_donchian_10_v5",
                "ready": True,
                "observed_at": "2026-10-01T00:00:00+00:00",
                "bar_time": "2026-09-30T20:00:00+00:00",
                "global_state": {"regime": "bull"},
                "candidates": [],
            },
            {
                "family": "breadth_donchian_10_v5",
                "ready": True,
                "observed_at": "2026-11-05T00:00:00+00:00",
                "bar_time": "2026-11-04T20:00:00+00:00",
                "global_state": {"regime": "neutral"},
                "candidates": [],
            },
        ],
    )
    config.write_text(
        json.dumps(
            {
                "observation_policy": {
                    "minimum_elapsed_days": 30,
                    "minimum_ready_4h_bars": 2,
                    "minimum_closed_hypothetical_trades": 1,
                    "minimum_profit_factor": 1.15,
                    "minimum_expectancy_r": 0.05,
                    "maximum_missed_signals": 0,
                    "maximum_invalidated_trades": 0,
                }
            }
        ),
        encoding="utf-8",
    )
    state = tmp_path / "v5_breadth_shadow_state.json"
    state.write_text(
        json.dumps(
            {
                "schema": "autotraderbot-v5-paper-shadow-state-v1",
                "closed_trades": 0,
                "missed_signals": 0,
                "invalidated_trades": 0,
                "open_positions": {},
                "blocked": {},
                "max_drawdown_pct": 0.0,
            }
        ),
        encoding="utf-8",
    )

    report = build_report(events, trades, config, state)

    assert report["checks"]["elapsed_days"] is True
    assert report["checks"]["ready_4h_bars"] is True
    assert report["checks"]["closed_hypothetical_trades"] is False
    assert report["prospective_shadow_passed"] is False
    assert report["execution_authority"] is False


def test_v5_shadow_report_can_pass_only_with_scored_forward_trades(tmp_path) -> None:
    events = tmp_path / "events.jsonl"
    trades = tmp_path / "trades.jsonl"
    config = tmp_path / "config.json"
    _write_jsonl(
        events,
        [
            {
                "family": "breadth_donchian_10_v5",
                "ready": True,
                "observed_at": "2026-10-01T00:00:00+00:00",
                "bar_time": "2026-09-30T20:00:00+00:00",
                "global_state": {"regime": "bull"},
                "candidates": [],
            },
            {
                "family": "breadth_donchian_10_v5",
                "ready": True,
                "observed_at": "2026-11-05T00:00:00+00:00",
                "bar_time": "2026-11-04T20:00:00+00:00",
                "global_state": {"regime": "bull"},
                "candidates": [],
            },
        ],
    )
    _write_jsonl(
        trades,
        [
            {"family": "breadth_donchian_10_v5", "closed": True, "net_r": 1.0},
            {"family": "breadth_donchian_10_v5", "closed": True, "net_r": 0.5},
            {"family": "breadth_donchian_10_v5", "closed": True, "net_r": -0.5},
        ],
    )
    config.write_text(
        json.dumps(
            {
                "observation_policy": {
                    "minimum_elapsed_days": 30,
                    "minimum_ready_4h_bars": 2,
                    "minimum_closed_hypothetical_trades": 3,
                    "minimum_profit_factor": 1.15,
                    "minimum_expectancy_r": 0.05,
                    "maximum_missed_signals": 0,
                    "maximum_invalidated_trades": 0,
                }
            }
        ),
        encoding="utf-8",
    )
    state = tmp_path / "v5_breadth_shadow_state.json"
    state.write_text(
        json.dumps(
            {
                "schema": "autotraderbot-v5-paper-shadow-state-v1",
                "closed_trades": 3,
                "missed_signals": 0,
                "invalidated_trades": 0,
                "open_positions": {},
                "blocked": {},
                "max_drawdown_pct": 1.0,
            }
        ),
        encoding="utf-8",
    )

    report = build_report(events, trades, config, state)

    assert report["trade_metrics"]["profit_factor"] == 3.0
    assert report["trade_metrics"]["expectancy_r"] > 0.05
    assert report["prospective_shadow_passed"] is True
    assert report["production_authority"] is False


def test_v5_shadow_report_blocks_invalidated_forward_sample(tmp_path) -> None:
    events = tmp_path / "events.jsonl"
    trades = tmp_path / "trades.jsonl"
    config = tmp_path / "config.json"
    state = tmp_path / "state.json"
    _write_jsonl(
        events,
        [
            {
                "family": "breadth_donchian_10_v5",
                "ready": True,
                "observed_at": "2026-10-01T00:00:00+00:00",
                "bar_time": "2026-09-30T20:00:00+00:00",
                "global_state": {"regime": "bull"},
                "candidates": [],
            },
            {
                "family": "breadth_donchian_10_v5",
                "ready": True,
                "observed_at": "2026-11-05T00:00:00+00:00",
                "bar_time": "2026-11-04T20:00:00+00:00",
                "global_state": {"regime": "bull"},
                "candidates": [],
            },
        ],
    )
    _write_jsonl(
        trades,
        [
            {"family": "breadth_donchian_10_v5", "closed": True, "net_r": 1.0},
            {"family": "breadth_donchian_10_v5", "closed": True, "net_r": 0.5},
            {"family": "breadth_donchian_10_v5", "closed": True, "net_r": -0.5},
        ],
    )
    config.write_text(
        json.dumps(
            {
                "observation_policy": {
                    "minimum_elapsed_days": 30,
                    "minimum_ready_4h_bars": 2,
                    "minimum_closed_hypothetical_trades": 3,
                    "minimum_profit_factor": 1.15,
                    "minimum_expectancy_r": 0.05,
                    "maximum_missed_signals": 0,
                    "maximum_invalidated_trades": 0,
                }
            }
        ),
        encoding="utf-8",
    )
    state.write_text(
        json.dumps(
            {
                "schema": "autotraderbot-v5-paper-shadow-state-v1",
                "closed_trades": 3,
                "missed_signals": 0,
                "invalidated_trades": 1,
                "open_positions": {},
                "blocked": {"unobserved_15m_gap": 1},
                "max_drawdown_pct": 1.0,
            }
        ),
        encoding="utf-8",
    )

    report = build_report(events, trades, config, state)

    assert report["checks"]["invalidated_trades"] is False
    assert report["paper_state"]["invalidated_trades"] == 1
    assert report["prospective_shadow_passed"] is False


def _passing_forward_sample(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    events = tmp_path / "events.jsonl"
    trades = tmp_path / "trades.jsonl"
    config = tmp_path / "config.json"
    state = tmp_path / "state.json"
    _write_jsonl(events, [
        {"family": "breadth_donchian_10_v5", "ready": True,
         "observed_at": "2026-10-01T00:00:00Z", "bar_time": "2026-09-30T20:00:00Z", "candidates": []},
        {"family": "breadth_donchian_10_v5", "ready": True,
         "observed_at": "2026-11-05T00:00:00Z", "bar_time": "2026-11-04T20:00:00Z", "candidates": []},
    ])
    _write_jsonl(trades, [
        {"family": "breadth_donchian_10_v5", "closed": True, "signal_key": "A", "net_r": 1.0},
        {"family": "breadth_donchian_10_v5", "closed": True, "signal_key": "B", "net_r": 0.5},
        {"family": "breadth_donchian_10_v5", "closed": True, "signal_key": "C", "net_r": -0.5},
    ])
    config.write_text(json.dumps({
        "observation_policy": {
            "minimum_elapsed_days": 30,
            "minimum_ready_4h_bars": 2,
            "minimum_closed_hypothetical_trades": 3,
            "minimum_profit_factor": 1.15,
            "minimum_expectancy_r": 0.05,
            "maximum_missed_signals": 0,
            "maximum_invalidated_trades": 0,
            "maximum_evidence_write_failures": 0,
        }
    }), encoding="utf-8")
    state.write_text(json.dumps({
        "schema": "autotraderbot-v5-paper-shadow-state-v1",
        "closed_trades": 3, "missed_signals": 0,
        "invalidated_trades": 0, "evidence_write_failures": 0,
        "open_positions": {}, "blocked": {},
    }), encoding="utf-8")
    return events, trades, config, state


def test_v5_shadow_report_rejects_missing_trade_evidence_even_if_pf_would_pass(tmp_path) -> None:
    events, trades, config, state = _passing_forward_sample(tmp_path)
    original = trades.read_text(encoding="utf-8").splitlines()
    trades.write_text("\n".join(original[:2]) + "\n", encoding="utf-8")
    report = build_report(events, trades, config, state)
    assert report["checks"]["trade_ledger_matches_state"] is False
    assert report["evidence_integrity"]["state_closed_trades"] == 3
    assert report["prospective_shadow_passed"] is False


def test_v5_shadow_report_rejects_malformed_jsonl_instead_of_skipping_loss(tmp_path) -> None:
    events, trades, config, state = _passing_forward_sample(tmp_path)
    trades.write_text(trades.read_text(encoding="utf-8") + "{bad-json\n", encoding="utf-8")
    report = build_report(events, trades, config, state)
    assert report["checks"]["trade_log_integrity"] is False
    assert report["evidence_integrity"]["invalid_trade_rows"] == 1
    assert report["prospective_shadow_passed"] is False


def test_v5_shadow_report_rejects_duplicate_trade_signal_keys(tmp_path) -> None:
    events, trades, config, state = _passing_forward_sample(tmp_path)
    rows = [json.loads(line) for line in trades.read_text(encoding="utf-8").splitlines()]
    rows[1]["signal_key"] = rows[0]["signal_key"]
    _write_jsonl(trades, rows)
    report = build_report(events, trades, config, state)
    assert report["checks"]["trade_log_integrity"] is False
    assert report["evidence_integrity"]["duplicate_signal_keys"] == 1
    assert report["prospective_shadow_passed"] is False


def test_v5_shadow_report_rejects_nonfinite_net_r(tmp_path) -> None:
    events, trades, config, state = _passing_forward_sample(tmp_path)
    rows = [json.loads(line) for line in trades.read_text(encoding="utf-8").splitlines()]
    rows[1]["net_r"] = float("nan")
    _write_jsonl(trades, rows)
    report = build_report(events, trades, config, state)
    assert report["checks"]["trade_log_integrity"] is False
    assert report["evidence_integrity"]["invalid_trade_records"] == 1
    assert report["prospective_shadow_passed"] is False


def test_v5_shadow_report_valid_forward_ledger_still_passes(tmp_path) -> None:
    events, trades, config, state = _passing_forward_sample(tmp_path)
    report = build_report(events, trades, config, state)
    assert report["checks"]["trade_log_integrity"] is True
    assert report["checks"]["trade_ledger_matches_state"] is True
    assert report["prospective_shadow_passed"] is True
    assert report["execution_authority"] is False
