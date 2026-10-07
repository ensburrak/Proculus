from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tools.v5_shadow_outcome_report import build_outcome_report


def _event(symbol: str = "BTC/USDT:USDT") -> dict:
    return {
        "schema": "autotraderbot-v5-breadth-shadow-v1",
        "family": "breadth_donchian_10_v5",
        "shadow_only": True,
        "execution_authority": False,
        "ready": True,
        "reason": "ready",
        "observed_at": "2026-10-01T04:05:00+00:00",
        "bar_time": "2026-10-01T00:00:00+00:00",
        "candidates": [
            {
                "family": "breadth_donchian_10_v5",
                "setup_id": "breadth_donchian_10_v5.entry.long.4h.v1",
                "symbol": symbol,
                "side": "long",
                "bar_time": "2026-10-01T00:00:00+00:00",
                "reference_price": 100.0,
                "atr_4h": 10.0,
                "stop_atr_mult": 2.0,
                "trail_atr_mult": 3.0,
                "breakeven_after_r": 0.5,
                "trail_after_r": 1.0,
                "decay_bars_15m": 192,
                "max_hold_bars_15m": 960,
                "hypothetical_risk_per_trade_pct": 0.0025,
                "hypothetical_wallet_cap_pct": 0.20,
                "hypothetical_leverage": 1.0,
                "shadow_only": True,
                "execution_authority": False,
            }
        ],
    }


def _frame() -> pd.DataFrame:
    start = pd.Timestamp("2026-10-01T04:00:00Z")
    return pd.DataFrame(
        {
            "timestamp": [start + pd.Timedelta(minutes=15 * i) for i in range(6)],
            "open": [100.0, 100.0, 120.0, 145.0, 112.0, 110.0],
            "high": [101.0, 125.0, 150.0, 146.0, 114.0, 111.0],
            "low": [99.0, 99.0, 118.0, 110.0, 108.0, 107.0],
            "close": [100.0, 120.0, 145.0, 112.0, 110.0, 108.0],
            "volume": [1000.0] * 6,
        }
    )


def _policy() -> dict:
    return {
        "family": "breadth_donchian_10_v5",
        "observation_gate": {
            "minimum_calendar_span_days": 0,
            "minimum_ready_4h_observations": 1,
            "minimum_candidate_events": 1,
            "minimum_candidate_symbols": 1,
            "require_zero_execution_authority_events": True,
            "maximum_recording_delay_minutes": 30,
        },
        "outcome_scoring": {
            "fee_bps_per_side": 5,
            "primary_slippage_bps_per_side": 5,
            "stress_slippage_bps_per_side": 15,
            "funding_bps_per_8h": 1,
        },
        "outcome_gate": {
            "primary_5bps": {
                "min_complete_trades": 1,
                "min_profit_factor": 0,
                "min_expectancy_r": -100,
            },
            "stress_15bps": {
                "min_complete_trades": 1,
                "min_profit_factor": 0,
                "min_expectancy_r_exclusive": -100,
            },
            "min_complete_symbols": 1,
        },
    }


def test_outcome_report_scores_only_prospective_safe_events(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _frame().to_parquet(data_dir / "BTC_USDT_USDT_15m.parquet", index=False)

    report = build_outcome_report([_event()], _policy(), data_dir)

    assert report["observation_gate"]["passed"] is True
    assert report["primary_5bps"]["complete_trades"] == 1
    assert report["stress_15bps"]["complete_trades"] == 1
    assert report["outcome_gate"]["passed"] is True
    assert report["execution_authority"] is False
    assert report["promotion_authority"] is False


def test_outcome_report_refuses_scoring_when_shadow_safety_is_violated(tmp_path: Path) -> None:
    event = _event()
    event["execution_authority"] = True

    report = build_outcome_report([event], _policy(), tmp_path)

    assert report["observation_gate"]["passed"] is False
    assert report["scoring_status"] == "blocked_shadow_safety_violation"
    assert report["primary_5bps"]["complete_trades"] == 0
    assert report["outcome_gate"]["passed"] is False
