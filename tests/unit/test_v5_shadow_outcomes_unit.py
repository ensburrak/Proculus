from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from research.v5_shadow_outcomes import (
    evaluate_outcome_gate,
    score_shadow_candidate,
    summarize_candidate_outcomes,
)


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


def _candidate(*, side: str = "long", bar_time: str = "2026-10-01T00:00:00Z") -> dict:
    return {
        "family": "breadth_donchian_10_v5",
        "setup_id": f"breadth_donchian_10_v5.entry.{side}.4h.v1",
        "symbol": "BTC/USDT:USDT",
        "side": side,
        "bar_time": bar_time,
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


def test_shadow_outcome_matches_v5_breakeven_then_trailing_contract() -> None:
    outcome = score_shadow_candidate(
        _candidate(),
        _frame(),
        fee_bps=5.0,
        slippage_bps=0.0,
        funding_bps_per_8h=0.0,
    )

    assert outcome["status"] == "complete"
    assert outcome["entry_time"] == "2026-10-01T04:00:00+00:00"
    assert outcome["entry_price"] == pytest.approx(100.0)
    assert outcome["exit_time"] == "2026-10-01T04:45:00+00:00"
    assert outcome["exit_reason"] == "v5_trailing_stop"
    assert outcome["exit_price"] == pytest.approx(115.0)
    assert outcome["r_value"] == pytest.approx(0.744625)


def test_shadow_outcome_stays_pending_when_future_window_is_incomplete() -> None:
    frame = _frame().iloc[:2].copy()
    outcome = score_shadow_candidate(
        _candidate(),
        frame,
        fee_bps=5.0,
        slippage_bps=5.0,
        funding_bps_per_8h=1.0,
    )

    assert outcome["status"] == "pending"
    assert outcome["r_value"] is None
    assert outcome["exit_reason"] == "insufficient_future_data"


def test_shadow_outcome_rejects_non_shadow_candidate() -> None:
    candidate = _candidate()
    candidate["execution_authority"] = True

    outcome = score_shadow_candidate(
        candidate,
        _frame(),
        fee_bps=5.0,
        slippage_bps=5.0,
        funding_bps_per_8h=1.0,
    )

    assert outcome["status"] == "invalid"
    assert outcome["exit_reason"] == "candidate_not_shadow_only"


def test_outcome_summary_counts_only_complete_unique_candidates() -> None:
    complete = {
        "status": "complete",
        "candidate_key": "a",
        "symbol": "BTC/USDT:USDT",
        "r_value": 0.5,
    }
    duplicate = dict(complete)
    pending = {
        "status": "pending",
        "candidate_key": "b",
        "symbol": "ETH/USDT:USDT",
        "r_value": None,
    }
    loss = {
        "status": "complete",
        "candidate_key": "c",
        "symbol": "ETH/USDT:USDT",
        "r_value": -0.25,
    }

    summary = summarize_candidate_outcomes([complete, duplicate, pending, loss])

    assert summary["complete_trades"] == 2
    assert summary["pending_trades"] == 1
    assert summary["profit_factor"] == pytest.approx(2.0)
    assert summary["expectancy_r"] == pytest.approx(0.125)
    assert summary["symbols"] == ["BTC/USDT:USDT", "ETH/USDT:USDT"]


def test_outcome_gate_requires_primary_stress_and_breadth() -> None:
    policy = {
        "outcome_gate": {
            "primary_5bps": {
                "min_complete_trades": 30,
                "min_profit_factor": 1.15,
                "min_expectancy_r": 0.05,
            },
            "stress_15bps": {
                "min_complete_trades": 20,
                "min_profit_factor": 1.05,
                "min_expectancy_r_exclusive": 0.0,
            },
            "min_complete_symbols": 4,
        }
    }
    primary = {
        "complete_trades": 35,
        "profit_factor": 1.30,
        "expectancy_r": 0.10,
        "symbols": ["BTC", "ETH", "SOL", "ADA"],
    }
    stress = {
        "complete_trades": 31,
        "profit_factor": 1.10,
        "expectancy_r": 0.03,
        "symbols": ["BTC", "ETH", "SOL", "ADA"],
    }

    result = evaluate_outcome_gate(primary, stress, policy)

    assert result["passed"] is True
    assert result["promotion_authority"] is False
    assert result["execution_authority"] is False
    assert all(result["checks"].values())


def test_shadow_outcome_does_not_score_across_missing_15m_bar() -> None:
    frame = _frame().drop(index=2).reset_index(drop=True)

    outcome = score_shadow_candidate(
        _candidate(),
        frame,
        fee_bps=5.0,
        slippage_bps=5.0,
        funding_bps_per_8h=1.0,
    )

    assert outcome["status"] == "pending"
    assert outcome["exit_reason"] == "market_data_gap"
    assert outcome["r_value"] is None
