from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from runtime.core_trend_shadow import (
    append_core_trend_shadow_event,
    build_core_trend_shadow_decision,
)


def _frame(*, rising: bool = True, rows: int = 260) -> pd.DataFrame:
    values = [100.0 + i * 0.25 for i in range(rows)]
    if not rising:
        values = list(reversed(values))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="4h", tz="UTC"),
            "open": values,
            "high": [v * 1.01 for v in values],
            "low": [v * 0.99 for v in values],
            "close": values,
            "volume": [1_000_000.0] * rows,
        }
    )


def _config() -> dict:
    return {
        "core_trend_shadow": {
            "enabled": True,
            "shadow_only": True,
            "order_authorized": False,
            "strategy_id": "core_trend_4h_ema50_200_voltarget.v1",
            "universe": ["BTC", "ETH"],
            "ema_fast": 50,
            "ema_slow": 200,
            "vol_window_bars": 180,
            "target_annualized_vol": 0.20,
            "per_symbol_exposure_cap": 1.0,
            "min_closed_bars": 220,
        }
    }


def test_core_trend_shadow_emits_long_without_order_authority() -> None:
    decision = build_core_trend_shadow_decision(
        {"symbol": "BTC/USDT:USDT", "mtf_data": {"4h": _frame(rising=True)}},
        _config(),
    )

    assert decision["status"] == "shadow_signal"
    assert decision["desired_direction"] == "long"
    assert decision["target_exposure"] > 0
    assert decision["shadow_only"] is True
    assert decision["order_authorized"] is False
    assert decision["research_contract"]["live_release"] is False


def test_core_trend_shadow_emits_short_without_order_authority() -> None:
    decision = build_core_trend_shadow_decision(
        {"symbol": "ETH/USDT:USDT", "mtf_data": {"4h": _frame(rising=False)}},
        _config(),
    )

    assert decision["status"] == "shadow_signal"
    assert decision["desired_direction"] == "short"
    assert decision["target_exposure"] < 0
    assert decision["order_authorized"] is False


def test_core_trend_shadow_fails_closed_for_short_history() -> None:
    decision = build_core_trend_shadow_decision(
        {"symbol": "BTC/USDT:USDT", "mtf_data": {"4h": _frame(rows=100)}},
        _config(),
    )

    assert decision["status"] == "insufficient_data"
    assert decision["desired_direction"] == "neutral"
    assert decision["target_exposure"] == 0.0
    assert decision["order_authorized"] is False


def test_core_trend_shadow_rejects_symbol_outside_research_universe() -> None:
    decision = build_core_trend_shadow_decision(
        {"symbol": "MEME/USDT:USDT", "mtf_data": {"4h": _frame()}},
        _config(),
    )

    assert decision["status"] == "not_eligible"
    assert decision["order_authorized"] is False


def test_core_trend_shadow_appends_forward_evidence(tmp_path: Path) -> None:
    destination = tmp_path / "shadow.jsonl"
    decision = build_core_trend_shadow_decision(
        {"symbol": "BTC/USDT:USDT", "mtf_data": {"4h": _frame()}},
        _config(),
    )

    append_core_trend_shadow_event(decision, destination=destination)
    rows = destination.read_text(encoding="utf-8").splitlines()

    assert len(rows) == 1
    payload = json.loads(rows[0])
    assert payload["strategy_id"] == "core_trend_4h_ema50_200_voltarget.v1"
    assert payload["order_authorized"] is False
    assert payload["recorded_at"]
