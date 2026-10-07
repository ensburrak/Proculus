from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from atomic_io import flush_async_io_queue
from decision import v5_breadth_shadow
from decision.v5_breadth_shadow import (
    PREREGISTERED_SYMBOLS,
    evaluate_v5_breadth_shadow,
)


def _trend_frame(
    *,
    start: str = "2026-01-01T00:00:00Z",
    periods: int = 220,
    direction: str = "up",
) -> pd.DataFrame:
    ts = pd.date_range(start, periods=periods, freq="4h", tz="UTC")
    if direction == "up":
        close = np.linspace(100.0, 220.0, periods)
    else:
        close = np.linspace(220.0, 100.0, periods)
    open_ = close * (0.998 if direction == "up" else 1.002)
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    volume = np.linspace(1000.0, 1500.0, periods)
    return pd.DataFrame(
        {
            "timestamp_4h": ts.tz_convert(None),
            "open_4h": open_,
            "high_4h": high,
            "low_4h": low,
            "close_4h": close,
            "volume_4h": volume,
        }
    )


def _items(direction: str = "up", periods: int = 220) -> list[dict]:
    return [
        {
            "symbol": symbol,
            "price": 100.0,
            "mtf_data": {"4h": _trend_frame(periods=periods, direction=direction)},
        }
        for symbol in PREREGISTERED_SYMBOLS
    ]


def test_v5_shadow_fails_closed_when_symbol_missing() -> None:
    items = _items()
    items.pop()

    payload = evaluate_v5_breadth_shadow(
        items,
        now=datetime(2026, 2, 10, tzinfo=UTC),
    )

    assert payload["ready"] is False
    assert payload["reason"] == "missing_preregistered_symbols"
    assert payload["execution_authority"] is False


def test_v5_shadow_fails_closed_when_4h_history_is_insufficient() -> None:
    payload = evaluate_v5_breadth_shadow(
        _items(periods=100),
        now=datetime(2026, 2, 10, tzinfo=UTC),
    )

    assert payload["ready"] is False
    assert payload["reason"] == "insufficient_completed_4h_history"
    assert payload["execution_authority"] is False


def test_v5_shadow_uses_only_completed_4h_bars() -> None:
    items = _items()
    last_frame = items[0]["mtf_data"]["4h"]
    incomplete_open = pd.Timestamp(last_frame.iloc[-1]["timestamp_4h"]).tz_localize("UTC")
    now = incomplete_open + timedelta(hours=2)

    payload = evaluate_v5_breadth_shadow(items, now=now)

    assert payload["ready"] is True
    assert payload["bar_time"] != incomplete_open.isoformat()
    assert pd.Timestamp(payload["bar_time"]) == incomplete_open - pd.Timedelta(hours=4)


def test_v5_shadow_emits_long_candidates_for_bull_breadth() -> None:
    frame = _trend_frame()
    last_open = pd.Timestamp(frame.iloc[-1]["timestamp_4h"]).tz_localize("UTC")
    payload = evaluate_v5_breadth_shadow(
        _items(direction="up"),
        now=(last_open + pd.Timedelta(hours=5)).to_pydatetime(),
    )

    assert payload["ready"] is True
    assert payload["global_state"]["regime"] == "bull"
    assert payload["global_state"]["breadth_above_ema200"] >= 0.625
    assert payload["global_state"]["median_adx"] >= 18.0
    assert payload["candidates"]
    assert all(candidate["side"] == "long" for candidate in payload["candidates"])
    assert all(candidate["shadow_only"] is True for candidate in payload["candidates"])
    assert all(candidate["execution_authority"] is False for candidate in payload["candidates"])


def test_v5_shadow_normalizes_runtime_symbol_forms() -> None:
    items = _items()
    items[0]["symbol"] = "BTC/USDT"
    frame = items[0]["mtf_data"]["4h"]
    last_open = pd.Timestamp(frame.iloc[-1]["timestamp_4h"]).tz_localize("UTC")

    payload = evaluate_v5_breadth_shadow(
        items,
        now=(last_open + pd.Timedelta(hours=5)).to_pydatetime(),
    )

    assert payload["ready"] is True
    assert "BTC/USDT:USDT" in payload["symbols"]


def test_v5_shadow_recorder_dedupes_after_process_state_reset(tmp_path) -> None:
    path = tmp_path / "shadow.jsonl"
    payload = {
        "schema": "autotraderbot-v5-breadth-shadow-v1",
        "family": "breadth_donchian_10_v5",
        "shadow_only": True,
        "execution_authority": False,
        "ready": True,
        "reason": "ready",
        "observed_at": "2026-10-08T04:05:00+00:00",
        "bar_time": "2026-10-08T00:00:00+00:00",
        "candidates": [],
    }

    v5_breadth_shadow._LAST_RECORDED_KEY = None
    assert v5_breadth_shadow.record_v5_shadow_evidence(payload, path=path) is True
    assert flush_async_io_queue(timeout=2.0) is True

    # Simulate a fresh Task Scheduler process where in-memory dedupe is gone.
    v5_breadth_shadow._LAST_RECORDED_KEY = None
    assert v5_breadth_shadow.record_v5_shadow_evidence(payload, path=path) is True
    assert flush_async_io_queue(timeout=2.0) is True

    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
