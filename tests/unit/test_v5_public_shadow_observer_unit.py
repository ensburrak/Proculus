from __future__ import annotations

import pandas as pd
import pytest

from tools.v5_public_shadow_observer import (
    merge_candle_cache,
    parse_okx_candles,
    validate_candle_cadence,
)


def _row(ts: int, close: float, confirm: str = "1") -> list[str]:
    return [
        str(ts),
        str(close - 1),
        str(close + 1),
        str(close - 2),
        str(close),
        "100",
        "1000",
        "1000",
        confirm,
    ]


def test_parse_okx_candles_uses_only_confirmed_rows_and_sorts() -> None:
    payload = [
        _row(1_700_000_900_000, 103.0, "0"),
        _row(1_700_000_600_000, 102.0),
        _row(1_700_000_000_000, 100.0),
        _row(1_700_000_300_000, 101.0),
    ]

    frame = parse_okx_candles(payload)

    assert list(frame["close"]) == [100.0, 101.0, 102.0]
    assert frame["timestamp"].is_monotonic_increasing


def test_merge_candle_cache_deduplicates_and_keeps_latest_window() -> None:
    start = pd.Timestamp("2026-10-01T00:00:00Z")
    existing = pd.DataFrame(
        {
            "timestamp": [start + pd.Timedelta(minutes=15 * i) for i in range(4)],
            "open": [1.0, 2.0, 3.0, 4.0],
            "high": [1.1, 2.1, 3.1, 4.1],
            "low": [0.9, 1.9, 2.9, 3.9],
            "close": [1.0, 2.0, 3.0, 4.0],
            "volume": [10.0] * 4,
        }
    )
    incoming = existing.iloc[2:].copy()
    incoming.loc[incoming.index[-1], "close"] = 44.0
    extra = existing.iloc[[-1]].copy()
    extra["timestamp"] = start + pd.Timedelta(minutes=60)
    extra["close"] = 5.0
    incoming = pd.concat([incoming, extra], ignore_index=True)

    merged = merge_candle_cache(existing, incoming, max_bars=4)

    assert len(merged) == 4
    assert merged["timestamp"].is_unique
    assert list(merged["close"]) == [2.0, 3.0, 44.0, 5.0]


def test_validate_candle_cadence_rejects_gap() -> None:
    start = pd.Timestamp("2026-10-01T00:00:00Z")
    frame = pd.DataFrame(
        {
            "timestamp": [
                start,
                start + pd.Timedelta(hours=4),
                start + pd.Timedelta(hours=12),
            ],
            "open": [1.0] * 3,
            "high": [1.0] * 3,
            "low": [1.0] * 3,
            "close": [1.0] * 3,
            "volume": [1.0] * 3,
        }
    )

    with pytest.raises(ValueError, match="cadence gap"):
        validate_candle_cadence(
            frame,
            expected=pd.Timedelta(hours=4),
            minimum_bars=3,
        )
