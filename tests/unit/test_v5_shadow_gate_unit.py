from __future__ import annotations

from datetime import UTC, datetime, timedelta

from research.v5_shadow_gate import summarize_shadow_evidence


POLICY = {
    "observation_gate": {
        "minimum_calendar_span_days": 30,
        "minimum_ready_4h_observations": 4,
        "minimum_candidate_events": 3,
        "minimum_candidate_symbols": 2,
        "require_zero_execution_authority_events": True,
        "maximum_recording_delay_minutes": 30,
        "required_evidence_source": {
            "venue": "okx",
            "market_data": "public_mainnet_rest",
            "authenticated": False,
            "order_endpoints_used": False,
        },
    }
}


def _event(day: int, *, symbol: str | None = None, authority: bool = False) -> dict:
    bar_open = datetime(2026, 10, 1, tzinfo=UTC) + timedelta(days=day)
    observed = bar_open + timedelta(hours=4, minutes=5)
    candidates = []
    if symbol is not None:
        candidates.append(
            {
                "family": "breadth_donchian_10_v5",
                "setup_id": f"breadth_donchian_10_v5.entry.long.4h.v1",
                "symbol": symbol,
                "side": "long",
                "bar_time": bar_open.isoformat(),
                "shadow_only": True,
                "execution_authority": False,
            }
        )
    return {
        "schema": "autotraderbot-v5-breadth-shadow-v1",
        "family": "breadth_donchian_10_v5",
        "shadow_only": True,
        "execution_authority": authority,
        "ready": True,
        "reason": "ready",
        "observed_at": observed.isoformat(),
        "bar_time": bar_open.isoformat(),
        "evidence_source": {
            "venue": "okx",
            "market_data": "public_mainnet_rest",
            "authenticated": False,
            "order_endpoints_used": False,
        },
        "candidates": candidates,
    }


def test_shadow_gate_passes_only_on_real_span_volume_and_symbol_breadth() -> None:
    records = [
        _event(0, symbol="BTC/USDT:USDT"),
        _event(10, symbol="ETH/USDT:USDT"),
        _event(20, symbol="BTC/USDT:USDT"),
        _event(31),
    ]

    report = summarize_shadow_evidence(records, POLICY)

    assert report["passed"] is True
    assert report["promotion_authority"] is False
    assert report["outcome_scoring_required"] is True
    assert report["candidate_event_count"] == 3
    assert report["candidate_symbols"] == ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    assert all(report["checks"].values())


def test_shadow_gate_fails_closed_on_any_execution_authority_record() -> None:
    records = [
        _event(0, symbol="BTC/USDT:USDT"),
        _event(10, symbol="ETH/USDT:USDT"),
        _event(20, symbol="BTC/USDT:USDT"),
        _event(31, authority=True),
    ]

    report = summarize_shadow_evidence(records, POLICY)

    assert report["passed"] is False
    assert report["checks"]["zero_execution_authority_events"] is False
    assert report["safety_violation_count"] == 1


def test_shadow_gate_deduplicates_same_candidate_identity() -> None:
    first = _event(0, symbol="BTC/USDT:USDT")
    duplicate = dict(first)
    duplicate["observed_at"] = (
        datetime(2026, 10, 1, 1, tzinfo=UTC).isoformat()
    )

    report = summarize_shadow_evidence([first, duplicate], POLICY)

    assert report["candidate_event_count"] == 1
    assert report["ready_observation_count"] == 1


def test_shadow_gate_rejects_backfilled_historical_records() -> None:
    stale = _event(0, symbol="BTC/USDT:USDT")
    stale["observed_at"] = (
        datetime(2026, 10, 3, tzinfo=UTC).isoformat()
    )

    report = summarize_shadow_evidence([stale], POLICY)

    assert report["passed"] is False
    assert report["checks"]["zero_backfilled_records"] is False
    assert report["backfilled_record_count"] == 1
    assert report["candidate_event_count"] == 0


def test_shadow_gate_not_ready_diagnostic_is_not_backfill() -> None:
    diagnostic = {
        "schema": "autotraderbot-v5-breadth-shadow-v1",
        "family": "breadth_donchian_10_v5",
        "shadow_only": True,
        "execution_authority": False,
        "ready": False,
        "reason": "insufficient_completed_4h_history",
        "observed_at": "2026-10-08T00:05:00+00:00",
        "evidence_source": {
            "venue": "okx",
            "market_data": "public_mainnet_rest",
            "authenticated": False,
            "order_endpoints_used": False,
        },
        "candidates": [],
    }

    report = summarize_shadow_evidence([diagnostic], POLICY)

    assert report["backfilled_record_count"] == 0
    assert report["checks"]["zero_backfilled_records"] is True


def test_shadow_gate_rejects_non_public_or_authenticated_source() -> None:
    event = _event(0, symbol="BTC/USDT:USDT")
    event["evidence_source"]["authenticated"] = True

    report = summarize_shadow_evidence([event], POLICY)

    assert report["source_violation_count"] == 1
    assert report["checks"]["required_evidence_source"] is False
    assert report["candidate_event_count"] == 0


def test_shadow_gate_rejects_candidate_from_different_bar() -> None:
    event = _event(0, symbol="BTC/USDT:USDT")
    event["candidates"][0]["bar_time"] = "2026-09-01T00:00:00+00:00"

    report = summarize_shadow_evidence([event], POLICY)

    assert report["invalid_record_count"] == 1
    assert report["candidate_event_count"] == 0
