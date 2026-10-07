from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Iterable


def _parse_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _candidate_key(candidate: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(candidate.get("family") or ""),
        str(candidate.get("setup_id") or ""),
        str(candidate.get("symbol") or ""),
        str(candidate.get("bar_time") or ""),
        str(candidate.get("side") or ""),
    )


def summarize_shadow_evidence(
    records: Iterable[dict[str, Any]],
    policy: dict[str, Any],
) -> dict[str, Any]:
    gate = policy.get("observation_gate")
    if not isinstance(gate, dict):
        raise ValueError("shadow policy missing observation_gate")

    min_span_days = float(gate.get("minimum_calendar_span_days") or 0)
    min_ready = int(gate.get("minimum_ready_4h_observations") or 0)
    min_candidates = int(gate.get("minimum_candidate_events") or 0)
    min_symbols = int(gate.get("minimum_candidate_symbols") or 0)
    require_zero_authority = bool(
        gate.get("require_zero_execution_authority_events", True)
    )
    maximum_recording_delay_minutes = float(
        gate.get("maximum_recording_delay_minutes") or 0
    )
    required_evidence_source = gate.get("required_evidence_source")
    if required_evidence_source is not None and not isinstance(
        required_evidence_source, dict
    ):
        raise ValueError("required_evidence_source must be an object")

    observed_times: list[datetime] = []
    ready_observation_keys: set[str] = set()
    not_ready_count = 0
    invalid_record_count = 0
    safety_violation_count = 0
    backfilled_record_count = 0
    source_violation_count = 0
    candidate_keys: set[tuple[str, str, str, str, str]] = set()
    candidate_symbols: set[str] = set()

    for raw in records:
        if not isinstance(raw, dict):
            invalid_record_count += 1
            continue
        observed_at = _parse_time(raw.get("observed_at"))
        if observed_at is None:
            invalid_record_count += 1
            continue
        is_ready = bool(raw.get("ready"))
        bar_time = _parse_time(raw.get("bar_time"))
        timely_record = True
        if is_ready and maximum_recording_delay_minutes > 0:
            if bar_time is None:
                timely_record = False
            else:
                bar_close = bar_time + timedelta(hours=4)
                delay_minutes = (observed_at - bar_close).total_seconds() / 60.0
                timely_record = 0.0 <= delay_minutes <= maximum_recording_delay_minutes
        if is_ready and not timely_record:
            backfilled_record_count += 1

        observed_times.append(observed_at)

        source_ok = True
        if isinstance(required_evidence_source, dict):
            source = raw.get("evidence_source")
            source_ok = isinstance(source, dict) and all(
                source.get(key) == expected
                for key, expected in required_evidence_source.items()
            )
            if not source_ok:
                source_violation_count += 1

        top_level_safe = (
            raw.get("shadow_only") is True
            and raw.get("execution_authority") is False
        )
        if not top_level_safe:
            safety_violation_count += 1

        if is_ready and timely_record and source_ok:
            bar_key = str(raw.get("bar_time") or "")
            if bar_key:
                ready_observation_keys.add(bar_key)
            else:
                invalid_record_count += 1
        else:
            not_ready_count += 1

        candidates = raw.get("candidates")
        if not isinstance(candidates, list):
            continue
        if not timely_record or not source_ok:
            continue
        record_bar_time = str(raw.get("bar_time") or "")
        record_family = str(raw.get("family") or "")
        for candidate in candidates:
            if not isinstance(candidate, dict):
                invalid_record_count += 1
                continue
            if (
                candidate.get("shadow_only") is not True
                or candidate.get("execution_authority") is not False
            ):
                safety_violation_count += 1
            if (
                str(candidate.get("bar_time") or "") != record_bar_time
                or str(candidate.get("family") or "") != record_family
            ):
                invalid_record_count += 1
                continue
            key = _candidate_key(candidate)
            if not all(key):
                invalid_record_count += 1
                continue
            candidate_keys.add(key)
            candidate_symbols.add(key[2])

    if observed_times:
        first_observed = min(observed_times)
        last_observed = max(observed_times)
        calendar_span_days = (
            last_observed - first_observed
        ).total_seconds() / 86_400.0
    else:
        first_observed = None
        last_observed = None
        calendar_span_days = 0.0

    checks = {
        "minimum_calendar_span_days": calendar_span_days >= min_span_days,
        "minimum_ready_4h_observations": len(ready_observation_keys) >= min_ready,
        "minimum_candidate_events": len(candidate_keys) >= min_candidates,
        "minimum_candidate_symbols": len(candidate_symbols) >= min_symbols,
        "zero_execution_authority_events": (
            safety_violation_count == 0 if require_zero_authority else True
        ),
        "zero_backfilled_records": backfilled_record_count == 0,
        "required_evidence_source": source_violation_count == 0,
    }

    return {
        "schema": "autotraderbot-v5-shadow-evidence-summary-v1",
        "family": str(policy.get("family") or "breadth_donchian_10_v5"),
        "passed": all(checks.values()),
        "promotion_authority": False,
        "execution_authority": False,
        "outcome_scoring_required": True,
        "checks": checks,
        "first_observed_at": (
            first_observed.isoformat() if first_observed is not None else None
        ),
        "last_observed_at": (
            last_observed.isoformat() if last_observed is not None else None
        ),
        "calendar_span_days": round(calendar_span_days, 6),
        "ready_observation_count": len(ready_observation_keys),
        "not_ready_observation_count": not_ready_count,
        "candidate_event_count": len(candidate_keys),
        "candidate_symbols": sorted(candidate_symbols),
        "candidate_symbol_count": len(candidate_symbols),
        "safety_violation_count": safety_violation_count,
        "backfilled_record_count": backfilled_record_count,
        "source_violation_count": source_violation_count,
        "invalid_record_count": invalid_record_count,
        "next_gate": (
            "prospective_shadow_outcome_scoring"
            if all(checks.values())
            else "continue_shadow_collection"
        ),
    }


__all__ = ["summarize_shadow_evidence"]
