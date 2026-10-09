#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENTS = ROOT / "reports" / "v5_breadth_shadow_events.jsonl"
DEFAULT_TRADES = ROOT / "reports" / "v5_breadth_shadow_trades.jsonl"
DEFAULT_CONFIG = ROOT / "config" / "v5_breadth_shadow_runtime.json"
DEFAULT_STATE = ROOT / "reports" / "v5_breadth_shadow_state.json"
DEFAULT_OUTPUT = ROOT / "reports" / "v5_breadth_shadow_status.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _load_jsonl_checked(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Read the evidence ledger without silently discarding damaged records.

    A missing, unreadable or malformed ledger is not admissible evidence for
    prospective promotion. Callers may still show partial diagnostics.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return [], 1
    out: list[dict[str, Any]] = []
    invalid_rows = 0
    for raw in lines:
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            invalid_rows += 1
            continue
        if isinstance(value, dict):
            out.append(value)
        else:
            invalid_rows += 1
    return out, invalid_rows


def _trade_evidence_integrity(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Flag duplicated/invalid V5 closes rather than accidentally inflating PF."""
    invalid = 0
    duplicates = 0
    seen: set[str] = set()
    for row in rows:
        if row.get("family") != "breadth_donchian_10_v5":
            continue
        if row.get("closed") is not True:
            invalid += 1
            continue
        try:
            net_r = float(row["net_r"])
        except (KeyError, TypeError, ValueError):
            invalid += 1
            continue
        if not math.isfinite(net_r):
            invalid += 1
        key = row.get("signal_key")
        if key is not None:
            if not isinstance(key, str) or not key.strip():
                invalid += 1
            elif key in seen:
                duplicates += 1
            else:
                seen.add(key)
    return {"invalid_trade_records": invalid, "duplicate_signal_keys": duplicates}


def _dt(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _trade_stats(trades: list[dict[str, Any]]) -> dict[str, Any]:
    values: list[float] = []
    for trade in trades:
        if str(trade.get("family") or "") != "breadth_donchian_10_v5":
            continue
        if trade.get("closed") is not True:
            continue
        value = trade.get("net_r")
        try:
            r_value = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(r_value):
            continue
        values.append(r_value)
    wins = [value for value in values if value > 0]
    losses = [value for value in values if value < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    return {
        "closed_trades": len(values),
        "win_rate_pct": round(100.0 * len(wins) / max(1, len(values)), 4),
        "profit_factor": round(gross_win / gross_loss, 5) if gross_loss > 0 else (999.0 if wins else 0.0),
        "expectancy_r": round(sum(values) / max(1, len(values)), 5),
    }


def build_report(
    events_path: Path,
    trades_path: Path,
    config_path: Path,
    state_path: Path | None = None,
) -> dict[str, Any]:
    config = _load_json(config_path)
    policy = config.get("observation_policy") if isinstance(config.get("observation_policy"), dict) else {}
    event_rows, invalid_event_rows = _load_jsonl_checked(events_path)
    trade_rows, invalid_trade_rows = _load_jsonl_checked(trades_path)
    integrity = _trade_evidence_integrity(trade_rows)
    events = [
        event
        for event in event_rows
        if str(event.get("family") or "") == "breadth_donchian_10_v5"
    ]
    observed = [_dt(event.get("observed_at")) for event in events]
    observed = [value for value in observed if value is not None]
    first = min(observed) if observed else None
    last = max(observed) if observed else None
    elapsed_days = (last - first).total_seconds() / 86400.0 if first and last else 0.0

    ready_by_bar: dict[str, dict[str, Any]] = {}
    not_ready = Counter()
    candidate_count = 0
    symbol_candidates = Counter()
    regimes = Counter()
    for event in events:
        if event.get("ready") is True:
            bar_time = str(event.get("bar_time") or "")
            if bar_time:
                ready_by_bar[bar_time] = event
            candidates = event.get("candidates") if isinstance(event.get("candidates"), list) else []
            candidate_count += len(candidates)
            for candidate in candidates:
                if isinstance(candidate, dict):
                    symbol = str(candidate.get("symbol") or "")
                    if symbol:
                        symbol_candidates[symbol] += 1
            state = event.get("global_state") if isinstance(event.get("global_state"), dict) else {}
            regime = str(state.get("regime") or "")
            if regime:
                regimes[regime] += 1
        else:
            not_ready[str(event.get("reason") or "unknown")] += 1

    trades = _trade_stats(trade_rows)
    resolved_state_path = state_path or (events_path.parent / "v5_breadth_shadow_state.json")
    state = _load_json(resolved_state_path)
    state_available = state.get("schema") == "autotraderbot-v5-paper-shadow-state-v1"
    state_closed = state.get("closed_trades")
    ledger_count_matches_state = (
        state_available
        and type(state_closed) is int
        and state_closed >= 0
        and state_closed == int(trades["closed_trades"])
    )
    missed_signals = int(_f(state.get("missed_signals"), 0))
    invalidated_trades = int(_f(state.get("invalidated_trades"), 0))
    evidence_write_failures = int(_f(state.get("evidence_write_failures"), 0))
    blocked = state.get("blocked") if isinstance(state.get("blocked"), dict) else {}
    max_drawdown_pct = _f(state.get("max_drawdown_pct"), 0.0)
    open_positions = len(state.get("open_positions") or {}) if isinstance(state.get("open_positions"), dict) else 0

    min_days = _f(policy.get("minimum_elapsed_days"), 30.0)
    min_bars = int(_f(policy.get("minimum_ready_4h_bars"), 120))
    min_trades = int(_f(policy.get("minimum_closed_hypothetical_trades"), 20))
    min_pf = _f(policy.get("minimum_profit_factor"), 1.15)
    min_er = _f(policy.get("minimum_expectancy_r"), 0.05)
    max_missed = int(_f(policy.get("maximum_missed_signals"), 0))
    max_invalidated = int(_f(policy.get("maximum_invalidated_trades"), 0))
    max_write_failures = int(_f(policy.get("maximum_evidence_write_failures"), 0))

    checks = {
        "paper_state_available": state_available,
        "event_log_integrity": invalid_event_rows == 0,
        "trade_log_integrity": (
            invalid_trade_rows == 0
            and integrity["invalid_trade_records"] == 0
            and integrity["duplicate_signal_keys"] == 0
        ),
        "trade_ledger_matches_state": ledger_count_matches_state,
        "elapsed_days": elapsed_days >= min_days,
        "ready_4h_bars": len(ready_by_bar) >= min_bars,
        "closed_hypothetical_trades": int(trades["closed_trades"]) >= min_trades,
        "profit_factor": float(trades["profit_factor"]) >= min_pf,
        "expectancy_r": float(trades["expectancy_r"]) >= min_er,
        "missed_signals": missed_signals <= max_missed,
        "invalidated_trades": invalidated_trades <= max_invalidated,
        "evidence_write_failures": evidence_write_failures <= max_write_failures,
    }
    blockers = [name for name, passed in checks.items() if not passed]

    return {
        "schema": "autotraderbot-v5-prospective-shadow-status-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "family": "breadth_donchian_10_v5",
        "shadow_only": True,
        "execution_authority": False,
        "production_authority": False,
        "events_path": str(events_path),
        "trades_path": str(trades_path),
        "state_path": str(resolved_state_path),
        "first_observed_at": first.isoformat() if first else None,
        "last_observed_at": last.isoformat() if last else None,
        "elapsed_days": round(elapsed_days, 4),
        "event_count": len(events),
        "ready_4h_bar_count": len(ready_by_bar),
        "candidate_count": candidate_count,
        "candidate_symbols": dict(sorted(symbol_candidates.items())),
        "regimes": dict(sorted(regimes.items())),
        "not_ready_reasons": dict(sorted(not_ready.items())),
        "trade_metrics": trades,
        "evidence_integrity": {
            "invalid_event_rows": invalid_event_rows,
            "invalid_trade_rows": invalid_trade_rows,
            "invalid_trade_records": integrity["invalid_trade_records"],
            "duplicate_signal_keys": integrity["duplicate_signal_keys"],
            "state_closed_trades": state_closed,
            "trade_ledger_matches_state": bool(ledger_count_matches_state),
        },
        "paper_state": {
            "available": state_available,
            "missed_signals": missed_signals,
            "invalidated_trades": invalidated_trades,
            "evidence_write_failures": evidence_write_failures,
            "open_positions": open_positions,
            "max_drawdown_pct": round(max_drawdown_pct, 6),
            "blocked": dict(sorted(blocked.items())),
        },
        "policy": {
            "minimum_elapsed_days": min_days,
            "minimum_ready_4h_bars": min_bars,
            "minimum_closed_hypothetical_trades": min_trades,
            "minimum_profit_factor": min_pf,
            "minimum_expectancy_r": min_er,
            "maximum_missed_signals": max_missed,
            "maximum_invalidated_trades": max_invalidated,
            "maximum_evidence_write_failures": max_write_failures,
        },
        "checks": checks,
        "prospective_shadow_passed": not blockers,
        "blockers": blockers,
        "next_gate": (
            "exact_repo_verification_and_operator_review"
            if not blockers
            else "continue_shadow_observation"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = build_report(args.events, args.trades, args.config, args.state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
