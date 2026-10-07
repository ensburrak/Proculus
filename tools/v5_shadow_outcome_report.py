#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atomic_io import atomic_write_json
from research.v5_shadow_gate import summarize_shadow_evidence
from research.v5_shadow_outcomes import (
    evaluate_outcome_gate,
    score_shadow_candidate,
    summarize_candidate_outcomes,
)

DEFAULT_EVENTS = ROOT / "reports" / "v5_breadth_shadow_events.jsonl"
DEFAULT_POLICY = ROOT / "config" / "v5_prospective_shadow_policy.json"
DEFAULT_DATA_DIR = ROOT / "data" / "fresh_v5_shadow"
DEFAULT_OUTPUT = ROOT / "reports" / "v5_prospective_shadow_outcomes.json"


def _candidate_key(candidate: dict[str, Any]) -> str:
    return "|".join(
        (
            str(candidate.get("family") or ""),
            str(candidate.get("setup_id") or ""),
            str(candidate.get("symbol") or ""),
            str(candidate.get("bar_time") or ""),
            str(candidate.get("side") or ""),
        )
    )


def _symbol_path(data_dir: Path, symbol: str) -> Path:
    base = symbol.split("/", 1)[0].strip().upper()
    return data_dir / f"{base}_USDT_USDT_15m.parquet"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid shadow JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"shadow JSONL record must be object at {path}:{line_number}"
                )
            records.append(payload)
    return records


def _safe_candidates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        candidates = record.get("candidates")
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            key = _candidate_key(candidate)
            if key and key not in unique:
                unique[key] = candidate
    return list(unique.values())


def _empty_summary() -> dict[str, Any]:
    return {
        "unique_candidates": 0,
        "complete_trades": 0,
        "pending_trades": 0,
        "invalid_trades": 0,
        "win_rate_pct": 0.0,
        "profit_factor": 0.0,
        "expectancy_r": 0.0,
        "symbols": [],
        "symbol_count": 0,
        "execution_authority": False,
        "promotion_authority": False,
    }


def build_outcome_report(
    records: list[dict[str, Any]],
    policy: dict[str, Any],
    data_dir: Path,
) -> dict[str, Any]:
    observation = summarize_shadow_evidence(records, policy)
    safety_blocked = (
        int(observation.get("safety_violation_count") or 0) > 0
        or int(observation.get("backfilled_record_count") or 0) > 0
        or int(observation.get("invalid_record_count") or 0) > 0
        or int(observation.get("source_violation_count") or 0) > 0
    )

    empty = _empty_summary()
    if safety_blocked:
        gate = {
            "passed": False,
            "checks": {"shadow_evidence_safe": False},
            "common_complete_symbols": [],
            "promotion_authority": False,
            "execution_authority": False,
            "next_gate": "repair_shadow_evidence_integrity",
        }
        return {
            "schema": "autotraderbot-v5-shadow-outcome-report-v1",
            "family": str(policy.get("family") or "breadth_donchian_10_v5"),
            "scoring_status": "blocked_shadow_safety_violation",
            "observation_gate": observation,
            "primary_5bps": dict(empty),
            "stress_15bps": dict(empty),
            "outcome_gate": gate,
            "promotion_review_ready": False,
            "execution_authority": False,
            "promotion_authority": False,
            "candidate_outcomes": {"primary_5bps": [], "stress_15bps": []},
        }

    scoring = policy.get("outcome_scoring")
    if not isinstance(scoring, dict):
        raise ValueError("shadow policy missing outcome_scoring")
    fee_bps = float(scoring.get("fee_bps_per_side") or 0.0)
    primary_slippage = float(
        scoring.get("primary_slippage_bps_per_side") or 0.0
    )
    stress_slippage = float(
        scoring.get("stress_slippage_bps_per_side") or 0.0
    )
    funding_bps = float(scoring.get("funding_bps_per_8h") or 0.0)

    candidates = _safe_candidates(records)
    frame_cache: dict[str, pd.DataFrame] = {}
    missing_files: set[str] = set()

    def frame_for(symbol: str) -> pd.DataFrame:
        if symbol in frame_cache:
            return frame_cache[symbol]
        path = _symbol_path(data_dir, symbol)
        if not path.exists():
            missing_files.add(str(path))
            frame_cache[symbol] = pd.DataFrame()
            return frame_cache[symbol]
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError, ImportError) as exc:
            raise ValueError(f"unable to read shadow market data {path}: {exc}") from exc
        frame_cache[symbol] = frame
        return frame

    primary_outcomes: list[dict[str, Any]] = []
    stress_outcomes: list[dict[str, Any]] = []
    for candidate in candidates:
        symbol = str(candidate.get("symbol") or "")
        frame = frame_for(symbol)
        primary_outcomes.append(
            score_shadow_candidate(
                candidate,
                frame,
                fee_bps=fee_bps,
                slippage_bps=primary_slippage,
                funding_bps_per_8h=funding_bps,
            )
        )
        stress_outcomes.append(
            score_shadow_candidate(
                candidate,
                frame,
                fee_bps=fee_bps,
                slippage_bps=stress_slippage,
                funding_bps_per_8h=funding_bps,
            )
        )

    primary = summarize_candidate_outcomes(primary_outcomes)
    stress = summarize_candidate_outcomes(stress_outcomes)
    outcome_gate = evaluate_outcome_gate(primary, stress, policy)
    review_ready = bool(observation.get("passed")) and bool(outcome_gate.get("passed"))

    return {
        "schema": "autotraderbot-v5-shadow-outcome-report-v1",
        "family": str(policy.get("family") or "breadth_donchian_10_v5"),
        "scoring_status": (
            "forward_gate_passed_manual_review_required"
            if review_ready
            else "interim_forward_scoring"
        ),
        "observation_gate": observation,
        "primary_5bps": primary,
        "stress_15bps": stress,
        "outcome_gate": outcome_gate,
        "promotion_review_ready": review_ready,
        "missing_market_data_files": sorted(missing_files),
        "execution_authority": False,
        "promotion_authority": False,
        "candidate_outcomes": {
            "primary_5bps": primary_outcomes,
            "stress_15bps": stress_outcomes,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--require-review-ready",
        action="store_true",
        help=(
            "Return exit code 2 until both the prospective observation gate "
            "and preregistered 5/15bps outcome gates pass."
        ),
    )
    args = parser.parse_args()

    try:
        records = _read_jsonl(args.events)
        policy = json.loads(args.policy.read_text(encoding="utf-8"))
        if not isinstance(policy, dict):
            raise ValueError("policy must be a JSON object")
        report = build_outcome_report(records, policy, args.data_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    report["events_path"] = str(args.events)
    report["policy_path"] = str(args.policy)
    report["data_dir"] = str(args.data_dir)
    report["record_count"] = len(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not atomic_write_json(args.output, report):
        raise RuntimeError(f"failed to atomically write {args.output}")

    print("V5_SHADOW_OUTCOMES=" + json.dumps(report, separators=(",", ":")))
    if args.require_review_ready and not bool(report.get("promotion_review_ready")):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
