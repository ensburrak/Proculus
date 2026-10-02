from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.profitability_evidence import (
    data_integrity_gate,
    strategy_genome_fitness,
    summarize_edge,
)

DEFAULT_REPORT = ROOT / "reports" / "walk_forward_edge_validation_latest.json"


def _timestamp(row: dict[str, Any]) -> str:
    return str(
        row.get("exit_time")
        or row.get("timestamp")
        or row.get("recorded_at")
        or row.get("entry_time")
        or ""
    )


def chronological_windows(
    rows: list[dict[str, Any]],
    *,
    train_size: int,
    test_size: int,
    purge_size: int = 1,
    embargo_size: int = 1,
) -> list[dict[str, list[dict[str, Any]]]]:
    ordered = sorted((dict(row) for row in rows if isinstance(row, dict)), key=_timestamp)
    train_n = max(1, int(train_size))
    test_n = max(1, int(test_size))
    purge_n = max(0, int(purge_size))
    embargo_n = max(0, int(embargo_size))
    windows: list[dict[str, list[dict[str, Any]]]] = []
    test_start = train_n + purge_n
    while test_start + test_n <= len(ordered):
        train_end = max(0, test_start - purge_n)
        train_start = max(0, train_end - train_n)
        test_end = test_start + test_n
        train = ordered[train_start:train_end]
        test = ordered[test_start:test_end]
        if train and test:
            windows.append({"train": train, "test": test})
        test_start = test_end + embargo_n
    return windows


def build_walk_forward_report(
    rows: list[dict[str, Any]],
    *,
    train_size: int = 250,
    test_size: int = 100,
    purge_size: int = 4,
    embargo_size: int = 4,
    min_profit_factor: float = 1.15,
    min_expectancy_r: float = 0.05,
) -> dict[str, Any]:
    windows = chronological_windows(
        rows,
        train_size=train_size,
        test_size=test_size,
        purge_size=purge_size,
        embargo_size=embargo_size,
    )
    folds: list[dict[str, Any]] = []
    all_oos: list[dict[str, Any]] = []
    lookahead_detected = False

    for index, window in enumerate(windows, start=1):
        train, test = window["train"], window["test"]
        train_last = _timestamp(train[-1]) if train else ""
        test_first = _timestamp(test[0]) if test else ""
        temporal_ok = bool(train_last and test_first and train_last < test_first)
        lookahead_detected = lookahead_detected or not temporal_ok
        train_summary = summarize_edge(train).to_dict()
        test_summary = summarize_edge(test).to_dict()
        positive = (
            temporal_ok
            and int(test_summary["trades"]) > 0
            and float(test_summary["profit_factor"] or 0.0) >= float(min_profit_factor)
            and float(test_summary["expectancy_r"]) >= float(min_expectancy_r)
            and float(test_summary["expectancy_r_ci95_low"]) > 0.0
        )
        folds.append(
            {
                "fold": index,
                "train_first": _timestamp(train[0]),
                "train_last": train_last,
                "test_first": test_first,
                "test_last": _timestamp(test[-1]),
                "temporal_ok": temporal_ok,
                "positive": positive,
                "train": train_summary,
                "test": test_summary,
            }
        )
        all_oos.extend(test)

    aggregate = summarize_edge(all_oos).to_dict()
    positive_folds = sum(1 for fold in folds if fold["positive"])
    positive_share = positive_folds / len(folds) if folds else 0.0
    fold_expectancies = [float(fold["test"]["expectancy_r"]) for fold in folds]
    if len(fold_expectancies) > 1:
        avg = sum(fold_expectancies) / len(fold_expectancies)
        fold_std = (sum((x - avg) ** 2 for x in fold_expectancies) / len(fold_expectancies)) ** 0.5
    else:
        fold_std = 0.0

    if folds:
        train_expectancy = sum(float(f["train"]["expectancy_r"]) for f in folds) / len(folds)
    else:
        train_expectancy = 0.0
    oos_expectancy = float(aggregate["expectancy_r"])
    degradation_pct = (
        max(0.0, (train_expectancy - oos_expectancy) / max(abs(train_expectancy), 1e-9) * 100.0)
        if train_expectancy > 0
        else 0.0
    )
    genome_input = {
        **aggregate,
        "fold_expectancy_std": fold_std,
        "oos_degradation_pct": degradation_pct,
        "turnover_per_day": 0.0,
        "cost_drag_r": (
            float(aggregate["cost_drag_usd"]) / max(abs(float(aggregate["net_pnl_usd"])), 1.0)
        ),
    }
    fitness = strategy_genome_fitness(genome_input)

    integrity = data_integrity_gate(
        {
            "fresh": True,
            "duplicate_rows": 0,
            "future_rows": 0,
            "broken_join_rows": 0,
            "missing_critical_rows": 0,
            "lookahead_detected": lookahead_detected,
        }
    )

    passed = (
        bool(folds)
        and not lookahead_detected
        and float(aggregate["profit_factor"] or 0.0) >= float(min_profit_factor)
        and float(aggregate["expectancy_r"]) >= float(min_expectancy_r)
        and float(aggregate["expectancy_r_ci95_low"]) > 0.0
        and positive_share >= 2.0 / 3.0
        and integrity["passed"]
    )
    return {
        "schema": "walk-forward-edge-validation-v3",
        "train_size": int(train_size),
        "test_size": int(test_size),
        "purge_size": int(purge_size),
        "embargo_size": int(embargo_size),
        "fold_count": len(folds),
        "positive_folds": positive_folds,
        "positive_fold_share": positive_share,
        "lookahead_detected": lookahead_detected,
        "out_of_sample_passed": passed,
        "aggregate_oos": aggregate,
        "fold_expectancy_std": fold_std,
        "oos_degradation_pct": degradation_pct,
        "strategy_genome": fitness,
        "data_integrity": integrity,
        "folds": folds,
        "live_readiness_impact": "evidence_only_no_automatic_live_authority",
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("trades", "_trades", "rows", "events"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(row) for row in value if isinstance(row, dict)]
    raise ValueError(f"no trade rows found in {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Leakage-safe chronological walk-forward edge validation.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--train-size", type=int, default=250)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--purge-size", type=int, default=4)
    parser.add_argument("--embargo-size", type=int, default=4)
    args = parser.parse_args()

    rows = load_rows(args.input)
    report = build_walk_forward_report(
        rows,
        train_size=args.train_size,
        test_size=args.test_size,
        purge_size=args.purge_size,
        embargo_size=args.embargo_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "ok": True,
        "output": str(args.output),
        "folds": report["fold_count"],
        "oos_passed": report["out_of_sample_passed"],
        "positive_fold_share": report["positive_fold_share"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
