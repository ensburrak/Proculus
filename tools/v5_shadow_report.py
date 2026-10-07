#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atomic_io import atomic_write_json
from research.v5_shadow_gate import summarize_shadow_evidence


DEFAULT_EVENTS = ROOT / "reports" / "v5_breadth_shadow_events.jsonl"
DEFAULT_POLICY = ROOT / "config" / "v5_prospective_shadow_policy.json"
DEFAULT_OUTPUT = ROOT / "reports" / "v5_prospective_shadow_status.json"


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Return exit code 2 until the preregistered evidence-volume gate passes.",
    )
    args = parser.parse_args()

    try:
        policy = json.loads(args.policy.read_text(encoding="utf-8"))
        if not isinstance(policy, dict):
            raise ValueError("policy must be a JSON object")
        records = _read_jsonl(args.events)
        report = summarize_shadow_evidence(records, policy)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    report["events_path"] = str(args.events)
    report["policy_path"] = str(args.policy)
    report["record_count"] = len(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not atomic_write_json(args.output, report):
        raise RuntimeError(f"failed to atomically write {args.output}")

    print("V5_SHADOW_STATUS=" + json.dumps(report, separators=(",", ":")))
    if args.require_ready and not bool(report.get("passed")):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
