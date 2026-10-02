from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.profitability_evidence import evaluate_release_gate

REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "profitability_release_latest.json"


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _config() -> dict[str, Any]:
    cfg = _load(ROOT / "config.json")
    value = cfg.get("profitability_evidence")
    return value if isinstance(value, dict) else {}


def _oos() -> dict[str, Any]:
    wf = _load(REPORTS / "walk_forward_edge_validation_latest.json")
    aggregate = wf.get("aggregate_oos") if isinstance(wf.get("aggregate_oos"), dict) else {}
    return {
        **aggregate,
        "positive_fold_share": _f(wf.get("positive_fold_share")),
        "fold_count": int(_f(wf.get("fold_count"))),
        "lookahead_detected": bool(wf.get("lookahead_detected", False)),
        "out_of_sample_passed": bool(wf.get("out_of_sample_passed", False)),
    }


def _shadow() -> dict[str, Any]:
    shadow = _load(REPORTS / "shadow_validation_latest.json")
    hours = _f(shadow.get("observed_hours"))
    if hours <= 0 and shadow.get("passed_30d") is True:
        hours = 30.0 * 24.0
    elif hours <= 0 and shadow.get("passed_72h") is True:
        hours = 72.0
    return {"hours": hours, "passed_72h": shadow.get("passed_72h"), "passed_30d": shadow.get("passed_30d")}


def _paper() -> dict[str, Any]:
    demo = _load(REPORTS / "demo_soak_latest.json")
    hours = _f(demo.get("observed_hours"))
    evidence = demo.get("evidence") if isinstance(demo.get("evidence"), dict) else {}
    if hours <= 0:
        hours = _f(evidence.get("observed_hours"))
    days = _f(demo.get("observed_days"))
    if days <= 0:
        days = hours / 24.0
    return {"days": days, "hours": hours, "passed": demo.get("passed")}


def _canary() -> dict[str, Any]:
    for name in ("live_canary_state.json", "canary_evidence_latest.json", "live_canary_latest.json"):
        payload = _load(REPORTS / name)
        if payload:
            days = _f(payload.get("observed_days"))
            if days <= 0:
                days = _f(payload.get("duration_days"))
            return {
                "days": days,
                "liquidations": int(_f(payload.get("liquidations"))),
                "source": name,
            }
    return {"days": 0.0, "liquidations": 0, "source": None}


def _integrity() -> dict[str, Any]:
    data = _load(REPORTS / "data_quality_acceptance_latest.json")
    passed = data.get("passed") is True
    if not data:
        wf = _load(REPORTS / "walk_forward_edge_validation_latest.json")
        integrity = wf.get("data_integrity") if isinstance(wf.get("data_integrity"), dict) else {}
        passed = integrity.get("passed") is True
        return {"passed": passed, "source": "walk_forward_edge_validation_latest.json"}
    return {"passed": passed, "source": "data_quality_acceptance_latest.json"}


def _tca() -> dict[str, Any]:
    payload = _load(REPORTS / "tca_latest.json")
    if not payload:
        return {"fills": 0, "source": None}
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else payload
    return {**summary, "fills": int(_f(summary.get("fills"))), "source": "tca_latest.json"}


def build() -> dict[str, Any]:
    cfg = _config()
    evidence = {
        "oos": _oos(),
        "shadow": _shadow(),
        "paper": _paper(),
        "canary": _canary(),
        "data_integrity": _integrity(),
        "tca": _tca(),
    }
    gate = evaluate_release_gate(evidence, cfg)
    report = {
        "schema": "profitability-release-v3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence": evidence,
        "gate": gate,
        "passed": bool(gate.get("live_allowed")),
        "blockers": list(gate.get("blockers") or []),
        "allowed_setups": [],
        "fail_closed": True,
    }
    return report


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    report = build()
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "passed": report["passed"],
        "status": "passed" if report["passed"] else "blocked",
        "blockers": report["blockers"],
        "report": str(OUTPUT),
    }, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
