from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.profitability_evidence import summarize_edge

REPORTS = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS / "setup_edge_observatory_latest.json"


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
        return rows
    payload = _load(path)
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("trades", "rows", "events", "trade_evidence"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(row) for row in value if isinstance(row, dict)]
    return []


def _group(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        setup_id = str(row.get("setup_id") or "").strip()
        if setup_id:
            grouped[setup_id].append(row)
    return dict(grouped)


def _eligible(summary: dict[str, Any], *, min_samples: int, min_pf: float, min_exp: float) -> bool:
    try:
        trades = int(summary.get("trades") or 0)
        pf = float(summary.get("profit_factor") or 0.0)
        exp = float(summary.get("expectancy_r") or 0.0)
        ci = float(summary.get("expectancy_r_ci95_low") or 0.0)
    except (TypeError, ValueError):
        return False
    return trades >= min_samples and pf >= min_pf and exp >= min_exp and ci > 0.0


def build_observatory(
    *,
    oos_rows: list[dict[str, Any]],
    paper_rows: list[dict[str, Any]],
    tca_report: dict[str, Any] | None,
    min_oos_samples: int = 100,
    min_paper_samples: int = 100,
    min_pf: float = 1.15,
    min_exp: float = 0.05,
) -> dict[str, Any]:
    oos_groups = _group(oos_rows)
    paper_groups = _group(paper_rows)
    setup_ids = sorted(set(oos_groups) | set(paper_groups))
    tca = tca_report if isinstance(tca_report, dict) else {}
    tca_summary = tca.get("summary") if isinstance(tca.get("summary"), dict) else tca
    try:
        real_tca_fills = int(tca_summary.get("fills") or 0) if isinstance(tca_summary, dict) else 0
    except (TypeError, ValueError):
        real_tca_fills = 0
    tca_scope = str(tca.get("scope") or "") if isinstance(tca, dict) else ""
    tca_is_hypothetical = "hypothetical" in tca_scope.lower() or "backtest" in tca_scope.lower()
    real_tca_ok = real_tca_fills > 0 and not tca_is_hypothetical

    setups: dict[str, Any] = {}
    allowed: list[str] = []
    for setup_id in setup_ids:
        oos_summary = summarize_edge(oos_groups.get(setup_id, [])).to_dict()
        paper_summary = summarize_edge(paper_groups.get(setup_id, [])).to_dict()
        oos_ok = _eligible(
            oos_summary,
            min_samples=min_oos_samples,
            min_pf=min_pf,
            min_exp=min_exp,
        )
        paper_ok = _eligible(
            paper_summary,
            min_samples=min_paper_samples,
            min_pf=min_pf,
            min_exp=min_exp,
        )
        setup_allowed = oos_ok and paper_ok and real_tca_ok
        blockers: list[str] = []
        if not oos_ok:
            blockers.append("oos_setup_edge")
        if not paper_ok:
            blockers.append("paper_setup_edge")
        if not real_tca_ok:
            blockers.append("real_tca")
        setups[setup_id] = {
            "oos": oos_summary,
            "paper": paper_summary,
            "oos_passed": oos_ok,
            "paper_passed": paper_ok,
            "real_tca_passed": real_tca_ok,
            "live_setup_allowed": setup_allowed,
            "blockers": blockers,
        }
        if setup_allowed:
            allowed.append(setup_id)

    return {
        "schema": "setup-edge-observatory-v1",
        "fail_closed": True,
        "thresholds": {
            "min_oos_samples": min_oos_samples,
            "min_paper_samples": min_paper_samples,
            "min_profit_factor": min_pf,
            "min_expectancy_r": min_exp,
            "require_positive_ci95_low": True,
            "require_real_tca": True,
        },
        "real_tca": {
            "fills": real_tca_fills,
            "scope": tca_scope,
            "hypothetical": tca_is_hypothetical,
            "passed": real_tca_ok,
        },
        "setup_count": len(setups),
        "allowed_setups": allowed,
        "setups": setups,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build fail-closed setup-level Edge Observatory.")
    parser.add_argument("--oos-trades", type=Path, default=REPORTS / "walk_forward_oos_trades.json")
    parser.add_argument("--paper-trades", type=Path, default=REPORTS / "paper_trades_latest.jsonl")
    parser.add_argument("--tca", type=Path, default=REPORTS / "tca_latest.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = build_observatory(
        oos_rows=_load_rows(args.oos_trades),
        paper_rows=_load_rows(args.paper_trades),
        tca_report=_load(args.tca),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "valid": True,
        "fail_closed": True,
        "allowed_setups": report["allowed_setups"],
        "setup_count": report["setup_count"],
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
