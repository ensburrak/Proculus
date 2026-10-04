from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENTS = ROOT / "reports" / "core_trend_shadow_events.jsonl"
DEFAULT_OUTPUT = ROOT / "reports" / "core_trend_shadow_report.json"


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        key = (
            str(row.get("strategy_id") or ""),
            str(row.get("policy_fingerprint_sha256") or ""),
            str(row.get("symbol") or ""),
            str(row.get("closed_bar_timestamp") or ""),
        )
        if not all(key) or key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


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


def _tca_is_real(tca: Any) -> tuple[bool, int, str]:
    if not isinstance(tca, dict):
        return False, 0, ""
    scope = str(tca.get("scope") or "")
    summary = tca.get("summary") if isinstance(tca.get("summary"), dict) else tca
    try:
        fills = int(summary.get("fills") or 0) if isinstance(summary, dict) else 0
    except (TypeError, ValueError):
        fills = 0
    lowered = scope.lower()
    hypothetical = "hypothetical" in lowered or "backtest" in lowered
    return fills > 0 and not hypothetical, fills, scope


def build_report(
    *,
    events: list[dict[str, Any]],
    policy: dict[str, Any],
    tca: dict[str, Any] | None = None,
    one_way_cost_bps: float = 20.0,
    funding_bp_per_4h: float = 0.5,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in events:
        if str(row.get("status") or "") != "shadow_signal":
            continue
        symbol = str(row.get("symbol") or "")
        if symbol:
            grouped[symbol].append(row)

    observations: list[dict[str, Any]] = []
    for symbol, rows in grouped.items():
        ordered = sorted(
            rows,
            key=lambda row: str(row.get("closed_bar_timestamp") or ""),
        )
        previous_target = 0.0
        for index in range(len(ordered) - 1):
            current = ordered[index]
            nxt = ordered[index + 1]
            entry = _as_float(current.get("entry_reference_price"))
            exit_ref = _as_float(nxt.get("entry_reference_price"))
            target = _as_float(current.get("target_exposure"))
            ts = _parse_time(current.get("closed_bar_timestamp"))
            next_ts = _parse_time(nxt.get("closed_bar_timestamp"))
            if (
                entry is None
                or exit_ref is None
                or target is None
                or ts is None
                or next_ts is None
                or entry <= 0
                or exit_ref <= 0
                or next_ts <= ts
            ):
                continue

            gross = target * (exit_ref / entry - 1.0)
            turnover = abs(target - previous_target)
            cost = turnover * (one_way_cost_bps / 10_000.0)
            funding = abs(target) * (funding_bp_per_4h / 10_000.0)
            net = gross - cost - funding
            observations.append(
                {
                    "symbol": symbol,
                    "timestamp": ts,
                    "next_timestamp": next_ts,
                    "target_exposure": target,
                    "turnover": turnover,
                    "gross_return": gross,
                    "cost_return": cost,
                    "funding_return": funding,
                    "net_return": net,
                }
            )
            previous_target = target

    net_values = [row["net_return"] for row in observations]
    wins = [value for value in net_values if value > 0]
    losses = [value for value in net_values if value < 0]
    profit_factor = (
        sum(wins) / abs(sum(losses))
        if losses
        else (999.0 if wins else 0.0)
    )
    avg_observation_bps = (
        float(np.mean(net_values)) * 10_000.0 if net_values else 0.0
    )

    by_timestamp: dict[datetime, list[float]] = defaultdict(list)
    by_day: dict[str, list[float]] = defaultdict(list)
    for row in observations:
        by_timestamp[row["timestamp"]].append(row["net_return"])
        by_day[row["timestamp"].date().isoformat()].append(row["net_return"])

    portfolio_returns = [
        float(np.mean(values))
        for _, values in sorted(by_timestamp.items())
        if values
    ]
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for value in portfolio_returns:
        equity *= max(0.0, 1.0 + value)
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = min(max_drawdown, equity / peak - 1.0)

    daily_returns = [
        float(np.mean(values))
        for _, values in sorted(by_day.items())
        if values
    ]
    if len(daily_returns) >= 2:
        arr = np.asarray(daily_returns, dtype=float)
        mean = float(arr.mean())
        se = float(arr.std(ddof=1) / math.sqrt(len(arr)))
        ci95_low_bps = (mean - 1.96 * se) * 10_000.0
        ci95_high_bps = (mean + 1.96 * se) * 10_000.0
    elif daily_returns:
        ci95_low_bps = ci95_high_bps = daily_returns[0] * 10_000.0
    else:
        ci95_low_bps = ci95_high_bps = 0.0

    unique_days = len(by_day)
    unique_symbols = len({row["symbol"] for row in observations})
    policy_fingerprints = sorted(
        {
            str(row.get("policy_fingerprint_sha256") or "")
            for row in events
            if str(row.get("policy_fingerprint_sha256") or "")
        }
    )
    policy_consistent = len(policy_fingerprints) == 1
    real_tca_ok, real_tca_fills, tca_scope = _tca_is_real(tca)

    thresholds = {
        "min_shadow_days": int(policy.get("min_shadow_days", 30) or 30),
        "min_observations": int(policy.get("min_observations", 500) or 500),
        "min_symbols": int(policy.get("min_symbols", 8) or 8),
        "min_profit_factor": float(policy.get("min_profit_factor", 1.20) or 1.20),
        "min_avg_observation_bps": float(
            policy.get("min_avg_observation_bps", 0.0) or 0.0
        ),
        "max_drawdown_pct": float(policy.get("max_drawdown_pct", 15.0) or 15.0),
        "require_positive_ci95_low": bool(
            policy.get("require_positive_ci95_low", True)
        ),
        "require_real_tca": bool(policy.get("require_real_tca", True)),
        "allow_live": bool(policy.get("allow_live", False)),
    }

    checks = {
        "policy_consistent": policy_consistent,
        "shadow_days": unique_days >= thresholds["min_shadow_days"],
        "observations": len(observations) >= thresholds["min_observations"],
        "symbols": unique_symbols >= thresholds["min_symbols"],
        "profit_factor": profit_factor >= thresholds["min_profit_factor"],
        "avg_observation": avg_observation_bps > thresholds["min_avg_observation_bps"],
        "drawdown": abs(max_drawdown) * 100.0 <= thresholds["max_drawdown_pct"],
        "ci95_low": (
            ci95_low_bps > 0.0
            if thresholds["require_positive_ci95_low"]
            else True
        ),
    }
    statistical_passed = all(checks.values())
    live_checks = {
        **checks,
        "real_tca": real_tca_ok if thresholds["require_real_tca"] else True,
        "manual_live_release": thresholds["allow_live"],
    }
    live_candidate = all(live_checks.values())

    blockers = [key for key, passed in live_checks.items() if not passed]

    return {
        "schema": "core-trend-shadow-forward-report-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "fail_closed": True,
        "research_only": True,
        "strategy_id": "core_trend_4h_ema50_200_voltarget.v1",
        "cost_model": {
            "one_way_cost_bps": one_way_cost_bps,
            "funding_bp_per_4h": funding_bp_per_4h,
        },
        "coverage": {
            "raw_events": len(events),
            "resolved_observations": len(observations),
            "unique_days": unique_days,
            "unique_symbols": unique_symbols,
            "policy_fingerprints": policy_fingerprints,
            "policy_consistent": policy_consistent,
        },
        "metrics": {
            "profit_factor": profit_factor,
            "avg_observation_bps": avg_observation_bps,
            "ci95_low_bps": ci95_low_bps,
            "ci95_high_bps": ci95_high_bps,
            "max_drawdown_pct": abs(max_drawdown) * 100.0,
            "compounded_return_pct": (equity - 1.0) * 100.0,
        },
        "thresholds": thresholds,
        "statistical_checks": checks,
        "statistical_passed": statistical_passed,
        "paper_candidate": statistical_passed,
        "real_tca": {
            "passed": real_tca_ok,
            "fills": real_tca_fills,
            "scope": tca_scope,
        },
        "live_candidate": live_candidate,
        "blockers": blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate forward-only Core Trend shadow evidence."
    )
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--tca", type=Path, default=ROOT / "reports" / "tca_latest.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--one-way-cost-bps", type=float, default=20.0)
    parser.add_argument("--funding-bp-per-4h", type=float, default=0.5)
    args = parser.parse_args()

    config = _load_json(args.config)
    if not isinstance(config, dict):
        raise SystemExit("invalid config")
    lane = (
        config.get("core_trend_shadow")
        if isinstance(config.get("core_trend_shadow"), dict)
        else {}
    )
    policy = (
        lane.get("forward_promotion")
        if isinstance(lane.get("forward_promotion"), dict)
        else {}
    )
    tca = _load_json(args.tca)

    report = build_report(
        events=_load_events(args.events),
        policy=policy,
        tca=tca if isinstance(tca, dict) else None,
        one_way_cost_bps=args.one_way_cost_bps,
        funding_bp_per_4h=args.funding_bp_per_4h,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "statistical_passed": report["statistical_passed"],
                "paper_candidate": report["paper_candidate"],
                "live_candidate": report["live_candidate"],
                "blockers": report["blockers"],
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
