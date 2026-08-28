from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict

from parameter_optimizer import run_multi_objective_parameter_optimization


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run verified multi-objective optimisation and auto-apply the best candidate."
    )
    parser.add_argument("--trials", type=int, default=100, help="Number of optimisation trials.")
    parser.add_argument(
        "--preset",
        choices=("balanced", "aggressive", "conservative"),
        default="balanced",
        help="Pareto threshold preset.",
    )
    return parser


def _summary(result: Dict[str, Any]) -> Dict[str, Any]:
    selected = result.get("selected") if isinstance(result, dict) else None
    return {
        "preset": result.get("preset") if isinstance(result, dict) else None,
        "candidate_count": result.get("candidate_count") if isinstance(result, dict) else 0,
        "verified_candidate_count": result.get("verified_candidate_count") if isinstance(result, dict) else 0,
        "invalid_trial_count": result.get("invalid_trial_count") if isinstance(result, dict) else 0,
        "selected_params": selected.get("params") if isinstance(selected, dict) else None,
        "selected_metrics": {
            "score": selected.get("score"),
            "sharpe_ratio": selected.get("sharpe_ratio"),
            "max_drawdown_pct": selected.get("max_drawdown_pct"),
            "win_rate_pct": selected.get("win_rate_pct"),
            "total_pnl_pct": selected.get("metadata", {}).get("total_pnl_pct") if isinstance(selected, dict) else None,
            "profit_factor": selected.get("metadata", {}).get("profit_factor") if isinstance(selected, dict) else None,
            "total_trades": selected.get("metadata", {}).get("total_trades") if isinstance(selected, dict) else None,
        }
        if isinstance(selected, dict)
        else None,
    }


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    result = run_multi_objective_parameter_optimization(n_trials=max(1, args.trials), preset=args.preset)
    print(json.dumps(_summary(result), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
