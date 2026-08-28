from __future__ import annotations


from core.exceptions import BEST_EFFORT_EXCEPTIONS
import itertools
import math
import random
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional


ReportRunner = Callable[[Dict[str, Any], Any], Awaitable[Any]]


def _global_timestamps(frames: Dict[str, Any]) -> List[Any]:
    all_times = set()
    for df in (frames or {}).values():
        try:
            all_times.update(df.index.tolist())
        except BEST_EFFORT_EXCEPTIONS:
            continue
    return sorted(all_times)


def _slice_frames_by_bounds(
    frames: Dict[str, Any],
    *,
    start: Any = None,
    end: Any = None,
    allowed_times: Optional[set[Any]] = None,
    min_rows: int = 20,
) -> Dict[str, Any]:
    sliced: Dict[str, Any] = {}
    for symbol, df in (frames or {}).items():
        try:
            subset = df
            if allowed_times is not None:
                subset = subset[subset.index.isin(allowed_times)]
            if start is not None:
                subset = subset[subset.index >= start]
            if end is not None:
                subset = subset[subset.index < end]
        except BEST_EFFORT_EXCEPTIONS:
            continue
        if len(subset) >= int(min_rows):
            sliced[symbol] = subset
    return sliced


def _candidate_overrides(cfg: Any) -> List[Dict[str, Any]]:
    base_sl = max(0.5, float(getattr(cfg, "sl_atr_mult", 1.5) or 1.5))
    base_tp = max(0.5, float(getattr(cfg, "tp_atr_mult", 2.5) or 2.5))
    base_hold = max(8, int(getattr(cfg, "max_hold_bars", 96) or 96))
    base_wallet = max(0.01, float(getattr(cfg, "wallet_alloc_pct", 0.10) or 0.10))
    raw = [
        {},
        {
            "sl_atr_mult": round(base_sl * 0.80, 4),
            "tp_atr_mult": round(base_tp, 4),
        },
        {
            "sl_atr_mult": round(base_sl * 1.20, 4),
            "tp_atr_mult": round(base_tp * 1.20, 4),
        },
        {
            "sl_atr_mult": round(base_sl, 4),
            "tp_atr_mult": round(base_tp * 0.80, 4),
        },
        {
            "max_hold_bars": max(8, int(round(base_hold * 0.75))),
        },
        {
            "max_hold_bars": max(8, int(round(base_hold * 1.25))),
            "wallet_alloc_pct": round(min(0.30, base_wallet * 0.80), 4),
        },
    ]
    unique: List[Dict[str, Any]] = []
    seen = set()
    for item in raw:
        key = tuple(sorted(item.items()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _report_score(report: Any) -> float:
    trades = int(getattr(report, "total_trades", 0) or 0)
    if trades <= 0:
        return -1_000_000.0
    sharpe = float(getattr(report, "sharpe_ratio", 0.0) or 0.0)
    pnl_pct = float(getattr(report, "total_pnl_pct", 0.0) or 0.0)
    profit_factor = float(getattr(report, "profit_factor", 0.0) or 0.0)
    max_dd = float(getattr(report, "max_drawdown_pct", 0.0) or 0.0)
    win_rate = float(getattr(report, "win_rate", 0.0) or 0.0)
    return (
        pnl_pct
        + sharpe * 10.0
        + max(0.0, profit_factor - 1.0) * 15.0
        + win_rate * 0.10
        - max_dd * 0.75
        + min(trades, 50) * 0.10
    )


def _trade_dicts(report: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for trade in list(getattr(report, "trades", []) or []):
        if isinstance(trade, dict):
            out.append(dict(trade))
            continue
        if hasattr(trade, "__dict__"):
            out.append(dict(trade.__dict__))
    return out


def _annualized_sharpe(trade_returns: Iterable[float]) -> float:
    values = [float(item) for item in trade_returns]
    if len(values) < 2:
        return 0.0
    mean_r = sum(values) / len(values)
    variance = sum((item - mean_r) ** 2 for item in values) / max(1, len(values) - 1)
    std_r = math.sqrt(variance) if variance > 0 else 0.0
    if std_r <= 1e-12:
        return 0.0
    return mean_r / std_r * math.sqrt(252)


async def _evaluate_candidates(
    frames: Dict[str, Any],
    cfg: Any,
    *,
    run_backtest: ReportRunner,
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    evaluations: List[Dict[str, Any]] = []
    for overrides in candidates:
        candidate_cfg = replace(cfg, **overrides)
        report = await run_backtest(frames, candidate_cfg)
        evaluations.append(
            {
                "overrides": dict(overrides),
                "config": candidate_cfg,
                "report": report,
                "score": round(_report_score(report), 6),
                "trade_count": int(getattr(report, "total_trades", 0) or 0),
            }
        )
    evaluations.sort(key=lambda item: item["score"], reverse=True)
    return evaluations


async def run_anchored_walk_forward_optimization(
    frames: Dict[str, Any],
    cfg: Any,
    *,
    run_backtest: ReportRunner,
    train_days: int = 30,
    test_days: int = 7,
    step_days: int = 7,
) -> Dict[str, Any]:
    timestamps = _global_timestamps(frames)
    if not timestamps:
        return {
            "method": "anchored_walk_forward_optimization",
            "fold_count": 0,
            "folds": [],
            "combined_oos_trades": [],
            "oos_is_score_ratio": 0.0,
            "robust": False,
            "candidate_count": 0,
        }

    start = timestamps[0]
    end = timestamps[-1]
    test_delta = timedelta(days=max(1, int(test_days)))
    step_delta = timedelta(days=max(1, int(step_days)))
    train_end = start + timedelta(days=max(1, int(train_days)))
    candidates = _candidate_overrides(cfg)

    folds: List[Dict[str, Any]] = []
    all_oos_trades: List[Dict[str, Any]] = []
    cumulative_is = 0.0
    cumulative_oos = 0.0
    fold_id = 0

    while train_end + test_delta <= end:
        fold_id += 1
        test_end = train_end + test_delta
        train_frames = _slice_frames_by_bounds(frames, start=start, end=train_end)
        test_frames = _slice_frames_by_bounds(frames, start=train_end, end=test_end)
        if not train_frames or not test_frames:
            train_end += step_delta
            continue

        train_evals = await _evaluate_candidates(
            train_frames,
            cfg,
            run_backtest=run_backtest,
            candidates=candidates,
        )
        best = train_evals[0]
        oos_report = await run_backtest(test_frames, best["config"])
        oos_score = _report_score(oos_report)
        cumulative_is += max(0.0, float(best["score"]))
        cumulative_oos += max(0.0, float(oos_score))
        oos_trades = _trade_dicts(oos_report)
        all_oos_trades.extend(oos_trades)
        folds.append(
            {
                "fold": fold_id,
                "train_start": str(start),
                "train_end": str(train_end),
                "test_start": str(train_end),
                "test_end": str(test_end),
                "best_params": dict(best["overrides"]),
                "candidate_count": len(candidates),
                "is_score": round(float(best["score"]), 6),
                "oos_score": round(float(oos_score), 6),
                "is_pnl_pct": round(float(getattr(best["report"], "total_pnl_pct", 0.0) or 0.0), 4),
                "oos_pnl_pct": round(float(getattr(oos_report, "total_pnl_pct", 0.0) or 0.0), 4),
                "is_trades": int(getattr(best["report"], "total_trades", 0) or 0),
                "oos_trades": int(getattr(oos_report, "total_trades", 0) or 0),
            }
        )
        train_end += step_delta

    ratio = round(cumulative_oos / max(cumulative_is, 1e-9), 4) if cumulative_is > 0 else 0.0
    return {
        "method": "anchored_walk_forward_optimization",
        "fold_count": len(folds),
        "folds": folds,
        "combined_oos_trades": all_oos_trades,
        "oos_is_score_ratio": ratio,
        "robust": bool(folds and ratio >= 0.70),
        "candidate_count": len(candidates),
        "parameter_keys": ["sl_atr_mult", "tp_atr_mult", "max_hold_bars", "wallet_alloc_pct"],
    }


async def run_combinatorial_purged_cross_validation(
    frames: Dict[str, Any],
    cfg: Any,
    *,
    run_backtest: ReportRunner,
    n_groups: int = 6,
    test_group_count: int = 2,
    embargo_groups: int = 1,
) -> Dict[str, Any]:
    timestamps = _global_timestamps(frames)
    if len(timestamps) < max(40, n_groups * 4):
        return {
            "method": "combinatorial_purged_cross_validation",
            "fold_count": 0,
            "overfit_probability_pct": 0.0,
            "overfit_detected": False,
            "folds": [],
        }

    n_groups = max(3, int(n_groups))
    test_group_count = min(max(1, int(test_group_count)), n_groups - 1)
    group_size = max(1, len(timestamps) // n_groups)
    groups: List[List[Any]] = []
    for idx in range(n_groups):
        start = idx * group_size
        end = len(timestamps) if idx == n_groups - 1 else min(len(timestamps), (idx + 1) * group_size)
        groups.append(timestamps[start:end])

    candidates = _candidate_overrides(cfg)
    fold_details: List[Dict[str, Any]] = []
    overfit_flags = 0

    for combo in itertools.combinations(range(n_groups), test_group_count):
        test_group_indexes = set(combo)
        embargo_indexes = set()
        for item in combo:
            for offset in range(1, max(0, embargo_groups) + 1):
                if item - offset >= 0:
                    embargo_indexes.add(item - offset)
                if item + offset < n_groups:
                    embargo_indexes.add(item + offset)

        train_times: List[Any] = []
        test_times: List[Any] = []
        for idx, group in enumerate(groups):
            if idx in test_group_indexes:
                test_times.extend(group)
            elif idx not in embargo_indexes:
                train_times.extend(group)

        train_frames = _slice_frames_by_bounds(frames, allowed_times=set(train_times))
        test_frames = _slice_frames_by_bounds(frames, allowed_times=set(test_times))
        if not train_frames or not test_frames:
            continue

        train_evals = await _evaluate_candidates(
            train_frames,
            cfg,
            run_backtest=run_backtest,
            candidates=candidates,
        )
        test_evals = await _evaluate_candidates(
            test_frames,
            cfg,
            run_backtest=run_backtest,
            candidates=candidates,
        )
        test_ranks = {tuple(sorted(item["overrides"].items())): idx for idx, item in enumerate(test_evals)}
        selected = train_evals[0]
        rank = test_ranks.get(tuple(sorted(selected["overrides"].items())), len(test_evals))
        overfit = bool(test_evals and rank >= math.ceil(len(test_evals) / 2))
        if overfit:
            overfit_flags += 1
        fold_details.append(
            {
                "test_groups": list(combo),
                "embargo_groups": sorted(embargo_indexes),
                "selected_params": dict(selected["overrides"]),
                "selected_is_score": round(float(selected["score"]), 6),
                "selected_oos_rank": int(rank + 1),
                "candidate_count": len(test_evals),
                "overfit_fold": overfit,
            }
        )

    fold_count = len(fold_details)
    overfit_probability = round(overfit_flags / max(1, fold_count) * 100.0, 2)
    return {
        "method": "combinatorial_purged_cross_validation",
        "fold_count": fold_count,
        "n_groups": n_groups,
        "test_group_count": test_group_count,
        "embargo_groups": embargo_groups,
        "overfit_probability_pct": overfit_probability,
        "overfit_detected": bool(fold_count and overfit_probability > 50.0),
        "folds": fold_details,
    }


def run_monte_carlo_permutation_test(
    trade_returns: Iterable[float],
    *,
    n_permutations: int = 10_000,
    seed: int = 42,
) -> Dict[str, Any]:
    returns = [float(item) for item in trade_returns]
    if len(returns) < 5:
        return {
            "method": "sign_flip_sharpe_test",
            "n_permutations": int(n_permutations),
            "p_value": 1.0,
            "statistically_significant": False,
            "original_sharpe": 0.0,
            "note": "insufficient_trade_count",
        }

    original_sharpe = _annualized_sharpe(returns)
    mean_r = sum(returns) / len(returns)
    centered = [item - mean_r for item in returns]
    rng = random.Random(seed)
    null_sharpes: List[float] = []
    for _ in range(max(10, int(n_permutations))):
        synthetic = [(-1.0 if rng.random() < 0.5 else 1.0) * item for item in centered]
        null_sharpes.append(_annualized_sharpe(synthetic))
    exceed_count = sum(1 for value in null_sharpes if value >= original_sharpe)
    p_value = round((exceed_count + 1) / (len(null_sharpes) + 1), 6)
    sorted_nulls = sorted(null_sharpes)
    lo_idx = max(0, int(len(sorted_nulls) * 0.025))
    hi_idx = min(len(sorted_nulls) - 1, int(len(sorted_nulls) * 0.975))
    return {
        "method": "sign_flip_sharpe_test",
        "requested_metric": "sharpe_ratio",
        "n_permutations": int(n_permutations),
        "original_sharpe": round(original_sharpe, 6),
        "p_value": p_value,
        "statistically_significant": bool(p_value < 0.05),
        "null_sharpe_95_ci": [round(sorted_nulls[lo_idx], 6), round(sorted_nulls[hi_idx], 6)],
        "note": "Trade-order shuffling is path-sensitive but Sharpe is order-invariant, so sign-flip is used for significance.",
    }


def _classify_regime_series(df: Any, *, lookback: int = 96, trend_threshold: float = 0.02) -> Any:
    close = df["close"].astype(float)
    returns = close.pct_change()
    trend = close.pct_change(max(2, int(lookback))).fillna(0.0)
    volatility = returns.rolling(max(8, int(lookback // 2))).std().fillna(0.0)
    vol_threshold = float(volatility.quantile(0.75)) if len(volatility) else 0.0
    regime = []
    for idx in range(len(df)):
        if vol_threshold > 0 and float(volatility.iloc[idx]) >= vol_threshold:
            regime.append("high_vol")
        elif float(trend.iloc[idx]) > trend_threshold:
            regime.append("bull")
        elif float(trend.iloc[idx]) < -trend_threshold:
            regime.append("bear")
        else:
            regime.append("sideways")
    return df.assign(_validation_regime=regime)["_validation_regime"]


def _trade_regime(symbol_frames: Dict[str, Any], trade: Dict[str, Any]) -> str:
    symbol = str(trade.get("symbol", ""))
    frame = symbol_frames.get(symbol)
    if frame is None or "_validation_regime" not in getattr(frame, "columns", []):
        return "unknown"
    raw_entry = trade.get("entry_time")
    if not raw_entry:
        return "unknown"
    try:
        entry_dt = datetime.fromisoformat(str(raw_entry).replace("Z", "+00:00"))
    except ValueError:
        return "unknown"
    try:
        position = int(frame.index.searchsorted(entry_dt, side="right")) - 1
    except BEST_EFFORT_EXCEPTIONS:
        return "unknown"
    if position < 0 or position >= len(frame):
        return "unknown"
    try:
        return str(frame["_validation_regime"].iloc[position])
    except BEST_EFFORT_EXCEPTIONS:
        return "unknown"


def run_regime_conditional_backtest(
    frames: Dict[str, Any],
    report: Any,
) -> Dict[str, Any]:
    if not frames or not getattr(report, "trades", None):
        return {
            "method": "regime_conditional_backtest",
            "regimes": {},
            "no_trade_regimes": [],
            "all_regimes_above_50": False,
        }

    prepared_frames: Dict[str, Any] = {}
    for symbol, df in frames.items():
        try:
            prepared_frames[symbol] = df.assign(_validation_regime=_classify_regime_series(df))
        except BEST_EFFORT_EXCEPTIONS:
            continue

    stats: Dict[str, Dict[str, Any]] = {}
    for trade in _trade_dicts(report):
        regime = _trade_regime(prepared_frames, trade)
        bucket = stats.setdefault(
            regime,
            {
                "trade_count": 0,
                "wins": 0,
                "total_pnl": 0.0,
                "returns": [],
            },
        )
        pnl = float(trade.get("pnl", 0.0) or 0.0)
        pnl_pct = float(trade.get("pnl_pct", 0.0) or 0.0)
        bucket["trade_count"] += 1
        bucket["wins"] += 1 if pnl > 0 else 0
        bucket["total_pnl"] += pnl
        bucket["returns"].append(pnl_pct / 100.0)

    regimes: Dict[str, Any] = {}
    no_trade_regimes: List[str] = []
    all_positive = True
    for name, bucket in stats.items():
        count = int(bucket["trade_count"])
        win_rate = round(bucket["wins"] / max(1, count) * 100.0, 2)
        avg_return = round(sum(bucket["returns"]) / max(1, len(bucket["returns"])) * 100.0, 4)
        sharpe = round(_annualized_sharpe(bucket["returns"]), 4)
        regimes[name] = {
            "trade_count": count,
            "win_rate": win_rate,
            "total_pnl": round(float(bucket["total_pnl"]), 4),
            "avg_trade_return_pct": avg_return,
            "sharpe_ratio": sharpe,
        }
        if count > 0 and win_rate < 45.0:
            no_trade_regimes.append(name)
        if count > 0 and win_rate <= 50.0:
            all_positive = False

    return {
        "method": "regime_conditional_backtest",
        "regimes": regimes,
        "no_trade_regimes": sorted(no_trade_regimes),
        "all_regimes_above_50": bool(regimes and all_positive),
    }


async def run_advanced_validation_suite(
    *,
    frames: Dict[str, Any],
    cfg: Any,
    run_backtest: ReportRunner,
    baseline_report: Any = None,
    walkforward_report: Any = None,
    train_days: int = 30,
    test_days: int = 7,
    step_days: int = 7,
    permutation_iterations: int = 10_000,
    cpcv_groups: int = 6,
    cpcv_test_groups: int = 2,
    cpcv_embargo_groups: int = 1,
) -> Dict[str, Any]:
    anchored = {}
    if walkforward_report is not None:
        anchored = dict((getattr(walkforward_report, "advanced_validation", {}) or {}).get("anchored_walk_forward", {}) or {})
    if not anchored:
        anchored = await run_anchored_walk_forward_optimization(
            frames,
            cfg,
            run_backtest=run_backtest,
            train_days=train_days,
            test_days=test_days,
            step_days=step_days,
        )

    trade_source = walkforward_report or baseline_report
    trade_returns = [
        float(trade.get("pnl_pct", 0.0) or 0.0) / 100.0
        for trade in _trade_dicts(trade_source) if isinstance(trade, dict)
    ] if trade_source is not None else []

    permutation = run_monte_carlo_permutation_test(
        trade_returns,
        n_permutations=permutation_iterations,
    )
    cpcv = await run_combinatorial_purged_cross_validation(
        frames,
        cfg,
        run_backtest=run_backtest,
        n_groups=cpcv_groups,
        test_group_count=cpcv_test_groups,
        embargo_groups=cpcv_embargo_groups,
    )
    regime = run_regime_conditional_backtest(frames, trade_source)

    summary = {
        "robust_walk_forward": bool(anchored.get("robust", False)),
        "oos_is_score_ratio": float(anchored.get("oos_is_score_ratio", 0.0) or 0.0),
        "overfit_probability_pct": float(cpcv.get("overfit_probability_pct", 0.0) or 0.0),
        "statistically_significant": bool(permutation.get("statistically_significant", False)),
        "no_trade_regimes": list(regime.get("no_trade_regimes", []) or []),
    }

    return {
        "anchored_walk_forward": {k: v for k, v in anchored.items() if k != "combined_oos_trades"},
        "cpcv": cpcv,
        "permutation_test": permutation,
        "regime_conditional": regime,
        "summary": summary,
    }


__all__ = [
    "run_anchored_walk_forward_optimization",
    "run_combinatorial_purged_cross_validation",
    "run_monte_carlo_permutation_test",
    "run_regime_conditional_backtest",
    "run_advanced_validation_suite",
]
