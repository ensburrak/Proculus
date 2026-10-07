from __future__ import annotations

import math
from typing import Any, Iterable

import pandas as pd


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


def _utc_timestamp(value: Any) -> pd.Timestamp | None:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp


def _normalized_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close"}
    if not required.issubset(frame.columns):
        return pd.DataFrame(columns=sorted(required))
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return (
        out.dropna(subset=["timestamp", "open", "high", "low", "close"])
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def _adverse_entry(raw: float, side: str, slippage: float) -> float:
    return raw * (1.0 + slippage) if side == "long" else raw * (1.0 - slippage)


def _adverse_exit(raw: float, side: str, slippage: float) -> float:
    return raw * (1.0 - slippage) if side == "long" else raw * (1.0 + slippage)


def score_shadow_candidate(
    candidate: dict[str, Any],
    frame: pd.DataFrame,
    *,
    fee_bps: float,
    slippage_bps: float,
    funding_bps_per_8h: float,
) -> dict[str, Any]:
    key = _candidate_key(candidate)
    symbol = str(candidate.get("symbol") or "")
    base = {
        "candidate_key": key,
        "family": str(candidate.get("family") or ""),
        "setup_id": str(candidate.get("setup_id") or ""),
        "symbol": symbol,
        "side": str(candidate.get("side") or "").strip().lower(),
        "bar_time": str(candidate.get("bar_time") or ""),
        "fee_bps_per_side": float(fee_bps),
        "slippage_bps_per_side": float(slippage_bps),
        "funding_bps_per_8h": float(funding_bps_per_8h),
        "execution_authority": False,
        "promotion_authority": False,
    }

    if candidate.get("shadow_only") is not True or candidate.get("execution_authority") is not False:
        return {
            **base,
            "status": "invalid",
            "exit_reason": "candidate_not_shadow_only",
            "r_value": None,
        }

    side = str(candidate.get("side") or "").strip().lower()
    if side not in {"long", "short"}:
        return {
            **base,
            "status": "invalid",
            "exit_reason": "invalid_side",
            "r_value": None,
        }

    bar_time = _utc_timestamp(candidate.get("bar_time"))
    if bar_time is None:
        return {
            **base,
            "status": "invalid",
            "exit_reason": "invalid_bar_time",
            "r_value": None,
        }
    decision_close = bar_time + pd.Timedelta(hours=4)

    try:
        atr = float(candidate.get("atr_4h"))
        stop_atr_mult = float(candidate.get("stop_atr_mult"))
        trail_atr_mult = float(candidate.get("trail_atr_mult"))
        breakeven_after_r = float(candidate.get("breakeven_after_r"))
        trail_after_r = float(candidate.get("trail_after_r"))
        decay_bars = int(candidate.get("decay_bars_15m"))
        max_hold_bars = int(candidate.get("max_hold_bars_15m"))
    except (TypeError, ValueError, OverflowError):
        return {
            **base,
            "status": "invalid",
            "exit_reason": "invalid_candidate_parameters",
            "r_value": None,
        }

    if (
        not all(
            math.isfinite(value)
            for value in (
                atr,
                stop_atr_mult,
                trail_atr_mult,
                breakeven_after_r,
                trail_after_r,
            )
        )
        or atr <= 0.0
        or stop_atr_mult <= 0.0
        or trail_atr_mult <= 0.0
        or decay_bars <= 0
        or max_hold_bars <= 0
    ):
        return {
            **base,
            "status": "invalid",
            "exit_reason": "invalid_candidate_parameters",
            "r_value": None,
        }

    candles = _normalized_frame(frame)
    if candles.empty:
        return {
            **base,
            "status": "pending",
            "exit_reason": "missing_market_data",
            "r_value": None,
        }

    times = pd.DatetimeIndex(candles["timestamp"])
    entry_idx = int(times.searchsorted(decision_close, side="left"))
    if entry_idx >= len(candles):
        return {
            **base,
            "status": "pending",
            "exit_reason": "entry_bar_not_available",
            "r_value": None,
        }

    fee = float(fee_bps) / 10_000.0
    slippage = float(slippage_bps) / 10_000.0
    funding_per_8h = float(funding_bps_per_8h) / 10_000.0
    entry_timestamp = pd.Timestamp(candles.iloc[entry_idx]["timestamp"])
    if entry_timestamp != decision_close:
        return {
            **base,
            "status": "pending",
            "exit_reason": "entry_bar_cadence_gap",
            "expected_entry_time": decision_close.isoformat(),
            "first_available_time": entry_timestamp.isoformat(),
            "r_value": None,
        }

    raw_entry = float(candles.iloc[entry_idx]["open"])
    fill = _adverse_entry(raw_entry, side, slippage)
    stop_distance = atr * stop_atr_mult
    risk_pct = stop_distance / max(fill, 1e-12)
    if not math.isfinite(risk_pct) or risk_pct <= 0.0:
        return {
            **base,
            "status": "invalid",
            "exit_reason": "invalid_stop_distance",
            "r_value": None,
        }

    current_stop = fill - stop_distance if side == "long" else fill + stop_distance
    current_stop_reason = "v5_initial_stop"
    peak = fill
    last_required_idx = entry_idx + max_hold_bars - 1
    max_available_idx = len(candles) - 1
    end_idx = min(max_available_idx, last_required_idx)
    bars = 0

    def complete(raw_exit: float, exit_idx: int, reason: str) -> dict[str, Any]:
        exit_price = _adverse_exit(raw_exit, side, slippage)
        gross_pct = (
            (exit_price - fill) / fill
            if side == "long"
            else (fill - exit_price) / fill
        )
        exit_fee_pct = fee * abs(exit_price / fill)
        funding_intervals = bars // 32
        funding_pct = funding_per_8h * funding_intervals
        net_pct = gross_pct - fee - exit_fee_pct - funding_pct
        r_value = net_pct / risk_pct
        return {
            **base,
            "status": "complete",
            "entry_time": pd.Timestamp(candles.iloc[entry_idx]["timestamp"]).isoformat(),
            "entry_price": fill,
            "exit_time": pd.Timestamp(candles.iloc[exit_idx]["timestamp"]).isoformat(),
            "exit_price": exit_price,
            "exit_reason": reason,
            "bars_held": bars,
            "gross_return_pct": gross_pct * 100.0,
            "net_return_pct": net_pct * 100.0,
            "risk_pct": risk_pct,
            "r_value": r_value,
            "funding_intervals_8h": funding_intervals,
        }

    previous_timestamp: pd.Timestamp | None = None
    for idx in range(entry_idx, end_idx + 1):
        timestamp = pd.Timestamp(candles.iloc[idx]["timestamp"])
        if (
            previous_timestamp is not None
            and timestamp - previous_timestamp != pd.Timedelta(minutes=15)
        ):
            return {
                **base,
                "status": "pending",
                "entry_time": entry_timestamp.isoformat(),
                "entry_price": fill,
                "exit_reason": "market_data_gap",
                "gap_after": previous_timestamp.isoformat(),
                "gap_before": timestamp.isoformat(),
                "r_value": None,
            }
        previous_timestamp = timestamp
        bars += 1
        row = candles.iloc[idx]
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        if side == "long":
            peak = max(peak, high)
            if low <= current_stop:
                return complete(current_stop, idx, current_stop_reason)
        else:
            peak = min(peak, low)
            if high >= current_stop:
                return complete(current_stop, idx, current_stop_reason)

        mfe_r = (
            (peak - fill) / stop_distance
            if side == "long"
            else (fill - peak) / stop_distance
        )
        cost_buffer = fill * (2.0 * fee + 2.0 * slippage)

        if mfe_r >= breakeven_after_r:
            breakeven = fill + cost_buffer if side == "long" else fill - cost_buffer
            updated_stop = (
                max(current_stop, breakeven)
                if side == "long"
                else min(current_stop, breakeven)
            )
            if updated_stop != current_stop:
                current_stop = updated_stop
                current_stop_reason = "v5_breakeven_stop"

        if mfe_r >= trail_after_r:
            trail = trail_atr_mult * atr
            candidate_stop = close - trail if side == "long" else close + trail
            updated_stop = (
                max(current_stop, candidate_stop)
                if side == "long"
                else min(current_stop, candidate_stop)
            )
            if updated_stop != current_stop:
                current_stop = updated_stop
                current_stop_reason = "v5_trailing_stop"

        if bars >= decay_bars and mfe_r < breakeven_after_r:
            return complete(close, idx, "v5_time_decay")

        if bars >= max_hold_bars:
            return complete(close, idx, "v5_max_hold")

    return {
        **base,
        "status": "pending",
        "entry_time": pd.Timestamp(candles.iloc[entry_idx]["timestamp"]).isoformat(),
        "entry_price": fill,
        "exit_reason": "insufficient_future_data",
        "bars_observed": bars,
        "bars_required_max": max_hold_bars,
        "r_value": None,
    }


def summarize_candidate_outcomes(outcomes: Iterable[dict[str, Any]]) -> dict[str, Any]:
    unique: dict[str, dict[str, Any]] = {}
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            continue
        key = str(outcome.get("candidate_key") or "")
        if key and key not in unique:
            unique[key] = outcome

    complete = [item for item in unique.values() if item.get("status") == "complete"]
    pending = [item for item in unique.values() if item.get("status") == "pending"]
    invalid = [item for item in unique.values() if item.get("status") == "invalid"]
    r_values = [
        float(item["r_value"])
        for item in complete
        if item.get("r_value") is not None and math.isfinite(float(item["r_value"]))
    ]
    wins = [value for value in r_values if value > 0.0]
    losses = [value for value in r_values if value < 0.0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    symbols = sorted(
        {
            str(item.get("symbol") or "")
            for item in complete
            if str(item.get("symbol") or "")
        }
    )
    return {
        "unique_candidates": len(unique),
        "complete_trades": len(complete),
        "pending_trades": len(pending),
        "invalid_trades": len(invalid),
        "win_rate_pct": (
            100.0 * len(wins) / len(r_values) if r_values else 0.0
        ),
        "profit_factor": (
            gross_win / gross_loss
            if gross_loss > 0.0
            else (999.0 if wins else 0.0)
        ),
        "expectancy_r": (
            sum(r_values) / len(r_values) if r_values else 0.0
        ),
        "symbols": symbols,
        "symbol_count": len(symbols),
        "execution_authority": False,
        "promotion_authority": False,
    }


def evaluate_outcome_gate(
    primary: dict[str, Any],
    stress: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    gate = policy.get("outcome_gate")
    if not isinstance(gate, dict):
        raise ValueError("shadow policy missing outcome_gate")
    primary_rule = gate.get("primary_5bps")
    stress_rule = gate.get("stress_15bps")
    if not isinstance(primary_rule, dict) or not isinstance(stress_rule, dict):
        raise ValueError("shadow policy outcome gate missing cost scenarios")

    primary_symbols = set(str(x) for x in primary.get("symbols") or [])
    stress_symbols = set(str(x) for x in stress.get("symbols") or [])
    common_symbols = primary_symbols & stress_symbols
    checks = {
        "primary_min_complete_trades": int(primary.get("complete_trades") or 0)
        >= int(primary_rule.get("min_complete_trades") or 0),
        "primary_profit_factor": float(primary.get("profit_factor") or 0.0)
        >= float(primary_rule.get("min_profit_factor") or 0.0),
        "primary_expectancy": float(primary.get("expectancy_r") or 0.0)
        >= float(primary_rule.get("min_expectancy_r") or 0.0),
        "stress_min_complete_trades": int(stress.get("complete_trades") or 0)
        >= int(stress_rule.get("min_complete_trades") or 0),
        "stress_profit_factor": float(stress.get("profit_factor") or 0.0)
        >= float(stress_rule.get("min_profit_factor") or 0.0),
        "stress_expectancy": float(stress.get("expectancy_r") or 0.0)
        > float(stress_rule.get("min_expectancy_r_exclusive") or 0.0),
        "minimum_complete_symbols": len(common_symbols)
        >= int(gate.get("min_complete_symbols") or 0),
        "zero_invalid_outcomes": (
            int(primary.get("invalid_trades") or 0) == 0
            and int(stress.get("invalid_trades") or 0) == 0
            if bool(gate.get("require_zero_invalid_outcomes", False))
            else True
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "common_complete_symbols": sorted(common_symbols),
        "promotion_authority": False,
        "execution_authority": False,
        "next_gate": (
            "exact_private_replay_and_manual_promotion_review"
            if all(checks.values())
            else "continue_prospective_shadow"
        ),
    }


__all__ = [
    "evaluate_outcome_gate",
    "score_shadow_candidate",
    "summarize_candidate_outcomes",
]
