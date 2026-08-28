# timing_filters.py
# =================
#
# Timing-based trade filters: hourly performance (A1), day-of-week (A2),
# monthly cycle (A3), and regime transition cooldown (A4).
#
# All filter functions return a dict with at least {"allowed": bool, "reason": str}.
# Size/confidence multipliers default to 1.0 when no adjustment is needed.

from __future__ import annotations


from core.exceptions import BEST_EFFORT_EXCEPTIONS
import json
import time
import calendar
from datetime import datetime, timezone
from pathlib import Path
from atomic_io import atomic_write_json
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
_TRADE_LOG_PATH = _ROOT / "metrics" / "trade_log.json"
_HOURLY_PERF_PATH = _ROOT / "data" / "hourly_performance.json"

# ---------------------------------------------------------------------------
# Module-level cache for hourly performance data
# ---------------------------------------------------------------------------
_hourly_cache: Optional[dict] = None
_hourly_cache_ts: float = 0.0
_HOURLY_CACHE_TTL: float = 300.0  # 5 minutes


def _load_hourly_cache() -> dict:
    """Load hourly performance JSON with a 300s TTL cache."""
    global _hourly_cache, _hourly_cache_ts
    now = time.monotonic()
    if _hourly_cache is not None and (now - _hourly_cache_ts) < _HOURLY_CACHE_TTL:
        return _hourly_cache
    try:
        data = json.loads(_HOURLY_PERF_PATH.read_text(encoding="utf-8"))
        _hourly_cache = data
        _hourly_cache_ts = now
        return data
    except BEST_EFFORT_EXCEPTIONS:
        return {}


# =========================================================================
# A1: Hourly Performance Filter
# =========================================================================

def analyze_hourly_performance(
    trade_log_path: str = "",
    output_path: str = "",
    min_trades_per_hour: int = 3,
    win_rate_threshold: float = 0.45,
    lookback_days: int = 90,
) -> dict:
    """
    Analyze trade_log.json grouped by UTC hour, write hourly_performance.json.

    Only considers trades from the last *lookback_days* (default 90 = ~3 months).
    Returns dict mapping hour (str) -> stats.
    """
    from datetime import timedelta
    from dateutil.parser import isoparse

    log_path = Path(trade_log_path) if trade_log_path else _TRADE_LOG_PATH
    out_path = Path(output_path) if output_path else _HOURLY_PERF_PATH

    trades = json.loads(log_path.read_text(encoding="utf-8"))

    # Cutoff: only last N days
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    # Group by UTC hour
    buckets: dict[int, list] = {h: [] for h in range(24)}
    total_included = 0
    for t in trades:
        ts_str = t.get("timestamp_open")
        if not ts_str:
            continue
        try:
            dt = isoparse(ts_str)
            # Ensure timezone-aware for comparison
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt < cutoff:
                continue
            hour = dt.hour
        except BEST_EFFORT_EXCEPTIONS:
            continue
        pnl_pct = t.get("pnl_pct")
        if pnl_pct is None:
            continue
        total_included += 1
        buckets[hour].append({
            "pnl_pct": float(pnl_pct),
            "win": float(pnl_pct) > 0,
        })

    result: dict = {}
    for h in range(24):
        entries = buckets[h]
        count = len(entries)
        if count == 0:
            result[str(h)] = {
                "count": 0,
                "win_rate": None,
                "avg_pnl_pct": None,
                "sufficient_data": False,
            }
            continue
        wins = sum(1 for e in entries if e["win"])
        wr = wins / count
        avg_pnl = sum(e["pnl_pct"] for e in entries) / count
        result[str(h)] = {
            "count": count,
            "win_rate": round(wr, 4),
            "avg_pnl_pct": round(avg_pnl, 4),
            "sufficient_data": count >= min_trades_per_hour,
        }

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lookback_days": lookback_days,
        "cutoff_date": cutoff.isoformat(),
        "total_trades_included": total_included,
        "min_trades_per_hour": min_trades_per_hour,
        "win_rate_threshold": win_rate_threshold,
        "hours": result,
    }
    atomic_write_json(out_path, payload)
    return result


def check_hour_filter(now_utc: Optional[datetime] = None) -> dict:
    """
    Check if the current UTC hour is historically profitable enough.

    Returns {"allowed": bool, "reason": str, "multiplier": float}
    """
    now = now_utc or datetime.now(timezone.utc)
    hour_key = str(now.hour)

    data = _load_hourly_cache()

    if not data.get("enabled", True):
        return {"allowed": True, "reason": f"Hour {now.hour}: hour_filter disabled", "multiplier": 1.0}

    hours = data.get("hours", {})
    threshold = data.get("win_rate_threshold", 0.45)

    info = hours.get(hour_key)
    if not info or not info.get("sufficient_data"):
        # Not enough data — fallback to session_filter
        try:
            from session_filter import is_trading_enabled
            if not is_trading_enabled(now):
                return {
                    "allowed": False,
                    "reason": f"Hour {now.hour}: insufficient data, session_filter blocked",
                    "multiplier": 1.0,
                }
        except BEST_EFFORT_EXCEPTIONS:
            pass
        return {"allowed": True, "reason": f"Hour {now.hour}: insufficient data, allowing", "multiplier": 1.0}

    wr = info.get("win_rate", 0.5)
    if wr < threshold:
        return {
            "allowed": False,
            "reason": f"Hour {now.hour}: win_rate={wr:.2%} < {threshold:.0%} (n={info['count']})",
            "multiplier": 1.0,
        }

    return {
        "allowed": True,
        "reason": f"Hour {now.hour}: win_rate={wr:.2%}, OK",
        "multiplier": 1.0,
    }


# =========================================================================
# A2: Day-of-Week Filter (data-validated)
# =========================================================================
_DOW_PERF_PATH = _ROOT / "data" / "daily_performance.json"
_dow_cache: Optional[dict] = None
_dow_cache_ts: float = 0.0


def analyze_daily_performance(
    trade_log_path: str = "",
    output_path: str = "",
    lookback_days: int = 90,
) -> dict:
    """
    Analyze trade_log.json grouped by day-of-week, write daily_performance.json.

    Returns dict mapping dow (str "0".."6") -> stats.
    """
    from datetime import timedelta
    from dateutil.parser import isoparse

    log_path = Path(trade_log_path) if trade_log_path else _TRADE_LOG_PATH
    out_path = Path(output_path) if output_path else _DOW_PERF_PATH

    trades = json.loads(log_path.read_text(encoding="utf-8"))
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    buckets: dict[int, list] = {d: [] for d in range(7)}
    for t in trades:
        ts_str = t.get("timestamp_open")
        if not ts_str:
            continue
        try:
            dt = isoparse(ts_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt < cutoff:
                continue
        except BEST_EFFORT_EXCEPTIONS:
            continue
        pnl_pct = t.get("pnl_pct")
        if pnl_pct is None:
            continue
        buckets[dt.weekday()].append(float(pnl_pct))

    result: dict = {}
    for d in range(7):
        entries = buckets[d]
        count = len(entries)
        if count == 0:
            result[str(d)] = {
                "name": day_names[d], "count": 0,
                "win_rate": None, "avg_pnl_pct": None,
            }
            continue
        wins = sum(1 for p in entries if p > 0)
        result[str(d)] = {
            "name": day_names[d],
            "count": count,
            "win_rate": round(wins / count, 4),
            "avg_pnl_pct": round(sum(entries) / count, 4),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lookback_days": lookback_days,
        "days": result,
    }
    atomic_write_json(out_path, payload)
    return result


def _load_dow_cache() -> dict:
    """Load daily performance data with 300s TTL cache."""
    global _dow_cache, _dow_cache_ts
    now = time.monotonic()
    if _dow_cache is not None and (now - _dow_cache_ts) < _HOURLY_CACHE_TTL:
        return _dow_cache
    try:
        data = json.loads(_DOW_PERF_PATH.read_text(encoding="utf-8"))
        _dow_cache = data
        _dow_cache_ts = now
        return data
    except BEST_EFFORT_EXCEPTIONS:
        return {}


# Structural risk rules per day (always applied regardless of data)
_DOW_STRUCTURAL_RULES: dict[int, dict] = {
    0: {"confidence_multiplier": 0.90, "size_multiplier": 1.0},   # Monday: gap risk
    4: {"confidence_multiplier": 1.0,  "size_multiplier": 0.70},  # Friday: weekend risk
    5: {"confidence_multiplier": 1.0,  "size_multiplier": 0.85},  # Saturday
    6: {"confidence_multiplier": 1.0,  "size_multiplier": 0.85},  # Sunday
}


def check_day_of_week_filter(
    now_utc: Optional[datetime] = None,
    spread_pct: float = 0.0,
) -> dict:
    """
    Day-of-week risk adjustments — combines structural rules with
    data-validated win rates from daily_performance.json.

    Returns {"allowed": bool, "confidence_multiplier": float,
             "size_multiplier": float, "reason": str}
    """
    now = now_utc or datetime.now(timezone.utc)
    dow = now.weekday()  # 0=Monday ... 6=Sunday
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    reasons: list[str] = []

    # Start with defaults
    conf_mult = 1.0
    size_mult = 1.0

    # 1. Apply structural rules
    structural = _DOW_STRUCTURAL_RULES.get(dow, {})
    if structural:
        conf_mult *= structural.get("confidence_multiplier", 1.0)
        size_mult *= structural.get("size_multiplier", 1.0)
        reasons.append(f"structural: conf={conf_mult:.2f}, size={size_mult:.2f}")

    # 2. Weekend spread check — hard block
    if dow in (5, 6) and spread_pct > 0.15:
        return {
            "allowed": False,
            "confidence_multiplier": conf_mult,
            "size_multiplier": size_mult,
            "reason": f"{day_names[dow]}: spread={spread_pct:.2%} > 0.15% — blocked",
        }

    # 3. Data-validated adjustments from daily_performance.json
    dow_data = _load_dow_cache()
    day_info = dow_data.get("days", {}).get(str(dow))
    if day_info and day_info.get("count", 0) >= 5:
        day_wr = day_info.get("win_rate")
        day_avg = day_info.get("avg_pnl_pct")
        if day_wr is not None:
            # If data shows this day is significantly worse than average,
            # apply additional penalty (scale: wr < 0.30 → extra 0.80x size)
            if day_wr < 0.30:
                size_mult *= 0.80
                reasons.append(f"data: wr={day_wr:.0%} < 30% → extra 0.80x")
            elif day_wr < 0.40:
                size_mult *= 0.90
                reasons.append(f"data: wr={day_wr:.0%} < 40% → extra 0.90x")
            # If data shows this day is clearly profitable, relax structural penalty slightly
            elif day_wr > 0.55 and day_avg is not None and day_avg > 0:
                # Don't exceed 1.0 but allow structural penalty to be softened
                if size_mult < 1.0:
                    size_mult = min(1.0, size_mult * 1.10)
                    reasons.append(f"data: wr={day_wr:.0%} > 55% → relaxed to {size_mult:.2f}")
    else:
        reasons.append("data: insufficient")

    reason_str = "; ".join(reasons) if reasons else "no adjustment"
    return {
        "allowed": True,
        "confidence_multiplier": round(conf_mult, 4),
        "size_multiplier": round(size_mult, 4),
        "reason": f"{day_names[dow]}: {reason_str}",
    }


# =========================================================================
# A3: Monthly Cycle Filter
# =========================================================================

def _is_last_friday_of_month(dt: datetime) -> bool:
    """Check if *dt* falls on the last Friday of its month."""
    if dt.weekday() != 4:  # Not a Friday
        return False
    _, days_in_month = calendar.monthrange(dt.year, dt.month)
    # Last Friday: no more Fridays left in the month
    return dt.day + 7 > days_in_month


def check_monthly_cycle_filter(now_utc: Optional[datetime] = None) -> dict:
    """
    Monthly calendar effects: options expiry (last Friday) and
    institutional rebalancing (first 3 days).

    Returns {"allowed": bool, "size_multiplier": float, "reason": str}
    """
    now = now_utc or datetime.now(timezone.utc)

    if _is_last_friday_of_month(now):
        return {
            "allowed": True,
            "size_multiplier": 0.60,
            "reason": "Last Friday of month (options expiry): size_multiplier=0.60",
        }

    if now.day <= 3:
        return {
            "allowed": True,
            "size_multiplier": 1.10,
            "reason": f"Day {now.day} of month (institutional rebalancing): size_multiplier=1.10",
        }

    return {
        "allowed": True,
        "size_multiplier": 1.0,
        "reason": "Normal monthly period",
    }


# =========================================================================
# A4: Regime Transition Filter
# =========================================================================

# Cooldown durations in hours for specific transitions
_TRANSITION_COOLDOWNS: dict[tuple[str, str], tuple[float, float]] = {
    # (from, to): (cooldown_hours, size_multiplier)
    ("RANGING", "HIGH_VOLATILITY"): (3.0, 0.50),
    ("TRENDING", "HIGH_VOLATILITY"): (3.0, 0.50),
    ("HIGH_VOLATILITY", "RANGING"): (4.0, 0.50),
}
_DEFAULT_COOLDOWN_HOURS = 2.0
_DEFAULT_COOLDOWN_MULT = 0.70


def check_regime_transition_filter(transition_info: Optional[dict] = None) -> dict:
    """
    Apply a cooldown after regime transitions.

    *transition_info* comes from ml/regime_detector.py's
    ``detect_regime_with_transition()`` and contains:
      - from_regime: str
      - to_regime: str
      - seconds_ago: float
      - cooldown_active: bool

    Returns {"allowed": bool, "size_multiplier": float,
             "cooldown_remaining_hours": float, "reason": str}
    """
    if not transition_info:
        return {
            "allowed": True,
            "size_multiplier": 1.0,
            "cooldown_remaining_hours": 0.0,
            "reason": "No transition info",
        }

    from_r = str(transition_info.get("from_regime", "")).upper()
    to_r = str(transition_info.get("to_regime", "")).upper()
    seconds_ago = float(transition_info.get("seconds_ago", 0))

    # Determine cooldown parameters for this transition pair
    key = (from_r, to_r)
    cooldown_hours, size_mult = _TRANSITION_COOLDOWNS.get(
        key, (_DEFAULT_COOLDOWN_HOURS, _DEFAULT_COOLDOWN_MULT)
    )
    cooldown_seconds = cooldown_hours * 3600

    remaining = max(0.0, cooldown_seconds - seconds_ago)
    remaining_hours = remaining / 3600

    if remaining > 0:
        return {
            "allowed": False,
            "size_multiplier": size_mult,
            "cooldown_remaining_hours": round(remaining_hours, 2),
            "reason": (
                f"Regime transition {from_r}->{to_r}: "
                f"cooldown {remaining_hours:.1f}h remaining, size_mult={size_mult}"
            ),
        }

    return {
        "allowed": True,
        "size_multiplier": 1.0,
        "cooldown_remaining_hours": 0.0,
        "reason": f"Regime transition {from_r}->{to_r}: cooldown expired",
    }
