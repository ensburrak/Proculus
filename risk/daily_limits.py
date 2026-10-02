from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "state" / "daily_limits.json"


@dataclass(frozen=True)
class LossLimitDecision:
    allowed: bool
    risk_scale: float
    hedge_only: bool
    reason: str
    daily_progress: float
    weekly_progress: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_config() -> dict[str, Any]:
    try:
        data = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _clock(cfg: dict[str, Any] | None = None) -> tuple[str, str]:
    config = cfg or _load_config()
    risk = config.get("risk") if isinstance(config.get("risk"), dict) else {}
    tz = ZoneInfo(str(risk.get("daily_loss_timezone") or "Europe/Istanbul"))
    now = datetime.now(tz)
    return now.strftime("%Y-%m-%d"), now.strftime("%Y-W%W")


def _read() -> dict[str, Any]:
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write(data: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATE_PATH)


def record_pnl(pnl_abs: float, *, config: dict[str, Any] | None = None) -> dict[str, Any]:
    day, week = _clock(config)
    state = _read()
    daily = state.get("daily") if isinstance(state.get("daily"), dict) else {}
    weekly = state.get("weekly") if isinstance(state.get("weekly"), dict) else {}
    daily[day] = float(daily.get(day, 0.0) or 0.0) + float(pnl_abs)
    weekly[week] = float(weekly.get(week, 0.0) or 0.0) + float(pnl_abs)
    state["daily"] = daily
    state["weekly"] = weekly
    state["updated_at"] = datetime.utcnow().isoformat() + "Z"
    _write(state)
    return state


def get_daily_realized_pnl(*, config: dict[str, Any] | None = None) -> float:
    day, _ = _clock(config)
    daily = _read().get("daily")
    if not isinstance(daily, dict):
        return 0.0
    try:
        return float(daily.get(day, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def evaluate_loss_limits(equity: float, *, config: dict[str, Any] | None = None) -> LossLimitDecision:
    cfg = config or _load_config()
    risk = cfg.get("risk") if isinstance(cfg.get("risk"), dict) else {}
    thresholds = risk.get("daily_limits") if isinstance(risk.get("daily_limits"), dict) else {}
    day, week = _clock(cfg)
    state = _read()
    daily_map = state.get("daily") if isinstance(state.get("daily"), dict) else {}
    weekly_map = state.get("weekly") if isinstance(state.get("weekly"), dict) else {}

    equity_f = max(float(equity), 1e-9)
    daily_limit = equity_f * float(risk.get("daily_loss_limit_pct", 0.005) or 0.005)
    weekly_limit = equity_f * float(risk.get("weekly_loss_limit_pct", 0.02) or 0.02)
    daily_pnl = float(daily_map.get(day, 0.0) or 0.0)
    weekly_pnl = float(weekly_map.get(week, 0.0) or 0.0)
    daily_progress = abs(min(daily_pnl, 0.0)) / max(daily_limit, 1e-9)
    weekly_progress = abs(min(weekly_pnl, 0.0)) / max(weekly_limit, 1e-9)

    if weekly_progress >= 1.0:
        return LossLimitDecision(False, 0.0, False, "weekly_loss_stop", daily_progress, weekly_progress)
    if daily_progress >= 1.0:
        return LossLimitDecision(False, 0.0, False, "daily_loss_stop", daily_progress, weekly_progress)

    hedge_at = float(thresholds.get("hedge_only_threshold", 0.75) or 0.75)
    reduce_at = float(thresholds.get("reduce_risk_threshold", 0.50) or 0.50)
    if daily_progress >= hedge_at:
        return LossLimitDecision(False, 0.0, True, "daily_hedge_only", daily_progress, weekly_progress)

    reduced = daily_progress >= reduce_at or weekly_progress >= 0.75
    return LossLimitDecision(True, 0.5 if reduced else 1.0, False, "ok", daily_progress, weekly_progress)


__all__ = ["LossLimitDecision", "evaluate_loss_limits", "get_daily_realized_pnl", "record_pnl"]
