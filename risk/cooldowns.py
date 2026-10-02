from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "state" / "cooldowns.json"


def _load_config() -> dict[str, Any]:
    try:
        data = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


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


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def cooldown_status(symbol: str, *, now: datetime | None = None) -> dict[str, Any]:
    current = now or datetime.now(timezone.utc)
    state = _read()
    symbols = state.get("symbols") if isinstance(state.get("symbols"), dict) else {}
    row = symbols.get(symbol) if isinstance(symbols.get(symbol), dict) else {}
    symbol_until = _parse_iso(row.get("cooldown_until"))
    global_until = _parse_iso(state.get("global_cooldown_until"))
    symbol_blocked = symbol_until is not None and symbol_until > current
    global_blocked = global_until is not None and global_until > current
    return {
        "blocked": symbol_blocked or global_blocked,
        "symbol_blocked": symbol_blocked,
        "global_blocked": global_blocked,
        "symbol_cooldown_until": symbol_until.isoformat() if symbol_until else None,
        "global_cooldown_until": global_until.isoformat() if global_until else None,
        "symbol_loss_streak": int(row.get("loss_streak", 0) or 0),
        "global_loss_streak": int(state.get("global_loss_streak", 0) or 0),
    }


def record_trade_outcome(
    symbol: str,
    *,
    pnl_pct: float | None = None,
    pnl_abs: float | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config or _load_config()
    risk = cfg.get("risk") if isinstance(cfg.get("risk"), dict) else {}
    settings = risk.get("cooldowns") if isinstance(risk.get("cooldowns"), dict) else {}
    now = datetime.now(timezone.utc)
    state = _read()
    symbols = state.get("symbols") if isinstance(state.get("symbols"), dict) else {}
    row = symbols.get(symbol) if isinstance(symbols.get(symbol), dict) else {}

    loss = False
    if pnl_abs is not None:
        loss = float(pnl_abs) < 0.0
    elif pnl_pct is not None:
        loss = float(pnl_pct) < 0.0

    if not loss:
        row["loss_streak"] = 0
        state["global_loss_streak"] = 0
    else:
        row["loss_streak"] = int(row.get("loss_streak", 0) or 0) + 1
        state["global_loss_streak"] = int(state.get("global_loss_streak", 0) or 0) + 1

        loss_pct_abs = abs(float(pnl_pct or 0.0))
        small = float(settings.get("small_loss_pct", 1.0) or 1.0)
        medium = float(settings.get("medium_loss_pct", 3.0) or 3.0)
        if loss_pct_abs >= medium:
            hours = float(settings.get("large_loss_cooldown_hours", 12) or 12)
        elif loss_pct_abs >= small:
            hours = float(settings.get("medium_loss_cooldown_hours", 4) or 4)
        else:
            hours = float(settings.get("small_loss_cooldown_hours", 1) or 1)

        streak_limit = int(settings.get("symbol_loss_streak", 3) or 3)
        if row["loss_streak"] >= streak_limit:
            hours = max(hours, float(settings.get("symbol_loss_streak_cooldown_hours", 4) or 4))
        row["cooldown_until"] = (now + timedelta(hours=max(0.0, hours))).isoformat()

        global_limit = int(settings.get("global_loss_count", 5) or 5)
        if int(state["global_loss_streak"]) >= global_limit:
            state["global_cooldown_until"] = (
                now + timedelta(hours=float(settings.get("global_cooldown_hours", 1) or 1))
            ).isoformat()
            state["global_loss_streak"] = 0

    row["last_outcome_at"] = now.isoformat()
    row["last_pnl_pct"] = pnl_pct
    row["last_pnl_abs"] = pnl_abs
    symbols[symbol] = row
    state["symbols"] = symbols
    state["updated_at"] = now.isoformat()
    _write(state)
    return cooldown_status(symbol, now=now)


def is_symbol_in_cooldown(symbol: str) -> bool:
    return bool(cooldown_status(symbol).get("blocked"))


__all__ = ["cooldown_status", "is_symbol_in_cooldown", "record_trade_outcome"]
