from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from execution.trade_logger import log_closed_trade
from risk.cooldowns import cooldown_status, record_trade_outcome
from risk.daily_limits import evaluate_loss_limits, record_pnl
from risk.stop_manager import calculate_adaptive_take_profit, compute_stop_loss

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "learning-probe-progress-v2"


def _probe_cfg(config: dict[str, Any]) -> dict[str, Any]:
    pipeline = config.get("pipeline_v2") if isinstance(config.get("pipeline_v2"), dict) else {}
    probe = pipeline.get("learning_probe_mode") if isinstance(pipeline.get("learning_probe_mode"), dict) else {}
    return probe


def _paper_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("paper_trading") if isinstance(config.get("paper_trading"), dict) else {}


def _state_path(config: dict[str, Any], override: Path | None = None) -> Path:
    if override is not None:
        return override
    raw = str(_probe_cfg(config).get("progress_file") or "state/learning_probe_progress.json")
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def _initial_state(config: dict[str, Any]) -> dict[str, Any]:
    paper = _paper_cfg(config)
    return {
        "schema": SCHEMA,
        "equity": float(paper.get("initial_capital", 10_000.0) or 10_000.0),
        "open_positions": {},
        "setups": {},
        "daily_risk_used_pct": {},
        "closed_trades": 0,
        "updated_at": None,
    }


def _load_state(config: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    target = _state_path(config, path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _initial_state(config)
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        return _initial_state(config)
    payload.setdefault("open_positions", {})
    payload.setdefault("setups", {})
    payload.setdefault("daily_risk_used_pct", {})
    payload.setdefault("closed_trades", 0)
    payload.setdefault("equity", float(_paper_cfg(config).get("initial_capital", 10_000.0) or 10_000.0))
    return payload


def _save_state(config: dict[str, Any], state: dict[str, Any], path: Path | None = None) -> None:
    target = _state_path(config, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    state["schema"] = SCHEMA
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(target)


def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        raw = float(value)
        if raw > 1e12:
            raw /= 1000.0
        try:
            return datetime.fromtimestamp(raw, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        try:
            raw = float(text)
        except ValueError:
            return None
        return _parse_ts(raw)


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _market_bar(item: dict[str, Any]) -> dict[str, Any] | None:
    raw = item.get("market_bar")
    if not isinstance(raw, dict):
        return None
    timestamp = _parse_ts(raw.get("timestamp"))
    high = _number(raw.get("high"))
    low = _number(raw.get("low"))
    close = _number(raw.get("close"))
    if timestamp is None or None in (high, low, close):
        return None
    return {
        "timestamp": timestamp,
        "high": float(high),
        "low": float(low),
        "close": float(close),
    }


def _day_key(ts: datetime, config: dict[str, Any]) -> str:
    risk = config.get("risk") if isinstance(config.get("risk"), dict) else {}
    tz = ZoneInfo(str(risk.get("daily_loss_timezone") or "Europe/Istanbul"))
    return ts.astimezone(tz).strftime("%Y-%m-%d")


def _setup_state(state: dict[str, Any], setup_id: str) -> dict[str, Any]:
    setups = state.setdefault("setups", {})
    row = setups.get(setup_id)
    if not isinstance(row, dict):
        row = {
            "closed_trades": 0,
            "wins": 0,
            "losses": 0,
            "sum_r": 0.0,
            "gross_win_r": 0.0,
            "gross_loss_r": 0.0,
            "recent_r": [],
            "profit_factor": 0.0,
            "expectancy_r": 0.0,
            "recent_profit_factor": 0.0,
            "recent_expectancy_r": 0.0,
            "frozen": False,
            "freeze_reason": None,
            "target_reached": False,
            "last_open_at": None,
            "last_close_at": None,
        }
        setups[setup_id] = row
    return row


def _update_setup_outcome(
    state: dict[str, Any],
    *,
    setup_id: str,
    r_multiple: float,
    closed_at: datetime,
    config: dict[str, Any],
) -> dict[str, Any]:
    probe = _probe_cfg(config)
    row = _setup_state(state, setup_id)
    row["closed_trades"] = int(row.get("closed_trades", 0) or 0) + 1
    row["sum_r"] = float(row.get("sum_r", 0.0) or 0.0) + float(r_multiple)
    if r_multiple > 0:
        row["wins"] = int(row.get("wins", 0) or 0) + 1
        row["gross_win_r"] = float(row.get("gross_win_r", 0.0) or 0.0) + float(r_multiple)
    elif r_multiple < 0:
        row["losses"] = int(row.get("losses", 0) or 0) + 1
        row["gross_loss_r"] = float(row.get("gross_loss_r", 0.0) or 0.0) + abs(float(r_multiple))

    gross_loss = float(row.get("gross_loss_r", 0.0) or 0.0)
    gross_win = float(row.get("gross_win_r", 0.0) or 0.0)
    row["profit_factor"] = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
    row["expectancy_r"] = float(row["sum_r"]) / max(1, int(row["closed_trades"]))

    min_freeze = max(1, int(probe.get("early_freeze_min_trades", 20) or 20))
    recent = list(row.get("recent_r") or [])
    recent.append(float(r_multiple))
    recent = recent[-max(50, min_freeze):]
    row["recent_r"] = recent
    recent_window = recent[-min_freeze:]
    if recent_window:
        recent_wins = sum(value for value in recent_window if value > 0)
        recent_losses = abs(sum(value for value in recent_window if value < 0))
        row["recent_profit_factor"] = (
            recent_wins / recent_losses
            if recent_losses > 0
            else (999.0 if recent_wins > 0 else 0.0)
        )
        row["recent_expectancy_r"] = sum(recent_window) / len(recent_window)

    if int(row["closed_trades"]) >= min_freeze:
        pf_floor = float(probe.get("early_freeze_recent_pf", 0.75) or 0.75)
        exp_floor = float(probe.get("early_freeze_expectancy_r", -0.25) or -0.25)
        if float(row["recent_profit_factor"]) < pf_floor and float(row["recent_expectancy_r"]) < exp_floor:
            row["frozen"] = True
            row["freeze_reason"] = "early_negative_edge"

    target = max(0, int(probe.get("target_trades_per_setup", 0) or 0))
    row["target_reached"] = bool(target and int(row["closed_trades"]) >= target)
    row["last_close_at"] = closed_at.isoformat()
    return row


def settle_open_positions(
    items: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    state_path: Path | None = None,
    persist_trade_log: bool = True,
) -> list[dict[str, Any]]:
    state = _load_state(config, state_path)
    opens = state.get("open_positions") if isinstance(state.get("open_positions"), dict) else {}
    item_map = {str(item.get("symbol") or ""): item for item in items}
    paper = _paper_cfg(config)
    fee_rate = max(0.0, float(paper.get("taker_fee", 0.0005) or 0.0005))
    slippage = max(0.0, float(paper.get("slippage_bps", 50.0) or 50.0)) / 10_000.0
    funding_rate = max(0.0, float(paper.get("funding_rate_8h", 0.0001) or 0.0001))
    closed: list[dict[str, Any]] = []

    for symbol, position in list(opens.items()):
        item = item_map.get(symbol)
        if item is None or not isinstance(position, dict):
            continue
        bar = _market_bar(item)
        opened_at = _parse_ts(position.get("opened_bar_at"))
        if bar is None or opened_at is None or bar["timestamp"] <= opened_at:
            continue

        side = str(position.get("side") or "").lower()
        stop = float(position["stop_loss"])
        target = float(position["take_profit"])
        stop_hit = bar["low"] <= stop if side == "long" else bar["high"] >= stop
        target_hit = bar["high"] >= target if side == "long" else bar["low"] <= target
        held_hours = max(0.0, (bar["timestamp"] - opened_at).total_seconds() / 3600.0)
        max_hold = max(0.25, float(position.get("max_hold_hours", 48.0) or 48.0))

        reason = None
        raw_exit = None
        if stop_hit:
            reason, raw_exit = "stop_loss", stop
        elif target_hit:
            reason, raw_exit = "take_profit", target
        elif held_hours >= max_hold:
            reason, raw_exit = "max_hold", bar["close"]
        if reason is None or raw_exit is None:
            continue

        entry = float(position["entry_price"])
        exit_fill = float(raw_exit) * (1.0 - slippage if side == "long" else 1.0 + slippage)
        sign = 1.0 if side == "long" else -1.0
        gross_return = (exit_fill - entry) / max(entry, 1e-12) * sign
        exit_fee_return = fee_rate * abs(exit_fill / max(entry, 1e-12))
        funding_events = int(held_hours // 8.0)
        funding_return = funding_rate * funding_events
        net_return = gross_return - fee_rate - exit_fee_return - funding_return
        initial_risk_frac = max(float(position.get("initial_risk_frac", 0.0) or 0.0), 1e-9)
        r_multiple = net_return / initial_risk_frac
        notional = float(position.get("notional_usd", 0.0) or 0.0)
        pnl_usd = notional * net_return

        state["equity"] = max(0.0, float(state.get("equity", 0.0) or 0.0) + pnl_usd)
        state["closed_trades"] = int(state.get("closed_trades", 0) or 0) + 1
        setup_id = str(position.get("setup_id") or "unknown")
        setup_row = _update_setup_outcome(
            state,
            setup_id=setup_id,
            r_multiple=r_multiple,
            closed_at=bar["timestamp"],
            config=config,
        )

        record = {
            "symbol": symbol,
            "side": side,
            "entry_price": entry,
            "exit_price": exit_fill,
            "size": float(position.get("size_units", 0.0) or 0.0),
            "pnl_usd": pnl_usd,
            "pnl_pct": net_return * 100.0,
            "timestamp_open": position.get("opened_bar_at"),
            "timestamp_close": bar["timestamp"].isoformat(),
            "master_confidence": position.get("confidence"),
            "leverage": int(position.get("leverage", 1) or 1),
            "risk_usd": float(position.get("risk_usd", 0.0) or 0.0),
            "tf": str(position.get("tf") or "15m"),
            "base_decision": side,
            "setup_id": setup_id,
            "regime": position.get("regime"),
            "strategy": position.get("strategy"),
            "r_multiple": r_multiple,
            "runtime_mode": "paper",
        }
        if persist_trade_log:
            log_closed_trade(record)
            record_pnl(pnl_usd, config=config)
            record_trade_outcome(
                symbol,
                pnl_pct=net_return * 100.0,
                pnl_abs=pnl_usd,
                config=config,
            )

        closed.append({
            **record,
            "exit_reason": reason,
            "held_hours": held_hours,
            "profit_factor": setup_row["profit_factor"],
            "expectancy_r": setup_row["expectancy_r"],
            "frozen": setup_row["frozen"],
            "target_reached": setup_row["target_reached"],
        })
        opens.pop(symbol, None)

    state["open_positions"] = opens
    if closed:
        _save_state(config, state, state_path)
    return closed


def authorize_learning_probe_entry(
    item: dict[str, Any],
    decision: dict[str, Any],
    config: dict[str, Any],
    *,
    state_path: Path | None = None,
) -> dict[str, Any]:
    mode = str(item.get("runtime_mode") or "paper").lower()
    learning = decision.get("learning_probe") if isinstance(decision.get("learning_probe"), dict) else {}
    if mode not in {"paper", "sim"} or learning.get("active") is not True:
        return {"allowed": True, "reason": "not_learning_probe", "risk_scale": float(decision.get("risk_scale", 1.0) or 1.0)}

    probe = _probe_cfg(config)
    state = _load_state(config, state_path)
    setup_id = str(decision.get("setup_id") or "")
    symbol = str(item.get("symbol") or "")
    bar = _market_bar(item)
    if not setup_id or not symbol or bar is None:
        return {"allowed": False, "reason": "probe_missing_setup_symbol_or_bar", "risk_scale": 0.0}

    setup = _setup_state(state, setup_id)
    if bool(setup.get("frozen")):
        return {"allowed": False, "reason": "probe_setup_frozen", "risk_scale": 0.0}
    if bool(setup.get("target_reached")):
        return {"allowed": False, "reason": "probe_target_reached", "risk_scale": 0.0}

    opens = state.get("open_positions") if isinstance(state.get("open_positions"), dict) else {}
    if symbol in opens:
        return {"allowed": False, "reason": "probe_symbol_already_open", "risk_scale": 0.0}
    max_positions = max(1, int(probe.get("max_concurrent_positions", 1) or 1))
    if len(opens) >= max_positions:
        return {"allowed": False, "reason": "probe_max_concurrent_positions", "risk_scale": 0.0}

    cooldown = cooldown_status(symbol)
    if cooldown.get("blocked"):
        return {"allowed": False, "reason": "runtime_loss_cooldown", "risk_scale": 0.0, "cooldown": cooldown}

    last_open = _parse_ts(setup.get("last_open_at"))
    cooldown_minutes = max(0.0, float(probe.get("setup_cooldown_minutes", 60) or 60))
    if last_open is not None and (bar["timestamp"] - last_open).total_seconds() < cooldown_minutes * 60.0:
        return {"allowed": False, "reason": "probe_setup_cooldown", "risk_scale": 0.0}

    equity = max(float(state.get("equity", 0.0) or 0.0), 1e-9)
    loss_gate = evaluate_loss_limits(equity, config=config)
    if not loss_gate.allowed:
        return {
            "allowed": False,
            "reason": loss_gate.reason,
            "risk_scale": 0.0,
            "loss_limits": loss_gate.to_dict(),
        }

    decision_scale = max(0.0, min(1.0, float(decision.get("risk_scale", 0.0) or 0.0)))
    effective_scale = decision_scale * float(loss_gate.risk_scale)
    base_risk = max(0.0, float(probe.get("risk_per_trade_pct", 0.0025) or 0.0025))
    proposed_risk_pct = base_risk * effective_scale
    day = _day_key(bar["timestamp"], config)
    daily_map = state.get("daily_risk_used_pct") if isinstance(state.get("daily_risk_used_pct"), dict) else {}
    used = max(0.0, float(daily_map.get(day, 0.0) or 0.0))
    budget = max(0.0, float(probe.get("daily_budget_pct", 0.005) or 0.005))
    if proposed_risk_pct <= 0.0:
        return {"allowed": False, "reason": "probe_zero_risk", "risk_scale": 0.0}
    if budget > 0.0 and used + proposed_risk_pct > budget + 1e-12:
        return {
            "allowed": False,
            "reason": "probe_daily_budget",
            "risk_scale": 0.0,
            "daily_budget_pct": budget,
            "daily_risk_used_pct": used,
        }

    return {
        "allowed": True,
        "reason": "ok",
        "risk_scale": effective_scale,
        "risk_budget_pct": proposed_risk_pct,
        "daily_budget_pct": budget,
        "daily_risk_used_pct": used,
    }


def register_learning_probe_entry(
    item: dict[str, Any],
    decision: dict[str, Any],
    config: dict[str, Any],
    *,
    state_path: Path | None = None,
) -> dict[str, Any]:
    gate = authorize_learning_probe_entry(item, decision, config, state_path=state_path)
    if gate.get("allowed") is not True:
        return gate

    state = _load_state(config, state_path)
    probe = _probe_cfg(config)
    paper = _paper_cfg(config)
    bar = _market_bar(item)
    assert bar is not None

    symbol = str(item.get("symbol") or "")
    setup_id = str(decision.get("setup_id") or "")
    side = str(decision.get("direction") or "").lower()
    raw_entry = _number(item.get("price"))
    if raw_entry is None or raw_entry <= 0 or side not in {"long", "short"}:
        return {"allowed": False, "reason": "probe_invalid_entry", "risk_scale": 0.0}

    slippage = max(0.0, float(paper.get("slippage_bps", 50.0) or 50.0)) / 10_000.0
    entry = raw_entry * (1.0 + slippage if side == "long" else 1.0 - slippage)
    ta = item.get("ta_pack") if isinstance(item.get("ta_pack"), dict) else {}
    atr = _number(ta.get("atr"))
    if atr is None or atr <= 0:
        atr_ratio = _number(ta.get("atr_ratio"))
        atr = entry * atr_ratio if atr_ratio is not None and atr_ratio > 0 else None
    if atr is None or atr <= 0:
        return {"allowed": False, "reason": "probe_missing_atr", "risk_scale": 0.0}

    stop_mult = max(0.1, float(decision.get("stop_atr_mult", 1.5) or 1.5))
    target_r = max(0.1, float(decision.get("tp_r_target", 2.5) or 2.5))
    stop = compute_stop_loss(side, entry, atr=atr, atr_mult=stop_mult)
    target = calculate_adaptive_take_profit(side, entry, atr=atr, atr_mult=stop_mult * target_r)
    initial_risk_frac = abs(entry - stop) / max(entry, 1e-12)
    if initial_risk_frac <= 0:
        return {"allowed": False, "reason": "probe_invalid_stop_distance", "risk_scale": 0.0}

    equity = max(float(state.get("equity", 0.0) or 0.0), 0.0)
    risk_pct = float(gate.get("risk_budget_pct", 0.0) or 0.0)
    risk_usd = equity * risk_pct
    max_wallet_pct = float((config.get("risk") or {}).get("max_wallet_pct", 0.20) or 0.20)
    notional = min(risk_usd / initial_risk_frac, equity * max_wallet_pct)
    fill_fraction = max(0.0, min(1.0, float(paper.get("partial_fill_rate", 1.0) or 1.0)))
    notional *= fill_fraction
    risk_usd = notional * initial_risk_frac
    size_units = notional / max(entry, 1e-12)

    state.setdefault("open_positions", {})[symbol] = {
        "symbol": symbol,
        "setup_id": setup_id,
        "strategy": decision.get("strategy"),
        "regime": decision.get("regime"),
        "side": side,
        "entry_price": entry,
        "opened_bar_at": bar["timestamp"].isoformat(),
        "stop_loss": stop,
        "take_profit": target,
        "initial_risk_frac": initial_risk_frac,
        "risk_usd": risk_usd,
        "notional_usd": notional,
        "size_units": size_units,
        "max_hold_hours": float(decision.get("max_hold_hours", 48.0) or 48.0),
        "confidence": decision.get("master_confidence"),
        "leverage": int(decision.get("lev", 1) or 1),
        "tf": str(item.get("tf") or "15m"),
        "risk_scale": float(gate.get("risk_scale", 0.0) or 0.0),
    }
    setup = _setup_state(state, setup_id)
    setup["last_open_at"] = bar["timestamp"].isoformat()

    day = _day_key(bar["timestamp"], config)
    daily = state.get("daily_risk_used_pct") if isinstance(state.get("daily_risk_used_pct"), dict) else {}
    daily[day] = float(daily.get(day, 0.0) or 0.0) + risk_pct
    state["daily_risk_used_pct"] = daily
    _save_state(config, state, state_path)
    return {
        **gate,
        "registered": True,
        "entry_price": entry,
        "stop_loss": stop,
        "take_profit": target,
        "notional_usd": notional,
        "risk_usd": risk_usd,
    }


def progress_snapshot(config: dict[str, Any], *, state_path: Path | None = None) -> dict[str, Any]:
    return _load_state(config, state_path)


__all__ = [
    "authorize_learning_probe_entry",
    "progress_snapshot",
    "register_learning_probe_entry",
    "settle_open_positions",
]
