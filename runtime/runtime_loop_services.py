from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from core.decision_pipeline import DecisionPipeline
from .execution_bridge import execute_decision
from .runtime_analysis_services import _analyze_one
from .paper_probe_ledger import (
    authorize_learning_probe_entry,
    register_learning_probe_entry,
    settle_open_positions,
)
from .stochrsi_parallel import build_stochrsi_parallel_decision
from .runtime_symbol_universe import resolve_runtime_symbols

ROOT = Path(__file__).resolve().parents[1]


def _load_config() -> dict[str, Any]:
    try:
        payload = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


async def trading_loop_async_service(exchange: Any, symbols: list[str], *, runtime_mode: str = "paper", once: bool = False) -> dict[str, dict[str, Any]]:
    cfg = _load_config()
    pipeline = DecisionPipeline(exchange)
    latest: dict[str, dict[str, Any]] = {}

    while True:
        items = []
        for symbol in symbols:
            try:
                item = await _analyze_one(exchange, symbol, runtime_mode=runtime_mode)
            except Exception:
                item = None
            if item is not None:
                items.append(item)

        if items:
            paper_mode = str(runtime_mode).lower() in {"paper", "sim"}
            settled_by_symbol: dict[str, list[dict[str, Any]]] = {}
            if paper_mode:
                for closed in settle_open_positions(items, cfg):
                    settled_by_symbol.setdefault(str(closed.get("symbol") or ""), []).append(closed)

            decisions = await pipeline.decide_batch(items)
            for item in items:
                symbol = str(item.get("symbol") or "")
                decision = dict(decisions.get(symbol, {"action": "hold", "reason": "missing decision"}))
                probe_gate: dict[str, Any] | None = None

                learning = decision.get("learning_probe") if isinstance(decision.get("learning_probe"), dict) else {}
                if (
                    paper_mode
                    and str(decision.get("action") or "").lower() == "enter"
                    and learning.get("active") is True
                ):
                    probe_gate = authorize_learning_probe_entry(item, decision, cfg)
                    decision["paper_probe_gate"] = probe_gate
                    if probe_gate.get("allowed") is not True:
                        execution = {
                            "status": "blocked_by_learning_probe_governor",
                            "order_sent": False,
                            "paper_probe_gate": probe_gate,
                        }
                    else:
                        decision["risk_scale"] = float(probe_gate.get("risk_scale", decision.get("risk_scale", 0.0)) or 0.0)
                        execution = await execute_decision(exchange, item, decision, cfg)
                        if execution.get("status") == "simulated":
                            registration = register_learning_probe_entry(item, decision, cfg)
                            execution = dict(execution)
                            execution["paper_probe_registration"] = registration
                else:
                    execution = await execute_decision(exchange, item, decision, cfg)

                decision["execution"] = execution
                if settled_by_symbol.get(symbol):
                    decision["paper_probe_settled"] = settled_by_symbol[symbol]

                stoch_decision = build_stochrsi_parallel_decision(item, cfg)
                if str(stoch_decision.get("action") or "").lower() == "enter":
                    if str(decision.get("action") or "").lower() == "enter":
                        stoch_execution = {
                            "status": "blocked_primary_symbol_slot_occupied",
                            "order_sent": False,
                        }
                    elif stoch_decision.get("order_authorized") is True:
                        stoch_execution = await execute_decision(exchange, item, stoch_decision, cfg)
                    else:
                        stoch_execution = {
                            "status": "blocked_by_stochrsi_mode_policy",
                            "order_sent": False,
                        }
                else:
                    stoch_execution = {"status": "not_applicable", "order_sent": False}
                stoch_decision = dict(stoch_decision)
                stoch_decision["execution"] = stoch_execution
                decision["stochrsi_parallel"] = stoch_decision
                latest[symbol] = decision

        if once:
            return latest
        delay = float((cfg.get("performance") or {}).get("loop_delay_sec", 1800) or 1800)
        await asyncio.sleep(max(1.0, delay))


async def trading_loop_async(exchange: Any, symbols: list[str] | None = None, *, runtime_mode: str = "paper", once: bool = False):
    cfg = _load_config()
    resolved_symbols = list(symbols) if symbols else await resolve_runtime_symbols(exchange, cfg)
    return await trading_loop_async_service(
        exchange,
        resolved_symbols,
        runtime_mode=runtime_mode,
        once=once,
    )


def _noop(*args: Any, **kwargs: Any) -> None:
    return None


_evaluate_kill_state = _noop
_handle_paused_state = _noop
_handle_user_control_commands = _noop
_initialize_strategy_state = _noop
_load_correlation_safe = _noop
_refresh_market_data = _noop
_refresh_runtime_balance = _noop
_refresh_strategy_state = _noop
_run_runtime_health_updates = _noop
_select_symbols_for_analysis = _noop
_shutdown_runtime = _noop
_start_prometheus_server_safe = _noop
_start_runtime_background_services = _noop
_sync_pause_flag_from_telegram = _noop
_sync_user_controller_snapshot = _noop
