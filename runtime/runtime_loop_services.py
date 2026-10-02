from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from core.decision_pipeline import DecisionPipeline
from .execution_bridge import execute_decision
from .runtime_analysis_services import _analyze_one
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
            decisions = await pipeline.decide_batch(items)
            for item in items:
                symbol = str(item.get("symbol") or "")
                decision = decisions.get(symbol, {"action": "hold", "reason": "missing decision"})
                execution = await execute_decision(exchange, item, decision, cfg)
                decision = dict(decision)
                decision["execution"] = execution

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
