from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from core.decision_pipeline import DecisionPipeline
from decision.stochrsi_parallel import evaluate_stochrsi90
from .execution_bridge import execute_decision
from .runtime_analysis_services import _analyze_one

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
                v2_decision = dict(
                    decisions.get(symbol, {"action": "hold", "reason": "missing decision"})
                )
                v2_execution = await execute_decision(exchange, item, v2_decision, cfg)
                v2_decision["execution"] = v2_execution

                stoch_decision = evaluate_stochrsi90(
                    item=item,
                    ta=item.get("ta_pack") if isinstance(item.get("ta_pack"), dict) else {},
                    config=cfg,
                )
                if (
                    str(stoch_decision.get("action") or "").lower() == "enter"
                    and stoch_decision.get("execution_allowed") is True
                ):
                    stoch_execution = await execute_decision(
                        exchange,
                        item,
                        stoch_decision,
                        cfg,
                    )
                else:
                    stoch_execution = {
                        "status": (
                            "blocked_by_stochrsi_runtime_policy"
                            if str(stoch_decision.get("action") or "").lower() == "enter"
                            else "not_applicable"
                        ),
                        "order_sent": False,
                    }
                stoch_decision = dict(stoch_decision)
                stoch_decision["execution"] = stoch_execution

                # Preserve the public V2 decision shape for existing callers.
                # Independent authorities are attached explicitly and never
                # blended back into the V2 direction/confidence calculation.
                v2_decision["parallel_decisions"] = {
                    "stochrsi": stoch_decision,
                }
                latest[symbol] = v2_decision

        if once:
            return latest
        delay = float((cfg.get("performance") or {}).get("loop_delay_sec", 1800) or 1800)
        await asyncio.sleep(max(1.0, delay))


async def trading_loop_async(exchange: Any, symbols: list[str] | None = None, *, runtime_mode: str = "paper", once: bool = False):
    return await trading_loop_async_service(exchange, symbols or ["BTC/USDT", "ETH/USDT", "SOL/USDT"], runtime_mode=runtime_mode, once=once)


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
