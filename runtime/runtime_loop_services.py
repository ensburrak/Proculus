from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from core.decision_pipeline import DecisionPipeline
from decision.cross_sectional_ranker import rank_enter_decisions
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
            raw_v2 = await pipeline.decide_batch(items)
            ranking_cfg = (
                (cfg.get("pipeline_v2") or {}).get("cross_sectional_ranking") or {}
                if isinstance(cfg.get("pipeline_v2"), dict)
                else {}
            )
            rank_enabled = ranking_cfg.get("enabled") is not False
            top_fraction = float(ranking_cfg.get("top_fraction", 0.15) or 0.15)
            min_candidates = int(ranking_cfg.get("min_candidates", 2) or 2)

            v2_decisions = (
                rank_enter_decisions(
                    raw_v2,
                    top_fraction=top_fraction,
                    min_candidates=min_candidates,
                )
                if rank_enabled
                else {symbol: dict(payload) for symbol, payload in raw_v2.items()}
            )

            raw_stoch: dict[str, dict[str, Any]] = {}
            for item in items:
                symbol = str(item.get("symbol") or "")
                raw_stoch[symbol] = evaluate_stochrsi90(
                    item=item,
                    ta=item.get("ta_pack") if isinstance(item.get("ta_pack"), dict) else {},
                    config=cfg,
                )
            stoch_decisions = (
                rank_enter_decisions(
                    raw_stoch,
                    top_fraction=top_fraction,
                    min_candidates=min_candidates,
                )
                if rank_enabled
                else {symbol: dict(payload) for symbol, payload in raw_stoch.items()}
            )

            for item in items:
                symbol = str(item.get("symbol") or "")
                v2_decision = dict(
                    v2_decisions.get(
                        symbol,
                        {"action": "hold", "reason": "missing decision"},
                    )
                )
                stoch_decision = dict(
                    stoch_decisions.get(
                        symbol,
                        {
                            "action": "hold",
                            "direction": "neutral",
                            "pipeline": "stochrsi_parallel",
                            "authority": "independent",
                            "risk_scale": 0.0,
                            "reason": "missing stochrsi decision",
                        },
                    )
                )

                v2_enter = str(v2_decision.get("action") or "").lower() == "enter"
                stoch_enter = (
                    str(stoch_decision.get("action") or "").lower() == "enter"
                    and stoch_decision.get("execution_allowed") is True
                )

                winner = "v2"
                if v2_enter and stoch_enter:
                    v2_conf = float(
                        v2_decision.get("master_confidence")
                        or v2_decision.get("confidence")
                        or v2_decision.get("score")
                        or 0.0
                    )
                    stoch_conf = float(
                        stoch_decision.get("master_confidence")
                        or stoch_decision.get("confidence")
                        or stoch_decision.get("score")
                        or 0.0
                    )
                    winner = "stochrsi" if stoch_conf > v2_conf else "v2"
                elif stoch_enter and not v2_enter:
                    winner = "stochrsi"

                if v2_enter and winner == "v2":
                    v2_execution = await execute_decision(
                        exchange,
                        item,
                        v2_decision,
                        cfg,
                    )
                elif v2_enter:
                    v2_execution = {
                        "status": "suppressed_by_submit_boundary",
                        "winner": winner,
                        "order_sent": False,
                    }
                else:
                    v2_execution = await execute_decision(
                        exchange,
                        item,
                        v2_decision,
                        cfg,
                    )
                v2_decision["execution"] = v2_execution

                if stoch_enter and winner == "stochrsi":
                    stoch_execution = await execute_decision(
                        exchange,
                        item,
                        stoch_decision,
                        cfg,
                    )
                elif stoch_enter:
                    stoch_execution = {
                        "status": "suppressed_by_submit_boundary",
                        "winner": winner,
                        "order_sent": False,
                    }
                else:
                    stoch_execution = {
                        "status": (
                            "blocked_by_stochrsi_runtime_policy"
                            if str(stoch_decision.get("action") or "").lower() == "enter"
                            else "not_applicable"
                        ),
                        "order_sent": False,
                    }
                stoch_decision["execution"] = stoch_execution

                # Public top-level remains the V2 decision for compatibility.
                # Independent authorities are visible under parallel_decisions;
                # only the final submit boundary arbitrates a same-symbol clash.
                v2_decision["parallel_decisions"] = {
                    "stochrsi": stoch_decision,
                }
                v2_decision["submit_boundary"] = {
                    "winner": winner if (v2_enter or stoch_enter) else "none",
                    "v2_enter": v2_enter,
                    "stochrsi_enter": stoch_enter,
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
