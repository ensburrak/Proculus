from __future__ import annotations

import inspect
from typing import Any


async def _await_maybe(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def execute_decision(exchange: Any, item: dict[str, Any], decision: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    if str(decision.get("action") or "").lower() != "enter":
        return {"status": "not_applicable", "order_sent": False}

    mode = str(item.get("runtime_mode") or "paper").lower()
    if mode in {"paper", "sim", "demo", "dry", "dry-run"}:
        return {
            "status": "simulated",
            "order_sent": False,
            "symbol": item.get("symbol"),
            "direction": decision.get("direction"),
            "risk_scale": decision.get("risk_scale"),
        }

    pipeline = config.get("pipeline_v2") if isinstance(config.get("pipeline_v2"), dict) else {}
    authority = config.get("ai_authority") if isinstance(config.get("ai_authority"), dict) else {}
    live_allowed = bool(pipeline.get("allow_live_exchange_side_effects", False)) and bool(authority.get("allow_live_exchange_side_effects", False))
    if not live_allowed:
        return {"status": "live_blocked_by_policy", "order_sent": False}

    amount = item.get("order_size")
    if amount is None:
        return {"status": "live_blocked_missing_order_size", "order_sent": False}
    try:
        amount_f = float(amount) * float(decision.get("risk_scale", 1.0) or 1.0)
    except (TypeError, ValueError):
        return {"status": "live_blocked_invalid_order_size", "order_sent": False}
    if amount_f <= 0:
        return {"status": "live_blocked_invalid_order_size", "order_sent": False}

    side = "buy" if str(decision.get("direction")).lower() == "long" else "sell"
    order = await _await_maybe(exchange.create_order(item["symbol"], "market", side, amount_f))
    return {"status": "submitted", "order_sent": True, "order": order}
