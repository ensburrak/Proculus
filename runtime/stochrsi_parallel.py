from __future__ import annotations

from typing import Any

from decision.stochrsi_opportunity import evaluate_stochrsi_opportunity


def build_stochrsi_parallel_decision(item: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    cfg=config.get("stochrsi_parallel") if isinstance(config.get("stochrsi_parallel"),dict) else {}
    if not bool(cfg.get("enabled",False)):
        return {"action":"hold","direction":"neutral","risk_scale":0.0,"pipeline":"stochrsi_parallel","reason":"disabled"}

    ta=item.get("ta_pack") if isinstance(item.get("ta_pack"),dict) else {}
    result=evaluate_stochrsi_opportunity(item=item,ta=ta)
    payload=result.to_dict()
    payload["pipeline"]="stochrsi_parallel"
    payload["decision_authority"]="independent"
    payload["merge_into_primary_pipeline"]=False

    mode=str(item.get("runtime_mode") or "paper").lower()
    if payload["action"]=="enter":
        if mode in {"paper","sim","dry","dry-run"}:
            authorized=bool(cfg.get("paper_orders_enabled",True))
        elif mode in {"demo","sandbox","testnet"}:
            authorized=bool(cfg.get("demo_orders_enabled",False))
        else:
            authorized=bool(cfg.get("live_orders_enabled",False))
        payload["order_authorized"]=authorized
        if not authorized:
            payload["execution_blocker"]="stochrsi_mode_not_authorized"
    else:
        payload["order_authorized"]=False
    return payload


__all__=["build_stochrsi_parallel_decision"]
