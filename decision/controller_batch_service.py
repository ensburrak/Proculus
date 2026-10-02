from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BatchDecisionRuntime:
    process_symbol_decision: Any = None
    config_file: Any = None
    resolve_variant_config: Any = None
    load_runtime_knobs: Any = None
    prepare_ai_batch_inputs: Any = None
    get_confidences_batch: Any = None
    is_llm_disable_error: Any = None
    classify_llm_batch_health: Any = None
    load_risk_models: Any = None
    get_calibration_params: Any = None
    get_logistic_weights: Any = None
    get_decision_weights: Any = None
    is_expert_score_mode: Any = None
    is_rl_in_ai: Any = None
    compute_symbol_signal: Any = None
    persist_last_decisions: Any = None
    log_decision_summary: Any = None


async def decide_batch(
    symbol_inputs: list[dict[str, Any]],
    *,
    variant: str | None = None,
    config_overrides: dict[str, Any] | None = None,
    runtime: BatchDecisionRuntime,
) -> dict[str, dict[str, Any]]:
    del variant
    out: dict[str, dict[str, Any]] = {}
    for item in symbol_inputs or []:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol") or "")
        ai_part = item.get("ai_part") if isinstance(item.get("ai_part"), dict) else {}
        result = runtime.process_symbol_decision(item=item, ai_part=ai_part, config_overrides=config_overrides)
        out[symbol] = result

    if callable(runtime.persist_last_decisions):
        runtime.persist_last_decisions(symbol_inputs, out, None)
    return out
