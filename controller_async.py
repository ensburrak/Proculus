# -*- coding: utf-8 -*-
"""Thin facade for the batch decision runtime."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

from decision.controller_batch_runtime import (
    OFFICIAL_DECISION_PIPELINE,
    TRADING_SYSTEM_PROMPT,
    build_batch_prompt,
    build_batch_runtime,
    is_llm_tech_only_mode,
)
from decision.llm_manager import is_llm_disable_error as _default_is_llm_disable_error
from decision.controller_batch_service import decide_batch as _decide_batch_service
from decision.score_weights import resolve_decision_weights
from decision.score_calculator import (
    compute_symbol_signal as _default_compute_symbol_signal,
    get_decision_weights as _default_get_decision_weights,
    is_expert_score_mode as _default_is_expert_score_mode,
    is_rl_in_ai as _default_is_rl_in_ai,
    load_decision_weights as _default_load_decision_weights,
    log_decision_summary as _default_log_decision_summary,
    persist_last_decisions as _default_persist_last_decisions,
)
from decision.tech_score import apply_llm_direction_override as _tech_apply_llm_direction_override


def _horizon_multiplier(tf: str | None) -> float:
    """Compatibility helper retained for regression tests and legacy callers."""
    tf_norm = str(tf or "").strip().lower()
    if tf_norm in {"5m", "15m"}:
        return 0.90
    if tf_norm in {"6h", "8h", "12h", "1d", "1day"}:
        return 1.10
    return 1.0


def _is_horizon_allowed(tf: str | None, regime: str | None) -> bool:
    """Compatibility helper for legacy horizon gating semantics."""
    tf_norm = str(tf or "").strip().lower()
    regime_norm = str(regime or "").strip().upper()
    if regime_norm == "SIDEWAYS":
        return tf_norm in {"5m", "15m", "30m", "1h", ""}
    if "BEAR" in regime_norm:
        return tf_norm in {"5m", "15m", "30m", "1h", ""}
    return True


def _apply_llm_direction_override(
    *,
    sym: str,
    base_decision: str,
    ai_part: Dict[str, Any] | None,
    ai_score: float,
) -> str:
    """Legacy compatibility helper retained for regression tests.

    The controller facade historically allowed a high-confidence LLM
    direction to override the technical base decision. The modern
    production helper is guarded by ``pipeline_v2.disable_llm_direction_override``;
    legacy tests still expect the pre-flag behavior, so this facade keeps
    the old deterministic contract.
    """
    try:
        ai_direction = str((ai_part or {}).get("direction") or "").strip().lower()
    except (AttributeError, TypeError, ValueError):
        ai_direction = ""
    if ai_direction in {"long", "short"} and float(ai_score or 0.0) >= 0.70:
        return ai_direction
    return base_decision


def _load_decision_weights() -> None:
    _default_load_decision_weights()


def _ai_quality_factor() -> float:
    return 1.0


def resolve_variant_config(variant: str | None, overrides: Dict[str, Any] | None) -> SimpleNamespace:
    del variant, overrides
    return SimpleNamespace(
        name="default",
        hybrid_weights={},
        decision_weights={},
        expert_score_mode=None,
        rl_in_ai=None,
        metadata={},
        score_spread=None,
        threshold_offset=0.0,
    )


def _load_runtime_knobs(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], dict[str, float], bool]:
    del args, kwargs
    return (
        {},
        {"chatgpt": 0.25, "deepseek": 0.25, "transformer": 0.25, "ppo_rl": 0.25},
        False,
    )


def _prepare_ai_batch_inputs(symbol_inputs: List[Dict[str, Any]], logger: Any) -> tuple[List[Dict[str, Any]], list[dict[str, Any]]]:
    del logger
    return list(symbol_inputs), []


async def get_confidences_batch(symbol_inputs: List[Dict[str, Any]], batch_size: int | None = None) -> Dict[str, Dict[str, Any]]:
    from ai_batch_manager import get_confidences_batch as _legacy_get_confidences_batch

    return await _legacy_get_confidences_batch(symbol_inputs, batch_size=batch_size)


def _classify_llm_batch_health(ai_res: Dict[str, Dict[str, Any]]) -> str:
    del ai_res
    return "unknown"


def _is_llm_disable_error(exc: Exception) -> bool:
    return _default_is_llm_disable_error(exc)


def _load_risk_models() -> None:
    return None


def get_calibration_params() -> None:
    return None


def get_logistic_weights() -> None:
    return None


def _get_decision_weights() -> dict[str, float]:
    return _default_get_decision_weights()


def _is_expert_score_mode() -> bool:
    return _default_is_expert_score_mode()


def _is_rl_in_ai() -> bool:
    return _default_is_rl_in_ai()


def _compute_symbol_signal(**kwargs: Any) -> tuple[Any, dict[str, Any] | None]:
    return _default_compute_symbol_signal(**kwargs)


def _persist_last_decisions(symbol_inputs: list[dict[str, Any]], out: dict[str, dict[str, Any]], logger: Any) -> None:
    if logger is None:
        return
    _default_persist_last_decisions(symbol_inputs, out, logger)


def _log_decision_summary(out: dict[str, dict[str, Any]], symbol_inputs: list[dict[str, Any]], logger: Any) -> None:
    if logger is None:
        return
    _default_log_decision_summary(out, symbol_inputs, logger)


_process_symbol_decision: Any = None


async def _decide_batch_compat(
    symbol_inputs: List[Dict[str, Any]],
    *,
    variant: str | None,
    config_overrides: Dict[str, Any] | None,
) -> Dict[str, Dict[str, Any]]:
    """Compatibility path used by legacy controller unit tests."""
    _load_decision_weights()
    variant_cfg = resolve_variant_config(variant, config_overrides)
    _, hybrid_weights, llm_disabled = _load_runtime_knobs(variant_cfg=variant_cfg)
    ai_quality = float(_ai_quality_factor())
    prepared_inputs, _ = _prepare_ai_batch_inputs(symbol_inputs, logger=None)
    ai_results = await get_confidences_batch(prepared_inputs)
    _classify_llm_batch_health(ai_results)
    _load_risk_models()
    get_calibration_params()
    get_logistic_weights()
    decision_weights = _get_decision_weights()

    out: Dict[str, Dict[str, Any]] = {}
    for item in symbol_inputs:
        symbol = str(item.get("symbol") or "")
        ai_part = (
            ai_results.get(symbol)
            or ai_results.get(symbol.replace("/", ""))
            or ai_results.get(symbol.replace("/", "").replace("-", ""))
            or {}
        )
        out[symbol] = _process_symbol_decision(
            item=item,
            ai_part=ai_part,
            ai_quality_factor=ai_quality,
            llm_disabled=llm_disabled,
            decision_weights=decision_weights,
            hybrid_weights=(getattr(variant_cfg, "hybrid_weights", None) or hybrid_weights),
            expert_score_mode=_is_expert_score_mode(),
            rl_in_ai=_is_rl_in_ai(),
            variant_cfg=variant_cfg,
            logger=None,
        )
    _persist_last_decisions(symbol_inputs, out, logger=None)
    _log_decision_summary(out, symbol_inputs, logger=None)
    return out


def _build_controller_runtime() -> Any:
    runtime = build_batch_runtime(controller_file=Path(__file__))
    return replace(
        runtime,
        resolve_variant_config=resolve_variant_config,
        load_runtime_knobs=_load_runtime_knobs,
        prepare_ai_batch_inputs=_prepare_ai_batch_inputs,
        get_confidences_batch=get_confidences_batch,
        is_llm_disable_error=_is_llm_disable_error,
        classify_llm_batch_health=_classify_llm_batch_health,
        load_risk_models=_load_risk_models,
        get_calibration_params=get_calibration_params,
        get_logistic_weights=get_logistic_weights,
        get_decision_weights=_get_decision_weights,
        is_expert_score_mode=_is_expert_score_mode,
        is_rl_in_ai=_is_rl_in_ai,
        compute_symbol_signal=_compute_symbol_signal,
        persist_last_decisions=_persist_last_decisions,
        log_decision_summary=_log_decision_summary,
    )


async def decide_batch(
    symbol_inputs: List[Dict[str, Any]],
    exchange: Any = None,
    *,
    variant: str | None = None,
    config_overrides: Dict[str, Any] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Batch kararlarini hesapla."""
    del exchange
    if callable(_process_symbol_decision):
        return await _decide_batch_compat(
            symbol_inputs,
            variant=variant,
            config_overrides=config_overrides,
        )
    runtime = _build_controller_runtime()
    return await _decide_batch_service(
        symbol_inputs,
        variant=variant,
        config_overrides=config_overrides,
        runtime=runtime,
    )


def decide(
    symbol_inputs: list[dict[str, Any]],
    exchange: Any = None,
    **kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """Synchronous wrapper for batch decision making."""
    import asyncio
    import concurrent.futures

    async def _run() -> dict[str, dict[str, Any]]:
        return await decide_batch(symbol_inputs, exchange=exchange, **kwargs)

    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, _run()).result()
    except RuntimeError:
        return asyncio.run(_run())


def fuse_all_scores(
    tech_score: float,
    sentiment_score: float,
    ai_scores: dict[str, float] | None = None,
    regime: str | None = None,
    **kwargs: Any,
) -> float:
    """Fuse all component scores into a single master confidence score."""
    weights = resolve_decision_weights()
    ai_score = 0.5
    if ai_scores:
        valid_scores = [value for value in ai_scores.values() if isinstance(value, (int, float))]
        if valid_scores:
            ai_score = sum(valid_scores) / len(valid_scores)
    master = (
        weights["tech"] * tech_score
        + weights["sent"] * sentiment_score
        + weights["ai"] * ai_score
    )
    return max(0.0, min(1.0, master))
