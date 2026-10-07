from __future__ import annotations

from core.exceptions import BEST_EFFORT_EXCEPTIONS
from core.feature_flags import flag_enabled
import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Dict, List


@dataclass(frozen=True)
class BatchDecisionRuntime:
    decision_state: Any
    config_file: Any
    load_decision_weights: Any
    ai_quality_factor: Any
    resolve_variant_config: Any
    load_runtime_knobs: Any
    prepare_ai_batch_inputs: Any
    get_confidences_batch: Any
    is_llm_disable_error: Any
    classify_llm_batch_health: Any
    record_llm_batch_state: Any
    send_ai_failure_notification: Any
    load_risk_models: Any
    get_calibration_params: Any
    get_logistic_weights: Any
    get_decision_weights: Any
    is_expert_score_mode: Any
    is_rl_in_ai: Any
    provider_failure_stats: Any
    process_symbol_decision: Any
    apply_no_trade_override: Any
    apply_confidence_penalty: Any
    compute_symbol_signal: Any
    enhanced_tech_score: Any
    sent_score: Any
    rl_score_for_symbol: Any
    ai_fusion_available: Any
    calculate_intelligent_fusion: Any
    enhancements_available: Any
    calculate_ai_agreement_bonus: Any
    check_mtf_alignment: Any
    safety_guard_available: Any
    check_trade_quality: Any
    get_current_regime: Any
    apply_symbol_postprocess: Any
    controller_file: Path
    get_rl_factor: Any
    get_macro_risk_multiplier: Any
    global_arbitrage_sensor: Any
    get_onchain_sentiment: Any
    adjust_confidence_with_imbalance: Any
    compute_meta_probability: Any
    is_horizon_allowed: Any
    finalize_symbol_decision: Any
    safe_sigmoid: Any
    classify_volatility: Any
    load_global_sentiment: Any
    apply_global_sentiment_adjustment: Any
    lev_from_master: Any
    is_trading_enabled: Any
    get_session_multiplier: Any
    get_symbol_threshold: Any
    get_regime_adjustment: Any
    meta_strategy_available: Any
    select_strategy: Any
    entry_optimizer_available: Any
    entry_optimizer: Any
    derive_provider_flags: Any
    build_decision_output: Any
    log_signal: Any
    persist_last_decisions: Any
    log_decision_summary: Any
    log: Any


def _load_pipeline_v2_config(config_file: Any) -> dict[str, Any]:
    path = Path(config_file)
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        pipeline_v2 = payload.get("pipeline_v2") if isinstance(payload, dict) else {}
        return pipeline_v2 if isinstance(pipeline_v2, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def _resolve_primary_pipeline(config_file: Any) -> str:
    pipeline_v2 = _load_pipeline_v2_config(config_file)
    if not bool(pipeline_v2.get("enabled", False)):
        return "v1"
    primary = str(pipeline_v2.get("primary_pipeline") or "v2").strip().lower()
    return primary if primary in {"v1", "v2"} else "v2"


def _decision_task_timeout_sec(config_file: Any) -> float:
    default_timeout = 180.0
    try:
        path = Path(config_file)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            return default_timeout
        performance = payload.get("performance")
        if not isinstance(performance, dict):
            performance = {}
        raw_timeout = performance.get("decision_task_timeout_sec", payload.get("decision_task_timeout_sec", default_timeout))
        timeout = float(raw_timeout)
        return timeout if timeout > 0.0 else 0.0
    except (OSError, json.JSONDecodeError, TypeError, ValueError, OverflowError):
        return default_timeout


def _write_runtime_heartbeat() -> None:
    try:
        from health_monitor import write_heartbeat

        write_heartbeat()
    except BEST_EFFORT_EXCEPTIONS:
        pass


def _run_v5_breadth_shadow(symbol_inputs: List[Dict[str, Any]], log: Any) -> None:
    """Evaluate V5 as prospective shadow evidence without touching execution decisions."""
    if not flag_enabled("ENABLE_V5_BREADTH_SHADOW", False):
        return

    try:
        from decision.v5_breadth_shadow import (
            DEFAULT_RUNTIME_EVIDENCE_PATH,
            evaluate_v5_breadth_shadow,
            record_v5_shadow_evidence,
        )

        payload = dict(evaluate_v5_breadth_shadow(symbol_inputs))
        if payload.get("execution_authority") is not False:
            log.warning(
                "[V5_SHADOW] evaluator attempted execution authority; forcing shadow-only"
            )
            payload["execution_authority"] = False
            payload["shadow_only"] = True
            payload["safety_override"] = "execution_authority_forced_false"

        recorded = record_v5_shadow_evidence(
            payload,
            path=DEFAULT_RUNTIME_EVIDENCE_PATH,
        )
        log.info(
            "[V5_SHADOW] evaluated ready=%s reason=%s candidates=%s recorded=%s",
            payload.get("ready"),
            payload.get("reason"),
            len(payload.get("candidates") or []),
            recorded,
        )
    except BEST_EFFORT_EXCEPTIONS as exc:
        log.warning(
            "[V5_SHADOW] V5 breadth shadow evaluation failed; primary decisions unchanged: %s",
            exc,
        )


def _normalize_hybrid_weights_for_decision(weights: Dict[str, float], defaults: Dict[str, float]) -> Dict[str, float]:
    try:
        candidate = dict(defaults)
        candidate.update(
            {
                str(key): float(value)
                for key, value in dict(weights or {}).items()
                if float(value) >= 0.0
            }
        )
        total = sum(candidate.values())
        if total <= 0.0:
            raise ValueError("hybrid weight sum is zero")
        return {key: value / total for key, value in candidate.items()}
    except (TypeError, ValueError):
        total = sum(defaults.values())
        return {key: value / total for key, value in defaults.items()}


def _unavailable_llm_providers(ai_part: Any, *, llm_disabled: bool) -> list[str]:
    providers = ai_part.get("providers") if isinstance(ai_part, dict) and isinstance(ai_part.get("providers"), dict) else {}
    unavailable: list[str] = []
    for provider, payload in providers.items():
        status = str(payload.get("status") if isinstance(payload, dict) else payload).strip().lower()
        if status not in {"ok", "available", "success"}:
            unavailable.append(str(provider))
    provider_status = str(ai_part.get("provider_status") if isinstance(ai_part, dict) else "").strip().lower()
    if not unavailable and (llm_disabled or provider_status in {"down", "disabled", "error", "all_down"}):
        unavailable.extend(["chatgpt", "deepseek"])
    return sorted(set(unavailable))


def _annotate_llm_degradation(
    decision: Dict[str, Any],
    *,
    ai_part: Any,
    llm_disabled: bool,
    symbol_tech_only: bool,
    batch_failure_reason: str,
) -> Dict[str, Any]:
    if not isinstance(decision, dict):
        return decision
    if not (llm_disabled or symbol_tech_only):
        return decision
    provider_status = str(ai_part.get("provider_status") if isinstance(ai_part, dict) else "").strip().lower()
    decision.setdefault("ai_route", "technical_only")
    decision.setdefault("candidate_route", "technical_only")
    decision.setdefault("llm_status", "degraded")
    decision.setdefault("llm_degraded", True)
    decision.setdefault(
        "llm_degraded_reason",
        batch_failure_reason or provider_status or "llm_unavailable",
    )
    decision.setdefault("unavailable_providers", _unavailable_llm_providers(ai_part, llm_disabled=llm_disabled))
    return decision


async def _process_symbol_decision_with_timeout(
    *,
    runtime: BatchDecisionRuntime,
    sym: str,
    timeout_sec: float,
    decision_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    if timeout_sec <= 0.0:
        return runtime.process_symbol_decision(**decision_kwargs)

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(runtime.process_symbol_decision, **decision_kwargs),
            timeout=timeout_sec,
        )
    except asyncio.TimeoutError:
        runtime.log.warning(
            "[DECISION] process_symbol_decision timeout for %s after %.1fs; emitting hold",
            sym,
            timeout_sec,
        )
        return {
            "symbol": sym,
            "action": "hold",
            "reason": "decision_timeout",
            "score": 0.0,
            "master_confidence": 0.0,
            "raw_score": 0.0,
            "provider_flags": {},
            "error": "decision_timeout",
        }


async def decide_batch(
    symbol_inputs: List[Dict[str, Any]],
    *,
    variant: str | None = None,
    config_overrides: Dict[str, Any] | None = None,
    runtime: BatchDecisionRuntime,
) -> Dict[str, Dict[str, Any]]:
    state = runtime.decision_state
    state.llm_tech_only_mode = False
    pipeline_v2_cfg = _load_pipeline_v2_config(runtime.config_file)
    primary_pipeline = _resolve_primary_pipeline(runtime.config_file)
    v2_primary = primary_pipeline == "v2"
    strict_expert_only = bool(pipeline_v2_cfg.get("strict_expert_only", True))
    llm_disabled = False
    llm_failure_for_circuit = False
    batch_failure_reason = ""
    if state.llm_circuit_breaker_until > time.time():
        llm_disabled = True
        llm_failure_for_circuit = True
        batch_failure_reason = "llm_circuit_breaker_active"
        state.llm_tech_only_mode = True
        runtime.log.warning(
            "[AI_SAFE] LLM circuit breaker active. %s",
            "Falling back to technical-only mode.",
        )

    runtime.load_decision_weights()
    ai_quality_factor = runtime.ai_quality_factor()
    variant_cfg = runtime.resolve_variant_config(variant, config_overrides)
    variant_label = variant_cfg.name if variant_cfg.name else "default"

    hw_default = {"deepseek": 0.35, "chatgpt": 0.35, "transformer": 0.10, "lightgbm": 0.15, "ppo_rl": 0.05}
    _, hybrid_weights, _disable_rl_master_fusion = runtime.load_runtime_knobs(
        runtime.config_file,
        runtime.log,
        hw_default,
    )
    del _disable_rl_master_fusion
    if variant_cfg.hybrid_weights:
        hybrid_weights.update(variant_cfg.hybrid_weights)
        hybrid_weights = _normalize_hybrid_weights_for_decision(hybrid_weights, hw_default)
    ai_items, missing_price_symbols = runtime.prepare_ai_batch_inputs(symbol_inputs, runtime.log)

    ai_res: Dict[str, Any] = {}
    if not llm_disabled:
        try:
            ai_res = await runtime.get_confidences_batch(ai_items)
        except (ConnectionError, TimeoutError, OSError, RuntimeError, ValueError, TypeError) as exc:
            llm_disabled = True
            llm_failure_for_circuit = True
            state.llm_tech_only_mode = True
            batch_failure_reason = f"llm_batch_error:{type(exc).__name__}"
            if runtime.is_llm_disable_error(exc):
                runtime.log.warning(
                    "[AI_SAFE] LLM disabled due to 429/402 or quota/rate-limit issue; %s",
                    "falling back to technical-only mode",
                )
            else:
                runtime.log.warning(
                    "[AI_SAFE] get_confidences_batch error: %s. %s",
                    exc,
                    "Falling back to technical-only mode.",
                )
            runtime.send_ai_failure_notification(f"[AI_SAFE] {batch_failure_reason}")
            ai_res = {}

    if not llm_disabled:
        try:
            llm_health = runtime.classify_llm_batch_health(ai_res)
            if llm_health == "empty":
                llm_disabled = True
                llm_failure_for_circuit = True
                state.llm_tech_only_mode = True
                batch_failure_reason = "llm_batch_empty"
                runtime.log.warning(
                    "[AI_SAFE] Empty LLM batch response. %s",
                    "Falling back to technical-only mode.",
                )
            elif llm_health == "all_down":
                llm_disabled = True
                llm_failure_for_circuit = True
                state.llm_tech_only_mode = True
                batch_failure_reason = "llm_providers_all_down"
                runtime.log.warning(
                    "[AI_SAFE] LLM providers are in fallback mode. %s",
                    "Falling back to technical-only mode.",
                )
            elif llm_health == "disabled":
                llm_disabled = True
                state.llm_tech_only_mode = True
                batch_failure_reason = "llm_providers_disabled_by_config"
                runtime.log.info(
                    "[AI_SAFE] LLM providers disabled by config. %s",
                    "Using technical/expert chain without recording provider failure.",
                )
        except (TypeError, ValueError, AttributeError, KeyError) as exc:
            runtime.log.debug("[DECISION] LLM fallback detection error: %s", exc)

    runtime.record_llm_batch_state(not llm_failure_for_circuit, batch_failure_reason if llm_failure_for_circuit else "")
    if llm_failure_for_circuit and batch_failure_reason:
        runtime.send_ai_failure_notification(f"[AI_SAFE] {batch_failure_reason}")

    runtime.load_risk_models()
    calibration_params = runtime.get_calibration_params()
    logistic_weights = runtime.get_logistic_weights()
    decision_weights = runtime.get_decision_weights()
    if variant_cfg.decision_weights:
        merged_dw = dict(decision_weights)
        merged_dw.update(variant_cfg.decision_weights)
        decision_weights = merged_dw
    expert_score_mode = variant_cfg.expert_score_mode if variant_cfg.expert_score_mode is not None else runtime.is_expert_score_mode()
    rl_in_ai = variant_cfg.rl_in_ai if variant_cfg.rl_in_ai is not None else runtime.is_rl_in_ai()

    out: Dict[str, Dict[str, Any]] = {}
    decision_timeout_sec = _decision_task_timeout_sec(runtime.config_file)

    def annotate_variant(decision: Dict[str, Any]) -> Dict[str, Any]:
        decision["model_variant"] = variant_label
        if variant_cfg.metadata:
            decision["variant_metadata"] = dict(variant_cfg.metadata)
        return decision

    for idx, item in enumerate(symbol_inputs):
        if idx % 5 == 0:
            _write_runtime_heartbeat()

        sym = item["symbol"]
        price = item.get("price")
        if price is None:
            price = item.get("current_price")
        if price is None:
            missing_price_symbols.append(str(sym))
            continue

        ai_part = ai_res.get(
            sym,
            {
                "confidence": 0.0,
                "direction": None,
                "action": None,
                "advisory_direction": "neutral",
                "advisory_action": "skip",
                "llm_role": "advisory_only",
                "quality_rank": 0.0,
                "veto_signal": True,
                "rationale": "llm_unavailable",
                "provider_status": "down",
            },
        )
        provider_total, provider_failed = runtime.provider_failure_stats(ai_part if isinstance(ai_part, dict) else {})
        failure_ratio = (provider_failed / provider_total) if provider_total else (1.0 if llm_disabled else 0.0)
        symbol_tech_only = (not llm_disabled) and provider_total > 0 and failure_ratio >= 0.67

        decision_kwargs = dict(
            item=item,
            ai_part=ai_part,
            ai_quality_factor=ai_quality_factor,
            llm_disabled=llm_disabled or symbol_tech_only,
            decision_weights=decision_weights,
            hybrid_weights=hybrid_weights,
            hw_default=hw_default,
            rl_in_ai=rl_in_ai,
            compute_symbol_signal=runtime.compute_symbol_signal,
            enhanced_tech_score=runtime.enhanced_tech_score,
            sent_score_fn=runtime.sent_score,
            rl_score_for_symbol=runtime.rl_score_for_symbol,
            ai_fusion_available=runtime.ai_fusion_available,
            calculate_intelligent_fusion=runtime.calculate_intelligent_fusion,
            enhancements_available=runtime.enhancements_available,
            calculate_ai_agreement_bonus=runtime.calculate_ai_agreement_bonus,
            check_mtf_alignment=runtime.check_mtf_alignment,
            safety_guard_available=runtime.safety_guard_available,
            check_trade_quality=runtime.check_trade_quality,
            get_current_regime=runtime.get_current_regime,
            apply_symbol_postprocess=runtime.apply_symbol_postprocess,
            root_file=runtime.controller_file,
            get_rl_factor=runtime.get_rl_factor,
            get_macro_risk_multiplier=runtime.get_macro_risk_multiplier,
            global_arbitrage_sensor=runtime.global_arbitrage_sensor,
            get_onchain_sentiment=runtime.get_onchain_sentiment,
            adjust_confidence_with_imbalance=runtime.adjust_confidence_with_imbalance,
            compute_meta_probability=runtime.compute_meta_probability,
            is_horizon_allowed=runtime.is_horizon_allowed,
            finalize_symbol_decision=runtime.finalize_symbol_decision,
            calibration_params=calibration_params,
            logistic_weights=logistic_weights,
            expert_score_mode=expert_score_mode,
            safe_sigmoid=runtime.safe_sigmoid,
            classify_volatility=runtime.classify_volatility,
            load_global_sentiment=runtime.load_global_sentiment,
            apply_global_sentiment_adjustment=runtime.apply_global_sentiment_adjustment,
            lev_from_master=runtime.lev_from_master,
            is_trading_enabled=runtime.is_trading_enabled,
            get_session_multiplier=runtime.get_session_multiplier,
            get_symbol_threshold=runtime.get_symbol_threshold,
            get_regime_adjustment=runtime.get_regime_adjustment,
            meta_strategy_available=runtime.meta_strategy_available,
            select_strategy=runtime.select_strategy,
            entry_optimizer_available=runtime.entry_optimizer_available,
            entry_optimizer=runtime.entry_optimizer,
            derive_provider_flags=runtime.derive_provider_flags,
            build_decision_output=runtime.build_decision_output,
            log_signal=runtime.log_signal,
            score_spread_override=variant_cfg.score_spread,
            threshold_offset=variant_cfg.threshold_offset,
            logger=runtime.log,
        )
        decision = await _process_symbol_decision_with_timeout(
            runtime=runtime,
            sym=sym,
            timeout_sec=decision_timeout_sec,
            decision_kwargs=decision_kwargs,
        )
        if symbol_tech_only:
            if not v2_primary:
                decision = runtime.apply_confidence_penalty(
                    decision,
                    multiplier=0.8,
                    reason="llm_provider_degraded_tech_only",
                )
        if symbol_tech_only and v2_primary:
            expert_signal = decision.get("expert_signal") if isinstance(decision, dict) else None
            expert_setup_id = expert_signal.get("setup_id") if isinstance(expert_signal, dict) else None
            if str(decision.get("action") or "").upper() == "ENTER" and not str(expert_setup_id or "").strip():
                decision = runtime.apply_no_trade_override(
                    decision,
                    reason="llm_advisory_only_no_expert_signal",
                )
        if v2_primary and strict_expert_only and isinstance(decision, dict):
            expert_signal = decision.get("expert_signal") if isinstance(decision.get("expert_signal"), dict) else {}
            technical_fallback = (
                str(expert_signal.get("source") or "").strip().lower() == "technical_fallback"
                or str(decision.get("decision_source") or "").strip().lower() == "expert_router_technical_fallback"
                or str(decision.get("setup_id") or "").strip().lower().startswith("technical_fallback.")
            )
            if str(decision.get("action") or "").upper() == "ENTER" and technical_fallback:
                decision = runtime.apply_no_trade_override(
                    decision,
                    reason="strict_expert_only_blocks_technical_fallback",
                )
        decision = _annotate_llm_degradation(
            decision,
            ai_part=ai_part,
            llm_disabled=llm_disabled,
            symbol_tech_only=symbol_tech_only,
            batch_failure_reason=batch_failure_reason,
        )
        out[sym] = annotate_variant(decision)

    for sym in missing_price_symbols:
        out.setdefault(
            sym,
            annotate_variant(
                {
                    "symbol": sym,
                    "action": "hold",
                    "reason": "missing_price",
                    "score": 0.0,
                    "master_confidence": 0.0,
                }
            ),
        )

    _run_v5_breadth_shadow(symbol_inputs, runtime.log)
    runtime.persist_last_decisions(symbol_inputs, out, runtime.log)
    runtime.log_decision_summary(out, symbol_inputs, runtime.log)
    _write_runtime_heartbeat()
    return out
