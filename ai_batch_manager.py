# -*- coding: utf-8 -*-
from __future__ import annotations

"""
ai_batch_manager.py
- ChatGPT + DeepSeek integration (50/50 default weighting)
- Token-efficient batch analysis with groups of 20 symbols
- Async and 429-safe provider handling
- Prediction logging to ``metrics/ai_predictions.json``
- Live signal snapshots in ``metrics/signals_last.json``
"""
import asyncio
import json
import logging
import os
import pathlib
import time
import uuid
from typing import Any, Dict, List

from ai.batch_llm_cache import LLMBatchCache
from ai.batch_logging import log_ai_prediction as _log_ai_prediction_impl
from ai.batch_logging import log_ai_predictions_batch as _log_ai_predictions_batch_impl
from ai.batch_logging import log_decision_audit as _log_decision_audit_impl
from ai.batch_logging import safe_json_parse as _safe_json_parse_impl
from ai.batch_logging import update_signals_file as _update_signals_file_impl
from ai.batch_logging import update_signals_file_batch as _update_signals_file_batch_impl
from ai.batch_manager_execution import (
    BatchExecutionDeps,
)
from ai.batch_manager_execution import analyze_batch as _analyze_batch_with_deps
from ai.batch_manager_execution import call_model as _call_model_with_deps
from ai.batch_manager_execution import get_confidences_batch as _get_confidences_batch_with_deps
from ai.batch_manager_rl import (
    BatchRlDeps,
)
from ai.batch_manager_rl import get_dynamic_weights as _get_dynamic_weights_with_deps
from ai.batch_manager_rl import get_ensemble_rl_signal as _get_ensemble_rl_signal_with_deps
from ai.batch_manager_rl import get_multi_agent_signal as _get_multi_agent_signal_with_deps
from ai.batch_manager_rl import get_rl_confidence_v3 as _get_rl_confidence_v3_with_deps
from ai.batch_manager_rl import get_rl_confidence_v4 as _get_rl_confidence_v4_with_deps
from ai.batch_manager_rl import get_rl_system_status as _get_rl_system_status_with_deps
from ai.batch_manager_rl import update_rl_trade_metrics as _update_rl_trade_metrics_with_deps
from ai.batch_orchestrator import analyze_batch as _analyze_batch_impl
from ai.batch_orchestrator import get_confidences_batch as _get_confidences_batch_impl
from ai.batch_prompts import DEFAULT_CHATGPT_SYSTEM_PROMPT_V2 as CHATGPT_SYSTEM_PROMPT_V2
from ai.batch_prompts import DEFAULT_DEEPSEEK_SYSTEM_PROMPT_V2 as DEEPSEEK_SYSTEM_PROMPT_V2
from ai.batch_prompts import build_batch_prompt as _build_batch_prompt
from ai.batch_prompts import (
    build_prompt_decision_ab_summary,
    load_prompt_bundle,
)
from ai.batch_rl import get_dynamic_weights as _get_dynamic_weights_impl
from ai.batch_rl import get_ensemble_rl_signal as _get_ensemble_rl_signal_impl
from ai.batch_rl import get_multi_agent_signal as _get_multi_agent_signal_impl
from ai.batch_rl import get_rl_confidence_v3 as _get_rl_confidence_v3_impl
from ai.batch_rl import get_rl_confidence_v4 as _get_rl_confidence_v4_impl
from ai.batch_rl import get_rl_system_status as _get_rl_system_status_impl
from ai.batch_rl import update_rl_trade_metrics as _update_rl_trade_metrics_impl
from ai.batch_runtime_state import (
    create_runtime_context,
    reload_runtime_context,
)
from ai.batch_runtime_state import set_deepseek_disable_logged as _set_runtime_deepseek_disable_logged
from ai.batch_runtime_state import sync_provider_state_from_globals as _sync_runtime_provider_state_from_globals
from ai.batch_runtime_state import sync_provider_state_to_context as _sync_runtime_provider_state_to_context
from ai.llm_cost_optimizer import (
    apply_llm_provider_policy_state,
    load_llm_cost_optimizer_state,
    select_llm_batch_size,
)
from ai.provider_runtime import call_model as _call_model_impl
from core.exceptions import BEST_EFFORT_EXCEPTIONS, DATA_EXCEPTIONS, IO_EXCEPTIONS, OPTIONAL_IMPORT_EXCEPTIONS
from decision.ai_batch_calibration import calibrate_confidence as _shared_calibrate_confidence
from decision.ai_batch_fallbacks import (
    build_disabled_provider_result,
    build_symbol_index,
    resolve_provider_result,
    should_use_model_fallback,
)
from decision.ai_batch_merge import build_merged_symbol_payload
from decision.ai_batch_status import build_provider_detail, classify_provider_status

AI_IMPORT_EXCEPTIONS = BEST_EFFORT_EXCEPTIONS + (ConnectionError, TimeoutError)
AI_RUNTIME_EXCEPTIONS = BEST_EFFORT_EXCEPTIONS + (ConnectionError, TimeoutError)
JSON_STORAGE_EXCEPTIONS = IO_EXCEPTIONS + (json.JSONDecodeError,) + DATA_EXCEPTIONS

_UNRESOLVED_TRANSFORMER = object()
_TRANSFORMER_FUNCS: tuple[Any, Any, Any] | None | object = _UNRESOLVED_TRANSFORMER


def _transformer_unavailable_signal(symbols: list[str]) -> dict[str, dict[str, Any]]:
    return {
        symbol: {
            "p_up": 0.5,
            "p_down": 0.5,
            "p_no_trade": 0.0,
            "confidence": 0.0,
            "direction": "neutral",
            "status": "unavailable",
            "reason": "transformer_import_failed",
        }
        for symbol in symbols
    }


def _resolve_transformer_funcs() -> tuple[Any, Any, Any] | None:
    global _TRANSFORMER_FUNCS
    if _TRANSFORMER_FUNCS is _UNRESOLVED_TRANSFORMER:
        try:
            from ml.transformer_inference import predict_prob
            from ml.transformer_inference import predict_prob_batch
            from ml.transformer_inference import predict_signal_batch

            _TRANSFORMER_FUNCS = (predict_prob, predict_prob_batch, predict_signal_batch)
        except AI_IMPORT_EXCEPTIONS:
            _TRANSFORMER_FUNCS = None
    return _TRANSFORMER_FUNCS if isinstance(_TRANSFORMER_FUNCS, tuple) else None


def _transformer_predict_prob(symbol: str, window: int = 180) -> float:
    funcs = _resolve_transformer_funcs()
    if funcs is None:
        return 0.5
    try:
        return float(funcs[0](symbol, window))
    except AI_RUNTIME_EXCEPTIONS:
        return 0.5


def _transformer_predict_prob_batch(symbols: list[str], window: int = 180) -> dict[str, float]:
    funcs = _resolve_transformer_funcs()
    if funcs is None:
        return {symbol: 0.5 for symbol in symbols}
    try:
        return {str(symbol): float(score) for symbol, score in funcs[1](symbols, window).items()}
    except AI_RUNTIME_EXCEPTIONS:
        return {symbol: 0.5 for symbol in symbols}


def _transformer_predict_signal_batch(symbols: list[str], window: int = 0) -> dict[str, dict[str, Any]]:
    funcs = _resolve_transformer_funcs()
    if funcs is None:
        return _transformer_unavailable_signal(symbols)
    try:
        return funcs[2](symbols, window)
    except AI_RUNTIME_EXCEPTIONS:
        return _transformer_unavailable_signal(symbols)


_UNRESOLVED_LIGHTGBM = object()
_LIGHTGBM_PREDICT_SIGNAL_BATCH: Any | object | None = _UNRESOLVED_LIGHTGBM


def _lightgbm_unavailable_signal(symbols: list[str]) -> dict[str, dict[str, Any]]:
    return {
        symbol: {
            "p_up": 0.5,
            "p_down": 0.5,
            "p_no_trade": 0.0,
            "confidence": 0.0,
            "direction": "neutral",
            "status": "unavailable",
            "reason": "lightgbm_import_failed",
        }
        for symbol in symbols
    }


def _resolve_lightgbm_predict_signal_batch() -> Any | None:
    global _LIGHTGBM_PREDICT_SIGNAL_BATCH
    if _LIGHTGBM_PREDICT_SIGNAL_BATCH is _UNRESOLVED_LIGHTGBM:
        try:
            from ml.lightgbm_inference import predict_signal_batch

            _LIGHTGBM_PREDICT_SIGNAL_BATCH = predict_signal_batch
        except AI_IMPORT_EXCEPTIONS:
            _LIGHTGBM_PREDICT_SIGNAL_BATCH = None
    return None if _LIGHTGBM_PREDICT_SIGNAL_BATCH is _UNRESOLVED_LIGHTGBM else _LIGHTGBM_PREDICT_SIGNAL_BATCH


def _lightgbm_predict_signal_batch(symbols: list[str], window: int = 0) -> dict[str, dict[str, Any]]:
    predict_signal_batch = _resolve_lightgbm_predict_signal_batch()
    if predict_signal_batch is None:
        return _lightgbm_unavailable_signal(symbols)
    try:
        return predict_signal_batch(symbols, window)
    except AI_RUNTIME_EXCEPTIONS:
        return _lightgbm_unavailable_signal(symbols)


# Import operational RL runtime facade.
try:
    from ml.rl_runtime import (
        get_operational_rl_prediction,
        get_operational_rl_stats,
        get_operational_rl_weights,
        update_operational_rl_metrics,
    )

    _RL_UNIFIED_AVAILABLE = True
except AI_IMPORT_EXCEPTIONS as _e:
    _RL_UNIFIED_AVAILABLE = False
    logging.getLogger(__name__).warning("Unified RL Manager not available: %s", _e)

# Optional RL adapter chain
try:
    from ml.rl_predict_enhanced import (
        get_regime_adjusted_weights,
        get_regime_adjusted_weights_for_regime,
        predict_prob_with_regime,
        predict_probs_with_regime,
    )

    _RL_ENHANCED_AVAILABLE = True
except AI_IMPORT_EXCEPTIONS:
    _RL_ENHANCED_AVAILABLE = False
    predict_prob_with_regime = None
    predict_probs_with_regime = None


_UNRESOLVED_RL_PREDICT = object()
_RL_PREDICT_FUNCS: tuple[Any, Any | None] | object = _UNRESOLVED_RL_PREDICT


def _resolve_rl_predict_funcs() -> tuple[Any, Any | None]:
    global _RL_PREDICT_FUNCS
    if _RL_PREDICT_FUNCS is _UNRESOLVED_RL_PREDICT:
        try:
            from ml.rl_predict import predict_prob
            from ml.rl_predict import predict_prob_batch

            _RL_PREDICT_FUNCS = (predict_prob, predict_prob_batch)
        except AI_IMPORT_EXCEPTIONS:
            _RL_PREDICT_FUNCS = (None, None)
    return _RL_PREDICT_FUNCS  # type: ignore[return-value]


def _rl_predict_prob(symbol: str, window: int = 60) -> float:
    predict_prob, _ = _resolve_rl_predict_funcs()
    if predict_prob is None:
        return 0.5
    try:
        return float(predict_prob(symbol, window))
    except AI_RUNTIME_EXCEPTIONS:
        return 0.5


def _rl_predict_prob_batch(symbols: list[str], window: int = 60) -> dict[str, float]:
    predict_prob, predict_prob_batch = _resolve_rl_predict_funcs()
    if predict_prob_batch is not None:
        try:
            return {str(symbol): float(score) for symbol, score in predict_prob_batch(symbols, window).items()}
        except AI_RUNTIME_EXCEPTIONS:
            pass
    if predict_prob is None:
        return {symbol: 0.5 for symbol in symbols}
    return {symbol: _rl_predict_prob(symbol, window) for symbol in symbols}


from logger import get_logger, log_event
from runtime_paths import LOGS_DIR as RUNTIME_LOGS_DIR
from runtime_paths import METRICS_DIR as RUNTIME_METRICS_DIR

# Provider fallback model stays explicit and centralized.
try:
    from settings import OPENAI_FALLBACK_MODEL
except ImportError:
    OPENAI_FALLBACK_MODEL = (os.getenv("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini").strip()

def _resolve_runtime_config_path() -> pathlib.Path:
    configured = str(os.getenv("APP_CONFIG_FILE") or "").strip()
    if configured:
        return pathlib.Path(configured).expanduser().resolve()
    return pathlib.Path(__file__).resolve().parent / "config.json"


_RUNTIME_CONFIG_PATH = _resolve_runtime_config_path()
_BATCH_RUNTIME_CONTEXT = create_runtime_context(ai_import_exceptions=AI_IMPORT_EXCEPTIONS)
reload_runtime_context(
    _BATCH_RUNTIME_CONTEXT,
    config_path=_RUNTIME_CONFIG_PATH,
)
HYBRID_WEIGHTS = dict(_BATCH_RUNTIME_CONTEXT.hybrid_weights)
ACTIVE_AI_MODE = _BATCH_RUNTIME_CONTEXT.active_ai_mode
ACTIVE_OPENAI_MODEL = _BATCH_RUNTIME_CONTEXT.active_openai_model
ACTIVE_DEEPSEEK_MODEL = _BATCH_RUNTIME_CONTEXT.active_deepseek_model
aoai = _BATCH_RUNTIME_CONTEXT.aoai
deepseek_client = _BATCH_RUNTIME_CONTEXT.deepseek_client
OPENAI_CONCURRENCY = _BATCH_RUNTIME_CONTEXT.openai_concurrency
_semaphore = _BATCH_RUNTIME_CONTEXT.semaphore
OPENAI_MIN_INTERVAL_SEC = _BATCH_RUNTIME_CONTEXT.openai_min_interval_sec
DEEPSEEK_MIN_INTERVAL_SEC = _BATCH_RUNTIME_CONTEXT.deepseek_min_interval_sec
DEEPSEEK_COOLDOWN_SEC = _BATCH_RUNTIME_CONTEXT.deepseek_cooldown_sec
_PROVIDER_STATE = _BATCH_RUNTIME_CONTEXT.provider_state
_PACERS = _BATCH_RUNTIME_CONTEXT.pacers
_OPENAI_DISABLED = _BATCH_RUNTIME_CONTEXT.openai_disabled
_OPENAI_DISABLE_REASON = _BATCH_RUNTIME_CONTEXT.openai_disable_reason
_DEEPSEEK_DISABLED_UNTIL = _BATCH_RUNTIME_CONTEXT.deepseek_disabled_until
_DEEPSEEK_DISABLE_REASON = _BATCH_RUNTIME_CONTEXT.deepseek_disable_reason
_DEEPSEEK_DISABLE_LOGGED = _BATCH_RUNTIME_CONTEXT.deepseek_disable_logged


def _build_rl_deps() -> BatchRlDeps:
    return BatchRlDeps(
        hybrid_weights=HYBRID_WEIGHTS,
        rl_unified_available=_RL_UNIFIED_AVAILABLE,
        rl_enhanced_available=_RL_ENHANCED_AVAILABLE,
        get_operational_rl_prediction=get_operational_rl_prediction if _RL_UNIFIED_AVAILABLE else None,
        get_operational_rl_weights=get_operational_rl_weights if _RL_UNIFIED_AVAILABLE else None,
        get_operational_rl_stats=get_operational_rl_stats if _RL_UNIFIED_AVAILABLE else None,
        update_operational_rl_metrics=update_operational_rl_metrics if _RL_UNIFIED_AVAILABLE else None,
        predict_prob_with_regime=globals().get("predict_prob_with_regime"),
        get_regime_adjusted_weights=globals().get("get_regime_adjusted_weights"),
        get_regime_adjusted_weights_for_regime=globals().get("get_regime_adjusted_weights_for_regime"),
        rl_predict_prob=globals().get("_rl_predict_prob"),
        runtime_exceptions=AI_RUNTIME_EXCEPTIONS,
        get_rl_confidence_v4_impl=_get_rl_confidence_v4_impl,
        get_rl_confidence_v3_impl=_get_rl_confidence_v3_impl,
        get_dynamic_weights_impl=_get_dynamic_weights_impl,
        get_ensemble_rl_signal_impl=_get_ensemble_rl_signal_impl,
        get_multi_agent_signal_impl=_get_multi_agent_signal_impl,
        update_rl_trade_metrics_impl=_update_rl_trade_metrics_impl,
        get_rl_system_status_impl=_get_rl_system_status_impl,
    )


# ────────────────────────────────────────────────────────────────
# v4.0 RL Helper Functions (Unified RL Manager - ALL FEATURES)
# ────────────────────────────────────────────────────────────────
# Features: SAC, Ensemble, Multi-Agent, Microstructure, Advanced Rewards


def _get_rl_confidence_v4(symbol: str, df=None) -> dict:
    return _get_rl_confidence_v4_with_deps(symbol, _build_rl_deps(), df=df)


def _get_rl_confidence_v3(symbol: str) -> tuple:
    return _get_rl_confidence_v3_with_deps(symbol, _build_rl_deps())


def _get_rl_confidence_v3_batch(symbols: list[str]) -> dict[str, tuple[float, str, float]]:
    multipliers = {
        "trending": 1.3,
        "ranging": 0.8,
        "high_volatility": 0.6,
        "unknown": 1.0,
    }
    if _RL_ENHANCED_AVAILABLE and predict_probs_with_regime is not None:
        try:
            results = predict_probs_with_regime(symbols, 60)
            return {
                symbol: (
                    float(result.get("final_confidence", 0.5)),
                    str(result.get("regime", "unknown")),
                    float(multipliers.get(str(result.get("regime", "unknown")), 1.0)),
                )
                for symbol, result in results.items()
            }
        except AI_RUNTIME_EXCEPTIONS:
            pass

    try:
        batch_scores = _rl_predict_prob_batch(symbols, 60)
        return {symbol: (float(batch_scores.get(symbol, 0.5)), "unknown", 1.0) for symbol in symbols}
    except AI_RUNTIME_EXCEPTIONS:
        pass

    return {symbol: _get_rl_confidence_v3(symbol) for symbol in symbols}


def _get_dynamic_weights(symbol: str, df=None, regime: str | None = None) -> dict:
    return _get_dynamic_weights_with_deps(symbol, _build_rl_deps(), df=df, regime=regime)


def _get_ensemble_rl_signal(symbol: str, df=None) -> dict:
    return _get_ensemble_rl_signal_with_deps(symbol, _build_rl_deps(), df=df)


def _get_multi_agent_signal(symbol: str, df=None, regime: str = "unknown") -> dict:
    return _get_multi_agent_signal_with_deps(symbol, _build_rl_deps(), df=df, regime=regime)


def _update_rl_trade_metrics(pnl: float, is_winning: bool):
    del is_winning
    _update_rl_trade_metrics_with_deps(pnl, _build_rl_deps())


def get_rl_system_status() -> dict:
    return _get_rl_system_status_with_deps(_build_rl_deps())


# ────────────────────────────────────────────────────────────────
# Logger
# ────────────────────────────────────────────────────────────────
log = get_logger("ai_batch_manager")

# ────────────────────────────────────────────────────────────────
# Kurumsal sistem promptları (ChatGPT ve DeepSeek için ayrı)
# ────────────────────────────────────────────────────────────────
# İki LLM'e farklı analitik çerçeveler verilir — böylece gerçek diversity sağlanır.
# Her iki prompt da kurumsal kalite ve nesnellik hedefler; user prompt tarafında
# teknik veriler korunur, gereksiz ham OHLCV tekrarları azaltılır.

_PROMPTS_PATH = pathlib.Path(__file__).resolve().parent / "prompts.json"
CHATGPT_SYSTEM_PROMPT, DEEPSEEK_SYSTEM_PROMPT, TRADING_SYSTEM_PROMPT = load_prompt_bundle(_PROMPTS_PATH)


def _log_provider_init(
    provider_name: str,
    client: Any,
    *,
    model_name: str,
    allowed_modes: set[str],
    weight_key: str,
) -> None:
    if ACTIVE_AI_MODE == "disabled":
        log.info("[AI_INIT] %s disabled by config (technical-only mode)", provider_name)
        return
    if ACTIVE_AI_MODE not in allowed_modes:
        log.info("[AI_INIT] %s disabled by ai_mode=%s", provider_name, ACTIVE_AI_MODE)
        return
    try:
        provider_weight = float(HYBRID_WEIGHTS.get(weight_key, 0.0))
    except (TypeError, ValueError, AttributeError):
        provider_weight = 0.0
    if provider_weight <= 0.0:
        log.info("[AI_INIT] %s disabled by config/weight (%s weight is 0)", provider_name, weight_key)
        return
    if client:
        log.info("[AI_INIT] %s enabled=True model=%s", provider_name, model_name)
        return
    log.info("[AI_INIT] %s unavailable (API key missing or invalid)", provider_name)


def _resolve_llm_timeout_sec() -> float:
    env_value = os.getenv("LLM_TIMEOUT_SEC") or os.getenv("AI_PROVIDER_CALL_TIMEOUT_SEC")
    if env_value:
        try:
            return max(1.0, min(float(env_value), 120.0))
        except (TypeError, ValueError):
            pass
    try:
        raw = json.loads(_RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and raw.get("llm_timeout_sec") is not None:
            return max(1.0, min(float(raw.get("llm_timeout_sec")), 120.0))
    except JSON_STORAGE_EXCEPTIONS:
        pass
    return 30.0


def _resolve_llm_timeout_policy() -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "enabled": True,
        "routine_timeout_sec": 30.0,
        "critical_timeout_sec": 45.0,
        "max_timeout_sec": 60.0,
        "critical_candidate_reasons": [
            "repaired_major_data",
            "open_position",
            "near_threshold",
        ],
    }
    try:
        raw = json.loads(_RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
        llm_cfg = raw.get("llm_cost_optimizer") if isinstance(raw, dict) else {}
        policy = llm_cfg.get("adaptive_timeout") if isinstance(llm_cfg, dict) else {}
        if isinstance(policy, dict):
            merged = dict(defaults)
            merged.update(policy)
            return merged
    except JSON_STORAGE_EXCEPTIONS:
        pass
    return defaults


def _resolve_llm_contract_config() -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "schema_version": "llm-defensive-v1",
        "validation_enabled": True,
        "apply_to_decision": False,
        "audit_enabled": True,
    }
    try:
        raw = json.loads(_RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
        configured = raw.get("llm_contract") if isinstance(raw, dict) else None
        if isinstance(configured, dict):
            merged = dict(defaults)
            merged.update(configured)
            # Execution application is not a configurable Phase 10 capability.
            merged["apply_to_decision"] = False
            return merged
    except JSON_STORAGE_EXCEPTIONS:
        pass
    return defaults


def _resolve_llm_provider_policy() -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "enabled": True,
        "mode": "critical_dual_routine_single",
        "routine_provider": "deepseek",
        "critical_dual_provider": True,
        "consensus_enter_dual": True,
        "routine_escalation_mode": "strict_evidence",
        "low_confidence_dual_scope": "trade_risk_only",
        "low_confidence_threshold": 0.55,
        "weak_skip_single_provider_enabled": False,
        "routine_no_order_single_provider_enabled": False,
        "prompt_ab_shadow_enabled": True,
        "prompt_ab_shadow_sample_per_cycle": 2,
        "escalate_on_parse_error": True,
        "escalate_on_timeout": True,
        "escalate_on_technical_conflict": True,
        "escalate_statuses": ["timeout", "error", "down", "down_cooldown", "disabled_no_key"],
        "critical_score_floor": 0.58,
        "critical_candidate_reasons": [
            "repaired_major_data",
            "open_position",
            "near_threshold",
            "recent_shadow_missed_enter",
        ],
        "always_on_symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
    }
    try:
        raw = json.loads(_RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
        llm_cfg = raw.get("llm_cost_optimizer") if isinstance(raw, dict) else {}
        policy = llm_cfg.get("provider_policy") if isinstance(llm_cfg, dict) else {}
        trade_cfg = raw.get("trade_parameters") if isinstance(raw, dict) else {}
        merged = dict(defaults)
        if isinstance(policy, dict):
            merged.update(policy)
        if isinstance(trade_cfg, dict) and isinstance(trade_cfg.get("always_on_symbols"), list):
            merged["always_on_symbols"] = [str(symbol) for symbol in trade_cfg.get("always_on_symbols", [])]
        if isinstance(llm_cfg, dict) and bool(llm_cfg.get("keep_both_llms")):
            merged["enabled"] = False
        return apply_llm_provider_policy_state(merged, state=load_llm_cost_optimizer_state())
    except JSON_STORAGE_EXCEPTIONS:
        pass
    return defaults


def _resolve_llm_result_cache_policy() -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "enabled": False,
        "mode": "stable_low_risk_one_cycle",
        "max_age_seconds": 2100,
        "min_confidence": 0.65,
        "stable_skip_score_enabled": True,
        "min_stable_skip_score": 0.85,
        "min_data_quality_score": 0.8,
        "near_threshold_score_floor": 0.58,
        "always_on_symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
    }
    try:
        raw = json.loads(_RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
        llm_cfg = raw.get("llm_cost_optimizer") if isinstance(raw, dict) else {}
        policy = llm_cfg.get("llm_result_cache") if isinstance(llm_cfg, dict) else {}
        trade_cfg = raw.get("trade_parameters") if isinstance(raw, dict) else {}
        merged = dict(defaults)
        if isinstance(policy, dict):
            merged.update(policy)
        if isinstance(trade_cfg, dict) and isinstance(trade_cfg.get("always_on_symbols"), list):
            merged["always_on_symbols"] = [str(symbol) for symbol in trade_cfg.get("always_on_symbols", [])]
        return merged
    except JSON_STORAGE_EXCEPTIONS:
        pass
    return defaults


_log_provider_init(
    "ChatGPT",
    aoai,
    model_name=ACTIVE_OPENAI_MODEL,
    allowed_modes={"chatgpt", "hybrid"},
    weight_key="chatgpt",
)
_log_provider_init(
    "DeepSeek",
    deepseek_client,
    model_name=ACTIVE_DEEPSEEK_MODEL,
    allowed_modes={"deepseek", "hybrid"},
    weight_key="deepseek",
)

# ────────────────────────────────────────────────────────────────
# File paths
# ────────────────────────────────────────────────────────────────
METRICS_DIR = RUNTIME_METRICS_DIR
AI_PRED_FILE = METRICS_DIR / "ai_predictions.json"
SIGNALS_FILE = METRICS_DIR / "signals_last.json"
LLM_CACHE_METRICS_FILE = METRICS_DIR / "llm_cache_metrics.json"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

_LLM_CACHE = LLMBatchCache(
    metrics_file=LLM_CACHE_METRICS_FILE,
    storage_exceptions=JSON_STORAGE_EXCEPTIONS,
    policy=_resolve_llm_result_cache_policy(),
)
_LLM_CACHE_STATS = _LLM_CACHE.stats
_LLM_CACHE_CONTEXT_REGIME = _LLM_CACHE.context_regime


_ORIGINAL_CALL_MODEL_WRAPPER = None


async def _dispatch_call_model_impl(
    *,
    state,
    pacers,
    semaphore,
    client,
    model_name,
    msgs,
    label="AI",
    parse_payload=None,
    log=None,
    deepseek_cooldown_sec=0.0,
    send_notification=None,
    deepseek_reasoning=None,
    retry=3,
    backoff=1.5,
):
    """Route provider calls through the monkeypatchable public wrapper in tests."""
    del state, pacers, semaphore, parse_payload, deepseek_cooldown_sec, send_notification
    if _ORIGINAL_CALL_MODEL_WRAPPER is not None and _call_model is not _ORIGINAL_CALL_MODEL_WRAPPER:
        return await _call_model(
            client,
            model_name,
            msgs,
            label=label,
            retry=retry,
            backoff=backoff,
        )
    return await _call_model_impl(
        state=_PROVIDER_STATE,
        pacers=_PACERS,
        semaphore=_semaphore,
        client=client,
        model_name=model_name,
        msgs=msgs,
        label=label,
        parse_payload=_safe_json_parse,
        log=log or globals().get("log"),
        deepseek_cooldown_sec=DEEPSEEK_COOLDOWN_SEC,
        send_notification=None,
        deepseek_reasoning=deepseek_reasoning,
        retry=retry,
        backoff=backoff,
    )


def _build_execution_deps() -> BatchExecutionDeps:
    llm_log_path = RUNTIME_LOGS_DIR / "llm_decisions.jsonl"
    return BatchExecutionDeps(
        llm_cache=_LLM_CACHE,
        provider_state=_PROVIDER_STATE,
        pacers=_PACERS,
        semaphore=_semaphore,
        aoai=aoai,
        deepseek_client=deepseek_client,
        active_ai_mode=ACTIVE_AI_MODE,
        active_openai_model=ACTIVE_OPENAI_MODEL,
        active_deepseek_model=ACTIVE_DEEPSEEK_MODEL,
        openai_fallback_model=OPENAI_FALLBACK_MODEL,
        deepseek_cooldown_sec=DEEPSEEK_COOLDOWN_SEC,
        hybrid_weights=HYBRID_WEIGHTS,
        chatgpt_system_prompt=CHATGPT_SYSTEM_PROMPT,
        deepseek_system_prompt=DEEPSEEK_SYSTEM_PROMPT,
        llm_log_path=llm_log_path,
        sync_provider_state_from_globals=_sync_provider_state_from_globals,
        sync_provider_state_to_globals=_sync_provider_state_to_globals,
        safe_json_parse=_safe_json_parse,
        analyze_batch_impl=_analyze_batch_impl,
        call_model_impl=_dispatch_call_model_impl,
        get_confidences_batch_impl=_get_confidences_batch_impl,
        build_batch_prompt=_build_batch_prompt,
        should_use_model_fallback=should_use_model_fallback,
        build_symbol_index=build_symbol_index,
        resolve_provider_result=resolve_provider_result,
        build_disabled_provider_result=build_disabled_provider_result,
        calibrate_confidence=_calibrate_confidence,
        transformer_predict_prob=_transformer_predict_prob,
        transformer_predict_prob_batch=_transformer_predict_prob_batch,
        transformer_predict_signal_batch=_transformer_predict_signal_batch,
        lightgbm_predict_signal_batch=_lightgbm_predict_signal_batch,
        rl_predict_prob=globals().get("_rl_predict_prob"),
        get_rl_confidence_v3=_get_rl_confidence_v3,
        get_rl_confidence_v3_batch=_get_rl_confidence_v3_batch,
        get_dynamic_weights=_get_dynamic_weights,
        build_merged_symbol_payload=build_merged_symbol_payload,
        build_provider_detail=build_provider_detail,
        classify_provider_status=classify_provider_status,
        log_decision_audit=_log_decision_audit,
        log_ai_prediction=_log_ai_prediction,
        log_ai_prediction_batch=_log_ai_prediction_batch,
        update_signals_file=_update_signals_file,
        update_signals_file_batch=_update_signals_file_batch,
        runtime_exceptions=AI_RUNTIME_EXCEPTIONS,
        json_storage_exceptions=JSON_STORAGE_EXCEPTIONS,
        get_deepseek_cooldown_state=lambda: (
            _DEEPSEEK_DISABLED_UNTIL,
            _DEEPSEEK_DISABLE_REASON,
            _DEEPSEEK_DISABLE_LOGGED,
        ),
        set_deepseek_disable_logged=_set_deepseek_disable_logged,
        llm_timeout_sec=_resolve_llm_timeout_sec(),
        llm_timeout_policy=_resolve_llm_timeout_policy(),
        llm_provider_policy=_resolve_llm_provider_policy(),
        llm_contract_config=_resolve_llm_contract_config(),
        log=log,
    )


# ────────────────────────────────────────────────────────────────
# AI prediction logger
# ────────────────────────────────────────────────────────────────
_DECISION_AUDIT_PATH = RUNTIME_LOGS_DIR / "decision_audit.jsonl"


def _sync_provider_state_from_globals() -> None:
    _sync_runtime_provider_state_from_globals(
        _BATCH_RUNTIME_CONTEXT,
        openai_disabled=_OPENAI_DISABLED,
        openai_disable_reason=_OPENAI_DISABLE_REASON,
        deepseek_disabled_until=_DEEPSEEK_DISABLED_UNTIL,
        deepseek_disable_reason=_DEEPSEEK_DISABLE_REASON,
        deepseek_disable_logged=_DEEPSEEK_DISABLE_LOGGED,
    )


def _sync_provider_state_to_globals() -> None:
    global _OPENAI_DISABLED, _OPENAI_DISABLE_REASON
    global _DEEPSEEK_DISABLED_UNTIL, _DEEPSEEK_DISABLE_REASON, _DEEPSEEK_DISABLE_LOGGED
    _sync_runtime_provider_state_to_context(_BATCH_RUNTIME_CONTEXT)
    _OPENAI_DISABLED = bool(_BATCH_RUNTIME_CONTEXT.openai_disabled)
    _OPENAI_DISABLE_REASON = _BATCH_RUNTIME_CONTEXT.openai_disable_reason
    _DEEPSEEK_DISABLED_UNTIL = float(_BATCH_RUNTIME_CONTEXT.deepseek_disabled_until)
    _DEEPSEEK_DISABLE_REASON = str(_BATCH_RUNTIME_CONTEXT.deepseek_disable_reason or "")
    _DEEPSEEK_DISABLE_LOGGED = bool(_BATCH_RUNTIME_CONTEXT.deepseek_disable_logged)


def _set_deepseek_disable_logged(value: bool) -> None:
    global _DEEPSEEK_DISABLE_LOGGED
    _set_runtime_deepseek_disable_logged(_BATCH_RUNTIME_CONTEXT, value)
    _DEEPSEEK_DISABLE_LOGGED = bool(value)


def _log_decision_audit(
    symbol: str,
    models: Dict[str, Any],
    combined_confidence: float,
    consensus_direction: str | None,
    consensus_action: str | None,
) -> None:
    _log_decision_audit_impl(
        audit_path=_DECISION_AUDIT_PATH,
        storage_exceptions=JSON_STORAGE_EXCEPTIONS,
        symbol=symbol,
        models=models,
        combined_confidence=combined_confidence,
        consensus_direction=consensus_direction,
        consensus_action=consensus_action,
    )


def _log_ai_prediction(
    symbol,
    model,
    confidence,
    action,
    base_decision,
    price=None,
    outcome=None,
    pnl=None,
    *,
    direction=None,
    decision_id=None,
    provider_status=None,
):
    _log_ai_prediction_impl(
        pred_file=AI_PRED_FILE,
        storage_exceptions=JSON_STORAGE_EXCEPTIONS,
        symbol=symbol,
        model=model,
        confidence=confidence,
        action=action,
        base_decision=base_decision,
        price=price,
        outcome=outcome,
        pnl=pnl,
        direction=direction,
        decision_id=decision_id,
        provider_status=provider_status,
    )


def _log_ai_prediction_batch(entries: list[dict[str, Any]]) -> None:
    _log_ai_predictions_batch_impl(
        pred_file=AI_PRED_FILE,
        storage_exceptions=JSON_STORAGE_EXCEPTIONS,
        entries=entries,
    )


# ────────────────────────────────────────────────────────────────
# Signal logger used by the dashboard signal stream
# ────────────────────────────────────────────────────────────────
def _update_signals_file(symbol, confidence, base_decision, rationale, model="hybrid"):
    _update_signals_file_impl(
        signals_file=SIGNALS_FILE,
        storage_exceptions=JSON_STORAGE_EXCEPTIONS,
        symbol=symbol,
        confidence=confidence,
        base_decision=base_decision,
        rationale=rationale,
        model=model,
    )


def _update_signals_file_batch(entries: list[dict[str, Any]]) -> None:
    _update_signals_file_batch_impl(
        signals_file=SIGNALS_FILE,
        storage_exceptions=JSON_STORAGE_EXCEPTIONS,
        entries=entries,
    )


# ────────────────────────────────────────────────────────────────
# JSON parser with resilient list/object extraction
# ────────────────────────────────────────────────────────────────
def _safe_json_parse(text: str):
    return _safe_json_parse_impl(text)


# ────────────────────────────────────────────────────────────────
# Confidence calibration
# ────────────────────────────────────────────────────────────────
def _calibrate_confidence(conf: float) -> float:
    """Local wrapper around the shared calibration helper."""
    return _shared_calibrate_confidence(conf)


# -------------------------------------------------------------------------
# AI Rating Mapper
#
# In addition to the raw confidence values returned by ChatGPT/DeepSeek, we map
# the qualitative fields "action" and "direction" into a numeric rating in
# the range [0.0, 1.0].  This provides the hybrid scoring logic with a more
# interpretable signal: strong buy/sell recommendations push the rating
# towards the extremes, whereas neutral or absent directions keep it near 0.5.
#
# The mapping uses two components:
#   - action_map: assigns a baseline strength to the model's disposition
#       enter → 1.0 (strong conviction to trade)
#       hold  → 0.5 (moderate conviction)
#       skip  → 0.25 (weak or no conviction)
#       exit  → 0.0 (no conviction / caution)
#   - dir_map: encodes the trade direction as +1 for long, −1 for short and
#       0 for unspecified.  Multiplying the baseline strength by the direction
#       yields a value in [−1, +1]; this is then shifted into [0, 1] by
#       rating = 0.5 + base/2.  Missing or neutral directions thus yield
#       0.5, which has no net influence.


def _map_ai_rating(res: Dict[str, Any]) -> float:
    """Map qualitative AI output into a numeric rating between 0 and 1."""
    action = str(res.get("action", "")).lower() if isinstance(res, dict) else ""
    direction = str(res.get("direction", "")).lower() if isinstance(res, dict) else ""
    # Baseline strength per action
    action_map = {
        "enter": 1.0,
        "hold": 0.5,
        "skip": 0.25,
        "exit": 0.0,
    }
    # Direction encoding
    dir_map = {
        "long": 1.0,
        "short": -1.0,
    }
    a_score = action_map.get(action, 0.5)
    d_score = dir_map.get(direction, 0.0)
    base = a_score * d_score
    rating = 0.5 + (base / 2.0)
    # Clamp into [0, 1]
    if rating < 0.0:
        rating = 0.0
    elif rating > 1.0:
        rating = 1.0
    return float(rating)


# ────────────────────────────────────────────────────────────────
# Batch prompt builder based on the trading system prompt
# ────────────────────────────────────────────────────────────────
# Prompt boundary invariant is preserved in ai.batch_prompts:
# --- BEGIN MARKET DATA ---
# --- END MARKET DATA ---
def _mk_batch_prompt(
    items: List[Dict[str, Any]],
    system_prompt: str | None = None,
    *,
    compact_override: bool | None = None,
) -> List[Dict[str, str]]:
    return _build_execution_deps().build_batch_prompt(
        items,
        system_prompt=system_prompt,
        default_system_prompt=CHATGPT_SYSTEM_PROMPT,
        compact_override=compact_override,
    )


# ────────────────────────────────────────────────────────────────
# Provider invocation
# ────────────────────────────────────────────────────────────────
async def _call_model(
    client,
    model_name,
    msgs,
    label="AI",
    retry=3,
    backoff=1.5,
    batch_items: List[Dict[str, Any]] | None = None,
):
    return await _call_model_with_deps(
        _build_execution_deps(),
        client,
        model_name,
        msgs,
        label=label,
        retry=retry,
        backoff=backoff,
        batch_items=batch_items,
    )


_ORIGINAL_CALL_MODEL_WRAPPER = _call_model


# ────────────────────────────────────────────────────────────────
# Batch analiz fonksiyonu
# ────────────────────────────────────────────────────────────────
async def _analyze_batch(batch: List[Dict[str, Any]]):
    return await _analyze_batch_with_deps(batch, _build_execution_deps())


# ────────────────────────────────────────────────────────────────
# Main batch entrypoint
# ────────────────────────────────────────────────────────────────
def _resolve_llm_batch_size(default: int = 20) -> int:
    try:
        payload = json.loads(pathlib.Path(_RUNTIME_CONFIG_PATH).read_text(encoding="utf-8"))
        router = payload.get("llm_candidate_router") if isinstance(payload, dict) else {}
        raw_size = router.get("batch_size") if isinstance(router, dict) else None
        if raw_size is None:
            return int(default)
        base_size = max(1, min(20, int(raw_size)))
        return select_llm_batch_size(default_batch_size=base_size, config_path=_RUNTIME_CONFIG_PATH)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return int(default)


async def get_confidences_batch(symbol_items: List[Dict[str, Any]], batch_size=None) -> Dict[str, Dict[str, Any]]:
    # 1. Check if decoupled is enabled
    decoupled = False
    try:
        raw = json.loads(pathlib.Path(_RUNTIME_CONFIG_PATH).read_text(encoding="utf-8"))
        decoupled = bool(raw.get("llm_cost_optimizer", {}).get("llm_decoupled", False))
    except Exception:  # broad-except-ok - fail-safe if runtime config is missing or invalid JSON
        pass

    inline_provider_allowed = str(os.getenv("ATB_ALLOW_INLINE_LLM_PROVIDER", "") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if decoupled or not inline_provider_allowed:
        import time

        import db_utils
        from ai.llm_direction_worker import publish_direction_candidate_snapshots
        from decision.llm_bias import build_neutral_llm_bias, normalize_llm_bias_payload

        # Load max age limit (default: 4 hours = 14400 seconds)
        max_age = 14400
        try:
            raw = json.loads(pathlib.Path(_RUNTIME_CONFIG_PATH).read_text(encoding="utf-8"))
            max_age = int(raw.get("llm_cost_optimizer", {}).get("llm_decoupled_max_age_seconds", 14400))
        except Exception:  # broad-except-ok - fail-safe fallback to default age if read fails
            pass

        try:
            publish_direction_candidate_snapshots(symbol_items)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            log.debug("[LLM_DIRECTION_WORKER] candidate snapshot publish failed: %s", exc)

        symbols = [item["symbol"] for item in symbol_items if "symbol" in item]
        try:
            cached_data = db_utils.get_llm_consensus(symbols)
        except Exception:  # broad-except-ok - decoupled LLM must fail neutral when cache storage is unavailable
            cached_data = {}

        final_results = {}
        missing_items = []

        now = time.time()
        for item in symbol_items:
            symbol = item.get("symbol")
            if not symbol:
                continue

            cached_res = cached_data.get(symbol)
            if cached_res and isinstance(cached_res, dict):
                normalized = normalize_llm_bias_payload(
                    str(symbol),
                    cached_res,
                    now=now,
                    timestamp=cached_res.get("_timestamp", cached_res.get("timestamp")),
                    max_age_seconds=max_age,
                    source="llm_direction_worker",
                    reason="cache_hit",
                )
                if not normalized.get("stale"):
                    normalized.pop("_timestamp", None)
                    final_results[symbol] = normalized
                    continue
                final_results[symbol] = normalized
                continue

            # Cache miss or stale: decoupled mode keeps the live decision loop
            # provider-free. The background direction worker is responsible for
            # refreshing this cache.
            missing_items.append(item)

        if not missing_items:
            return final_results

        for item in missing_items:
            sym = item.get("symbol")
            if sym:
                final_results[sym] = build_neutral_llm_bias(
                    str(sym),
                    reason="cache_miss_or_stale",
                    now=now,
                    max_age_seconds=max_age,
                )

        return final_results

    effective_batch_size = _resolve_llm_batch_size() if batch_size is None else int(batch_size)
    return await _get_confidences_batch_with_deps(
        symbol_items,
        _build_execution_deps(),
        batch_size=max(1, min(20, effective_batch_size)),
    )


# ────────────────────────────────────────────────────────────────
# Config Hot-Reload Mechanism
# ────────────────────────────────────────────────────────────────
def load_config():
    """
    Reloads HYBRID_WEIGHTS from config.json.
    Called by main_bot_async.py loop to support hot-tuning.
    """
    global ACTIVE_AI_MODE, ACTIVE_DEEPSEEK_MODEL, ACTIVE_OPENAI_MODEL, HYBRID_WEIGHTS
    try:
        reload_runtime_context(
            _BATCH_RUNTIME_CONTEXT,
            config_path=_RUNTIME_CONFIG_PATH,
        )
        ACTIVE_AI_MODE = _BATCH_RUNTIME_CONTEXT.active_ai_mode
        ACTIVE_OPENAI_MODEL = _BATCH_RUNTIME_CONTEXT.active_openai_model
        ACTIVE_DEEPSEEK_MODEL = _BATCH_RUNTIME_CONTEXT.active_deepseek_model
        HYBRID_WEIGHTS = dict(_BATCH_RUNTIME_CONTEXT.hybrid_weights)
    except AI_RUNTIME_EXCEPTIONS as e:
        log_event(
            log,
            logging.WARNING,
            "ai_config_reload_failed",
            "[AI] Config reload failed",
            error=str(e),
        )
