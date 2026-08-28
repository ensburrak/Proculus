from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Dict, List

import ai_batch_manager as _abm
from logger import log_event

from ai_batch_manager import (
    HYBRID_WEIGHTS,
    _call_model,
    _calibrate_confidence,
    _get_dynamic_weights,
    _get_rl_confidence_v3,
    _log_ai_prediction,
    _map_ai_rating,
    _mk_batch_prompt,
    _rl_predict_prob,
    _transformer_predict_prob_batch,
    _transformer_predict_signal_batch,
    _update_signals_file,
    aoai,
    deepseek_client,
    log,
)

LLM_CACHE_AVAILABLE = False
RECOVERABLE_EXCEPTIONS = _abm.AI_RUNTIME_EXCEPTIONS
TRADING_MODE = "hybrid"


def _get_technical_fallback_confidence(symbol: str, features: Any | None = None) -> float:
    del symbol, features
    return 0.5


def cache_get(symbol: str, features: Any, base_decision: str) -> None:
    del symbol, features, base_decision
    return None


def cache_put(*args: Any, **kwargs: Any) -> None:
    del args, kwargs


def cache_stats() -> dict[str, float]:
    return {"hits": 0, "misses": 0, "hit_rate_pct": 0.0, "estimated_cost_savings_pct": 0.0}


async def analyze_batch_service(batch: List[Dict[str, Any]]):
    results: Dict[str, Dict[str, Any]] = {}

    # ═══════════════════════════════════════════════════════════════════
    # [v4.1] TECHNICAL-ONLY MODE: Skip all LLM calls, use TA-based scoring
    # ═══════════════════════════════════════════════════════════════════
    if TRADING_MODE == "technical_only":
        log.info("[TECHNICAL_ONLY] LLM calls skipped, using technical analysis")
        for it in batch:
            sym = it["symbol"]
            tech_conf = _get_technical_fallback_confidence(sym, it.get("features"))
            results[sym] = {
                "chatgpt": {
                    "symbol": sym,
                    "confidence": tech_conf,
                    "rationale": "technical_only_mode"
                },
                "deepseek": {
                    "symbol": sym,
                    "confidence": tech_conf,
                    "rationale": "technical_only_mode"
                },
                "_provider_status": {
                    "chatgpt": "skipped_technical_mode",
                    "deepseek": "skipped_technical_mode"
                }
            }
        return results

    # ═══════════════════════════════════════════════════════════════════
    # [2026-02-09] LLM CACHE CHECK: Use cached responses when available
    # ═══════════════════════════════════════════════════════════════════
    cached_symbols: List[str] = []
    uncached_batch: List[Dict[str, Any]] = []

    if LLM_CACHE_AVAILABLE:
        for it in batch:
            sym = it["symbol"]
            features = it.get("features", {})
            base_decision = it.get("base_decision", "skip")

            cached = cache_get(sym, features, base_decision)
            if cached:
                # Use cached response
                cached_symbols.append(sym)
                results[sym] = {
                    "chatgpt": {
                        "symbol": sym,
                        "confidence": cached["confidence"],
                        "direction": cached.get("direction"),
                        "action": cached.get("action"),
                        "rationale": cached["rationale"]
                    },
                    "deepseek": {
                        "symbol": sym,
                        "confidence": cached["confidence"],
                        "direction": cached.get("direction"),
                        "action": cached.get("action"),
                        "rationale": cached["rationale"]
                    },
                    "_cached": True,
                    "_cache_age_sec": cached.get("cache_age_sec", 0)
                }
            else:
                uncached_batch.append(it)

        if cached_symbols:
            log_event(
                log,
                logging.INFO,
                "llm_cache_batch_hit",
                "[LLM_CACHE] Batch cache lookup completed",
                cached_symbols=len(cached_symbols),
                uncached_symbols=len(uncached_batch),
            )
    else:
        uncached_batch = batch

    # If all symbols were cached, return early (huge cost savings!)
    if not uncached_batch:
        log.info("[LLM_CACHE] 100% cache hit - no LLM calls needed!")
        return results

    # Generate prompt only for uncached symbols
    msgs = _mk_batch_prompt(uncached_batch)

    # Track provider-level status for this batch. These are attached to each
    # per-symbol result so the controller can distinguish between a partial
    # outage (one provider down) and a full LLM outage (both down).
    gpt_status: str = "disabled"
    gpt_err: str | None = None
    ds_status: str = "disabled"
    ds_err: str | None = None

    # ChatGPT cagrisi
    if aoai:
        log.info("[ChatGPT] batch cagrisi basliyor (%d coin)", len(uncached_batch))
        gpt_items, gpt_status, gpt_err = await _call_model(
            aoai,
            _abm.ACTIVE_OPENAI_MODEL,
            msgs,
            label="ChatGPT",
        )
        log.info("[ChatGPT] batch yaniti alindi (%d kayit) status=%s", len(gpt_items), gpt_status)
        if not gpt_items:
            log.warning("[ChatGPT] bos yanit veya fallback kullaniliyor")

        # Gelenleri symbole gore map et
        gpt_by_symbol: Dict[str, Dict[str, Any]] = {}
        for r in gpt_items:
            sym = r.get("symbol")
            if not sym:
                continue
            r = dict(r)

            # reason -> rationale map
            if "rationale" not in r and "reason" in r:
                r["rationale"] = r["reason"]

            # master_confidence -> confidence alias
            raw_conf = r.get("confidence", r.get("master_confidence", 0.5))
            r["confidence"] = _calibrate_confidence(raw_conf)

            gpt_by_symbol[str(sym).upper()] = r

        # Batch'teki her symbol icin garanti entry
        for it in uncached_batch:
            sym = it["symbol"]
            key = str(sym).upper()
            r = gpt_by_symbol.get(key)
            if not r:
                log.warning("[ChatGPT] response missing for symbol=%s, fallback kullanilacak", sym)
                if gpt_status != "ok":
                    reason = f"chatgpt_{gpt_status}"
                else:
                    reason = "missing-from-gpt"
                r = {
                    "symbol": sym,
                    "confidence": _get_technical_fallback_confidence(sym, it.get("features")),
                    "rationale": reason
                }
            results.setdefault(sym, {})["chatgpt"] = r
    else:
        gpt_status = "disabled_no_key"
        log.warning("[ChatGPT] disabled (OPENAI_API_KEY bos veya gecersiz)")
        for it in uncached_batch:
            sym = it["symbol"]
            results.setdefault(sym, {})["chatgpt"] = {
                "symbol": sym,
                "confidence": _get_technical_fallback_confidence(sym),
                "rationale": "no-openai"
            }

    # DeepSeek cagrisi
    if deepseek_client:
        log.info("[DeepSeek] batch cagrisi basliyor (%d coin)", len(uncached_batch))
        now = time.monotonic()
        ds_items: list[dict] = []
        if now < _abm._DEEPSEEK_DISABLED_UNTIL:
            ds_status = "down_cooldown"
            # Log only once per cooldown window to avoid spam.
            if not _abm._DEEPSEEK_DISABLE_LOGGED:
                _abm._DEEPSEEK_DISABLE_LOGGED = True
                remaining = int(max(0.0, _abm._DEEPSEEK_DISABLED_UNTIL - now))
                log.warning(
                    "[DeepSeek] disabled by cooldown (%ds remaining). reason=%s",
                    remaining,
                    _abm._DEEPSEEK_DISABLE_REASON,
                )
        else:
            ds_items, ds_status, ds_err = await _call_model(
                deepseek_client,
                _abm.ACTIVE_DEEPSEEK_MODEL,
                msgs,
                label="DeepSeek",
            )

        log.info("[DeepSeek] batch yaniti alindi (%d kayit) status=%s", len(ds_items), ds_status)
        if not ds_items:
            log.warning("[DeepSeek] bos yanit veya disabled (status=%s)", ds_status)

        ds_by_symbol: Dict[str, Dict[str, Any]] = {}
        for r in ds_items:
            sym = r.get("symbol")
            if not sym:
                continue
            r = dict(r)

            # reason -> rationale map
            if "rationale" not in r and "reason" in r:
                r["rationale"] = r["reason"]

            # master_confidence -> confidence alias
            raw_conf = r.get("confidence", r.get("master_confidence", 0.5))
            r["confidence"] = _calibrate_confidence(raw_conf)

            ds_by_symbol[str(sym).upper()] = r

        for it in uncached_batch:
            sym = it["symbol"]
            key = str(sym).upper()
            r = ds_by_symbol.get(key)
            if not r:
                # DeepSeek bos / yetersiz kaldiginda deterministik degrade.
                # Avoid generic "fallback" token so the controller does not
                # misclassify a partial outage as a full LLM outage.
                reason = f"deepseek_{ds_status}"
                r = {
                    "symbol": sym,
                    "confidence": _get_technical_fallback_confidence(sym, it.get("features")),
                    "rationale": reason
                }
            results.setdefault(sym, {})["deepseek"] = r
    else:
        ds_status = "disabled_no_key"
        log.warning("[DeepSeek] disabled (DEEPSEEK_API_KEY bos veya gecersiz)")
        for it in uncached_batch:
            sym = it["symbol"]
            results.setdefault(sym, {})["deepseek"] = {
                "symbol": sym,
                "confidence": _get_technical_fallback_confidence(sym),
                "rationale": "no-deepseek"
            }


    # Ortalama birlestirme
    merged: Dict[str, Dict[str, Any]] = {}

    # Run transformer once for all symbols (single forward pass).
    tf_raw_batch: Dict[str, float] = {}
    tf_signal_batch: Dict[str, Dict[str, Any]] = {}
    try:
        tf_signal_batch = _transformer_predict_signal_batch([it["symbol"] for it in uncached_batch], 0)
    except RECOVERABLE_EXCEPTIONS:
        try:
            tf_raw_batch = _transformer_predict_prob_batch([it["symbol"] for it in uncached_batch], 60)
        except RECOVERABLE_EXCEPTIONS:
            tf_raw_batch = {}

    for it in uncached_batch:
        sym = it["symbol"]
        cg = results.get(sym, {}).get("chatgpt", {"confidence": 0.25, "rationale": "chatgpt_missing"})
        ds = results.get(sym, {}).get("deepseek", {"confidence": 0.25, "rationale": "deepseek_missing"})

        # ═══════════════════════════════════════════════════════════
        # V3.0: REGIME-AWARE TRANSFORMER + RL PREDICTIONS
        # ═══════════════════════════════════════════════════════════
        
        # Fetch Transformer probability
        base_decision = str(it.get("base_decision", "unknown")).lower()
        tf_signal = tf_signal_batch.get(sym) if isinstance(tf_signal_batch, dict) else None
        tf_status = "ok"
        model_governance = {}
        if isinstance(tf_signal, dict):
            p_up = float(tf_signal.get("p_up", 0.5))
            p_down = float(tf_signal.get("p_down", 0.5))
            p_no_trade = float(tf_signal.get("p_no_trade", 0.0))
            tf_status = str(tf_signal.get("status") or "ok")
            model_governance = tf_signal.get("model_governance") if isinstance(tf_signal.get("model_governance"), dict) else {}
            if tf_status == "no_trade" or (p_no_trade >= max(p_up, p_down) and p_no_trade > 0.4):
                tf_raw = 0.5
                tf_status = "no_trade"
            elif base_decision == "short":
                tf_raw = p_down
            elif base_decision == "long":
                tf_raw = p_up
            else:
                tf_raw = max(p_up, p_down)
        else:
            p_up = p_down = p_no_trade = None
            tf_raw = float(tf_raw_batch.get(sym, 0.5))
        tf_conf = _calibrate_confidence(tf_raw) if tf_status == "ok" else 0.0
        
        # Fetch RL probability with v3.0 regime awareness
        rl_conf = 0.5
        rl_raw = 0.5  # Default value to avoid UnboundLocalError
        regime = "unknown"
        regime_mult = 1.0
        try:
            rl_conf_v3, regime, regime_mult = _get_rl_confidence_v3(sym)
            rl_raw = rl_conf_v3
            rl_conf = _calibrate_confidence(rl_conf_v3)
        except RECOVERABLE_EXCEPTIONS as e:
            # Fallback to base RL if v3.0 fails
            try:
                rl_raw = _rl_predict_prob(sym, 60)
                rl_conf = _calibrate_confidence(rl_raw)
            except RECOVERABLE_EXCEPTIONS:
                rl_conf = 0.5
        
        # Get dynamic weights based on market regime and Historical RELIABILITY
        try:
            from ai.fusion.state import build_reliability_weighted_weights

            dynamic_weights = _get_dynamic_weights(sym)
            weighted_weights = build_reliability_weighted_weights(
                dynamic_weights,
                model_scores={
                    "chatgpt": cg.get("confidence") if gpt_status == "ok" else None,
                    "deepseek": ds.get("confidence") if ds_status == "ok" else None,
                    "transformer": tf_conf,
                    "ppo_rl": rl_conf,
                },
            )
            w_chat = float(weighted_weights.get("chatgpt", 0.0))
            w_deep = float(weighted_weights.get("deepseek", 0.0))
            w_tf = float(weighted_weights.get("transformer", 0.0))
            w_rl = float(weighted_weights.get("rl", 0.0))
        except RECOVERABLE_EXCEPTIONS:
            # Y1 FIX: Fallback ağırlıkları HYBRID_WEIGHTS ile aynı
            w_chat = float(HYBRID_WEIGHTS.get("chatgpt", 0.40))
            w_deep = float(HYBRID_WEIGHTS.get("deepseek", 0.40))
            w_tf = float(HYBRID_WEIGHTS.get("transformer", 0.10))
            w_rl = float(HYBRID_WEIGHTS.get("ppo_rl", 0.10))

        # If a provider is not healthy for this batch, remove its weight rather
        # than dragging the average down with a fake-neutral "0.25" placeholder.
        if gpt_status != "ok":
            w_chat = 0.0
        if ds_status != "ok":
            w_deep = 0.0
        if tf_status != "ok":
            w_tf = 0.0
        if isinstance(model_governance, dict) and (
            model_governance.get("accepted") is False
            or model_governance.get("schema_match") is False
            or model_governance.get("calibrated") is False
            or model_governance.get("data_fresh") is False
        ):
            w_tf = 0.0
        w_tf = min(float(w_tf), 0.10)
        w_rl = min(float(w_rl), 0.05)
        if str(regime or "").strip().lower() == "unknown" and abs(float(rl_raw) - 0.5) < 1e-9:
            w_rl = 0.0

        sum_w = w_chat + w_deep + w_tf + w_rl
        if sum_w <= 0:
            sum_w = 1.0
        
        # Use individual calibrated confidences from each model
        c_chat = float(cg.get("confidence", 0.25))
        c_deep = float(ds.get("confidence", 0.25))
        
        # Combined with regime-aware weights
        weighted_conf = (w_chat * c_chat + w_deep * c_deep + w_tf * tf_conf + w_rl * rl_conf) / sum_w

        # Compute a qualitative rating by blending the directions/actions of LLMs.
        # Transformer/RL does not provide textual direction, so we omit it here in qualitative rating.
        # Weight the rating components proportionally to their hybrid weights.
        try:
            cg_rating = _map_ai_rating(cg)
        except RECOVERABLE_EXCEPTIONS:
            cg_rating = 0.5
        try:
            ds_rating = _map_ai_rating(ds)
        except RECOVERABLE_EXCEPTIONS:
            ds_rating = 0.5
        if (w_chat + w_deep) > 0:
            rating_sum_w = w_chat + w_deep
            avg_rating = (w_chat * cg_rating + w_deep * ds_rating) / rating_sum_w
            # Blend the numeric weighted confidence and the qualitative rating.
            combined_conf = 0.7 * weighted_conf + 0.3 * avg_rating
        else:
            # No LLM ratings available; use the numeric signal only.
            avg_rating = 0.5
            combined_conf = weighted_conf


        # Build a rationale combining sources with regime info
        rationale = (
            f"GPT({gpt_status}):{cg.get('rationale')} | DS({ds_status}):{ds.get('rationale')} | "
            f"TF:{tf_conf:.3f} | RL:{rl_conf:.3f} [regime:{regime},x{regime_mult:.2f}]"
        )

        # Determine a consensus direction and action when both models agree.
        agg_direction = None
        agg_action = None
        try:
            cg_dir = cg.get("direction")
            ds_dir = ds.get("direction")
            if isinstance(cg_dir, str) and isinstance(ds_dir, str):
                if cg_dir.lower() == ds_dir.lower():
                    agg_direction = cg_dir.lower()
            cg_act = cg.get("action")
            ds_act = ds.get("action")
            if isinstance(cg_act, str) and isinstance(ds_act, str):
                if cg_act.lower() == ds_act.lower():
                    agg_action = cg_act.lower()
        except RECOVERABLE_EXCEPTIONS:
            # Silently ignore malformed fields
            agg_direction = None
            agg_action = None

        any_ok = (gpt_status == "ok") or (ds_status == "ok")
        if gpt_status == "ok" and ds_status == "ok":
            provider_status = "ok"
        elif any_ok:
            provider_status = "partial"
        else:
            provider_status = "down"

        merged[sym] = {
            "confidence": round(combined_conf, 3),
            "rationale": rationale,
            "direction": agg_direction,
            "action": agg_action,
            "provider_status": provider_status,
            "provider_status": provider_status,
            "transformer_confidence": round(float(tf_conf), 6),
            "transformer_p_up": round(float(p_up), 6) if p_up is not None else None,
            "transformer_p_down": round(float(p_down), 6) if p_down is not None else None,
            "transformer_p_no_trade": round(float(p_no_trade), 6) if p_no_trade is not None else None,
            "model_governance": model_governance,
            "ml_no_trade_veto": bool(tf_status == "no_trade"),
            "rl_confidence": round(float(rl_conf), 6),
            "providers": {
                "chatgpt": {
                    "status": gpt_status,
                    "error": gpt_err,
                    "confidence": round(c_chat, 3),
                    "direction": cg.get("direction"),
                    "action": cg.get("action"),
                    "rationale": cg.get("rationale"),
                },
                "deepseek": {
                    "status": ds_status,
                    "error": ds_err,
                    "confidence": round(c_deep, 3),
                    "direction": ds.get("direction"),
                    "action": ds.get("action"),
                    "rationale": ds.get("rationale"),
                },
            },
        }

        # Tahmin logunu yaz (Hybrid)
        base_decision = it.get("base_decision", "unknown")
        price = it.get("price", None)
        _log_ai_prediction(
            symbol=sym,
            model="hybrid",
            confidence=combined_conf,
            action="enter",
            base_decision=base_decision,
            price=price
        )
        
        # Log individual model predictions for performance monitoring
        if gpt_status == "ok":
             _log_ai_prediction(
                symbol=sym,
                model="chatgpt",
                confidence=c_chat,
                action=cg.get("action", "unknown"),
                base_decision=base_decision,
                price=price
            )
        
        if ds_status == "ok":
             _log_ai_prediction(
                symbol=sym,
                model="deepseek",
                confidence=c_deep,
                action=ds.get("action", "unknown"),
                base_decision=base_decision,
                price=price
            )

        if tf_conf is not None:
             _log_ai_prediction(
                symbol=sym,
                model="transformer",
                confidence=tf_conf,
                action="enter" if tf_raw > 0.5 else "skip", 
                base_decision=base_decision,
                price=price
            )
            
        if rl_conf is not None:
             _log_ai_prediction(
                symbol=sym,
                model="rl_ppo",
                confidence=rl_conf,
                action="enter" if rl_raw > 0.5 else "skip", 
                base_decision=base_decision,
                price=price
            )

        # Dashboard sinyal update
        _update_signals_file(
            symbol=sym,
            confidence=combined_conf,
            base_decision=base_decision,
            rationale=rationale
        )

        # ═══════════════════════════════════════════════════════════════════
        # [2026-02-09] CACHE NEW RESPONSES for future cost savings
        # ═══════════════════════════════════════════════════════════════════
        if LLM_CACHE_AVAILABLE and gpt_status == "ok" and ds_status == "ok":
            try:
                cache_put(
                    symbol=sym,
                    features=it.get("features", {}),
                    base_decision=base_decision,
                    response={
                        "confidence": combined_conf,
                        "direction": agg_direction,
                        "action": agg_action,
                        "rationale": rationale,
                    }
                )
            except RECOVERABLE_EXCEPTIONS as e:
                log_event(
                    log,
                    logging.DEBUG,
                    "llm_cache_write_failed",
                    "[LLM_CACHE] Failed to cache response",
                    symbol=sym,
                    error=str(e),
                )

    # Log cache statistics periodically
    if LLM_CACHE_AVAILABLE:
        stats = cache_stats()
        if stats.get("hits", 0) + stats.get("misses", 0) > 0:
            log_event(
                log,
                logging.INFO,
                "llm_cache_stats",
                "[LLM_CACHE] Stats updated",
                hits=int(stats.get("hits", 0)),
                misses=int(stats.get("misses", 0)),
                hit_rate_pct=float(stats.get("hit_rate_pct", 0.0)),
                estimated_cost_savings_pct=float(stats.get("estimated_cost_savings_pct", 0.0)),
            )

    return merged

