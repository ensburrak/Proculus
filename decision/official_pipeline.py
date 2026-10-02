from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .meta_quality_gate import evaluate_meta_quality_gate
from .regime_policy import get_regime_policy, normalize_regime
from .strategy_router import route_to_expert

CONFIG_FILE = Path(__file__).resolve().parents[1] / "config.json"


def _load_config() -> dict[str, Any]:
    try:
        payload = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _f(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _infer_regime(item: dict[str, Any], ta: dict[str, Any]) -> str:
    explicit = item.get("regime") or ta.get("regime")
    if explicit:
        return normalize_regime(explicit)

    mtf = item.get("mtf_features")
    if isinstance(mtf, dict):
        h4 = mtf.get("4h") if isinstance(mtf.get("4h"), dict) else {}
        h1 = mtf.get("1h") if isinstance(mtf.get("1h"), dict) else {}
        h4f, h4s = _f(h4.get("ema_fast")), _f(h4.get("ema_slow"))
        h1f, h1s = _f(h1.get("ema_fast")), _f(h1.get("ema_slow"))
        h4m = _f(h4.get("macd_hist"))
        if None not in (h4f, h4s, h1f, h1s):
            if h4f > h4s and h1f > h1s and (h4m is None or h4m >= 0):
                return "bull"
            if h4f < h4s and h1f < h1s and (h4m is None or h4m <= 0):
                return "bear"

    adx = _f(ta.get("adx"))
    if adx is not None and adx < 18:
        return "range"
    return "unknown"


def _runtime_mode(item: dict[str, Any], cfg: dict[str, Any]) -> str:
    explicit = str(item.get("runtime_mode") or item.get("mode") or "").lower()
    if explicit:
        return explicit
    if isinstance(cfg.get("paper_trading"), dict) and cfg["paper_trading"].get("enabled"):
        return "paper"
    return "live"


def _shock_detected(item: dict[str, Any], ta: dict[str, Any]) -> bool:
    candidates = [ta.get("atr_ratio"), ta.get("atr_pct"), ta.get("vol_z")]
    mtf = item.get("mtf_features")
    if isinstance(mtf, dict):
        for tf in ("15m", "1h", "4h"):
            row = mtf.get(tf)
            if isinstance(row, dict):
                candidates.extend([row.get("atr_ratio"), row.get("vol_z")])
    vals = [_f(v) for v in candidates]
    atr_values = [v for v in vals[:2] if v is not None]
    if any(v > 0.05 for v in atr_values):
        return True
    return any(v is not None and abs(v) >= 3.0 for v in vals[2:])


def _edge_validated(item: dict[str, Any]) -> bool:
    if item.get("edge_validated") is True:
        return True
    edge = item.get("edge")
    return isinstance(edge, dict) and str(edge.get("status") or "").lower() in {"approved", "validated", "confirmed"}


def _adaptive_size(*, regime: str, confidence: float, meta_scale: float, cfg: dict[str, Any]) -> float:
    sizing = cfg.get("adaptive_position_sizing") if isinstance(cfg.get("adaptive_position_sizing"), dict) else dict(cfg or {})
    caps = sizing.get("regime_size_caps") if isinstance(sizing.get("regime_size_caps"), dict) else {}
    defaults = {"bull": 1.0, "bear": 1.0, "range": 0.8, "compression": 0.75, "transition": 0.0, "shock": 0.0, "conflict": 0.0, "unknown": 0.0}
    defaults.update({str(k): float(v) for k, v in caps.items() if isinstance(v, (int, float))})
    regime_cap = max(0.0, min(1.0, defaults.get(regime, 0.0)))
    floor = float(sizing.get("confidence_floor", 0.60) or 0.60)
    full = float(sizing.get("full_size_confidence", 0.85) or 0.85)
    min_scale = float(sizing.get("confidence_floor_size_scale", 0.50) or 0.50)
    if confidence <= floor:
        conf_scale = min_scale
    elif confidence >= full:
        conf_scale = 1.0
    else:
        conf_scale = min_scale + (1.0 - min_scale) * ((confidence - floor) / max(full - floor, 1e-9))
    return max(0.0, min(1.0, regime_cap, conf_scale, meta_scale))


def _hold(symbol: str, regime: str, reason: str, **extra: Any) -> dict[str, Any]:
    payload = {
        "symbol": symbol,
        "action": "hold",
        "direction": "neutral",
        "base_decision": "neutral",
        "master_confidence": 0.0,
        "lev": 0,
        "risk_scale": 0.0,
        "regime": regime,
        "pipeline": "v2",
        "reason": reason,
    }
    payload.update(extra)
    return payload


def process_symbol_decision(*, item: dict[str, Any], ai_part: dict[str, Any] | None = None, config_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = _load_config()
    if isinstance(config_overrides, dict):
        cfg = {**cfg, **config_overrides}
    pipeline = cfg.get("pipeline_v2") if isinstance(cfg.get("pipeline_v2"), dict) else {}
    symbol = str(item.get("symbol") or "")
    ta = item.get("ta_pack") if isinstance(item.get("ta_pack"), dict) else {}
    regime = _infer_regime(item, ta)
    if _shock_detected(item, ta):
        regime = "shock"

    policy = get_regime_policy(regime)
    if policy["action"] == "no_trade":
        return _hold(symbol, regime, f"{regime}: hard no-trade")

    runtime_mode = _runtime_mode(item, cfg)
    probe_cfg = pipeline.get("learning_probe_mode") if isinstance(pipeline.get("learning_probe_mode"), dict) else {}
    probe_modes = {str(v).lower() for v in (probe_cfg.get("eligible_runtime_modes") or ["paper", "sim"])}
    probe_active = bool(probe_cfg.get("enabled", False)) and runtime_mode in probe_modes

    router_min_confidence = float(policy["min_confidence"])
    if probe_active:
        overrides = probe_cfg.get("regime_min_confidence_override") if isinstance(probe_cfg.get("regime_min_confidence_override"), dict) else {}
        try:
            router_min_confidence = float(overrides.get(regime, router_min_confidence))
        except (TypeError, ValueError):
            pass

    expert = route_to_expert(
        regime=regime,
        item=item,
        ta=ta,
        min_confidence_override=router_min_confidence,
    )
    if expert is None:
        return _hold(symbol, regime, "no regime-compatible confirmed setup")

    if expert.direction not in policy["directions"]:
        return _hold(symbol, regime, "expert direction rejected by regime policy")

    if probe_active:
        tracked_raw = probe_cfg.get("tracked_setups")
        tracked_setups = {
            str(value)
            for value in tracked_raw
            if str(value)
        } if isinstance(tracked_raw, list) else set()
        if tracked_setups and expert.setup_id not in tracked_setups:
            return _hold(
                symbol,
                regime,
                "learning probe setup not tracked",
                candidate_setup=expert.to_dict(),
                learning_probe={
                    "active": True,
                    "tracked": False,
                    "frozen": False,
                    "target_trades_per_setup": int(probe_cfg.get("target_trades_per_setup", 0) or 0),
                },
            )

        frozen_raw = probe_cfg.get("frozen_setups")
        frozen_setups = {
            str(value)
            for value in frozen_raw
            if str(value)
        } if isinstance(frozen_raw, list) else set()
        if expert.setup_id in frozen_setups:
            return _hold(
                symbol,
                regime,
                "learning probe setup frozen by negative OOS evidence",
                candidate_setup=expert.to_dict(),
                learning_probe={
                    "active": True,
                    "tracked": (not tracked_setups) or expert.setup_id in tracked_setups,
                    "frozen": True,
                    "freeze_reason": str(probe_cfg.get("frozen_setup_reason") or ""),
                    "target_trades_per_setup": int(probe_cfg.get("target_trades_per_setup", 0) or 0),
                },
            )

    strict_edge = bool(pipeline.get("strict_edge_evidence", True))
    edge_ok = _edge_validated(item)
    cold_start = bool(pipeline.get("allow_edge_cold_start", False)) and runtime_mode in {"paper", "sim", "demo", "dry-run", "dry"}
    probe_without_edge = probe_active and bool(probe_cfg.get("allow_sample_collection_without_edge", False))
    edge_scale = 1.0
    edge_mode = "validated"
    if strict_edge and not edge_ok:
        if probe_without_edge:
            probe_cap = float(probe_cfg.get("max_size_scale", 0.25) or 0.25)
            cold_cap = float(probe_cfg.get("cold_start_size_scale_cap", probe_cap) or probe_cap)
            edge_scale = max(0.01, min(probe_cap, cold_cap, 0.25))
            edge_mode = "learning_probe"
        elif cold_start:
            edge_scale = max(0.10, min(0.25, float(pipeline.get("edge_cold_start_size_scale", 0.25) or 0.25)))
            edge_mode = "cold_start"
        else:
            return _hold(
                symbol,
                regime,
                "strict edge evidence missing",
                candidate_setup=expert.to_dict(),
                edge_contract={"validated": False, "strict": True, "mode": "blocked"},
            )

    meta_cfg = pipeline.get("meta_quality_gate") if isinstance(pipeline.get("meta_quality_gate"), dict) else {}
    ai_part = dict(ai_part or {})
    meta_p = _f(item.get("meta_quality_probability"))
    if meta_p is None:
        meta_p = _f(ai_part.get("meta_quality_probability"))
    calibrated = bool(item.get("meta_model_calibrated") or ai_part.get("meta_model_calibrated"))
    meta = evaluate_meta_quality_gate(probability=meta_p, calibrated=calibrated, config=meta_cfg, runtime_mode=runtime_mode)
    if meta.enforced and not meta.allowed:
        return _hold(symbol, regime, "meta quality veto", setup_id=expert.setup_id, meta_quality=meta.to_dict())

    confidence = max(0.0, min(1.0, float(expert.confidence)))
    min_confidence = router_min_confidence
    if confidence < min_confidence:
        return _hold(symbol, regime, "expert confidence below regime threshold", setup_id=expert.setup_id)

    sizing_cfg = pipeline.get("adaptive_position_sizing") if isinstance(pipeline.get("adaptive_position_sizing"), dict) else {}
    risk_scale = _adaptive_size(regime=regime, confidence=confidence, meta_scale=meta.size_scale, cfg=sizing_cfg)
    risk_scale = min(risk_scale, edge_scale)
    if probe_active:
        risk_scale = min(risk_scale, float(probe_cfg.get("max_size_scale", 0.25) or 0.25))
    if risk_scale <= 0.0:
        return _hold(symbol, regime, "risk sizing resolved to zero", setup_id=expert.setup_id)

    max_lev = int(policy["max_leverage"])
    trading_cfg = cfg.get("trading") if isinstance(cfg.get("trading"), dict) else {}
    max_lev = min(max_lev, int(trading_cfg.get("max_leverage", max_lev) or max_lev))
    if probe_active:
        max_lev = min(max_lev, int(probe_cfg.get("max_leverage", 1) or 1))
    leverage = max(1, max_lev)
    if risk_scale < 1.0:
        leverage = 1

    return {
        "symbol": symbol,
        "action": "enter",
        "direction": expert.direction,
        "base_decision": expert.direction,
        "master_confidence": confidence,
        "score": confidence,
        "lev": leverage,
        "risk_scale": round(risk_scale, 4),
        "regime": regime,
        "setup_id": expert.setup_id,
        "strategy": expert.strategy,
        "pipeline": "v2",
        "reason": " | ".join(expert.reasoning),
        "meta_quality": meta.to_dict(),
        "edge_contract": {"validated": edge_ok, "strict": strict_edge, "size_scale": edge_scale, "mode": edge_mode},
        "learning_probe": {
            "active": probe_active,
            "tracked": (not probe_active) or not isinstance(probe_cfg.get("tracked_setups"), list) or expert.setup_id in {str(v) for v in probe_cfg.get("tracked_setups", [])},
            "frozen": False,
            "target_trades_per_setup": int(probe_cfg.get("target_trades_per_setup", 0) or 0) if probe_active else None,
            "max_size_scale": float(probe_cfg.get("max_size_scale", 0.25) or 0.25) if probe_active else None,
        },
        "ai_authority": {"directional": False, "role": "quality_advisory_only"},
    }
