from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

CONFIG_FILE = Path(__file__).resolve().parents[1] / "config.json"


def _load_config() -> dict[str, Any]:
    try:
        payload = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _f(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed else None


def _hold(symbol: str, regime: str, reason: str, **extra: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "symbol": symbol,
        "action": "hold",
        "direction": "neutral",
        "base_decision": "neutral",
        "confidence": 0.0,
        "risk_scale": 0.0,
        "max_leverage": 0.0,
        "regime": regime,
        "pipeline": "stochrsi_parallel",
        "authority": "independent",
        "model_confirmation_required": False,
        "reason": reason,
    }
    result.update(extra)
    return result


def _candidate_side(
    regime: str,
    *,
    k: float,
    d: float,
    prev_k: float | None,
    prev_d: float | None,
) -> tuple[str | None, str | None]:
    if prev_k is None or prev_d is None:
        return None, None

    long_threshold = 30.0 if regime == "bull" else 10.0
    short_threshold = 70.0 if regime == "bear" else 90.0
    long_cross = prev_k < prev_d and k >= d and k <= long_threshold
    short_cross = prev_k > prev_d and k <= d and k >= short_threshold

    if regime == "bull":
        return ("long", "stochrsi90_cross_up") if long_cross else (None, None)
    if regime == "bear":
        return ("short", "stochrsi90_cross_down") if short_cross else (None, None)
    if regime == "range":
        if long_cross:
            return "long", "stochrsi90_cross_up"
        if short_cross:
            return "short", "stochrsi90_cross_down"
    return None, None


def compute_stochrsi90_snapshot(close_values: Any) -> dict[str, Any]:
    values: list[float] = []
    try:
        iterable = list(close_values)
    except TypeError:
        iterable = []
    for raw in iterable:
        parsed = _f(raw)
        if parsed is not None:
            values.append(parsed)

    # Wilder RSI(90) + StochRSI(90) + K(3) + D(3) needs roughly 184
    # closed observations before two fully-smoothed K/D points exist.
    if len(values) < 185:
        return {
            "stoch_rsi_warmup_ok": False,
            "stoch_rsi_reason": "stochrsi90_insufficient_closed_candles",
        }

    close = pd.Series(values, dtype="float64")
    delta = close.diff()
    gain = delta.clip(lower=0.0).fillna(0.0)
    loss = (-delta.clip(upper=0.0)).fillna(0.0)
    avg_gain = gain.ewm(alpha=1.0 / 90.0, adjust=False, min_periods=90).mean()
    avg_loss = loss.ewm(alpha=1.0 / 90.0, adjust=False, min_periods=90).mean()
    rs = avg_gain / avg_loss.replace(0.0, float("nan"))
    rsi = 100.0 - (100.0 / (1.0 + rs))

    low = rsi.rolling(90, min_periods=90).min()
    high = rsi.rolling(90, min_periods=90).max()
    width = (high - low).replace(0.0, float("nan"))
    raw = ((rsi - low) / width) * 100.0
    k = raw.rolling(3, min_periods=3).mean().clip(0.0, 100.0)
    d = k.rolling(3, min_periods=3).mean().clip(0.0, 100.0)

    points = pd.DataFrame({"k": k, "d": d}).dropna()
    if len(points) < 2:
        return {
            "stoch_rsi_warmup_ok": False,
            "stoch_rsi_reason": "stochrsi90_not_fully_smoothed",
        }

    previous = points.iloc[-2]
    current = points.iloc[-1]
    return {
        "stoch_rsi_warmup_ok": True,
        "stoch_rsi_reason": "ok",
        "stoch_rsi_90_prev_k": float(previous["k"]),
        "stoch_rsi_90_prev_d": float(previous["d"]),
        "stoch_rsi_90_k": float(current["k"]),
        "stoch_rsi_90_d": float(current["d"]),
    }


def evaluate_stochrsi90(
    *,
    item: dict[str, Any] | None,
    ta: dict[str, Any] | None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    item_dict = item if isinstance(item, dict) else {}
    ta_dict = ta if isinstance(ta, dict) else {}
    cfg = config if isinstance(config, dict) else _load_config()
    stoch_cfg = cfg.get("stochrsi_parallel") if isinstance(cfg.get("stochrsi_parallel"), dict) else {}

    symbol = str(item_dict.get("symbol") or "")
    regime = str(item_dict.get("regime") or ta_dict.get("regime") or "unknown").lower()
    runtime_mode = str(item_dict.get("runtime_mode") or "paper").lower()

    if stoch_cfg and stoch_cfg.get("enabled") is False:
        return _hold(symbol, regime, "stochrsi_parallel_disabled")

    never_trade = set(
        str(value).lower()
        for value in (
            (cfg.get("pipeline_v2") or {})
            .get("learning_probe_mode", {})
            .get("never_trade_regimes", ["shock", "conflict", "unknown"])
        )
    )
    if regime in never_trade:
        return _hold(symbol, regime, f"regime_block:{regime}")
    if regime not in {"bull", "bear", "range"}:
        return _hold(symbol, regime, f"no_stochrsi90_profile:{regime}")

    if ta_dict.get("stoch_rsi_warmup_ok") is not True:
        return _hold(
            symbol,
            regime,
            str(ta_dict.get("stoch_rsi_reason") or "stochrsi90_warmup_not_ok"),
        )

    k = _f(ta_dict.get("stoch_rsi_90_k"))
    d = _f(ta_dict.get("stoch_rsi_90_d"))
    prev_k = _f(ta_dict.get("stoch_rsi_90_prev_k"))
    prev_d = _f(ta_dict.get("stoch_rsi_90_prev_d"))
    if k is None or d is None:
        return _hold(symbol, regime, "missing_stochrsi90")

    volume_ratio = _f(
        ta_dict.get("volume_spike_ratio")
        if ta_dict.get("volume_spike_ratio") is not None
        else ta_dict.get("volume_expansion_ratio")
    )
    if volume_ratio is not None and volume_ratio >= 3.0:
        return _hold(symbol, regime, "volume_spike_veto")

    side, signal_reason = _candidate_side(
        regime,
        k=k,
        d=d,
        prev_k=prev_k,
        prev_d=prev_d,
    )
    if side is None:
        return _hold(
            symbol,
            regime,
            "no_stochrsi90_trade_setup",
            audit={"k": k, "d": d, "prev_k": prev_k, "prev_d": prev_d},
        )

    score = 0.40
    reasons = [str(signal_reason)]

    ema_row = ta_dict.get("ema") if isinstance(ta_dict.get("ema"), dict) else {}
    ema_fast = _f(
        ta_dict.get("ema_fast")
        if ta_dict.get("ema_fast") is not None
        else ema_row.get("fast")
    )
    ema_slow = _f(
        ta_dict.get("ema_slow")
        if ta_dict.get("ema_slow") is not None
        else ema_row.get("slow")
    )
    if ema_fast is not None and ema_slow is not None:
        aligned = (side == "long" and ema_fast >= ema_slow) or (
            side == "short" and ema_fast <= ema_slow
        )
        range_flat = (
            regime == "range"
            and abs(ema_fast - ema_slow) / max(abs(ema_slow), 1e-9) <= 0.01
        )
        if aligned or range_flat:
            score += 0.20
            reasons.append("ema_filter_ok")

    adx = _f(ta_dict.get("adx"))
    plus_di = _f(ta_dict.get("plus_di"))
    minus_di = _f(ta_dict.get("minus_di"))
    if adx is not None and plus_di is not None and minus_di is not None:
        if regime == "range" and adx <= 22.0:
            score += 0.20
            reasons.append("range_adx_ok")
        elif side == "long" and adx >= 18.0 and plus_di > minus_di:
            score += 0.20
            reasons.append("adx_dmi_long_ok")
        elif side == "short" and adx >= 18.0 and minus_di > plus_di:
            score += 0.20
            reasons.append("adx_dmi_short_ok")

    cmf = _f(ta_dict.get("cmf"))
    if cmf is not None and (
        (side == "long" and cmf >= 0.05)
        or (side == "short" and cmf <= -0.05)
    ):
        score += 0.15
        reasons.append("cmf_confirmation_ok")

    atr_pct = _f(
        ta_dict.get("atr14_pct")
        if ta_dict.get("atr14_pct") is not None
        else ta_dict.get("atr_pct")
    )
    if atr_pct is not None and 0.001 <= atr_pct <= 0.04:
        score += 0.05
        reasons.append("atr_tradable")

    score = max(0.0, min(1.0, score))
    if score < 0.72:
        return _hold(
            symbol,
            regime,
            "stochrsi90_quality_below_trade_threshold",
            confidence=round(score, 4),
            candidate_direction=side,
            audit={"k": k, "d": d, "prev_k": prev_k, "prev_d": prev_d},
        )

    paper_allowed = bool(stoch_cfg.get("paper_orders_enabled", True))
    demo_allowed = bool(stoch_cfg.get("demo_orders_enabled", False))
    live_allowed = bool(stoch_cfg.get("live_orders_enabled", False))
    execution_allowed = (
        paper_allowed
        if runtime_mode in {"paper", "sim"}
        else demo_allowed
        if runtime_mode in {"demo", "sandbox", "testnet"}
        else live_allowed
    )

    risk_scale = 0.25 if regime in {"range"} else 0.25
    setup_id = f"stochrsi90.{regime}.{side}.15m.v1"
    return {
        "symbol": symbol,
        "action": "enter",
        "direction": side,
        "base_decision": side,
        "confidence": round(score, 4),
        "master_confidence": round(score, 4),
        "risk_scale": risk_scale,
        "max_leverage": 1.0,
        "lev": 1,
        "regime": regime,
        "setup_id": setup_id,
        "strategy": "stochrsi90_independent",
        "pipeline": "stochrsi_parallel",
        "authority": "independent",
        "model_confirmation_required": False,
        "execution_allowed": execution_allowed,
        "reason": " | ".join(reasons),
        "audit": {
            "k": k,
            "d": d,
            "prev_k": prev_k,
            "prev_d": prev_d,
            "volume_spike_ratio": volume_ratio,
        },
    }


__all__ = ["compute_stochrsi90_snapshot", "evaluate_stochrsi90"]
