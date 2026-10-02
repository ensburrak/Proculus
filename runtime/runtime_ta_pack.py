from __future__ import annotations

from typing import Any

from decision.stochrsi_parallel import compute_stochrsi90_snapshot


def _last(df: Any, *names: str) -> float | None:
    if df is None or getattr(df, "empty", True):
        return None
    for name in names:
        if name in getattr(df, "columns", []):
            try:
                value = float(df[name].iloc[-1])
                return value if value == value else None
            except (TypeError, ValueError, IndexError):
                continue
    return None


def _tail(df: Any, *names: str, limit: int = 8) -> list[float]:
    if df is None or getattr(df, "empty", True):
        return []
    for name in names:
        if name in getattr(df, "columns", []):
            out = []
            for raw in df[name].tail(limit).tolist():
                try:
                    value = float(raw)
                    if value == value:
                        out.append(value)
                except (TypeError, ValueError):
                    continue
            return out
    return []


def build_mtf_features(multi_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for tf_key, df in (multi_data or {}).items():
        tf = str(tf_key).lower()
        if df is None or getattr(df, "empty", True):
            continue
        out[tf] = {
            "rsi": _last(df, "rsi", f"RSI_{tf}", f"rsi_{tf}"),
            "adx": _last(df, "adx", f"ADX_{tf}", f"adx_{tf}"),
            "atr": _last(df, "atr", f"ATR_{tf}", f"atr_{tf}"),
            "atr_ratio": _last(df, "atr_ratio", f"ATR_Ratio_{tf}", f"atr_ratio_{tf}", f"NATR_{tf}"),
            "ema_fast": _last(df, "ema_fast", f"EMA_FAST_{tf}", f"EMA_20_{tf}", f"EMA20_{tf}"),
            "ema_slow": _last(df, "ema_slow", f"EMA_SLOW_{tf}", f"EMA_50_{tf}", f"EMA50_{tf}"),
            "ema200": _last(df, "ema200", "ema_200", f"EMA_200_{tf}", f"EMA200_{tf}"),
            "macd_hist": _last(df, "macd_hist", f"MACD_Hist_{tf}", f"macd_hist_{tf}"),
            "vol_z": _last(df, "vol_z", f"VOL_Z_{tf}", f"vol_z_{tf}"),
            "recent_opens": _tail(df, "open", f"open_{tf}"),
            "recent_highs": _tail(df, "high", f"high_{tf}"),
            "recent_lows": _tail(df, "low", f"low_{tf}"),
            "recent_closes": _tail(df, "close", f"close_{tf}"),
            "recent_volumes": _tail(df, "volume", f"volume_{tf}"),
        }
    return out


def build_ta_pack_from_multidata(multi_data: dict[str, Any], tf_main: str = "15m") -> dict[str, Any]:
    features = build_mtf_features(multi_data)
    row = features.get(str(tf_main).lower()) or {}
    fast, slow = row.get("ema_fast"), row.get("ema_slow")
    base = "neutral"
    if isinstance(fast, (int, float)) and isinstance(slow, (int, float)):
        if fast > slow:
            base = "long"
        elif fast < slow:
            base = "short"
    closes = row.get("recent_closes") or []
    main_df = (multi_data or {}).get(str(tf_main).lower())
    close_history = []
    if main_df is not None and not getattr(main_df, "empty", True):
        for raw in getattr(main_df, "get", lambda *_args, **_kwargs: [])("close", []):
            try:
                value = float(raw)
                if value == value:
                    close_history.append(value)
            except (TypeError, ValueError):
                continue
    stochrsi = compute_stochrsi90_snapshot(close_history)
    payload = {
        "base_decision": base,
        "price": closes[-1] if closes else None,
        "rsi": row.get("rsi"),
        "adx": row.get("adx"),
        "atr": row.get("atr"),
        "atr_ratio": row.get("atr_ratio"),
        "atr_pct": row.get("atr_ratio"),
        "vol_z": row.get("vol_z"),
        "ema_fast": fast,
        "ema_slow": slow,
        "ema": {"fast": fast, "slow": slow},
        "ema200": row.get("ema200"),
        "macd_hist": row.get("macd_hist"),
        "plus_di": _last(main_df, "plus_di", "dmi_plus", "DI_plus", "+DI"),
        "minus_di": _last(main_df, "minus_di", "dmi_minus", "DI_minus", "-DI"),
        "cmf": _last(main_df, "cmf", "CMF"),
        "volume_spike_ratio": _last(
            main_df,
            "volume_spike_ratio",
            "volume_expansion_ratio",
            "VOL_EXPANSION_RATIO",
        ),
    }
    payload.update(stochrsi)
    return payload


def safe_last(df: Any, column: str, default: Any = None) -> Any:
    value = _last(df, column)
    return default if value is None else value
