from __future__ import annotations

from typing import Any


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


def _stoch_rsi_tail(df: Any, rsi_period: int, stoch_period: int) -> dict[str, float | None]:
    """Compute closed-candle StochRSI K/D without a TA-Lib dependency."""
    empty={"k":None,"d":None,"prev_k":None,"prev_d":None}
    if df is None or getattr(df,"empty",True) or "close" not in getattr(df,"columns",[]):
        return empty
    try:
        close=df["close"].astype(float)
        need=rsi_period+stoch_period+8
        if len(close)<need:
            return empty
        delta=close.diff()
        gain=delta.clip(lower=0.0)
        loss=-delta.clip(upper=0.0)
        avg_gain=gain.ewm(alpha=1.0/rsi_period,adjust=False,min_periods=rsi_period).mean()
        avg_loss=loss.ewm(alpha=1.0/rsi_period,adjust=False,min_periods=rsi_period).mean()
        rs=avg_gain/avg_loss.replace(0.0,float("nan"))
        rsi=100.0-(100.0/(1.0+rs))
        lo=rsi.rolling(stoch_period,min_periods=stoch_period).min()
        hi=rsi.rolling(stoch_period,min_periods=stoch_period).max()
        raw=100.0*(rsi-lo)/(hi-lo).replace(0.0,float("nan"))
        k=raw.rolling(3,min_periods=3).mean()
        d=k.rolling(3,min_periods=3).mean()
        valid_k=k.dropna(); valid_d=d.dropna()
        if len(valid_k)<2 or len(valid_d)<2:
            return empty
        return {
            "k":float(valid_k.iloc[-1]),
            "d":float(valid_d.iloc[-1]),
            "prev_k":float(valid_k.iloc[-2]),
            "prev_d":float(valid_d.iloc[-2]),
        }
    except Exception:
        return empty


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

    main_df=None
    for key,value in (multi_data or {}).items():
        if str(key).lower()==str(tf_main).lower():
            main_df=value
            break
    s90=_stoch_rsi_tail(main_df,90,90)
    s80=_stoch_rsi_tail(main_df,80,80)

    return {
        "base_decision": base,
        "price": closes[-1] if closes else None,
        "rsi": row.get("rsi"),
        "adx": row.get("adx"),
        "atr": row.get("atr"),
        "atr_ratio": row.get("atr_ratio"),
        "vol_z": row.get("vol_z"),
        "ema": {"fast": fast, "slow": slow},
        "ema200": row.get("ema200"),
        "macd_hist": row.get("macd_hist"),
        "stoch_rsi_90_k":s90["k"],
        "stoch_rsi_90_d":s90["d"],
        "stoch_rsi_90_prev_k":s90["prev_k"],
        "stoch_rsi_90_prev_d":s90["prev_d"],
        "stoch_rsi_80_k":s80["k"],
        "stoch_rsi_80_d":s80["d"],
        "stoch_rsi_80_prev_k":s80["prev_k"],
        "stoch_rsi_80_prev_d":s80["prev_d"],
        "stoch_rsi_warmup_ok": all(v is not None for v in s90.values()),
    }


def safe_last(df: Any, column: str, default: Any = None) -> Any:
    value = _last(df, column)
    return default if value is None else value


__all__=["build_mtf_features","build_ta_pack_from_multidata","safe_last"]
