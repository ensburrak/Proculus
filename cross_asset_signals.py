# -*- coding: utf-8 -*-
from __future__ import annotations

# cross_asset_signals.py
# =====================
#
# Cross-asset signal filters for crypto trading (GRUP B):
#   B1: BTC Dominance — altcoin vs BTC season detection
#   B2: DXY (Dollar Index) correlation
#   B3: S&P 500 / VIX correlation
#   B4: Stablecoin flow proxy (exchange funding/taker metrics)
#   B5: ETH/BTC ratio — altcoin season indicator
#
# All data fetched from free APIs (Binance, CoinGecko, Fear&Greed).
# Each signal returns a dict with confidence/size multipliers.
# A combined function aggregates all signals for bot_engine integration.



from core.exceptions import BEST_EFFORT_EXCEPTIONS
import json
import logging
import time
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from atomic_io import atomic_write_json

try:
    from logger import get_logger
    log = get_logger(__name__)
except ImportError:
    log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent
_CROSS_ASSET_PATH = _ROOT / "data" / "cross_asset_signals.json"

# -------------------------------------------------------------------------
# Module-level cache
# -------------------------------------------------------------------------
_signal_cache: Optional[dict] = None
_signal_cache_ts: float = 0.0
_CACHE_TTL: float = 600.0  # 10 minutes


def _safe_get(url: str, params: dict = None, timeout: int = 8) -> Optional[dict]:
    """HTTP GET with error swallowing — returns None on failure."""
    try:
        import requests
        resp = requests.get(url, params=params or {}, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except BEST_EFFORT_EXCEPTIONS as exc:
        log.debug("[CROSS_ASSET] HTTP error %s: %s", url, exc)
        return None


# =========================================================================
# DATA FETCHERS
# =========================================================================

def _fetch_btc_dominance() -> Optional[dict]:
    """Fetch BTC dominance from CoinGecko global endpoint (free, no key)."""
    data = _safe_get("https://api.coingecko.com/api/v3/global")
    if not data:
        return None
    market_data = data.get("data", {})
    btc_d = market_data.get("market_cap_percentage", {}).get("bitcoin")
    eth_d = market_data.get("market_cap_percentage", {}).get("ethereum")
    total_mcap = market_data.get("total_market_cap", {}).get("usd")
    if btc_d is None:
        return None
    return {
        "btc_dominance_pct": round(float(btc_d), 2),
        "eth_dominance_pct": round(float(eth_d), 2) if eth_d else None,
        "total_market_cap_usd": total_mcap,
    }


def _fetch_ethbtc_klines(limit: int = 30) -> Optional[list]:
    """Fetch ETH/BTC daily klines from Binance spot (free)."""
    data = _safe_get(
        "https://api.binance.com/api/v3/klines",
        params={"symbol": "ETHBTC", "interval": "1d", "limit": limit},
    )
    if not data or not isinstance(data, list):
        return None
    # Each kline: [openTime, open, high, low, close, volume, ...]
    closes = []
    for k in data:
        try:
            closes.append(float(k[4]))
        except (IndexError, ValueError, TypeError):
            continue
    return closes if len(closes) >= 5 else None


def _fetch_fear_greed() -> Optional[dict]:
    """Fetch Fear & Greed Index (free, no key)."""
    data = _safe_get("https://api.alternative.me/fng/?limit=1")
    if not data:
        return None
    entries = data.get("data", [])
    if not entries:
        return None
    entry = entries[0]
    return {
        "value": int(entry.get("value", 50)),
        "classification": entry.get("value_classification", "Neutral"),
    }


def _fetch_btcusdt_klines(limit: int = 30) -> Optional[list]:
    """Fetch BTC/USDT daily klines from Binance spot (free)."""
    data = _safe_get(
        "https://api.binance.com/api/v3/klines",
        params={"symbol": "BTCUSDT", "interval": "1d", "limit": limit},
    )
    if not data or not isinstance(data, list):
        return None
    closes = []
    for k in data:
        try:
            closes.append(float(k[4]))
        except (IndexError, ValueError, TypeError):
            continue
    return closes if len(closes) >= 5 else None


def _ema(values: list, period: int) -> list:
    """Compute EMA over a list of floats."""
    if len(values) < period:
        return values[:]
    multiplier = 2.0 / (period + 1)
    ema_vals = [sum(values[:period]) / period]
    for v in values[period:]:
        ema_vals.append(v * multiplier + ema_vals[-1] * (1 - multiplier))
    return ema_vals


def _trend_direction(values: list, period: int = 20) -> str:
    """Determine UP/DOWN/SIDEWAYS from EMA trend of last values."""
    if len(values) < 3:
        return "SIDEWAYS"
    ema = _ema(values, min(period, len(values) - 1))
    if len(ema) < 3:
        return "SIDEWAYS"
    recent = ema[-3:]
    if recent[-1] > recent[-2] > recent[-3]:
        return "UP"
    if recent[-1] < recent[-2] < recent[-3]:
        return "DOWN"
    return "SIDEWAYS"


# =========================================================================
# B1: BTC DOMINANCE FILTER
# =========================================================================

def check_btc_dominance(
    symbol: str,
    direction: str,
    btc_dom_data: Optional[dict] = None,
) -> dict:
    """
    Adjust confidence based on BTC dominance trend.

    - BTC.D rising + altcoin long → confidence -15%
    - BTC.D falling + altcoin long → confidence +10%
    - BTC.D rising + BTC long → confidence +5%

    Returns {"confidence_multiplier": float, "reason": str, "btc_dominance_pct": float|None}
    """
    if not btc_dom_data:
        return {"confidence_multiplier": 1.0, "reason": "No BTC.D data", "btc_dominance_pct": None}

    btc_d = btc_dom_data.get("btc_dominance_pct")
    btc_d_trend = btc_dom_data.get("btc_d_trend", "SIDEWAYS")
    if btc_d is None:
        return {"confidence_multiplier": 1.0, "reason": "No BTC.D value", "btc_dominance_pct": None}

    is_btc = symbol.upper().startswith("BTC")
    is_long = direction.lower() in ("long", "buy")

    mult = 1.0
    reason_parts = [f"BTC.D={btc_d:.1f}% trend={btc_d_trend}"]

    if btc_d_trend == "UP":
        if not is_btc and is_long:
            mult = 0.85  # BTC.D rising + altcoin long → penalize
            reason_parts.append("altcoin long penalized (-15%)")
        elif is_btc and is_long:
            mult = 1.05  # BTC.D rising + BTC long → slight boost
            reason_parts.append("BTC long boosted (+5%)")
        elif not is_btc and not is_long:
            mult = 1.05  # BTC.D rising + altcoin short → slight boost
            reason_parts.append("altcoin short boosted (+5%)")
    elif btc_d_trend == "DOWN":
        if not is_btc and is_long:
            mult = 1.10  # BTC.D falling + altcoin long → boost
            reason_parts.append("altcoin long boosted (+10%)")
        elif is_btc and is_long:
            mult = 0.95  # BTC.D falling + BTC long → slight penalty
            reason_parts.append("BTC long slightly penalized (-5%)")

    # [B1e] TOTAL2/BTC ratio signal: high ratio = altcoin strength
    t2_ratio = btc_dom_data.get("total2_btc_ratio")
    if t2_ratio is not None:
        if t2_ratio > 1.2 and not is_btc and is_long:
            mult *= 1.03  # Strong altcoin market → extra altcoin boost
            reason_parts.append(f"TOTAL2/BTC={t2_ratio:.2f} altcoin strength +3%")
        elif t2_ratio < 0.7 and is_btc and is_long:
            mult *= 1.03  # Weak altcoin market → BTC dominance boost
            reason_parts.append(f"TOTAL2/BTC={t2_ratio:.2f} BTC dominance +3%")

    return {
        "confidence_multiplier": round(mult, 4),
        "reason": "; ".join(reason_parts),
        "btc_dominance_pct": btc_d,
        "total2_btc_ratio": t2_ratio,
    }


# =========================================================================
# B2: DXY (DOLLAR INDEX) CORRELATION
# =========================================================================

def check_dxy_signal(
    direction: str,
    macro_state: Optional[dict] = None,
) -> dict:
    """
    Adjust confidence based on DXY trend from macro_sensor.

    - DXY strengthening → crypto long confidence -10%
    - DXY weakening → crypto long confidence +10%

    Returns {"confidence_multiplier": float, "reason": str}
    """
    if not macro_state:
        try:
            from analysis.macro_sensor import get_macro_state
            macro_state = get_macro_state()
        except (ImportError, Exception):
            return {"confidence_multiplier": 1.0, "reason": "No macro state"}

    dxy_trend = macro_state.get("dxy_trend", "SIDEWAYS")
    dxy_chg = macro_state.get("dxy_change_pct")
    treasury_10y = macro_state.get("treasury_10y")
    is_long = direction.lower() in ("long", "buy")

    mult = 1.0
    reasons = [f"DXY trend={dxy_trend}"]

    if dxy_trend == "UP":  # Dollar strengthening
        if is_long:
            mult = 0.90  # Crypto long penalized
        else:
            mult = 1.05  # Crypto short slightly boosted
    elif dxy_trend == "DOWN":  # Dollar weakening
        if is_long:
            mult = 1.10  # Crypto long boosted
        else:
            mult = 0.95  # Crypto short slightly penalized

    # [B2a] DXY daily change intensity — stronger move = stronger adjustment
    if dxy_chg is not None:
        _abs_chg = abs(float(dxy_chg))
        if _abs_chg > 1.0:  # >1% daily move is significant
            _intensity = min(0.05, (_abs_chg - 0.5) * 0.03)
            if float(dxy_chg) > 0 and is_long:
                mult -= _intensity  # Extra penalty for strong dollar move
            elif float(dxy_chg) < 0 and is_long:
                mult += _intensity  # Extra boost for weak dollar move
            reasons.append(f"DXY chg={dxy_chg:+.2f}%")

    # [B2e] 10Y Treasury yield — high yields = risk-off for crypto
    if treasury_10y is not None:
        _yield = float(treasury_10y)
        if _yield > 4.5 and is_long:
            mult *= 0.95  # High real rates disfavor crypto long
            reasons.append(f"10Y={_yield:.2f}% high→-5%")
        elif _yield < 3.5 and is_long:
            mult *= 1.03  # Low rates favor risk-on crypto
            reasons.append(f"10Y={_yield:.2f}% low→+3%")

    return {
        "confidence_multiplier": round(mult, 4),
        "reason": "; ".join(reasons),
        "dxy_value": macro_state.get("dxy_value"),
        "treasury_10y": treasury_10y,
    }


# =========================================================================
# B3: S&P 500 / VIX CORRELATION
# =========================================================================

# [B3e] Rolling correlation cache
_BTC_DAILY_RETURNS: list = []
_SP500_DAILY_RETURNS: list = []
_LAST_CORR_DATE: str = ""


def _compute_btc_sp500_correlation(macro_state: Optional[dict] = None) -> Optional[float]:
    """
    [B3e] Compute 30-day rolling correlation between BTC and S&P 500 daily returns.

    Uses BTC klines from Binance and S&P 500 history from macro_sensor.
    Returns Pearson correlation coefficient or None if insufficient data.
    """
    global _BTC_DAILY_RETURNS, _SP500_DAILY_RETURNS, _LAST_CORR_DATE
    try:
        # Load BTC daily returns
        btc_closes = _fetch_btcusdt_klines(limit=32)
        if not btc_closes or len(btc_closes) < 5:
            return None

        btc_returns = []
        for i in range(1, len(btc_closes)):
            if btc_closes[i - 1] > 0:
                btc_returns.append((btc_closes[i] - btc_closes[i - 1]) / btc_closes[i - 1])

        # Load S&P 500 history from macro_sensor's accumulated data
        try:
            from analysis.macro_sensor import _sp500_history
            if len(_sp500_history) < 5:
                return None
            sp_returns = []
            for i in range(1, len(_sp500_history)):
                if _sp500_history[i - 1] > 0:
                    sp_returns.append((_sp500_history[i] - _sp500_history[i - 1]) / _sp500_history[i - 1])
        except (ImportError, Exception):
            return None

        # Align lengths (use minimum common length, max 30)
        n = min(len(btc_returns), len(sp_returns), 30)
        if n < 5:
            return None

        btc_r = btc_returns[-n:]
        sp_r = sp_returns[-n:]

        # Pearson correlation
        mean_b = sum(btc_r) / n
        mean_s = sum(sp_r) / n
        cov = sum((btc_r[i] - mean_b) * (sp_r[i] - mean_s) for i in range(n))
        var_b = sum((x - mean_b) ** 2 for x in btc_r)
        var_s = sum((x - mean_s) ** 2 for x in sp_r)
        denom = (var_b * var_s) ** 0.5
        if denom < 1e-12:
            return None
        return round(cov / denom, 4)
    except BEST_EFFORT_EXCEPTIONS:
        return None


def check_vix_signal(
    direction: str,
    macro_state: Optional[dict] = None,
) -> dict:
    """
    Adjust confidence based on VIX level.

    - VIX > 35 → only short or NO_TRADE (long blocked)
    - VIX > 25 → long confidence -15%
    - VIX < 15 → risk-on, long bias +10%

    Returns {"allowed": bool, "confidence_multiplier": float, "size_multiplier": float,
             "reason": str, "vix_level": float|None}
    """
    if not macro_state:
        try:
            from analysis.macro_sensor import get_macro_state
            macro_state = get_macro_state()
        except (ImportError, Exception):
            return {"allowed": True, "confidence_multiplier": 1.0,
                    "size_multiplier": 1.0, "reason": "No macro state", "vix_level": None}

    regime = macro_state.get("regime", "NEUTRAL")
    vix = macro_state.get("vix_level")  # Real VIX if available
    vix_real = macro_state.get("vix_real")  # [B3] Actual CBOE VIX from FMP
    risk_mult = macro_state.get("risk_multiplier", 1.0)
    is_long = direction.lower() in ("long", "buy")

    # [B3] Prefer real VIX thresholds when available
    _effective_vix = vix_real if vix_real is not None else vix

    conf_mult = 1.0
    size_mult = 1.0
    allowed = True

    if _effective_vix is not None and _effective_vix > 35:
        # VIX > 35 → only short or NO_TRADE
        if is_long:
            allowed = False
            conf_mult = 0.40
        size_mult = 0.40
        reason = f"VIX={_effective_vix:.1f} EXTREME: {'long blocked' if not allowed else 'short only'}, size*=0.40"
    elif _effective_vix is not None and _effective_vix > 25:
        # VIX > 25 → long confidence -15%
        if is_long:
            conf_mult = 0.85
        size_mult = 0.70
        reason = f"VIX={_effective_vix:.1f} HIGH: conf*={conf_mult:.2f}, size*=0.70"
    elif regime == "CRISIS":
        if is_long:
            allowed = False
            conf_mult = 0.50
        size_mult = 0.50
        reason = f"VIX CRISIS regime (proxy): {'long blocked' if not allowed else 'short allowed'}, size*=0.50"
    elif regime == "CAUTION":
        if is_long:
            conf_mult = 0.85
        size_mult = 0.80
        reason = f"VIX CAUTION regime: conf*={conf_mult:.2f}, size*=0.80"
    elif _effective_vix is not None and _effective_vix < 15:
        if is_long:
            conf_mult = 1.10
        reason = f"VIX={_effective_vix:.1f} LOW: risk-on, conf*={conf_mult:.2f}"
    elif regime == "OPTIMISTIC":
        if is_long:
            conf_mult = 1.10
        reason = f"VIX OPTIMISTIC regime: risk-on, conf*={conf_mult:.2f}"
    else:
        reason = f"VIX NEUTRAL regime: no adjustment"

    # [B3d] S&P 500 overnight movement adjustment
    sp500_chg = macro_state.get("sp500_change_pct")
    if sp500_chg is not None:
        _sp_chg = float(sp500_chg)
        if _sp_chg < -2.0 and is_long:
            conf_mult *= 0.92  # S&P big drop → crypto long penalty
            reason += f" | S&P500 chg={_sp_chg:+.1f}% → -8%"
        elif _sp_chg > 2.0 and is_long:
            conf_mult *= 1.05  # S&P big rally → crypto long boost
            reason += f" | S&P500 chg={_sp_chg:+.1f}% → +5%"

    # [B3e] BTC-S&P 30-day rolling correlation
    _corr = _compute_btc_sp500_correlation(macro_state)
    if _corr is not None and abs(_corr) > 0.5:
        # High correlation → S&P signal carries more weight
        if sp500_chg is not None:
            _extra_weight = (abs(_corr) - 0.5) * 0.10  # max ~5% extra
            if float(sp500_chg) < -1.0 and is_long:
                conf_mult *= max(0.90, 1.0 - _extra_weight)
                reason += f" | corr={_corr:.2f} amplified"
            elif float(sp500_chg) > 1.0 and is_long:
                conf_mult *= min(1.10, 1.0 + _extra_weight)
                reason += f" | corr={_corr:.2f} amplified"

    return {
        "allowed": allowed,
        "confidence_multiplier": round(conf_mult, 4),
        "size_multiplier": round(size_mult, 4),
        "reason": reason,
        "vix_level": _effective_vix,
        "vix_real": vix_real,
        "sp500_change_pct": sp500_chg,
        "btc_sp500_correlation": _corr,
    }


# =========================================================================
# B4: STABLECOIN FLOW PROXY
# =========================================================================

def check_stablecoin_flow(
    onchain_metrics: Optional[dict] = None,
) -> dict:
    """
    Proxy for stablecoin flow using exchange funding rates and taker imbalance.

    Positive funding + positive taker imbalance → bullish (inflow proxy)
    Negative funding + negative taker imbalance → bearish (outflow proxy)

    Returns {"sentiment_adjustment": float, "reason": str}
    """
    if not onchain_metrics:
        try:
            path = _ROOT / "data" / "onchain_metrics.json"
            if path.exists():
                onchain_metrics = json.loads(path.read_text(encoding="utf-8"))
        except BEST_EFFORT_EXCEPTIONS:
            pass

    if not onchain_metrics:
        return {"sentiment_adjustment": 0.0, "reason": "No onchain data"}

    metrics = onchain_metrics.get("metrics", onchain_metrics)
    # Aggregate across BTC and ETH
    total_funding = 0.0
    total_taker = 0.0
    total_oi_trend = 0.0
    total_ls_imb = 0.0
    total_tt_imb = 0.0
    count = 0
    for sym_key in ("BTC", "ETH"):
        m = metrics.get(sym_key, {})
        if isinstance(m, dict):
            fb = m.get("funding_bias", 0)
            ti = m.get("taker_imbalance", 0)
            oi = m.get("oi_trend", 0)
            ls = m.get("long_short_imbalance", 0)
            tt = m.get("top_trader_imbalance", 0)
            if isinstance(fb, (int, float)):
                total_funding += fb
            if isinstance(ti, (int, float)):
                total_taker += ti
            if isinstance(oi, (int, float)):
                total_oi_trend += oi
            if isinstance(ls, (int, float)):
                total_ls_imb += ls
            if isinstance(tt, (int, float)):
                total_tt_imb += tt
            count += 1

    if count == 0:
        return {"sentiment_adjustment": 0.0, "reason": "No symbol metrics"}

    avg_funding = total_funding / count
    avg_taker = total_taker / count
    avg_oi = total_oi_trend / count
    avg_ls = total_ls_imb / count
    avg_tt = total_tt_imb / count

    # [B4] Enhanced combined signal with 5 components:
    #   funding_bias (20%) + taker_imbalance (20%) + oi_trend (20%)
    #   + long_short_imbalance (20%) + top_trader_imbalance (20%)
    combined = (avg_funding * 0.20 + avg_taker * 0.20 + avg_oi * 0.20
                + avg_ls * 0.20 + avg_tt * 0.20)
    combined = max(-1.0, min(1.0, combined))

    # Scale to sentiment adjustment: [-0.10, +0.10]
    adjustment = combined * 0.10

    reason = (f"funding={avg_funding:.3f}, taker={avg_taker:.3f}, oi={avg_oi:.3f}, "
              f"ls={avg_ls:.3f}, top_trader={avg_tt:.3f}, adj={adjustment:+.4f}")
    return {
        "sentiment_adjustment": round(adjustment, 4),
        "long_short_ratio_avg": round(avg_ls + 1.0, 4),
        "top_trader_ratio_avg": round(avg_tt + 1.0, 4),
        "oi_trend_avg": round(avg_oi, 4),
        "reason": reason,
    }


# =========================================================================
# B5: ETH/BTC RATIO SIGNAL
# =========================================================================

def check_ethbtc_signal(
    symbol: str,
    direction: str,
    ethbtc_data: Optional[dict] = None,
) -> dict:
    """
    Adjust confidence based on ETH/BTC ratio trend.

    - ETH/BTC rising → altcoin season, altcoin long boosted
    - ETH/BTC falling → BTC season, BTC long boosted, altcoin long penalized
    - ETH/BTC below 30d MA → altcoin short boosted

    Returns {"confidence_multiplier": float, "reason": str}
    """
    if not ethbtc_data:
        return {"confidence_multiplier": 1.0, "reason": "No ETH/BTC data"}

    trend = ethbtc_data.get("ethbtc_trend", "SIDEWAYS")
    below_ma = ethbtc_data.get("below_30d_ma", False)
    is_btc = symbol.upper().startswith("BTC")
    is_eth = symbol.upper().startswith("ETH")
    is_long = direction.lower() in ("long", "buy")

    mult = 1.0
    reasons = [f"ETH/BTC trend={trend}"]

    if trend == "UP":
        # Altcoin season
        if not is_btc and is_long:
            mult = 1.08  # Altcoin long boosted
            reasons.append("altcoin season: long +8%")
        elif is_btc and is_long:
            mult = 0.95  # BTC long penalized
            reasons.append("altcoin season: BTC long -5%")
    elif trend == "DOWN":
        # BTC season
        if is_btc and is_long:
            mult = 1.08  # BTC long boosted
            reasons.append("BTC season: BTC long +8%")
        elif not is_btc and is_long:
            mult = 0.92  # Altcoin long penalized
            reasons.append("BTC season: altcoin long -8%")

    # Below 30d MA: altcoin shorts boosted
    if below_ma and not is_btc and not is_long:
        mult *= 1.05
        reasons.append("below 30d MA: altcoin short +5%")

    return {
        "confidence_multiplier": round(mult, 4),
        "reason": "; ".join(reasons),
    }


# =========================================================================
# DATA UPDATE — fetch all cross-asset data and write to disk
# =========================================================================


def update_cross_asset_data() -> dict:
    """
    Fetch all cross-asset data from free APIs and write to data/cross_asset_signals.json.
    Called periodically by auto_updater.

    Returns summary dict.
    """
    result: dict = {}
    now_iso = datetime.now(timezone.utc).isoformat()

    # B1: BTC Dominance
    btc_dom = _fetch_btc_dominance()
    if btc_dom:
        result["btc_dominance"] = btc_dom
    else:
        result["btc_dominance"] = None

    # B1+: BTC dominance trend — accumulate daily snapshots for real 20-day EMA
    btc_d_trend = "SIDEWAYS"
    _btc_d_history_path = _ROOT / "data" / "btc_dominance_history.json"
    _btc_d_history: list = []
    try:
        if _btc_d_history_path.exists():
            _btc_d_history = json.loads(_btc_d_history_path.read_text(encoding="utf-8"))
            if not isinstance(_btc_d_history, list):
                _btc_d_history = []
    except BEST_EFFORT_EXCEPTIONS:
        _btc_d_history = []

    if btc_dom and btc_dom.get("btc_dominance_pct") is not None:
        _today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        # Append today's BTC.D if not already recorded today
        if not _btc_d_history or _btc_d_history[-1].get("date") != _today:
            _btc_d_history.append({
                "date": _today,
                "btc_d": btc_dom["btc_dominance_pct"],
            })
        else:
            _btc_d_history[-1]["btc_d"] = btc_dom["btc_dominance_pct"]
        # Keep last 60 days
        _btc_d_history = _btc_d_history[-60:]
        try:
            atomic_write_json(_btc_d_history_path, _btc_d_history)
        except BEST_EFFORT_EXCEPTIONS:
            pass

        # Compute real 20-day EMA trend from historical BTC.D values
        _btc_d_values = [float(h["btc_d"]) for h in _btc_d_history if h.get("btc_d") is not None]
        if len(_btc_d_values) >= 5:
            btc_d_trend = _trend_direction(_btc_d_values, period=20)
        else:
            # Fallback: use BTC price trend as proxy when insufficient history
            btc_closes = _fetch_btcusdt_klines(limit=30)
            if btc_closes:
                btc_price_trend = _trend_direction(btc_closes, period=20)
                btc_d_val = btc_dom.get("btc_dominance_pct", 50)
                if btc_d_val > 52 and btc_price_trend == "UP":
                    btc_d_trend = "UP"
                elif btc_d_val < 48 or btc_price_trend == "DOWN":
                    btc_d_trend = "DOWN"

    # [B1e] TOTAL2/BTC ratio — altcoin market cap strength indicator
    total2_btc_ratio = None
    if btc_dom:
        _total_mcap = btc_dom.get("total_market_cap_usd")
        _btc_d_pct = btc_dom.get("btc_dominance_pct")
        if _total_mcap and _btc_d_pct and _btc_d_pct > 0:
            _btc_mcap = _total_mcap * (_btc_d_pct / 100.0)
            _altcoin_mcap = _total_mcap - _btc_mcap
            if _btc_mcap > 0:
                total2_btc_ratio = round(_altcoin_mcap / _btc_mcap, 4)
        btc_dom["total2_btc_ratio"] = total2_btc_ratio

    result["btc_d_trend"] = btc_d_trend
    result["total2_btc_ratio"] = total2_btc_ratio
    if btc_dom:
        btc_dom["btc_d_trend"] = btc_d_trend

    # B3: Fear & Greed (proxy for VIX sentiment)
    fng = _fetch_fear_greed()
    result["fear_greed"] = fng

    # B5: ETH/BTC ratio
    ethbtc_closes = _fetch_ethbtc_klines(limit=30)
    ethbtc_data: Optional[dict] = None
    if ethbtc_closes:
        ethbtc_trend = _trend_direction(ethbtc_closes, period=20)
        # 30-day simple MA
        ma30 = sum(ethbtc_closes) / len(ethbtc_closes)
        current = ethbtc_closes[-1]
        ethbtc_data = {
            "ethbtc_trend": ethbtc_trend,
            "current": round(current, 6),
            "ma30": round(ma30, 6),
            "below_30d_ma": current < ma30,
        }
    result["ethbtc"] = ethbtc_data

    # B4: Stablecoin flow (read existing onchain metrics)
    flow = check_stablecoin_flow()
    result["stablecoin_flow"] = flow

    # Write to disk
    payload = {
        "updated_at": now_iso,
        "data": result,
    }
    atomic_write_json(_CROSS_ASSET_PATH, payload)

    global _signal_cache, _signal_cache_ts
    _signal_cache = payload
    _signal_cache_ts = time.monotonic()

    return result


def _load_signal_cache() -> dict:
    """Load cross-asset signals with TTL cache."""
    global _signal_cache, _signal_cache_ts
    now = time.monotonic()
    if _signal_cache is not None and (now - _signal_cache_ts) < _CACHE_TTL:
        return _signal_cache
    try:
        raw = json.loads(_CROSS_ASSET_PATH.read_text(encoding="utf-8"))
        _signal_cache = raw
        _signal_cache_ts = now
        return raw
    except BEST_EFFORT_EXCEPTIONS:
        return {}


# =========================================================================
# COMBINED SIGNAL — single entry point for bot_engine
# =========================================================================

def get_cross_asset_adjustment(
    symbol: str,
    direction: str,
    macro_state: Optional[dict] = None,
) -> dict:
    """
    Compute combined cross-asset confidence and size multipliers.

    Called from bot_engine.py after timing filters, before order sizing.

    Returns {
        "confidence_multiplier": float,   # product of B1+B2+B3+B5
        "size_multiplier": float,          # from B3 (VIX)
        "sentiment_adjustment": float,     # from B4 (stablecoin proxy)
        "allowed": bool,                   # from B3 (VIX crisis blocks longs)
        "reason": str,
        "details": dict,                   # individual signal results
    }
    """
    cached = _load_signal_cache()
    data = cached.get("data", {})

    details: dict = {}
    conf_mult = 1.0
    size_mult = 1.0
    allowed = True
    reasons: list[str] = []

    # B1: BTC Dominance
    btc_dom = data.get("btc_dominance")
    if btc_dom and isinstance(btc_dom, dict):
        btc_dom["btc_d_trend"] = data.get("btc_d_trend", "SIDEWAYS")
    b1 = check_btc_dominance(symbol, direction, btc_dom)
    details["btc_dominance"] = b1
    conf_mult *= b1.get("confidence_multiplier", 1.0)
    if b1.get("confidence_multiplier", 1.0) != 1.0:
        reasons.append(f"B1:{b1['reason']}")

    # B2: DXY
    b2 = check_dxy_signal(direction, macro_state)
    details["dxy"] = b2
    conf_mult *= b2.get("confidence_multiplier", 1.0)
    if b2.get("confidence_multiplier", 1.0) != 1.0:
        reasons.append(f"B2:{b2['reason']}")

    # B3: VIX
    b3 = check_vix_signal(direction, macro_state)
    details["vix"] = b3
    conf_mult *= b3.get("confidence_multiplier", 1.0)
    size_mult *= b3.get("size_multiplier", 1.0)
    if not b3.get("allowed", True):
        allowed = False
        reasons.append(f"B3:VIX BLOCKED")
    elif b3.get("confidence_multiplier", 1.0) != 1.0:
        reasons.append(f"B3:{b3['reason']}")

    # B4: Stablecoin flow
    b4 = data.get("stablecoin_flow") or check_stablecoin_flow()
    details["stablecoin_flow"] = b4
    sent_adj = b4.get("sentiment_adjustment", 0.0) if b4 else 0.0

    # B5: ETH/BTC ratio
    ethbtc = data.get("ethbtc")
    b5 = check_ethbtc_signal(symbol, direction, ethbtc)
    details["ethbtc"] = b5
    conf_mult *= b5.get("confidence_multiplier", 1.0)
    if b5.get("confidence_multiplier", 1.0) != 1.0:
        reasons.append(f"B5:{b5['reason']}")

    return {
        "confidence_multiplier": round(conf_mult, 4),
        "size_multiplier": round(size_mult, 4),
        "sentiment_adjustment": round(sent_adj, 4),
        "allowed": allowed,
        "reason": " | ".join(reasons) if reasons else "No cross-asset adjustment",
        "details": details,
    }

