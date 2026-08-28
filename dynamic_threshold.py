# -*- coding: utf-8 -*-
"""
dynamic_threshold.py
====================

[2026-01-15 PROFESSIONAL UPGRADE]

Dinamik entry esigi accountlama modulu. Trade entry esigini (master confidence 
threshold) piyasa kosullarina gore settinglar.

Kullanilan Veri Kaynaklari (Sadece guvenilir):
- OKX/Binance API: Fiyat, hacim, volatilite
- AI Modelleri: ChatGPT/DeepSeek konsensusu
- Rejim Detektoru: Mevcut piyasa rejimi

Kullanilmayan:
- Harici API'ler (unreliable)
- Sosyal medya verileri

Threshold Mantigi:
- BULL trend + AI konsensusu yuksek → dusuk esik (0.55)
- BEAR trend + AI konsensusu dusuk → yuksek esik (0.75)
- SIDEWAYS + belirsizlik → cok yuksek esik (0.80)
"""

from __future__ import annotations


from core.exceptions import BEST_EFFORT_EXCEPTIONS
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any

from atomic_io import safe_read_json
from core.config_loader import load_config as load_app_config
from decision.parameter_registry import get_threshold_registry_version, get_threshold_set

# Online learning state file for recent_win_rate feedback
_ONLINE_LEARNING_STATE = Path(__file__).resolve().parent / "metrics" / "online_learning_state.json"

logger = logging.getLogger(__name__)

# Default thresholds
# [2026-01-20] Genis ve gercekci aralik - piyasa kosullarina gore adapte olur
_DYNAMIC_THRESHOLD_BOUNDS = get_threshold_set("dynamic_threshold_bounds")
THRESHOLD_VERSION = get_threshold_registry_version()
DEFAULT_THRESHOLD = float(_DYNAMIC_THRESHOLD_BOUNDS.get("default", 0.63))
MIN_THRESHOLD = float(_DYNAMIC_THRESHOLD_BOUNDS.get("min", 0.50))   # Cok guclu sinyal + uygun rejim
MAX_THRESHOLD = float(_DYNAMIC_THRESHOLD_BOUNDS.get("max", 0.85))   # Zayif sinyal veya riskli piyasa


def get_threshold_metadata() -> dict[str, Any]:
    """Return the active threshold registry metadata for decision audit logs."""
    return {
        "threshold_version": THRESHOLD_VERSION,
        "default_threshold": DEFAULT_THRESHOLD,
        "min_threshold": MIN_THRESHOLD,
        "max_threshold": MAX_THRESHOLD,
        "source": "parameter_registry.dynamic_threshold_bounds",
    }


def _configured_min_master_score(default: float) -> float:
    try:
        cfg = load_app_config()
        thresholds = cfg.get("thresholds", {})
        if isinstance(thresholds, dict):
            candidate = thresholds.get("min_master_score", thresholds.get("min_confidence"))
            if candidate is not None:
                return float(candidate)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return float(default)


def _read_recent_win_rate() -> Optional[float]:
    """Read recent_win_rate from online learning state file (disk-based, no import dependency)."""
    try:
        data = safe_read_json(_ONLINE_LEARNING_STATE, default={})
        if isinstance(data, dict):
            wr = data.get("recent_win_rate")
            if wr is not None:
                return float(wr)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    return None


def get_ai_consensus_score(ai_scores: Optional[Dict[str, float]] = None) -> float:
    """
    AI modellerinin konsensus skorunu accountla.

    Tum AI modellerini kullanir: ChatGPT, DeepSeek, Transformer, RL
    Notr skorlar (0.495-0.505) haric tutulur.
    """
    if not ai_scores:
        return 0.5

    active_scores = []

    # Tum modelleri check et
    for model_name in ["chatgpt", "deepseek", "transformer", "rl"]:
        score = ai_scores.get(model_name)
        if score is not None and isinstance(score, (int, float)):
            score = float(score)
            # Sadece tam notr degilse dahil et (dar aralik)
            if score < 0.495 or score > 0.505:
                active_scores.append(score)

    if not active_scores:
        return 0.5

    return sum(active_scores) / len(active_scores)


def get_volatility_factor(atr: Optional[float] = None, price: Optional[float] = None) -> float:
    """
    Volatilite faktoru accountla.
    
    Yuksek volatilite = yuksek risk = yuksek esik
    Dusuk volatilite = dusuk risk = dusuk esik
    """
    if atr is None or price is None or price <= 0:
        return 1.0  # Notr
    
    try:
        atr_pct = (atr / price) * 100
        
        if atr_pct < 1.0:
            # Dusuk volatilite - esigi dusur
            return 0.90
        elif atr_pct < 2.0:
            # Normal volatilite
            return 1.0
        elif atr_pct < 4.0:
            # Orta-yuksek volatilite
            return 1.10
        else:
            # Yuksek volatilite - esigi yukselt
            return 1.20
    except BEST_EFFORT_EXCEPTIONS:
        return 1.0


def calculate_dynamic_threshold(
    regime: Optional[str] = None,
    regime_confidence: Optional[float] = None,
    ai_scores: Optional[Dict[str, float]] = None,
    atr: Optional[float] = None,
    current_price: Optional[float] = None,
    recent_win_rate: Optional[float] = None,
    direction: Optional[str] = None,
) -> float:
    """
    [PROFESSIONAL FIX #21] Dinamik entry esigi accountla.
    
    Bu fonksiyon, sabit bir esik yerine piyasa kosullarina gore
    dinamik bir esik return.
    
    Args:
        regime: "BULL", "BEAR", "SIDEWAYS", etc.
        regime_confidence: 0-1 arasi rejim guveni
        ai_scores: AI model skorlari {"chatgpt": float, "deepseek": float, ...}
        atr: Average True Range
        current_price: Mevcut fiyat
        recent_win_rate: Son 20 trade'in win rate'i (0-1)
        direction: "long" veya "short"
    
    Returns:
        float: 0.50-0.85 arasi dinamik esik
    """
    threshold = DEFAULT_THRESHOLD
    adjustments = []
    
    # =========================================================================
    # 1. Rejim Bazli Ayarlama (daha yumusak valueler)
    # =========================================================================
    if regime:
        regime = regime.upper()
        direction = (direction or "long").lower()

        if regime in ("BULL", "STRONG_BULL"):
            if direction == "long":
                # BULL + Long = uyum, esik dusur
                threshold -= 0.05
                adjustments.append(f"BULL+Long: -0.05")
            else:
                # BULL + Short = karsit, esik yukselt
                threshold += 0.06
                adjustments.append(f"BULL+Short: +0.06")

        elif regime in ("BEAR", "STRONG_BEAR"):
            if direction == "short":
                # BEAR + Short = uyum, esik dusur
                threshold -= 0.05
                adjustments.append(f"BEAR+Short: -0.05")
            else:
                # BEAR + Long = karsit, esik yukselt
                threshold += 0.06
                adjustments.append(f"BEAR+Long: +0.06")

        elif regime == "SIDEWAYS":
            # Yatay piyasa - dikkatli ol
            threshold += 0.08
            adjustments.append(f"SIDEWAYS: +0.08")

    # Rejim guveni yuksekse daha cesur ol
    if regime_confidence is not None and regime_confidence > 0.75:
        threshold -= 0.02
        adjustments.append(f"High regime conf: -0.02")
    
    # =========================================================================
    # 2. AI Konsensusu Bazli Ayarlama (daha yumusak)
    # =========================================================================
    ai_consensus = get_ai_consensus_score(ai_scores)

    if ai_consensus >= 0.70:
        # Guclu AI konsensusu
        threshold -= 0.03
        adjustments.append(f"Strong AI consensus ({ai_consensus:.2f}): -0.03")
    elif ai_consensus >= 0.60:
        # Iyi AI konsensusu
        threshold -= 0.01
    elif ai_consensus < 0.45:
        # Zayif AI konsensusu
        threshold += 0.05
        adjustments.append(f"Weak AI consensus ({ai_consensus:.2f}): +0.05")
    
    # =========================================================================
    # 3. Volatilite Bazli Ayarlama
    # =========================================================================
    vol_factor = get_volatility_factor(atr, current_price)
    if vol_factor != 1.0:
        old_threshold = threshold
        threshold *= vol_factor
        adjustments.append(f"Volatility factor: ×{vol_factor:.2f}")
    
    # =========================================================================
    # 4. Son Performans Bazli Ayarlama (daha yumusak)
    # =========================================================================
    # learning_probe aktif iken win-rate penalty bypass edilir; düşük win
    # rate probe'un doğal sonucudur, eşiği yükseltmek probe'u kilitler.
    try:
        from decision.learning_probe import should_suppress_dynamic_threshold as _suppress_dyn_inner

        _dyn_suppress = _suppress_dyn_inner()
    except ImportError:
        _dyn_suppress = False
    if recent_win_rate is not None and not _dyn_suppress:
        if recent_win_rate >= 0.60:
            # Performing well - can be slightly aggressive
            threshold -= 0.02
            adjustments.append(f"Good win rate ({recent_win_rate:.0%}): -0.02")
        elif recent_win_rate < 0.40:
            # Performing poorly - be more conservative
            threshold += 0.05
            adjustments.append(f"Poor win rate ({recent_win_rate:.0%}): +0.05")
    elif _dyn_suppress:
        adjustments.append("learning_probe: win-rate penalty suppressed")
    
    # =========================================================================
    # 5. Final Clamp ve Log
    # =========================================================================
    threshold = max(MIN_THRESHOLD, min(MAX_THRESHOLD, threshold))
    
    if adjustments:
        logger.info(f"[DYN_THRESHOLD] Base={DEFAULT_THRESHOLD:.2f} | Adjustments: {', '.join(adjustments)} | Final={threshold:.2f}")
    
    return round(threshold, 3)


# =============================================================================
# Convenience function
# =============================================================================

def get_threshold(
    symbol: str,
    default: float = DEFAULT_THRESHOLD,
    **kwargs
) -> float:
    """
    Legacy-compatible threshold getter.

    Eger ek parametreler verilmemisse config-based base threshold'u
    recent_win_rate ile ayarlar. Parametreler verilmisse dinamik
    accountlama yapar.
    """
    if not kwargs:
        base = _configured_min_master_score(default)
        win_rate = _read_recent_win_rate()
        # Suppress the win-rate inflation branch in learning_probe mode:
        # during evidence-gathering a <40% win rate is expected and
        # raising the threshold here defeats the whole probe.
        try:
            from decision.learning_probe import should_suppress_dynamic_threshold as _suppress_dyn

            suppress = _suppress_dyn()
        except ImportError:
            suppress = False
        if win_rate is not None and not suppress:
            if win_rate >= 0.60:
                base -= 0.02
                logger.info("[DYN_THRESHOLD] Win rate %.0f%% >= 60%% → threshold -0.02", win_rate * 100)
            elif win_rate < 0.40:
                base += 0.05
                logger.info("[DYN_THRESHOLD] Win rate %.0f%% < 40%% → threshold +0.05", win_rate * 100)
        elif suppress:
            logger.debug(
                "[DYN_THRESHOLD] learning_probe active → win-rate inflation suppressed (wr=%s)",
                f"{win_rate:.2f}" if win_rate is not None else "n/a",
            )
        return max(MIN_THRESHOLD, min(MAX_THRESHOLD, base))

    if "recent_win_rate" not in kwargs:
        win_rate = _read_recent_win_rate()
        if win_rate is not None:
            kwargs["recent_win_rate"] = win_rate
    return calculate_dynamic_threshold(**kwargs)


def get_market_regime_multiplier(regime: Optional[str] = None) -> Dict[str, float]:
    """
    Rejime gore position boyutu ve leverage carpanlari return.
    
    Args:
        regime: Mevcut piyasa rejimi
    
    Returns:
        {
            "position_multiplier": float,  # 0.25 - 1.0
            "leverage_cap": int,           # Max leverage
            "sl_multiplier": float         # Stop loss carpani
        }
    """
    multipliers = {
        "STRONG_BULL": {"position_multiplier": 1.0, "leverage_cap": 1, "sl_multiplier": 1.0},
        "BULL": {"position_multiplier": 1.0, "leverage_cap": 1, "sl_multiplier": 1.0},
        "WEAK_BULL": {"position_multiplier": 0.8, "leverage_cap": 1, "sl_multiplier": 1.1},
        "SIDEWAYS": {"position_multiplier": 0.6, "leverage_cap": 1, "sl_multiplier": 1.2},
        "WEAK_BEAR": {"position_multiplier": 0.5, "leverage_cap": 1, "sl_multiplier": 1.3},
        "BEAR": {"position_multiplier": 0.4, "leverage_cap": 1, "sl_multiplier": 1.4},
        "STRONG_BEAR": {"position_multiplier": 0.25, "leverage_cap": 1, "sl_multiplier": 1.5},
        "CRISIS": {"position_multiplier": 0.1, "leverage_cap": 1, "sl_multiplier": 2.0},
    }
    
    regime = (regime or "SIDEWAYS").upper()
    return multipliers.get(regime, multipliers["SIDEWAYS"])


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DYNAMIC THRESHOLD TEST")
    print("=" * 60)
    
    # Test 1: BULL + Long + Strong AI
    t1 = calculate_dynamic_threshold(
        regime="BULL",
        direction="long",
        ai_scores={"chatgpt": 0.75, "deepseek": 0.72},
        recent_win_rate=0.55
    )
    print(f"BULL + Long + AI consensus: {t1}")
    
    # Test 2: SIDEWAYS + Weak AI
    t2 = calculate_dynamic_threshold(
        regime="SIDEWAYS",
        direction="long",
        ai_scores={"chatgpt": 0.48, "deepseek": 0.52},
        recent_win_rate=0.35
    )
    print(f"SIDEWAYS + Weak AI + Poor performance: {t2}")
    
    # Test 3: BEAR + Short
    t3 = calculate_dynamic_threshold(
        regime="BEAR",
        direction="short",
        ai_scores={"chatgpt": 0.30, "deepseek": 0.35},
    )
    print(f"BEAR + Short: {t3}")
