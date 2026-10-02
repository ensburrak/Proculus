# -*- coding: utf-8 -*-
"""
meta_strategy_selector.py
=========================

[2026-01-15 PROFESSIONAL UPGRADE]

Piyasa rejimine göre en uygun stratejiyi seçen modül.

Stratejiler:
1. TREND - Momentum takip (BULL/BEAR rejimlerinde)
2. MEAN_REVERSION - Ortalamaya dönüş (SIDEWAYS/RANGE rejimde)
3. BREAKOUT - Kırılım stratejisi (range sonlarında)
4. DEFENSIVE - Koruyucu mod (yüksek belirsizlikte)

Veri Kaynakları (Sadece güvenilir):
- OKX/Binance: Fiyat, hacim, volatilite
- AI modelleri: ChatGPT/DeepSeek tahmini
- Rejim detektörü: Mevcut piyasa durumu
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)
CONFIG_FILE = Path(__file__).resolve().parent / "config.json"


def _pipeline_v2_enabled(path: Path = CONFIG_FILE) -> bool:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            return False
        pipeline_v2 = payload.get("pipeline_v2")
        return bool(pipeline_v2.get("enabled", False)) if isinstance(pipeline_v2, dict) else False
    except (OSError, json.JSONDecodeError, ValueError, TypeError, AttributeError):
        return False


def calculate_momentum(prices: np.ndarray, lookback: int = 20) -> float:
    """Basit momentum hesapla (son lookback mum üzerinden)."""
    if len(prices) < lookback + 1:
        return 0.0
    
    returns = np.diff(prices[-lookback:]) / (prices[-lookback:-1] + 1e-8)
    return float(np.sum(returns))


def calculate_volatility(prices: np.ndarray, lookback: int = 20) -> float:
    """Volatilite hesapla (return std)."""
    if len(prices) < lookback + 1:
        return 0.0
    
    returns = np.diff(prices[-lookback:]) / (prices[-lookback:-1] + 1e-8)
    return float(np.std(returns))


def detect_range_breakout(prices: np.ndarray, lookback: int = 50, threshold: float = 0.02) -> Optional[str]:
    """
    Range kırılımı tespit et.
    
    Returns:
        "breakout_up" - Üst kırılım
        "breakout_down" - Alt kırılım
        None - Kırılım yok
    """
    if len(prices) < lookback:
        return None
    
    window = prices[-lookback:]
    high = np.max(window[:-1])  # Son mum hariç
    low = np.min(window[:-1])
    last = prices[-1]
    
    if last > high * (1 + threshold):
        return "breakout_up"
    if last < low * (1 - threshold):
        return "breakout_down"
    
    return None


def select_strategy(
    prices: Optional[Iterable[float]] = None,
    regime: Optional[str] = None,
    regime_confidence: Optional[float] = None,
    ai_scores: Optional[Dict[str, float]] = None,
    atr: Optional[float] = None,
    current_price: Optional[float] = None,
    lookback: int = 60,
    vol_threshold: float = 0.02,
) -> Dict[str, Any]:
    """
    [PROFESSIONAL FIX #22] Piyasa koşullarına göre strateji seç.
    
    Args:
        prices: Son fiyat serisi
        regime: Mevcut piyasa rejimi ("BULL", "BEAR", "SIDEWAYS", "RANGE", ...)
        regime_confidence: Rejim güven skoru (0-1)
        ai_scores: AI model skorları
        atr: Average True Range
        current_price: Mevcut fiyat
        lookback: Analiz penceresi
        vol_threshold: Volatilite eşiği
    
    Returns:
        {
            "strategy": str,  # "trend", "mean_reversion", "breakout", "defensive"
            "confidence": float,
            "direction_bias": str,  # "long", "short", "neutral"
            "reasoning": str
        }
    """
    if _pipeline_v2_enabled():
        return {
            "strategy": None,
            "confidence": None,
            "direction_bias": None,
            "reasoning": "pipeline_v2 enabled -> shadow audit only; decision authority owned by strategy router",
            "router_managed": True,
            "shadow_only": True,
            "audit_label": "meta_strategy_shadow",
            "operator_summary": "router-owned decision path; meta strategy kept for audit only",
        }

    result = {
        "action": "watch",
        "strategy": "mean_reversion",
        "confidence": 0.5,
        "direction_bias": "neutral",
        "risk_size_cap": 0.8,
        "reasoning": "",
    }
    
    # Fiyat verisini numpy array'e çevir
    if prices is not None:
        prices_arr = np.asarray(list(prices), dtype=float)
    else:
        prices_arr = None
    
    reasons = []
    
    # =========================================================================
    # 1. Rejim Bazlı Strateji Seçimi (Birincil)
    # =========================================================================
    normalized_regime = str(regime or "UNKNOWN").upper()
    confidence = float(regime_confidence or 0.5)

    if normalized_regime in ("BULL", "STRONG_BULL", "WEAK_BULL"):
        result.update(
            action="trade",
            strategy="trend",
            direction_bias="long",
            confidence=min(0.85, 0.60 + confidence * 0.30),
            risk_size_cap=1.0,
        )
        reasons.append(f"{normalized_regime} regime -> TREND-LONG")
    elif normalized_regime in ("BEAR", "STRONG_BEAR", "WEAK_BEAR"):
        result.update(
            action="trade",
            strategy="trend",
            direction_bias="short",
            confidence=min(0.85, 0.60 + confidence * 0.30),
            risk_size_cap=1.0,
        )
        reasons.append(f"{normalized_regime} regime -> TREND-SHORT")
    elif normalized_regime in ("SIDEWAYS", "RANGE", "RANGING"):
        result.update(
            action="trade",
            strategy="mean_reversion",
            direction_bias="neutral",
            confidence=max(0.55, min(0.75, confidence)),
            risk_size_cap=0.80,
        )
        reasons.append(f"{normalized_regime} regime -> MEAN_REVERSION")
    elif normalized_regime in ("COMPRESSION", "SQUEEZE"):
        result.update(
            action="watch",
            strategy="breakout",
            direction_bias="neutral",
            confidence=max(0.60, min(0.80, confidence)),
            risk_size_cap=0.75,
        )
        reasons.append(f"{normalized_regime} regime -> WAIT_FOR_BREAKOUT_CONFIRMATION")
    elif normalized_regime in ("TRANSITION", "EARLY_TREND"):
        result.update(
            action="no_trade",
            strategy="confirmation",
            direction_bias="neutral",
            confidence=max(0.70, confidence),
            risk_size_cap=0.0,
        )
        reasons.append(f"{normalized_regime} regime -> NO_TRADE_UNTIL_CONFIRMATION")
    elif normalized_regime in ("VOLATILE", "CRISIS", "SHOCK", "CONFLICT", "UNKNOWN"):
        result.update(
            action="no_trade",
            strategy="defensive",
            direction_bias="neutral",
            confidence=max(0.90, confidence),
            risk_size_cap=0.0,
        )
        reasons.append(f"{normalized_regime} regime -> HARD_NO_TRADE")

    # =========================================================================
    # 2. Breakout Tespiti (Override)
    # =========================================================================
    breakout_allowed_regimes = {"SIDEWAYS", "RANGE", "RANGING", "COMPRESSION", "SQUEEZE"}
    if (
        prices_arr is not None
        and len(prices_arr) >= lookback
        and normalized_regime in breakout_allowed_regimes
    ):
        breakout = detect_range_breakout(prices_arr, lookback)
        if breakout:
            result["action"] = "trade"
            result["strategy"] = "breakout"
            result["direction_bias"] = "long" if breakout == "breakout_up" else "short"
            result["confidence"] = min(0.80, max(float(result["confidence"]), 0.72))
            result["risk_size_cap"] = min(float(result["risk_size_cap"]), 0.75)
            reasons.append(f"Confirmed breakout: {breakout}")
    
    # =========================================================================
    # 3. AI / ML Quality Advisory — NEVER direction authority
    # =========================================================================
    if ai_scores and result["action"] == "trade":
        active_scores = [
            float(score)
            for score in (ai_scores.get("chatgpt"), ai_scores.get("deepseek"))
            if isinstance(score, (int, float))
        ]
        if active_scores:
            avg_ai = sum(active_scores) / len(active_scores)
            conflict = (
                (result["direction_bias"] == "long" and avg_ai < 0.35)
                or (result["direction_bias"] == "short" and avg_ai > 0.65)
            )
            if conflict:
                result["confidence"] *= 0.85
                result["risk_size_cap"] = min(float(result["risk_size_cap"]), 0.75)
                reasons.append(f"AI advisory conflict -> quality downscale (AI={avg_ai:.2f})")
            else:
                reasons.append(f"AI advisory recorded only (AI={avg_ai:.2f}); no direction boost")

    # =========================================================================
    # 4. Volatilite Kontrolü
    # =========================================================================
    if atr is not None and current_price is not None and current_price > 0:
        atr_pct = atr / current_price
        
        if atr_pct > 0.05:
            result["action"] = "no_trade"
            result["strategy"] = "defensive"
            result["direction_bias"] = "neutral"
            result["risk_size_cap"] = 0.0
            result["confidence"] = max(float(result["confidence"]), 0.90)
            reasons.append(f"Extreme volatility (ATR={atr_pct:.1%}) -> HARD_NO_TRADE")
        elif atr_pct > 0.03 and result["action"] == "trade":
            result["confidence"] *= 0.90
            result["risk_size_cap"] = min(float(result["risk_size_cap"]), 0.75)
            reasons.append(f"High volatility (ATR={atr_pct:.1%}) -> size cap")
    
    # =========================================================================
    # 5. Final
    # =========================================================================
    result["confidence"] = round(max(0.3, min(0.95, result["confidence"])), 3)
    result["reasoning"] = " | ".join(reasons) if reasons else "Default"
    
    logger.info(
        "[META_STRATEGY] action=%s strategy=%s bias=%s conf=%.2f size_cap=%.2f | %s",
        result["action"],
        str(result["strategy"]).upper(),
        result["direction_bias"],
        result["confidence"],
        float(result["risk_size_cap"]),
        result["reasoning"],
    )
    
    return result


def get_strategy_multipliers(strategy: str) -> Dict[str, float]:
    """
    Strateji için parametre çarpanlarını döndür.
    
    Bu çarpanlar risk_manager ve controller tarafından kullanılabilir.
    """
    multipliers = {
        "trend": {
            "position_size": 1.0,
            "stop_loss_distance": 1.0,
            "take_profit_distance": 1.0,
            "max_holding_time": 1.0,
        },
        "mean_reversion": {
            "position_size": 0.8,  # Daha küçük pozisyon
            "stop_loss_distance": 0.8,  # Daha sıkı stop
            "take_profit_distance": 0.7,  # Daha yakın TP
            "max_holding_time": 0.5,  # Daha kısa tutma
        },
        "breakout": {
            "position_size": 0.75,
            "stop_loss_distance": 1.0,
            "take_profit_distance": 1.25,
            "max_holding_time": 1.0,
        },
        "confirmation": {
            "position_size": 0.0,
            "stop_loss_distance": 1.0,
            "take_profit_distance": 1.0,
            "max_holding_time": 0.5,
        },
        "defensive": {
            "position_size": 0.0,
            "stop_loss_distance": 1.0,
            "take_profit_distance": 1.0,
            "max_holding_time": 0.3,
        },
    }
    
    return multipliers.get(strategy, multipliers["mean_reversion"])


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("META STRATEGY SELECTOR TEST")
    print("=" * 60)
    
    # Test 1: BULL regime
    result1 = select_strategy(
        regime="BULL",
        regime_confidence=0.75,
        ai_scores={"chatgpt": 0.72, "deepseek": 0.68}
    )
    print(f"\nTest 1 (BULL): {result1}")
    
    # Test 2: SIDEWAYS with breakout
    prices = [100 + i*0.1 for i in range(60)]  # Yavaş yükseliş
    prices.append(115)  # Breakout!
    result2 = select_strategy(
        prices=prices,
        regime="SIDEWAYS",
    )
    print(f"\nTest 2 (SIDEWAYS+Breakout): {result2}")
    
    # Test 3: High volatility
    result3 = select_strategy(
        regime="BULL",
        ai_scores={"chatgpt": 0.60, "deepseek": 0.58},
        atr=5000,
        current_price=100000
    )
    print(f"\nTest 3 (High Vol): {result3}")
