# -*- coding: utf-8 -*-
"""
entry_optimizer.py - Multi-Timeframe Entry Confirmation System
================================================================

[2026-01-15] Kötü entry timing'i önlemek için MTF analizi.

Entry Kuralları:
- 15m: Entry trigger (MACD crossover, RSI extreme, vb.)
- 1h: Trend confirmation (fiyat > EMA50)
- 4h: Major trend alignment (counter-trend engelleme)

Minimum 2/3 timeframe uyumlu olmalı.

Kullanım:
    from entry_optimizer import EntryOptimizer
    
    optimizer = EntryOptimizer()
    result = optimizer.check_entry(multi_data, direction="long")
    if result["approved"]:
        # Entry yapılabilir
"""

from __future__ import annotations


from core.exceptions import BEST_EFFORT_EXCEPTIONS
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import talib as ta
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

try:
    from logger import get_logger
    log = get_logger("entry_opt")
except ImportError:
    log = logging.getLogger("entry_opt")
    log.setLevel(logging.INFO)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TimeframeCheck:
    """Single timeframe check result."""
    timeframe: str
    passed: bool
    score: float  # 0-1
    signal: str  # Description
    details: Dict = field(default_factory=dict)


@dataclass
class EntryCheckResult:
    """Complete entry check result."""
    approved: bool
    overall_score: float  # 0-1
    aligned_count: int
    required_count: int
    timeframe_checks: List[TimeframeCheck] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class OrderTypeDecision:
    """Limit vs market order decision."""
    order_type: str  # "limit" or "market"
    limit_price: Optional[float] = None
    timeout_seconds: int = 120  # Fallback to market after timeout
    reasoning: str = ""
    spread_pct: float = 0.0


# =============================================================================
# ENTRY OPTIMIZER
# =============================================================================

class EntryOptimizer:
    """
    Multi-timeframe entry confirmation.

    Prevents bad entries by requiring:
    - Entry trigger on lower timeframe (15m)
    - Trend confirmation on medium timeframe (1h)
    - Major trend alignment on higher timeframe (4h)

    [2026-01-20 UPDATE] MTF verisi eksik olduğunda daha toleranslı davranır.
    Kolon isimleri suffix'li olabilir (close_15m, close_1h, vb.)
    """

    # Timeframe weights
    TIMEFRAME_WEIGHTS = {
        "15m": 0.30,
        "1h": 0.40,
        "4h": 0.30
    }

    # Minimum alignment required
    # [2026-01-20] Daha toleranslı: 2/3 yerine 1/3 yeterli, score threshold düşürüldü
    MIN_ALIGNED_COUNT = 1  # 1 out of 3 (veri eksikliği durumunda)
    MIN_OVERALL_SCORE = 0.45  # Daha düşük eşik - veri eksikliğinde adil
    
    def __init__(self):
        pass

    def _get_column(self, df: pd.DataFrame, col_name: str, tf: str = None) -> Optional[np.ndarray]:
        """
        [2026-01-20] DataFrame'den kolon al - suffix'li ve suffix'siz isimleri destekler.
        Örnek: 'close' veya 'close_15m' veya 'Close_15m'
        """
        if df is None or not hasattr(df, 'columns'):
            return None

        # Deneme sırası: exact match, suffixed, case-insensitive
        candidates = [col_name]
        if tf:
            candidates.append(f"{col_name}_{tf}")
            candidates.append(f"{col_name.lower()}_{tf}")
            candidates.append(f"{col_name.capitalize()}_{tf}")

        for cand in candidates:
            if cand in df.columns:
                try:
                    return df[cand].values
                except BEST_EFFORT_EXCEPTIONS:
                    continue

        # Son çare: kısmi eşleşme
        col_lower = col_name.lower()
        for c in df.columns:
            if col_lower in c.lower():
                try:
                    return df[c].values
                except BEST_EFFORT_EXCEPTIONS:
                    continue

        return None

    def check_entry(
        self,
        multi_data: Dict[str, pd.DataFrame],
        direction: Literal["long", "short"],
        current_price: float = None
    ) -> EntryCheckResult:
        """
        Check if entry is approved based on MTF analysis.
        
        Args:
            multi_data: Dict of {timeframe: df} with OHLCV + indicators
            direction: "long" or "short"
            current_price: Current price (optional, uses last close if not provided)
            
        Returns:
            EntryCheckResult with approval decision
        """
        if not PANDAS_AVAILABLE:
            return EntryCheckResult(
                approved=True,
                overall_score=0.5,
                aligned_count=0,
                required_count=2,
                reasoning=["Pandas not available - skipping MTF check"]
            )
        
        checks = []
        reasoning = []
        warnings = []
        
        # Check 15m - Entry Trigger
        if "15m" in multi_data:
            check = self._check_entry_trigger(multi_data["15m"], direction)
            checks.append(check)
            if check.passed:
                reasoning.append(f"15m: ✅ {check.signal}")
            else:
                reasoning.append(f"15m: ❌ {check.signal}")
        
        # Check 1h - Trend Confirmation
        if "1h" in multi_data:
            check = self._check_trend_confirmation(multi_data["1h"], direction)
            checks.append(check)
            if check.passed:
                reasoning.append(f"1h: ✅ {check.signal}")
            else:
                reasoning.append(f"1h: ❌ {check.signal}")
        
        # Check 4h - Major Trend
        if "4h" in multi_data:
            check = self._check_major_trend(multi_data["4h"], direction)
            checks.append(check)
            if check.passed:
                reasoning.append(f"4h: ✅ {check.signal}")
            else:
                reasoning.append(f"4h: ❌ {check.signal}")
                if check.details.get("counter_trend"):
                    warnings.append("⚠️ Counter-trend trade!")
        
        # Calculate overall score
        aligned_count = sum(1 for c in checks if c.passed)
        
        # Weighted score
        total_weight = 0.0
        weighted_score = 0.0
        for check in checks:
            weight = self.TIMEFRAME_WEIGHTS.get(check.timeframe, 0.33)
            weighted_score += check.score * weight
            total_weight += weight
        
        overall_score = weighted_score / total_weight if total_weight > 0 else 0.5
        
        # Decision
        approved = (
            aligned_count >= self.MIN_ALIGNED_COUNT and
            overall_score >= self.MIN_OVERALL_SCORE
        )
        
        return EntryCheckResult(
            approved=approved,
            overall_score=overall_score,
            aligned_count=aligned_count,
            required_count=self.MIN_ALIGNED_COUNT,
            timeframe_checks=checks,
            reasoning=reasoning,
            warnings=warnings
        )
    
    def decide_order_type(
        self,
        df_15m: Optional[pd.DataFrame],
        direction: Literal["long", "short"],
        current_price: float,
        spread_pct: float = 0.0,
        volatility_pct: float = 0.0,
    ) -> OrderTypeDecision:
        """
        [FAZA 5.4] Decide between limit and market order.

        Rules:
        - Low volatility + tight spread → limit order (better fill)
        - High volatility or wide spread → market order (ensure fill)
        - Limit order timeout: 120s default, shorter in high vol

        Args:
            df_15m: 15-minute OHLCV data
            direction: Trade direction
            current_price: Current market price
            spread_pct: Current bid-ask spread as percentage
            volatility_pct: Recent volatility (ATR/price)
        """
        if not PANDAS_AVAILABLE or current_price <= 0:
            return OrderTypeDecision(
                order_type="market",
                reasoning="No data available - defaulting to market",
            )

        # Thresholds
        SPREAD_LIMIT_THRESHOLD = 0.05   # %0.05 spread → limit possible
        VOLATILITY_LIMIT_THRESHOLD = 1.5  # %1.5 ATR → too volatile for limit

        # Calculate volatility from data if not provided
        if volatility_pct <= 0 and df_15m is not None and len(df_15m) >= 14:
            try:
                close = self._get_column(df_15m, "close", "15m")
                high = self._get_column(df_15m, "high", "15m")
                low = self._get_column(df_15m, "low", "15m")
                if close is not None and high is not None and low is not None:
                    tr = np.maximum(
                        high[-14:] - low[-14:],
                        np.maximum(
                            np.abs(high[-14:] - np.roll(close[-14:], 1)[1:14]),
                            np.abs(low[-14:] - np.roll(close[-14:], 1)[1:14])
                        )
                    )
                    atr = np.nanmean(tr)
                    volatility_pct = (atr / current_price) * 100 if current_price > 0 else 0
            except BEST_EFFORT_EXCEPTIONS:
                pass

        # Decision logic
        use_limit = (
            spread_pct < SPREAD_LIMIT_THRESHOLD and
            volatility_pct < VOLATILITY_LIMIT_THRESHOLD and
            volatility_pct > 0  # Need valid volatility data
        )

        if use_limit:
            # Place limit order slightly better than current price
            offset_pct = max(0.01, spread_pct * 0.5)  # Half the spread improvement
            if direction == "long":
                limit_price = current_price * (1 - offset_pct / 100)
            else:
                limit_price = current_price * (1 + offset_pct / 100)

            # Shorter timeout in higher volatility
            if volatility_pct > 1.0:
                timeout = 60
            elif volatility_pct > 0.5:
                timeout = 90
            else:
                timeout = 120

            return OrderTypeDecision(
                order_type="limit",
                limit_price=round(limit_price, 8),
                timeout_seconds=timeout,
                reasoning=f"Low vol ({volatility_pct:.2f}%) + tight spread ({spread_pct:.3f}%) → limit order",
                spread_pct=spread_pct,
            )
        else:
            reason_parts = []
            if spread_pct >= SPREAD_LIMIT_THRESHOLD:
                reason_parts.append(f"wide spread ({spread_pct:.3f}%)")
            if volatility_pct >= VOLATILITY_LIMIT_THRESHOLD:
                reason_parts.append(f"high vol ({volatility_pct:.2f}%)")
            if volatility_pct <= 0:
                reason_parts.append("no volatility data")

            return OrderTypeDecision(
                order_type="market",
                reasoning=f"Market order: {', '.join(reason_parts) or 'default'}",
                spread_pct=spread_pct,
            )

    def _check_entry_trigger(
        self,
        df: pd.DataFrame,
        direction: str
    ) -> TimeframeCheck:
        """
        Check entry trigger on 15m.
        Looks for: MACD crossover, RSI extreme, volume spike
        [2026-01-20] Suffix'li kolon desteği eklendi (close_15m, vb.)
        """
        if df is None or (hasattr(df, '__len__') and len(df) < 20):
            return TimeframeCheck(
                timeframe="15m",
                passed=True,  # [2026-01-20] Veri yoksa geç
                score=0.6,
                signal="No data - allowing"
            )

        try:
            # [2026-01-20 FIX] Suffix'li kolon desteği
            close = self._get_column(df, "close", "15m")
            if close is None:
                return TimeframeCheck(
                    timeframe="15m",
                    passed=True,  # Allow on missing data
                    score=0.6,
                    signal="MTF data incomplete - allowing"
                )
            
            # Get indicators
            if TALIB_AVAILABLE:
                macd, macd_signal, macd_hist = ta.MACD(close, 12, 26, 9)
                rsi = ta.RSI(close, 14)
            else:
                macd_hist = np.zeros_like(close)
                rsi = np.full_like(close, 50.0)
            
            last_macd_hist = macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0
            prev_macd_hist = macd_hist[-2] if len(macd_hist) > 1 and not np.isnan(macd_hist[-2]) else 0
            last_rsi = rsi[-1] if not np.isnan(rsi[-1]) else 50
            
            triggers = []
            score = 0.5
            
            if direction == "long":
                # MACD crossover (histogram turning positive)
                if last_macd_hist > 0 and prev_macd_hist <= 0:
                    triggers.append("MACD bullish crossover")
                    score += 0.25
                elif last_macd_hist > 0:
                    score += 0.1
                
                # RSI oversold bounce
                if last_rsi < 35:
                    triggers.append("RSI oversold (<35)")
                    score += 0.15
                elif last_rsi < 45:
                    score += 0.05
                
                # Price momentum
                if len(close) >= 3:
                    if close[-1] > close[-2] > close[-3]:
                        triggers.append("Price momentum up")
                        score += 0.1
            else:
                # SHORT
                if last_macd_hist < 0 and prev_macd_hist >= 0:
                    triggers.append("MACD bearish crossover")
                    score += 0.25
                elif last_macd_hist < 0:
                    score += 0.1
                
                if last_rsi > 65:
                    triggers.append("RSI overbought (>65)")
                    score += 0.15
                elif last_rsi > 55:
                    score += 0.05
                
                if len(close) >= 3:
                    if close[-1] < close[-2] < close[-3]:
                        triggers.append("Price momentum down")
                        score += 0.1
            
            score = min(1.0, max(0.0, score))
            passed = score >= 0.6
            
            signal = ", ".join(triggers) if triggers else "No strong trigger"
            
            return TimeframeCheck(
                timeframe="15m",
                passed=passed,
                score=score,
                signal=signal,
                details={"rsi": last_rsi, "macd_hist": last_macd_hist}
            )
            
        except BEST_EFFORT_EXCEPTIONS as e:
            log.warning(f"15m check error: {e}")
            return TimeframeCheck(
                timeframe="15m",
                passed=False,
                score=0.5,
                signal=f"Error: {e}"
            )
    
    def _check_trend_confirmation(
        self,
        df: pd.DataFrame,
        direction: str
    ) -> TimeframeCheck:
        """
        Check trend confirmation on 1h.
        Looks for: Price vs EMA50, trend direction
        [2026-01-20] Suffix'li kolon desteği eklendi
        """
        if df is None or (hasattr(df, '__len__') and len(df) < 50):
            return TimeframeCheck(
                timeframe="1h",
                passed=True,  # [2026-01-20] Veri yoksa geç
                score=0.6,
                signal="No data - allowing"
            )

        try:
            # [2026-01-20 FIX] Suffix'li kolon desteği
            close = self._get_column(df, "close", "1h")
            if close is None:
                return TimeframeCheck(
                    timeframe="1h",
                    passed=True,  # Allow on missing data
                    score=0.6,
                    signal="MTF data incomplete - allowing"
                )
            
            if TALIB_AVAILABLE:
                ema20 = ta.EMA(close, 20)
                ema50 = ta.EMA(close, 50)
            else:
                ema20 = pd.Series(close).ewm(span=20).mean().values
                ema50 = pd.Series(close).ewm(span=50).mean().values
            
            last_close = close[-1]
            last_ema20 = ema20[-1] if not np.isnan(ema20[-1]) else last_close
            last_ema50 = ema50[-1] if not np.isnan(ema50[-1]) else last_close
            
            confirmations = []
            score = 0.5
            
            if direction == "long":
                if last_close > last_ema50:
                    confirmations.append("Price > EMA50")
                    score += 0.2
                if last_ema20 > last_ema50:
                    confirmations.append("EMA20 > EMA50 (uptrend)")
                    score += 0.2
                if last_close > last_ema20:
                    confirmations.append("Price > EMA20")
                    score += 0.1
            else:
                if last_close < last_ema50:
                    confirmations.append("Price < EMA50")
                    score += 0.2
                if last_ema20 < last_ema50:
                    confirmations.append("EMA20 < EMA50 (downtrend)")
                    score += 0.2
                if last_close < last_ema20:
                    confirmations.append("Price < EMA20")
                    score += 0.1
            
            score = min(1.0, max(0.0, score))
            passed = score >= 0.6
            
            signal = ", ".join(confirmations) if confirmations else "No trend confirmation"
            
            return TimeframeCheck(
                timeframe="1h",
                passed=passed,
                score=score,
                signal=signal,
                details={
                    "price_vs_ema50": "above" if last_close > last_ema50 else "below"
                }
            )
            
        except BEST_EFFORT_EXCEPTIONS as e:
            log.warning(f"1h check error: {e}")
            return TimeframeCheck(
                timeframe="1h",
                passed=False,
                score=0.5,
                signal=f"Error: {e}"
            )
    
    def _check_major_trend(
        self,
        df: pd.DataFrame,
        direction: str
    ) -> TimeframeCheck:
        """
        Check major trend on 4h.
        Prevents counter-trend trades.
        [2026-01-20] Suffix'li kolon desteği eklendi
        """
        if df is None or (hasattr(df, '__len__') and len(df) < 50):
            return TimeframeCheck(
                timeframe="4h",
                passed=True,  # Allow if no data
                score=0.6,
                signal="No 4h data - allowing"
            )

        try:
            # [2026-01-20 FIX] Suffix'li kolon desteği
            close = self._get_column(df, "close", "4h")
            high = self._get_column(df, "high", "4h")
            low = self._get_column(df, "low", "4h")

            if close is None:
                return TimeframeCheck(
                    timeframe="4h",
                    passed=True,
                    score=0.6,
                    signal="MTF data incomplete - allowing"
                )
            
            if TALIB_AVAILABLE:
                ema50 = ta.EMA(close, 50)
                ema200 = ta.EMA(close, 200)
                # [2026-01-20 FIX] high/low değişkenlerini kullan
                if high is not None and low is not None:
                    adx = ta.ADX(high, low, close, 14)
                else:
                    adx = np.full_like(close, 25.0)  # Varsayılan ADX
            else:
                ema50 = pd.Series(close).ewm(span=50).mean().values
                ema200 = pd.Series(close).ewm(span=200).mean().values
                adx = np.full_like(close, 25.0)
            
            last_ema50 = ema50[-1] if not np.isnan(ema50[-1]) else close[-1]
            last_ema200 = ema200[-1] if not np.isnan(ema200[-1]) else close[-1]
            last_adx = adx[-1] if not np.isnan(adx[-1]) else 25.0
            
            # Determine major trend
            if last_ema50 > last_ema200:
                major_trend = "bullish"
            elif last_ema50 < last_ema200:
                major_trend = "bearish"
            else:
                major_trend = "neutral"
            
            score = 0.5
            is_counter_trend = False
            
            if direction == "long":
                if major_trend == "bullish":
                    score = 0.8
                    signal = "Major trend bullish - aligned ✅"
                elif major_trend == "neutral":
                    score = 0.6
                    signal = "Major trend neutral - acceptable"
                else:
                    score = 0.3
                    is_counter_trend = True
                    signal = "⚠️ Major trend bearish - COUNTER-TREND!"
            else:
                if major_trend == "bearish":
                    score = 0.8
                    signal = "Major trend bearish - aligned ✅"
                elif major_trend == "neutral":
                    score = 0.6
                    signal = "Major trend neutral - acceptable"
                else:
                    score = 0.3
                    is_counter_trend = True
                    signal = "⚠️ Major trend bullish - COUNTER-TREND!"
            
            # Strong trend bonus/penalty
            if last_adx > 30:
                if not is_counter_trend:
                    score += 0.1  # Bonus for going with strong trend
                else:
                    score -= 0.1  # Extra penalty for counter-trend in strong trend
            
            score = min(1.0, max(0.0, score))
            passed = score >= 0.5  # More lenient for 4h
            
            return TimeframeCheck(
                timeframe="4h",
                passed=passed,
                score=score,
                signal=signal,
                details={
                    "major_trend": major_trend,
                    "counter_trend": is_counter_trend,
                    "adx": last_adx
                }
            )
            
        except BEST_EFFORT_EXCEPTIONS as e:
            log.warning(f"4h check error: {e}")
            return TimeframeCheck(
                timeframe="4h",
                passed=True,  # Allow on error
                score=0.5,
                signal=f"Error: {e}"
            )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

import threading as _threading_eo
_optimizer_instance: Optional[EntryOptimizer] = None
_optimizer_instance_lock = _threading_eo.Lock()


def get_entry_optimizer() -> EntryOptimizer:
    """Get global optimizer instance."""
    global _optimizer_instance
    if _optimizer_instance is None:
        with _optimizer_instance_lock:
            if _optimizer_instance is None:
                _optimizer_instance = EntryOptimizer()
    return _optimizer_instance


def check_mtf_entry(
    multi_data: Dict[str, pd.DataFrame],
    direction: str
) -> Dict:
    """
    Quick check function for MTF entry.
    
    Returns:
        {
            "approved": bool,
            "score": float,
            "aligned": int,
            "reasoning": list
        }
    """
    optimizer = get_entry_optimizer()
    result = optimizer.check_entry(multi_data, direction)
    
    return {
        "approved": result.approved,
        "score": result.overall_score,
        "aligned": result.aligned_count,
        "required": result.required_count,
        "reasoning": result.reasoning,
        "warnings": result.warnings
    }


def decide_order_type(
    df_15m: Optional[pd.DataFrame],
    direction: str,
    current_price: float,
    spread_pct: float = 0.0,
    volatility_pct: float = 0.0,
) -> Dict:
    """Quick helper for order type decision."""
    optimizer = get_entry_optimizer()
    result = optimizer.decide_order_type(df_15m, direction, current_price, spread_pct, volatility_pct)
    return {
        "order_type": result.order_type,
        "limit_price": result.limit_price,
        "timeout_seconds": result.timeout_seconds,
        "reasoning": result.reasoning,
        "spread_pct": result.spread_pct,
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Entry Optimizer Test")
    print("=" * 50)
    
    optimizer = EntryOptimizer()
    print("Optimizer initialized")
    print(f"Min aligned: {optimizer.MIN_ALIGNED_COUNT}")
    print(f"Min score: {optimizer.MIN_OVERALL_SCORE}")
