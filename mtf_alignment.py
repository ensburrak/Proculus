# -*- coding: utf-8 -*-
"""
mtf_alignment.py - Multi-Timeframe Alignment Enforcer
======================================================

[2026-01-14] Yeni modül.

Multi-timeframe (MTF) analizi ile trade yönünün onaylanmasını sağlar.
Bir trade'in açılabilmesi için birden fazla timeframe'in aynı yönde
olması gerekir. Bu, yanlış yönlü trade'lerin önlenmesine yardımcı olur.

Örnek senaryo:
- 5m: LONG sinyali
- 15m: LONG sinyali
- 1h: SHORT sinyali
- 4h: LONG sinyali

Sonuç: 3/4 = %75 alignment. Long açılabilir (threshold %60 ise).

Kullanım:
    from mtf_alignment import MTFAlignmentEnforcer

    enforcer = MTFAlignmentEnforcer()
    result = enforcer.check_alignment(multi_data, direction="long")

    if result["approved"]:
        # Trade aç
    else:
        # Trade'i atla
"""

from core.exceptions import BEST_EFFORT_EXCEPTIONS
import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from logger import get_logger
    log = get_logger("mtf_alignment")
except ImportError:
    import logging
    log = logging.getLogger("mtf_alignment")


DirectionType = Literal["long", "short", "neutral"]


@dataclass
class TimeframeSignal:
    """Tek bir timeframe için sinyal bilgisi."""
    timeframe: str
    direction: DirectionType
    strength: float  # 0-1 arası sinyal gücü
    indicators: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


@dataclass
class AlignmentResult:
    """MTF alignment analizi sonucu."""
    approved: bool
    alignment_score: float  # 0-1
    aligned_count: int
    total_count: int
    required_count: int
    direction: DirectionType
    timeframe_signals: List[TimeframeSignal] = field(default_factory=list)
    veto_reason: Optional[str] = None
    reasoning: str = ""


class MTFAlignmentEnforcer:
    """
    Multi-Timeframe Alignment Enforcer.

    Trade yönünün birden fazla timeframe tarafından onaylanmasını sağlar.
    Yanlış yönlü trade'lerin önlenmesine yardımcı olur.
    """

    # Timeframe öncelikleri (yüksek = daha önemli)
    TIMEFRAME_WEIGHTS = {
        "5m": 0.15,   # Kısa vadeli gürültü
        "15m": 0.20,  # Orta-kısa vadeli
        "1h": 0.30,   # Ana trade timeframe
        "4h": 0.35    # Trend yönü belirleyici
    }

    # Veto timeframe'leri (bunlar karşıtsa trade açılmaz)
    VETO_TIMEFRAMES = ["4h"]  # 4h karşıtsa kesinlikle trade açma

    # [FAZA 6.4] Graduated confidence adjustments
    # aligned_count -> confidence multiplier
    GRADUATED_CONFIDENCE = {
        4: 1.00,   # 4/4 TF aligned -> tam confidence
        3: 0.90,   # 3/4 TF aligned -> %90 confidence
        2: 0.75,   # 2/4 TF aligned -> %75 confidence (minimum gecerli)
        1: 0.50,   # 1/4 TF aligned -> %50 confidence (rejected)
        0: 0.30,   # 0/4 TF aligned -> %30 confidence (rejected)
    }

    def __init__(
        self,
        min_alignment_ratio: float = 0.60,
        min_aligned_count: int = 2,
        enable_veto: bool = True,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            min_alignment_ratio: Minimum alignment oranı (0-1)
            min_aligned_count: Minimum aligned timeframe sayısı (FAZA 6.4: default=2)
            enable_veto: Veto timeframe'lerini etkinleştir
            weights: Özel timeframe ağırlıkları
        """
        self.min_alignment_ratio = min_alignment_ratio
        self.min_aligned_count = min_aligned_count
        self.enable_veto = enable_veto
        self.weights = weights or self.TIMEFRAME_WEIGHTS.copy()

    def check_alignment(
        self,
        multi_data: Dict[str, pd.DataFrame],
        direction: DirectionType,
        current_price: Optional[float] = None
    ) -> AlignmentResult:
        """
        Multi-timeframe alignment kontrolü yapar.

        Args:
            multi_data: Her timeframe için DataFrame
                        Keys: "5m", "15m", "1h", "4h" gibi
                        Values: OHLCV + indicators DataFrame
            direction: Trade yönü ("long" veya "short")
            current_price: Mevcut fiyat (opsiyonel, EMA kontrolü için)

        Returns:
            AlignmentResult: Alignment analizi sonucu
        """
        direction = str(direction or "long").lower()
        if direction not in ("long", "short"):
            direction = "long"

        tf_signals: List[TimeframeSignal] = []
        reasoning_parts = []

        # Her timeframe için sinyal analizi
        for tf, df in multi_data.items():
            if df is None or df.empty:
                continue

            tf_key = self._normalize_timeframe(tf)
            if tf_key is None:
                continue

            signal = self._analyze_timeframe(tf_key, df, current_price)
            tf_signals.append(signal)

        if not tf_signals:
            return AlignmentResult(
                approved=False,
                alignment_score=0.0,
                aligned_count=0,
                total_count=0,
                required_count=self.min_aligned_count,
                direction=direction,
                veto_reason="Hicbir timeframe verisi yok",
                reasoning="Veri eksikligi"
            )

        # Aligned count hesapla
        aligned_count = sum(1 for s in tf_signals if s.direction == direction)
        total_count = len(tf_signals)

        # Agirlikli alignment score
        weighted_score = self._calculate_weighted_score(tf_signals, direction)

        # [FAZA 6.4] Timeframe'leri isimle index'le
        tf_map = {s.timeframe: s for s in tf_signals}
        has_4h = "4h" in tf_map
        has_1h = "1h" in tf_map
        has_15m = "15m" in tf_map
        has_5m = "5m" in tf_map

        # Veto kontrolu
        veto_reason = None
        if self.enable_veto:
            veto_reason = self._check_veto(tf_signals, direction)

        # [FAZA 6.4a] Minimum 2 TF data zorunlu + 15m veya 1h zorunlu
        # Sadece yüksek TF (4h, 1d) ile karar vermek tehlikeli — kısa TF onayı gerekli
        _has_short_tf = has_15m or has_5m or has_1h  # En az bir kısa/orta TF olmalı
        insufficient_data = total_count < 2 or not _has_short_tf

        # Approval karari
        meets_ratio = weighted_score >= self.min_alignment_ratio
        meets_count = aligned_count >= self.min_aligned_count
        no_veto = veto_reason is None

        approved = meets_ratio and meets_count and no_veto and not insufficient_data

        # [FAZA 6.4] Graduated confidence multiplier
        grad_mult = self.GRADUATED_CONFIDENCE.get(aligned_count, 0.30)

        # [FAZA 6.4] Ozel durum bonuslari/penaltilari
        bonus = 0.0
        bonus_reasons = []

        # Tum TF'ler uyumlu -> bonus %10
        if aligned_count == total_count and total_count >= 3:
            bonus += 0.10
            bonus_reasons.append("all_tf_aligned(+10%)")

        # 4h + 1h aligned -> bonus %5
        if has_4h and has_1h:
            s4h = tf_map["4h"]
            s1h = tf_map["1h"]
            if s4h.direction == direction and s1h.direction == direction:
                bonus += 0.05
                bonus_reasons.append("4h+1h_aligned(+5%)")

        # 4h karsi ama 1h/15m uyumlu -> confidence %20 dusus
        if has_4h and tf_map["4h"].direction != direction and tf_map["4h"].direction != "neutral":
            lower_aligned = sum(1 for tf in ("1h", "15m", "5m") if tf in tf_map and tf_map[tf].direction == direction)
            if lower_aligned >= 2 and veto_reason is not None:
                # 4h karsi ama alt TF'ler uyumlu -> veto'yu kaldir, %20 penalty uygula
                veto_reason = None
                approved = meets_ratio and meets_count
                bonus -= 0.20
                bonus_reasons.append("4h_against_lower_aligned(-20%)")

        # Yetersiz data -> %50 confidence dusus (tam veto degil)
        if insufficient_data:
            bonus -= 0.50
            bonus_reasons.append(f"insufficient_data({total_count}tf, -50%)")

        # Adjusted score = weighted_score * grad_mult + bonus
        adjusted_score = max(0.0, min(1.0, weighted_score * grad_mult + bonus))

        # Reasoning olustur
        for s in tf_signals:
            mark = "+" if s.direction == direction else "-"
            reasoning_parts.append(f"{s.timeframe}:{s.direction}({s.strength:.0%}){mark}")

        reasoning = (
            f"Alignment={weighted_score:.0%} ({aligned_count}/{total_count}) "
            f"grad={grad_mult:.2f} adj={adjusted_score:.2f}"
        )
        if bonus_reasons:
            reasoning += " | " + ", ".join(bonus_reasons)
        reasoning += " | " + " ".join(reasoning_parts)

        if veto_reason:
            reasoning += f" | VETO: {veto_reason}"

        return AlignmentResult(
            approved=approved,
            alignment_score=round(adjusted_score, 3),
            aligned_count=aligned_count,
            total_count=total_count,
            required_count=self.min_aligned_count,
            direction=direction,
            timeframe_signals=tf_signals,
            veto_reason=veto_reason,
            reasoning=reasoning
        )

    def _normalize_timeframe(self, tf: str) -> Optional[str]:
        """Timeframe string'ini normalize et."""
        tf = str(tf).lower().strip()

        # Yaygın formatları standartlaştır
        mappings = {
            "5m": "5m", "5min": "5m", "5": "5m",
            "15m": "15m", "15min": "15m", "15": "15m",
            "1h": "1h", "60m": "1h", "60min": "1h", "60": "1h",
            "4h": "4h", "240m": "4h", "240min": "4h", "240": "4h",
            "1d": "1d", "d": "1d", "daily": "1d"
        }

        return mappings.get(tf)

    def _analyze_timeframe(
        self,
        timeframe: str,
        df: pd.DataFrame,
        current_price: Optional[float] = None
    ) -> TimeframeSignal:
        """
        Tek bir timeframe için yön ve güç analizi.

        Kullanılan indikatörler:
        - EMA crossover (20/50)
        - MACD histogram
        - RSI pozisyonu
        - ADX trend gücü
        """
        indicators = {}
        bullish_signals = 0
        bearish_signals = 0
        total_signals = 0

        reasoning_parts = []

        # === EMA ANALIZI ===
        ema_fast_col = self._find_column(df, ["EMA_20", f"EMA_20_{timeframe}", "ema_20", "ema20"])
        ema_slow_col = self._find_column(df, ["EMA_50", f"EMA_50_{timeframe}", "ema_50", "ema50"])

        if ema_fast_col and ema_slow_col:
            ema_fast = df[ema_fast_col].iloc[-1]
            ema_slow = df[ema_slow_col].iloc[-1]

            if not pd.isna(ema_fast) and not pd.isna(ema_slow):
                total_signals += 1
                indicators["ema_fast"] = float(ema_fast)
                indicators["ema_slow"] = float(ema_slow)

                if ema_fast > ema_slow:
                    bullish_signals += 1
                    reasoning_parts.append("EMA+")
                else:
                    bearish_signals += 1
                    reasoning_parts.append("EMA-")

        # === MACD ANALIZI ===
        macd_hist_col = self._find_column(df, ["MACD_Hist", f"MACD_Hist_{timeframe}", "macd_hist", "macd_histogram"])

        if macd_hist_col:
            macd_hist = df[macd_hist_col].iloc[-1]

            if not pd.isna(macd_hist):
                total_signals += 1
                indicators["macd_hist"] = float(macd_hist)

                if macd_hist > 0:
                    bullish_signals += 1
                    reasoning_parts.append("MACD+")
                else:
                    bearish_signals += 1
                    reasoning_parts.append("MACD-")

        # === RSI ANALIZI ===
        rsi_col = self._find_column(df, ["RSI", f"RSI_{timeframe}", "rsi", "rsi_14"])

        if rsi_col:
            rsi = df[rsi_col].iloc[-1]

            if not pd.isna(rsi):
                total_signals += 1
                indicators["rsi"] = float(rsi)

                if rsi > 50 and rsi < 70:
                    bullish_signals += 1
                    reasoning_parts.append("RSI+")
                elif rsi < 50 and rsi > 30:
                    bearish_signals += 1
                    reasoning_parts.append("RSI-")
                elif rsi >= 70:
                    # Overbought - nötr veya bearish bias
                    bearish_signals += 0.5
                    reasoning_parts.append("RSI_OB")
                elif rsi <= 30:
                    # Oversold - nötr veya bullish bias
                    bullish_signals += 0.5
                    reasoning_parts.append("RSI_OS")

        # === PRICE vs EMA 200 ===
        ema_200_col = self._find_column(df, ["EMA_200", f"EMA_200_{timeframe}", "ema_200", "ema200"])
        close_col = self._find_column(df, ["close", "Close", "CLOSE"])

        if ema_200_col and close_col:
            ema_200 = df[ema_200_col].iloc[-1]
            close = df[close_col].iloc[-1]

            if not pd.isna(ema_200) and not pd.isna(close):
                total_signals += 1
                indicators["ema_200"] = float(ema_200)
                indicators["close"] = float(close)

                if close > ema_200:
                    bullish_signals += 1
                    reasoning_parts.append("P>EMA200")
                else:
                    bearish_signals += 1
                    reasoning_parts.append("P<EMA200")

        # === ADX TREND GÜCÜ ===
        adx_col = self._find_column(df, ["ADX", f"ADX_{timeframe}", "adx", "adx_14"])

        if adx_col:
            adx = df[adx_col].iloc[-1]

            if not pd.isna(adx):
                indicators["adx"] = float(adx)

                if adx < 20:
                    # Zayıf trend - sinyallerin gücünü azalt
                    indicators["trend_strength"] = "weak"
                elif adx > 30:
                    indicators["trend_strength"] = "strong"
                else:
                    indicators["trend_strength"] = "moderate"

        # === YÖN VE GÜÇ HESAPLA ===
        if total_signals == 0:
            return TimeframeSignal(
                timeframe=timeframe,
                direction="neutral",
                strength=0.0,
                indicators=indicators,
                reasoning="No indicators available"
            )

        bullish_ratio = bullish_signals / total_signals
        bearish_ratio = bearish_signals / total_signals

        if bullish_ratio > 0.55:
            direction = "long"
            strength = bullish_ratio
        elif bearish_ratio > 0.55:
            direction = "short"
            strength = bearish_ratio
        else:
            direction = "neutral"
            strength = max(bullish_ratio, bearish_ratio)

        # ADX ile güç modifikasyonu
        trend_str = indicators.get("trend_strength", "moderate")
        if trend_str == "weak":
            strength *= 0.7
        elif trend_str == "strong":
            strength = min(1.0, strength * 1.2)

        return TimeframeSignal(
            timeframe=timeframe,
            direction=direction,
            strength=round(strength, 3),
            indicators=indicators,
            reasoning=" ".join(reasoning_parts)
        )

    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """DataFrame'de mevcut olan ilk sütunu bul."""
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _calculate_weighted_score(
        self,
        signals: List[TimeframeSignal],
        target_direction: str
    ) -> float:
        """
        Ağırlıklı alignment skoru hesapla.

        Daha yüksek timeframe'ler daha fazla ağırlık taşır.
        """
        total_weight = 0.0
        aligned_weight = 0.0

        for signal in signals:
            weight = self.weights.get(signal.timeframe, 0.1)
            total_weight += weight

            if signal.direction == target_direction:
                # Tam alignment
                aligned_weight += weight * signal.strength
            elif signal.direction == "neutral":
                # Nötr = yarım puan
                aligned_weight += weight * 0.5 * signal.strength

        if total_weight == 0:
            return 0.0

        return aligned_weight / total_weight

    def _check_veto(
        self,
        signals: List[TimeframeSignal],
        target_direction: str
    ) -> Optional[str]:
        """
        Veto timeframe'lerini kontrol et.

        Eğer kritik bir timeframe (örn. 4h) trade yönünün karşısındaysa,
        trade veto edilir.
        """
        opposite = "short" if target_direction == "long" else "long"

        for signal in signals:
            if signal.timeframe in self.VETO_TIMEFRAMES:
                if signal.direction == opposite and signal.strength > 0.6:
                    return f"{signal.timeframe} güçlü karşıt sinyal ({signal.direction}, strength={signal.strength:.0%})"

        return None


class MTFAlignmentConfig:
    """MTF Alignment konfigürasyonu yönetimi."""

    CONFIG_FILE = Path(__file__).resolve().parent / "config.json"

    @classmethod
    def load_from_config(cls) -> MTFAlignmentEnforcer:
        """config.json'dan MTF ayarlarını yükle."""
        try:
            if cls.CONFIG_FILE.exists():
                with open(cls.CONFIG_FILE, "r", encoding="utf-8") as f:
                    config = json.load(f)

                mtf_config = config.get("mtf_alignment", {})

                return MTFAlignmentEnforcer(
                    min_alignment_ratio=mtf_config.get("min_alignment_ratio", 0.60),
                    min_aligned_count=mtf_config.get("min_aligned_count", 2),
                    enable_veto=mtf_config.get("enable_veto", True),
                    weights=mtf_config.get("weights")
                )
        except BEST_EFFORT_EXCEPTIONS as e:
            log.warning(f"[MTF] Config yükleme hatası: {e}, default kullanılıyor")

        return MTFAlignmentEnforcer()


# =============================================================================
# ENTEGRASYON FONKSİYONLARI
# =============================================================================

def enforce_mtf_alignment(
    multi_data: Dict[str, pd.DataFrame],
    direction: str,
    min_aligned: int = 2,
    current_price: Optional[float] = None
) -> Dict[str, Any]:
    """
    Basit wrapper fonksiyon - eski kod ile uyumluluk için.

    Args:
        multi_data: Timeframe -> DataFrame mapping
        direction: "long" veya "short"
        min_aligned: Minimum aligned timeframe sayısı
        current_price: Mevcut fiyat

    Returns:
        {
            "aligned": bool,
            "score": float,
            "count": int,
            "details": dict,
            "veto": bool,
            "reasoning": str
        }
    """
    enforcer = MTFAlignmentEnforcer(min_aligned_count=min_aligned)
    result = enforcer.check_alignment(multi_data, direction, current_price)

    # [FAZA 6.4] Graduated confidence multiplier
    grad_mult = enforcer.GRADUATED_CONFIDENCE.get(result.aligned_count, 0.30)

    return {
        "aligned": result.approved,
        "approved": result.approved,
        "score": result.alignment_score,
        "count": result.aligned_count,
        "total": result.total_count,
        "confidence_multiplier": grad_mult,
        "details": {s.timeframe: s.direction for s in result.timeframe_signals},
        "veto": result.veto_reason is not None,
        "veto_reason": result.veto_reason,
        "reasoning": result.reasoning
    }


def get_mtf_enforcer() -> MTFAlignmentEnforcer:
    """Global MTF enforcer instance döndür (config'den yüklenmiş)."""
    return MTFAlignmentConfig.load_from_config()


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MTF ALIGNMENT ENFORCER - TEST")
    print("=" * 60)

    # Örnek veri oluştur
    np.random.seed(42)

    def create_sample_df(bullish: bool = True) -> pd.DataFrame:
        """Test için örnek DataFrame oluştur."""
        n = 100
        base = 50000 if bullish else 50000
        trend = np.linspace(0, 1000 if bullish else -1000, n)
        noise = np.random.randn(n) * 100

        close = base + trend + noise

        df = pd.DataFrame({
            "close": close,
            "EMA_20": pd.Series(close).ewm(span=20).mean(),
            "EMA_50": pd.Series(close).ewm(span=50).mean(),
            "EMA_200": pd.Series(close).ewm(span=200).mean(),
            "RSI": 60 if bullish else 40,
            "MACD_Hist": 50 if bullish else -50,
            "ADX": 35
        })

        return df

    # Test case 1: Tüm TF'ler bullish
    print("\n[TEST 1] Tüm timeframe'ler bullish:")
    multi_data_bullish = {
        "5m": create_sample_df(bullish=True),
        "15m": create_sample_df(bullish=True),
        "1h": create_sample_df(bullish=True),
        "4h": create_sample_df(bullish=True),
    }
    result1 = enforce_mtf_alignment(multi_data_bullish, "long")
    print(f"  Approved: {result1['approved']}")
    print(f"  Score: {result1['score']:.2%}")
    print(f"  Reasoning: {result1['reasoning']}")

    # Test case 2: Mixed signals
    print("\n[TEST 2] Karışık sinyaller:")
    multi_data_mixed = {
        "5m": create_sample_df(bullish=True),
        "15m": create_sample_df(bullish=False),
        "1h": create_sample_df(bullish=True),
        "4h": create_sample_df(bullish=False),  # VETO!
    }
    result2 = enforce_mtf_alignment(multi_data_mixed, "long")
    print(f"  Approved: {result2['approved']}")
    print(f"  Score: {result2['score']:.2%}")
    print(f"  Veto: {result2['veto']}")
    print(f"  Reasoning: {result2['reasoning']}")

    # Test case 3: Short direction
    print("\n[TEST 3] Short yönü, bearish TF'ler:")
    multi_data_bearish = {
        "5m": create_sample_df(bullish=False),
        "15m": create_sample_df(bullish=False),
        "1h": create_sample_df(bullish=False),
        "4h": create_sample_df(bullish=False),
    }
    result3 = enforce_mtf_alignment(multi_data_bearish, "short")
    print(f"  Approved: {result3['approved']}")
    print(f"  Score: {result3['score']:.2%}")
    print(f"  Reasoning: {result3['reasoning']}")

    print("\n" + "=" * 60)
