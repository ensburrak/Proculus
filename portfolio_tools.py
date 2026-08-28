# -*- coding: utf-8 -*-
"""
portfolio_tools.py
==================
Portföy korelasyon analizi ve risk araçları.
Async ve senkron API desteği.
"""
from __future__ import annotations


from core.exceptions import BEST_EFFORT_EXCEPTIONS
import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

try:
    import pandas as pd
    import numpy as np
    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False


def calculate_correlation(
    exchange: Any,
    symbols: List[str],
    timeframe: str = "4h",
    limit: int = 100,
) -> "pd.DataFrame":
    """
    Coin'lerin kapanış fiyatları arasındaki Pearson korelasyon matrisini hesapla.

    Args:
        exchange: ccxt exchange instance
        symbols: Sembol listesi
        timeframe: OHLCV timeframe
        limit: Bar sayısı

    Returns:
        Korelasyon matrisi (DataFrame)
    """
    if not _HAS_DEPS:
        raise ImportError("pandas ve numpy gerekli: pip install pandas numpy")

    price_df = pd.DataFrame()

    for symbol in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            price_df[symbol] = df["close"]
            log.debug("[PORTFOLIO] %s: %d bar yüklendi", symbol, len(df))
        except BEST_EFFORT_EXCEPTIONS as e:
            # [FIX] Hangi sembolün başarısız olduğu loglanıyor
            log.warning("[PORTFOLIO] %s veri çekilemedi: %s", symbol, e)
            continue

    if price_df.empty:
        log.warning("[PORTFOLIO] Hiçbir sembol için veri çekilemedi")
        return pd.DataFrame()

    price_df.dropna(inplace=True)
    correlation_matrix = price_df.corr(method="pearson")

    log.info("[PORTFOLIO] %dx%d korelasyon matrisi hesaplandı", *correlation_matrix.shape)
    return correlation_matrix


async def calculate_correlation_async(
    exchange: Any,
    symbols: List[str],
    timeframe: str = "4h",
    limit: int = 100,
) -> "pd.DataFrame":
    """[FIX] Async korelasyon hesaplama — bot event loop'unu bloke etmez.

    Args: calculate_correlation ile aynı.
    Returns: Korelasyon matrisi (DataFrame)
    """
    import asyncio

    if not _HAS_DEPS:
        raise ImportError("pandas ve numpy gerekli")

    price_df = pd.DataFrame()

    for symbol in symbols:
        try:
            # async ccxt exchange desteği
            if hasattr(exchange, "fetch_ohlcv") and asyncio.iscoroutinefunction(exchange.fetch_ohlcv):
                ohlcv = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            else:
                ohlcv = await asyncio.to_thread(
                    exchange.fetch_ohlcv, symbol, timeframe, None, limit
                )
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            price_df[symbol] = df["close"]
            log.debug("[PORTFOLIO-ASYNC] %s: %d bar", symbol, len(df))
        except BEST_EFFORT_EXCEPTIONS as e:
            log.warning("[PORTFOLIO-ASYNC] %s veri çekilemedi: %s", symbol, e)
            continue

    if price_df.empty:
        return pd.DataFrame()

    price_df.dropna(inplace=True)
    return price_df.corr(method="pearson")


def get_high_correlation_pairs(
    corr_matrix: "pd.DataFrame",
    threshold: float = 0.7,
) -> List[Dict[str, Any]]:
    """[FIX] Yüksek korelasyon çiftlerini bul — risk yönetimi için.

    Returns:
        [{"pair": ("BTC/USDT", "ETH/USDT"), "correlation": 0.85}, ...]
    """
    if not _HAS_DEPS or corr_matrix.empty:
        return []

    pairs = []
    symbols = corr_matrix.columns.tolist()
    for i, s1 in enumerate(symbols):
        for j, s2 in enumerate(symbols):
            if j <= i:
                continue
            corr = corr_matrix.loc[s1, s2]
            if abs(corr) >= threshold:
                pairs.append({"pair": (s1, s2), "correlation": round(float(corr), 4)})

    pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    log.info("[PORTFOLIO] %d yüksek korelasyon çifti bulundu (>%.2f)", len(pairs), threshold)
    return pairs
