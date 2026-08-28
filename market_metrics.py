# -*- coding: utf-8 -*-
"""
market_metrics.py - Professional Market Metrics Provider
=========================================================

[2026-01-15] Profesyonel seviye piyasa metrikleri.

Sağlanan Metrikler:
1. Funding Rate (OKX API)
2. Open Interest Delta
3. CVD (Cumulative Volume Delta)
4. Long/Short Ratio
5. Exchange Netflow (cache-based)

Kullanım:
    from market_metrics import MarketMetricsProvider
    
    provider = MarketMetricsProvider(exchange)
    metrics = await provider.get_all_metrics("BTC/USDT")
"""

from __future__ import annotations


from core.exceptions import BEST_EFFORT_EXCEPTIONS
import asyncio
import inspect
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple
import logging

try:
    from state_manager import get_metric
except BEST_EFFORT_EXCEPTIONS:
    def get_metric(key: str, default=None):  # type: ignore
        return default

try:
    from ops_data_provider import fetch_options_metrics
except BEST_EFFORT_EXCEPTIONS:
    fetch_options_metrics = None  # type: ignore

try:
    from liquidation_data_provider import fetch_liquidation_heatmap
except BEST_EFFORT_EXCEPTIONS:
    fetch_liquidation_heatmap = None  # type: ignore

# Logger
try:
    from logger import get_logger
    log = get_logger("market_metrics")
except ImportError:
    log = logging.getLogger("market_metrics")
    log.setLevel(logging.INFO)

# Data processing
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None
    np = None


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FundingMetrics:
    """Funding rate metrics."""
    current_rate: float = 0.0  # Current funding rate
    predicted_rate: float = 0.0  # Predicted next funding
    avg_8h_rate: float = 0.0  # 8h average
    avg_24h_rate: float = 0.0  # 24h average
    signal: str = "neutral"  # bullish/bearish/neutral
    strength: float = 0.0  # 0-1


@dataclass
class OpenInterestMetrics:
    """Open Interest metrics."""
    current_oi: float = 0.0
    oi_change_1h: float = 0.0  # % change
    oi_change_4h: float = 0.0
    oi_change_24h: float = 0.0
    oi_trend: str = "stable"  # rising/falling/stable
    price_oi_divergence: bool = False  # OI vs price divergence


@dataclass
class CVDMetrics:
    """Cumulative Volume Delta metrics."""
    cvd_1h: float = 0.0
    cvd_4h: float = 0.0
    cvd_24h: float = 0.0
    cvd_trend: str = "neutral"  # buying/selling/neutral
    cvd_strength: float = 0.0  # 0-1


@dataclass
class LongShortMetrics:
    """Long/Short ratio metrics."""
    ratio: float = 1.0  # >1 = more longs
    top_trader_ratio: float = 1.0
    signal: str = "neutral"  # contrarian signals
    crowd_position: str = "balanced"  # long_heavy/short_heavy/balanced


@dataclass
class MarketMetrics:
    """Combined market metrics."""
    symbol: str
    timestamp: str
    
    funding: FundingMetrics = field(default_factory=FundingMetrics)
    open_interest: OpenInterestMetrics = field(default_factory=OpenInterestMetrics)
    cvd: CVDMetrics = field(default_factory=CVDMetrics)
    long_short: LongShortMetrics = field(default_factory=LongShortMetrics)
    
    # Combined signals
    overall_sentiment: str = "neutral"  # bullish/bearish/neutral
    sentiment_score: float = 0.5  # 0=very bearish, 1=very bullish
    confidence: float = 0.5
    volume_24h: float = 0.0
    basis_bps: float = 0.0
    liquidation_pressure: float = 0.0
    oi_acceleration: float = 0.0
    cross_exchange_spread_pct: float = 0.0
    cross_exchange_confirmed: bool = False
    cross_exchange_status: str = "unknown"
    options_iv: float = 0.0
    metric_status: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "funding": {
                "current_rate": self.funding.current_rate,
                "signal": self.funding.signal,
                "strength": self.funding.strength
            },
            "open_interest": {
                "current_oi": self.open_interest.current_oi,
                "oi_change_24h": self.open_interest.oi_change_24h,
                "oi_trend": self.open_interest.oi_trend
            },
            "cvd": {
                "cvd_24h": self.cvd.cvd_24h,
                "cvd_trend": self.cvd.cvd_trend
            },
            "long_short": {
                "ratio": self.long_short.ratio,
                "signal": self.long_short.signal
            },
            "overall_sentiment": self.overall_sentiment,
            "sentiment_score": self.sentiment_score,
            "volume_24h": self.volume_24h,
            "basis_bps": self.basis_bps,
            "liquidation_pressure": self.liquidation_pressure,
            "oi_acceleration": self.oi_acceleration,
            "cross_exchange_spread_pct": self.cross_exchange_spread_pct,
            "cross_exchange_confirmed": self.cross_exchange_confirmed,
            "cross_exchange_status": self.cross_exchange_status,
            "options_iv": self.options_iv,
            "metric_status": dict(self.metric_status),
        }


# =============================================================================
# MARKET METRICS PROVIDER
# =============================================================================

class MarketMetricsProvider:
    """
    Professional market metrics provider.
    
    Fetches and analyzes:
    - Funding rates
    - Open Interest
    - CVD (from OHLCV)
    - Long/Short ratios
    """
    
    # Cache settings
    CACHE_TTL = 60  # 1 minute
    METRIC_TIMEOUT_SECONDS = 2.0
    METRIC_STALE_TTL_SECONDS = 300.0
    
    def __init__(
        self,
        exchange: Any = None,
        *,
        metric_timeout_seconds: float | None = None,
        metric_stale_ttl_seconds: float | None = None,
        time_fn: Callable[[], float] | None = None,
    ):
        self.exchange = exchange
        self._cache: Dict[str, Tuple[float, MarketMetrics]] = {}
        self._metric_cache: Dict[Tuple[str, str], Tuple[float, Any]] = {}
        self._metric_status: Dict[str, str] = {}
        self._oi_history: Dict[str, List[Tuple[float, float]]] = {}
        self._time_fn = time_fn or time.time
        self._metric_timeout_seconds = self._positive_float(
            metric_timeout_seconds,
            env_key="MARKET_METRICS_TIMEOUT_SECONDS",
            default=self.METRIC_TIMEOUT_SECONDS,
            minimum=0.05,
        )
        self._metric_stale_ttl_seconds = self._positive_float(
            metric_stale_ttl_seconds,
            env_key="MARKET_METRICS_STALE_TTL_SECONDS",
            default=self.METRIC_STALE_TTL_SECONDS,
            minimum=1.0,
        )

    @staticmethod
    def _positive_float(value: Any, *, env_key: str, default: float, minimum: float) -> float:
        candidate = value
        if candidate is None:
            candidate = os.getenv(env_key)
        try:
            parsed = float(candidate)
        except (TypeError, ValueError, OverflowError):
            parsed = float(default)
        return max(float(minimum), parsed)

    async def _await_maybe(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _cached_metric_value(self, symbol: str, metric_name: str) -> Any | None:
        cached = self._metric_cache.get((symbol, metric_name))
        if cached is None:
            return None
        ts, value = cached
        if self._time_fn() - ts <= self._metric_stale_ttl_seconds:
            return value
        return None

    async def _metric_with_timeout(
        self,
        symbol: str,
        metric_name: str,
        producer: Callable[[], Awaitable[Any]],
        fallback: Any,
        *,
        status: Dict[str, str] | None = None,
    ) -> Any:
        metric_status = status if status is not None else self._metric_status
        try:
            value = await asyncio.wait_for(producer(), timeout=self._metric_timeout_seconds)
        except asyncio.TimeoutError:
            cached = self._cached_metric_value(symbol, metric_name)
            if cached is not None:
                metric_status[metric_name] = "stale_cache_used"
                return cached
            metric_status[metric_name] = "metric_timeout"
            return fallback
        except BEST_EFFORT_EXCEPTIONS:
            cached = self._cached_metric_value(symbol, metric_name)
            if cached is not None:
                metric_status[metric_name] = "stale_cache_used_after_error"
                return cached
            metric_status[metric_name] = "metric_error"
            return fallback
        self._metric_cache[(symbol, metric_name)] = (self._time_fn(), value)
        metric_status[metric_name] = "ok"
        return value

    def _record_oi_snapshot(self, symbol: str, current_oi: float) -> Tuple[float, float, float]:
        now_ts = time.time()
        history = self._oi_history.setdefault(symbol, [])
        history.append((now_ts, current_oi))
        cutoff_24h = now_ts - 86400
        history[:] = [(ts, val) for ts, val in history if ts >= cutoff_24h]

        def _pct_change(window_sec: int) -> float:
            baseline = next((val for ts, val in history if ts >= now_ts - window_sec), None)
            if baseline is None or baseline <= 0:
                return 0.0
            return ((current_oi - baseline) / baseline) * 100.0

        return _pct_change(3600), _pct_change(4 * 3600), _pct_change(24 * 3600)

    async def _get_volume_24h(self, symbol: str) -> float:
        if not self.exchange:
            return 0.0
        try:
            ticker = await self._await_maybe(self.exchange.fetch_ticker(symbol))
            if isinstance(ticker, dict):
                quote_volume = float(ticker.get("quoteVolume") or 0.0)
                if quote_volume > 0:
                    return quote_volume
                base_volume = float(ticker.get("baseVolume") or ticker.get("volume") or 0.0)
                last_price = float(ticker.get("last") or ticker.get("close") or 0.0)
                if base_volume > 0 and last_price > 0:
                    return base_volume * last_price
        except BEST_EFFORT_EXCEPTIONS:
            pass
        return 0.0

    async def _get_basis_bps(self, symbol: str, funding: FundingMetrics) -> float:
        if not self.exchange:
            return float((funding.current_rate or 0.0) * 30000.0)
        base = (symbol or "").split("/")[0].upper()
        quote = "USDT"
        spot_symbol = f"{base}/{quote}"
        try:
            perp_ticker = await self._await_maybe(self.exchange.fetch_ticker(symbol))
            spot_ticker = await self._await_maybe(self.exchange.fetch_ticker(spot_symbol))
            perp_last = float((perp_ticker or {}).get("last") or (perp_ticker or {}).get("close") or 0.0)
            spot_last = float((spot_ticker or {}).get("last") or (spot_ticker or {}).get("close") or 0.0)
            if perp_last > 0 and spot_last > 0:
                return ((perp_last - spot_last) / spot_last) * 10000.0
        except BEST_EFFORT_EXCEPTIONS:
            pass
        return float((funding.current_rate or 0.0) * 30000.0)

    async def _get_cross_exchange_parity(self, symbol: str) -> Tuple[float, bool, str]:
        if not self.exchange:
            return 0.0, False, "primary_unavailable"
        try:
            from analyzer import get_binance_async, normalize_symbol_for_binance_fallback, binance_supports_symbol

            primary_ticker = await self._await_maybe(self.exchange.fetch_ticker(symbol))
            primary_last = float((primary_ticker or {}).get("last") or (primary_ticker or {}).get("close") or 0.0)
            if primary_last <= 0:
                return 0.0, False, "primary_missing"

            binance = await get_binance_async()
            if not binance:
                return 0.0, False, "peer_unavailable"
            binance_symbol = normalize_symbol_for_binance_fallback(symbol)
            if not binance_supports_symbol(getattr(binance, "markets", None), binance_symbol):
                return 0.0, False, "peer_unsupported"
            peer_ticker = await self._await_maybe(binance.fetch_ticker(binance_symbol))
            peer_last = float((peer_ticker or {}).get("last") or (peer_ticker or {}).get("close") or 0.0)
            if peer_last <= 0:
                return 0.0, False, "peer_missing"
            spread_pct = abs(primary_last - peer_last) / max((primary_last + peer_last) / 2.0, 1e-9)
            confirmed = bool(spread_pct <= 0.003)
            return float(spread_pct), confirmed, "confirmed" if confirmed else "dislocated"
        except BEST_EFFORT_EXCEPTIONS:
            return 0.0, False, "error"

    async def _get_options_iv(self, symbol: str) -> float:
        live = get_metric("options_metrics")
        base = (symbol or "").split("/")[0].upper()
        if isinstance(live, dict):
            candidate = live.get(base) or live.get(symbol) or live
            if isinstance(candidate, dict):
                iv = candidate.get("iv_index", candidate.get("implied_volatility"))
                try:
                    return float(iv or 0.0)
                except BEST_EFFORT_EXCEPTIONS:
                    pass
        if fetch_options_metrics is None:
            return 0.0
        try:
            payload = await asyncio.to_thread(fetch_options_metrics, base)
            if isinstance(payload, dict):
                return float(payload.get("iv_index") or payload.get("implied_volatility") or 0.0)
        except BEST_EFFORT_EXCEPTIONS:
            pass
        return 0.0

    async def _get_liquidation_pressure(self, symbol: str, volume_24h: float) -> float:
        live = get_metric("liquidation_intensity")
        base = (symbol or "").split("/")[0].upper()
        if isinstance(live, dict):
            candidate = live.get(base) or live.get(symbol)
            if isinstance(candidate, dict):
                raw = candidate.get("pressure", candidate.get("intensity", candidate.get("score")))
            else:
                raw = candidate
            try:
                return max(0.0, min(1.0, float(raw or 0.0)))
            except BEST_EFFORT_EXCEPTIONS:
                pass
        if fetch_liquidation_heatmap is None:
            return 0.0
        try:
            heatmap = await asyncio.to_thread(fetch_liquidation_heatmap, base)
            if isinstance(heatmap, dict) and heatmap:
                total = 0.0
                for value in heatmap.values():
                    try:
                        total += abs(float(value or 0.0))
                    except BEST_EFFORT_EXCEPTIONS:
                        continue
                denom = max(float(volume_24h or 0.0), 1_000_000.0)
                return max(0.0, min(1.0, (total / denom) * 25.0))
        except BEST_EFFORT_EXCEPTIONS:
            pass
        return 0.0

    def _calculate_oi_acceleration(self, oi: OpenInterestMetrics) -> float:
        fast = float(oi.oi_change_1h or 0.0)
        slow = float(oi.oi_change_4h or 0.0) / 4.0
        return fast - slow
    
    async def get_all_metrics(self, symbol: str) -> MarketMetrics:
        """Get all metrics for a symbol."""
        # Check cache
        cache_key = symbol
        if cache_key in self._cache:
            cache_time, cached_metrics = self._cache[cache_key]
            if time.time() - cache_time < self.CACHE_TTL:
                return cached_metrics
        
        # Fetch all metrics with bounded latency. Optional market metrics may
        # fall back to the last healthy value, but the analysis loop must not
        # wait indefinitely for an exchange/API provider.
        metric_status: Dict[str, str] = {}
        funding = await self._metric_with_timeout(symbol, "funding", lambda: self._get_funding_metrics(symbol), FundingMetrics(), status=metric_status)
        oi = await self._metric_with_timeout(symbol, "open_interest", lambda: self._get_oi_metrics(symbol), OpenInterestMetrics(), status=metric_status)
        cvd = await self._metric_with_timeout(symbol, "cvd", lambda: self._get_cvd_metrics(symbol), CVDMetrics(), status=metric_status)
        ls = await self._metric_with_timeout(symbol, "long_short", lambda: self._get_long_short_metrics(symbol), LongShortMetrics(), status=metric_status)
        volume_24h = await self._metric_with_timeout(symbol, "volume_24h", lambda: self._get_volume_24h(symbol), 0.0, status=metric_status)
        basis_bps = await self._metric_with_timeout(symbol, "basis_bps", lambda: self._get_basis_bps(symbol, funding), 0.0, status=metric_status)
        cross_exchange_spread_pct, cross_exchange_confirmed, cross_exchange_status = await self._metric_with_timeout(
            symbol,
            "cross_exchange",
            lambda: self._get_cross_exchange_parity(symbol),
            (0.0, False, "metric_timeout"),
            status=metric_status,
        )
        liquidation_pressure = await self._metric_with_timeout(
            symbol,
            "liquidation_pressure",
            lambda: self._get_liquidation_pressure(symbol, volume_24h),
            0.0,
            status=metric_status,
        )
        oi_acceleration = self._calculate_oi_acceleration(oi)
        options_iv = await self._metric_with_timeout(symbol, "options_iv", lambda: self._get_options_iv(symbol), 0.0, status=metric_status)
        
        # Calculate combined sentiment
        sentiment_score, overall_sentiment = self._calculate_combined_sentiment(
            funding, oi, cvd, ls
        )
        
        metrics = MarketMetrics(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            funding=funding,
            open_interest=oi,
            cvd=cvd,
            long_short=ls,
            overall_sentiment=overall_sentiment,
            sentiment_score=sentiment_score,
            confidence=0.7 if self.exchange else 0.3,
            volume_24h=volume_24h,
            basis_bps=basis_bps,
            liquidation_pressure=liquidation_pressure,
            oi_acceleration=oi_acceleration,
            cross_exchange_spread_pct=cross_exchange_spread_pct,
            cross_exchange_confirmed=cross_exchange_confirmed,
            cross_exchange_status=cross_exchange_status,
            options_iv=options_iv,
            metric_status=dict(metric_status),
        )
        
        # Cache
        self._cache[cache_key] = (time.time(), metrics)
        
        return metrics
    
    async def _get_funding_metrics(self, symbol: str) -> FundingMetrics:
        """Fetch funding rate from exchange."""
        try:
            if not self.exchange:
                return FundingMetrics()
            
            # OKX funding rate
            # Convert symbol format: BTC/USDT → BTC-USDT-SWAP
            okx_symbol = symbol.replace("/", "-") + "-SWAP"
            
            try:
                # Try ccxt unified method first
                if hasattr(self.exchange, 'fetch_funding_rate'):
                    funding_data = await self._await_maybe(self.exchange.fetch_funding_rate(symbol))
                    current_rate = float(funding_data.get('fundingRate', 0))
                else:
                    # Fallback: OKX public API
                    response = await self._await_maybe(self.exchange.public_get_public_funding_rate({
                        'instId': okx_symbol
                    }))
                    data = response.get('data', [{}])[0]
                    current_rate = float(data.get('fundingRate', 0))
            except BEST_EFFORT_EXCEPTIONS:
                current_rate = 0.0
            
            # Determine signal
            # Positive funding = longs pay shorts = bearish bias (crowded long)
            # Negative funding = shorts pay longs = bullish bias (crowded short)
            if current_rate > 0.0005:  # >0.05% = very high
                signal = "bearish"
                strength = min(1.0, current_rate / 0.001)
            elif current_rate > 0.0001:  # >0.01%
                signal = "slightly_bearish"
                strength = current_rate / 0.0005
            elif current_rate < -0.0005:
                signal = "bullish"
                strength = min(1.0, abs(current_rate) / 0.001)
            elif current_rate < -0.0001:
                signal = "slightly_bullish"
                strength = abs(current_rate) / 0.0005
            else:
                signal = "neutral"
                strength = 0.0
            
            return FundingMetrics(
                current_rate=current_rate,
                signal=signal,
                strength=strength
            )
            
        except BEST_EFFORT_EXCEPTIONS as e:
            log.warning(f"Funding rate fetch error for {symbol}: {e}")
            return FundingMetrics()
    
    async def _get_oi_metrics(self, symbol: str) -> OpenInterestMetrics:
        """Fetch Open Interest from exchange."""
        try:
            if not self.exchange:
                return OpenInterestMetrics()
            
            okx_symbol = symbol.replace("/", "-") + "-SWAP"
            
            try:
                if hasattr(self.exchange, 'fetch_open_interest'):
                    oi_data = await self._await_maybe(self.exchange.fetch_open_interest(symbol))
                    current_oi = float(oi_data.get('openInterest', 0))
                else:
                    # OKX API
                    response = await self._await_maybe(self.exchange.public_get_public_open_interest({
                        'instType': 'SWAP',
                        'instId': okx_symbol
                    }))
                    data = response.get('data', [{}])[0]
                    current_oi = float(data.get('oi', 0))
            except BEST_EFFORT_EXCEPTIONS:
                current_oi = 0.0
            
            oi_change_1h, oi_change_4h, oi_change_24h = self._record_oi_snapshot(symbol, current_oi)
            if oi_change_24h > 5.0:
                oi_trend = "rising"
            elif oi_change_24h < -5.0:
                oi_trend = "falling"
            else:
                oi_trend = "stable"
            return OpenInterestMetrics(
                current_oi=current_oi,
                oi_change_1h=oi_change_1h,
                oi_change_4h=oi_change_4h,
                oi_change_24h=oi_change_24h,
                oi_trend=oi_trend,
            )
            
        except BEST_EFFORT_EXCEPTIONS as e:
            log.warning(f"OI fetch error for {symbol}: {e}")
            return OpenInterestMetrics()
    
    async def _get_cvd_metrics(self, symbol: str) -> CVDMetrics:
        """Calculate CVD from OHLCV data."""
        try:
            if not self.exchange or not PANDAS_AVAILABLE:
                return CVDMetrics()
            
            # Fetch 1h OHLCV for 24h of data
            ohlcv = await self._await_maybe(self.exchange.fetch_ohlcv(symbol, timeframe='1h', limit=24))
            if not ohlcv or len(ohlcv) < 10:
                return CVDMetrics()
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Calculate CVD approximation
            # Delta = volume * (close - open) / (high - low)
            df['range'] = df['high'] - df['low']
            df['range'] = df['range'].replace(0, 0.0001)  # Avoid division by zero
            df['delta'] = df['volume'] * (df['close'] - df['open']) / df['range']
            df['cvd'] = df['delta'].cumsum()
            
            # Get CVD values
            cvd_1h = df['delta'].iloc[-1] if len(df) >= 1 else 0
            cvd_4h = df['delta'].iloc[-4:].sum() if len(df) >= 4 else 0
            cvd_24h = df['delta'].sum()
            
            # Normalize to -1 to +1
            max_cvd = abs(df['delta']).max() * 24 if abs(df['delta']).max() > 0 else 1
            cvd_normalized = cvd_24h / max_cvd if max_cvd > 0 else 0
            
            # Determine trend
            if cvd_normalized > 0.3:
                cvd_trend = "buying"
                strength = min(1.0, cvd_normalized)
            elif cvd_normalized < -0.3:
                cvd_trend = "selling"
                strength = min(1.0, abs(cvd_normalized))
            else:
                cvd_trend = "neutral"
                strength = abs(cvd_normalized)
            
            return CVDMetrics(
                cvd_1h=cvd_1h,
                cvd_4h=cvd_4h,
                cvd_24h=cvd_24h,
                cvd_trend=cvd_trend,
                cvd_strength=strength
            )
            
        except BEST_EFFORT_EXCEPTIONS as e:
            log.warning(f"CVD calculation error for {symbol}: {e}")
            return CVDMetrics()
    
    async def _get_long_short_metrics(self, symbol: str) -> LongShortMetrics:
        """Fetch Long/Short ratio from exchange."""
        try:
            if not self.exchange:
                return LongShortMetrics()
            
            # OKX long/short ratio endpoint
            try:
                # This may not be available in all ccxt versions
                # Try OKX specific endpoint
                response = await self._await_maybe(self.exchange.public_get_public_long_short_ratio({
                    'instId': symbol.replace("/", "-"),
                    'period': '1H'
                }))
                data = response.get('data', [{}])[0]
                ratio = float(data.get('longShortRatio', 1.0))
            except BEST_EFFORT_EXCEPTIONS:
                ratio = 1.0  # Default balanced
            
            # Determine signal (contrarian)
            # >1.5 = too many longs = bearish signal
            # <0.67 = too many shorts = bullish signal
            if ratio > 2.0:
                signal = "bearish"  # Crowded long
                crowd = "long_heavy"
            elif ratio > 1.3:
                signal = "slightly_bearish"
                crowd = "long_heavy"
            elif ratio < 0.5:
                signal = "bullish"  # Crowded short
                crowd = "short_heavy"
            elif ratio < 0.77:
                signal = "slightly_bullish"
                crowd = "short_heavy"
            else:
                signal = "neutral"
                crowd = "balanced"
            
            return LongShortMetrics(
                ratio=ratio,
                signal=signal,
                crowd_position=crowd
            )
            
        except BEST_EFFORT_EXCEPTIONS as e:
            log.warning(f"Long/Short ratio fetch error for {symbol}: {e}")
            return LongShortMetrics()
    
    def _calculate_combined_sentiment(
        self,
        funding: FundingMetrics,
        oi: OpenInterestMetrics,
        cvd: CVDMetrics,
        ls: LongShortMetrics
    ) -> Tuple[float, str]:
        """Calculate combined sentiment from all metrics."""
        scores = []
        weights = []
        
        # Funding signal (weight: 0.30)
        funding_score_map = {
            "bullish": 0.8,
            "slightly_bullish": 0.6,
            "neutral": 0.5,
            "slightly_bearish": 0.4,
            "bearish": 0.2
        }
        if funding.signal in funding_score_map:
            scores.append(funding_score_map[funding.signal])
            weights.append(0.30)
        
        # CVD trend (weight: 0.30)
        cvd_score_map = {
            "buying": 0.7 + 0.2 * cvd.cvd_strength,
            "neutral": 0.5,
            "selling": 0.3 - 0.2 * cvd.cvd_strength
        }
        if cvd.cvd_trend in cvd_score_map:
            scores.append(cvd_score_map[cvd.cvd_trend])
            weights.append(0.30)
        
        # Long/Short ratio (weight: 0.25) - contrarian
        ls_score_map = {
            "bullish": 0.75,
            "slightly_bullish": 0.6,
            "neutral": 0.5,
            "slightly_bearish": 0.4,
            "bearish": 0.25
        }
        if ls.signal in ls_score_map:
            scores.append(ls_score_map[ls.signal])
            weights.append(0.25)
        
        # OI trend (weight: 0.15)
        oi_score_map = {
            "rising": 0.6,  # Can be bullish or bearish depending on price
            "stable": 0.5,
            "falling": 0.4
        }
        if oi.oi_trend in oi_score_map:
            scores.append(oi_score_map[oi.oi_trend])
            weights.append(0.15)
        
        # Calculate weighted average
        if scores and weights:
            total_weight = sum(weights)
            sentiment_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        else:
            sentiment_score = 0.5
        
        # Determine overall sentiment
        if sentiment_score > 0.65:
            overall = "bullish"
        elif sentiment_score > 0.55:
            overall = "slightly_bullish"
        elif sentiment_score < 0.35:
            overall = "bearish"
        elif sentiment_score < 0.45:
            overall = "slightly_bearish"
        else:
            overall = "neutral"
        
        return sentiment_score, overall


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_provider_instance: Optional[MarketMetricsProvider] = None
_provider_instance_lock = None


def get_provider(exchange: Any = None) -> MarketMetricsProvider:
    """Get global provider instance."""
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = MarketMetricsProvider(exchange)
    elif exchange is not None:
        _provider_instance.exchange = exchange
    return _provider_instance


def clear_metrics_cache(exchange: Any = None) -> int:
    """Clear cached market metrics and return the number of evicted entries."""
    provider = get_provider(exchange)
    cleared = len(provider._cache) + len(provider._metric_cache)
    provider._cache.clear()
    provider._metric_cache.clear()
    return cleared


async def get_market_metrics(symbol: str, exchange: Any = None) -> MarketMetrics:
    """Convenience function to get metrics."""
    provider = get_provider(exchange)
    return await provider.get_all_metrics(symbol)


def get_funding_signal(symbol: str, exchange: Any = None) -> Tuple[str, float]:
    """Sync wrapper for funding signal."""
    try:
        import asyncio
        provider = get_provider(exchange)
        metrics = asyncio.run(provider.get_all_metrics(symbol))
        return metrics.funding.signal, metrics.funding.strength
    except BEST_EFFORT_EXCEPTIONS:
        return "neutral", 0.0


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Market Metrics Provider Test")
    print("=" * 50)
    
    # Test without exchange
    provider = MarketMetricsProvider()
    
    async def test():
        metrics = await provider.get_all_metrics("BTC/USDT")
        print(f"Metrics: {metrics.to_dict()}")
    
    asyncio.run(test())
