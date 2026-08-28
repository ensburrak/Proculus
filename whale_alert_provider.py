# -*- coding: utf-8 -*-
"""
whale_alert_provider.py
=======================

[2026-01-16 PROFESSIONAL FIX #48 - V2]

Profesyonel Büyük İşlem (Whale Trade) Provider.

OKX ve Binance API'leri ile:
1. Büyük işlemleri tespit et
2. Alış/Satış dengesini analiz et
3. Trade kararlarını güçlendir

Etherscan kullanmıyoruz - mevcut exchange API'lerini kullanıyoruz.

Kullanım:
    from whale_alert_provider import get_whale_signal, adjust_confidence_with_whale_data

    signal, strength = get_whale_signal("BTC/USDT", exchange)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.adaptive_backoff import AdaptiveBackoffState
from core.exceptions import BEST_EFFORT_EXCEPTIONS
from runtime_paths import METRICS_DIR

try:
    import aiohttp
except ImportError:
    aiohttp = None

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

WHALE_CACHE_FILE = METRICS_DIR / "whale_trades_cache.json"

# Binance API
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# Thresholds
MIN_WHALE_TRADE_USD = 100000  # $100K minimum for "whale trade" (fallback)
CACHE_TTL = 60  # 1 minute
ADAPTIVE_CACHE_TTL = CACHE_TTL
_HTTP_BACKOFF_UNTIL = 0.0
_HTTP_BACKOFF_STATE = AdaptiveBackoffState(base_ttl=CACHE_TTL, max_delay=30.0)
WHALE_TIME_WINDOW_SECONDS = 60  # [FIX] Son 60 saniyeyi tara (eski: ~3-5 saniye)
WHALE_CONFIRMATION_DELAY_SEC = 1800  # [FAZA 8.3d] 30 dakika teyit bekleme

# [FAZA 8.3d] Whale movement timestamp tracker (symbol -> last_whale_ts)
_whale_movement_timestamps: Dict[str, float] = {}


@dataclass
class _HTTPResult:
    status_code: int
    headers: Dict[str, str]
    payload: Any

    def json(self) -> Any:
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _sync_legacy_backoff_globals() -> None:
    global ADAPTIVE_CACHE_TTL, _HTTP_BACKOFF_UNTIL
    snapshot = _HTTP_BACKOFF_STATE.snapshot()
    ADAPTIVE_CACHE_TTL = int(snapshot["adaptive_cache_ttl"])
    _HTTP_BACKOFF_UNTIL = float(snapshot["backoff_until"])


async def _http_get_with_backoff(url: str, **kwargs: Any) -> _HTTPResult | None:
    """Async GET helper with thread-safe adaptive 401/429/5xx backoff."""
    if aiohttp is None:
        return None
    timeout = kwargs.pop("timeout", 10)
    if _HTTP_BACKOFF_STATE.should_skip():
        _sync_legacy_backoff_globals()
        return None
    for attempt in range(3):
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=timeout, **kwargs) as response:
                status = int(getattr(response, "status", getattr(response, "status_code", 0)) or 0)
                headers = dict(getattr(response, "headers", {}) or {})
                payload = await response.json()
        decision = _HTTP_BACKOFF_STATE.record_response(status, headers, attempt=attempt)
        _sync_legacy_backoff_globals()
        result = _HTTPResult(status_code=status, headers=headers, payload=payload)
        if decision.should_backoff:
            if attempt < 2:
                await asyncio.sleep(decision.delay_seconds)
                continue
        return result
    return result


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class WhaleTrade:
    """Single large trade."""

    timestamp: str
    symbol: str
    side: str  # "buy" or "sell"
    price: float
    quantity: float
    amount_usd: float
    is_maker: bool  # Market maker or taker
    transfer_type: str = "on_exchange"  # [FAZA 8.3] "to_exchange", "from_exchange", "on_exchange"


@dataclass
class WhaleMetrics:
    """Aggregated whale metrics for a symbol."""

    symbol: str
    total_buy_volume_usd: float
    total_sell_volume_usd: float
    net_flow_usd: float  # Buy - Sell (positive = buying pressure)
    whale_trade_count: int
    buy_count: int
    sell_count: int
    signal: str  # "bullish", "bearish", "neutral"
    strength: float  # 0-1
    last_update: str


# =============================================================================
# BINANCE LARGE TRADES PROVIDER
# =============================================================================


async def fetch_binance_large_trades(
    symbol: str,
    limit: int = 1000,
    min_usd: float = MIN_WHALE_TRADE_USD,
    current_price: float = 0.0,
) -> List[WhaleTrade]:
    """
    [FIX] Son 60 saniyeyi kapsayan balina işlem tarayıcı.

    Eski: limit=500 → son ~3-5 saniye (BTC gibi yüksek hacimli piyasalarda).
    Yeni: aggTrades + startTime ile son 60 saniyeyi tarar.

    Ayrıca minimum balina eşiği dinamik:
    - current_price > 0 → eşik = current_price × 1.0 BTC ($100K BTC → min $100K)
    - current_price = 0 → fallback $100K
    """
    if aiohttp is None:
        return []

    trades = []

    # [FIX] Dinamik minimum balina eşiği
    if current_price > 0:
        dynamic_min = current_price * 1.0  # En az 1 BTC'lik işlem
        min_usd = max(min_usd, dynamic_min)

    try:
        binance_symbol = symbol.replace("/", "").upper()

        # [FIX] aggTrades + startTime ile son 60 saniyeyi tara
        import time as _time

        start_time_ms = int((_time.time() - WHALE_TIME_WINDOW_SECONDS) * 1000)
        url = (
            f"https://fapi.binance.com/fapi/v1/aggTrades"
            f"?symbol={binance_symbol}&startTime={start_time_ms}&limit={limit}"
        )

        response = await _http_get_with_backoff(url, timeout=10)
        if response is None:
            return []
        if response.status_code != 200:
            logger.debug(f"Binance aggTrades API error: {response.status_code}")
            return []

        data = response.json()

        for trade in data:
            price = float(trade.get("p", 0))
            qty = float(trade.get("q", 0))
            amount_usd = price * qty

            if amount_usd < min_usd:
                continue

            is_buyer_maker = trade.get("m", False)
            side = "sell" if is_buyer_maker else "buy"

            trades.append(
                WhaleTrade(
                    timestamp=datetime.fromtimestamp(
                        trade.get("T", 0) / 1000,
                        tz=timezone.utc,
                    )
                    .isoformat()
                    .replace("+00:00", "Z"),
                    symbol=symbol,
                    side=side,
                    price=price,
                    quantity=qty,
                    amount_usd=amount_usd,
                    is_maker=is_buyer_maker,
                )
            )

        logger.debug(f"[WHALE] Found {len(trades)} large trades in last {WHALE_TIME_WINDOW_SECONDS}s for {symbol}")

    except BEST_EFFORT_EXCEPTIONS as e:
        logger.warning(f"Binance large trades fetch error: {e}")

    return trades


async def fetch_binance_aggregated_trades(
    symbol: str,
    hours: int = 1,
) -> Dict:
    """
    Get aggregated trade statistics from Binance.
    """
    if aiohttp is None:
        return {}

    try:
        binance_symbol = symbol.replace("/", "").upper()

        # Get 24h ticker for quick stats
        url = f"https://fapi.binance.com/fapi/v1/ticker/24hr?symbol={binance_symbol}"

        response = await _http_get_with_backoff(url, timeout=10)
        if response is None:
            return {}
        if response.status_code != 200:
            return {}

        data = response.json()

        return {
            "volume_24h": float(data.get("volume", 0)),
            "quote_volume_24h": float(data.get("quoteVolume", 0)),
            "price_change_pct": float(data.get("priceChangePercent", 0)),
            "high_24h": float(data.get("highPrice", 0)),
            "low_24h": float(data.get("lowPrice", 0)),
            "trade_count_24h": int(data.get("count", 0)),
        }

    except BEST_EFFORT_EXCEPTIONS as e:
        logger.debug(f"Binance ticker error: {e}")
        return {}


# =============================================================================
# OKX LARGE TRADES PROVIDER
# =============================================================================


async def fetch_okx_large_trades(
    exchange: Any,
    symbol: str,
    limit: int = 100,
    min_usd: float = MIN_WHALE_TRADE_USD,
) -> List[WhaleTrade]:
    """
    Fetch recent trades from OKX exchange.
    Uses ccxt exchange instance.
    """
    trades = []

    try:
        if not exchange:
            return []

        # Fetch recent trades via ccxt
        recent_trades = exchange.fetch_trades(symbol, limit=limit)

        for trade in recent_trades:
            price = float(trade.get("price", 0))
            amount = float(trade.get("amount", 0))
            cost = float(trade.get("cost", 0)) or (price * amount)

            if cost < min_usd:
                continue

            trades.append(
                WhaleTrade(
                    timestamp=trade.get(
                        "datetime",
                        datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    ),
                    symbol=symbol,
                    side=trade.get("side", "unknown"),
                    price=price,
                    quantity=amount,
                    amount_usd=cost,
                    is_maker=trade.get("takerOrMaker") == "maker",
                )
            )

        logger.debug(f"[WHALE] OKX: Found {len(trades)} large trades for {symbol}")

    except BEST_EFFORT_EXCEPTIONS as e:
        logger.debug(f"OKX large trades error: {e}")

    return trades


# =============================================================================
# WHALE METRICS CALCULATOR
# =============================================================================


def calculate_whale_metrics(
    trades: List[WhaleTrade],
    symbol: str,
) -> WhaleMetrics:
    """Calculate aggregated whale metrics from trades."""

    total_buy = 0.0
    total_sell = 0.0
    buy_count = 0
    sell_count = 0

    withdrawal_volume = 0.0  # [FAZA 8.3] from_exchange = hodl signal (bullish)
    deposit_volume = 0.0  # [FAZA 8.3] to_exchange = sell pressure (bearish)

    for trade in trades:
        if trade.symbol.upper().replace("/", "") != symbol.upper().replace("/", ""):
            continue

        if trade.side.lower() == "buy":
            total_buy += trade.amount_usd
            buy_count += 1
        else:
            total_sell += trade.amount_usd
            sell_count += 1

        # [FAZA 8.3] Track transfer direction volumes
        if trade.transfer_type == "from_exchange":
            withdrawal_volume += trade.amount_usd
        elif trade.transfer_type == "to_exchange":
            deposit_volume += trade.amount_usd

    net_flow = total_buy - total_sell
    total_volume = total_buy + total_sell

    # [FAZA 8.3] Transfer direction adjustment: withdrawals are bullish, deposits bearish
    transfer_net = withdrawal_volume - deposit_volume
    if total_volume > 0 and abs(transfer_net) > MIN_WHALE_TRADE_USD * 0.5:
        # Withdrawals boost buy pressure, deposits boost sell pressure
        net_flow += transfer_net * 0.3  # 30% weight for transfer signal

    # Determine signal
    if total_volume < MIN_WHALE_TRADE_USD:
        signal = "neutral"
        strength = 0.0
    else:
        ratio = net_flow / max(total_volume, 1)

        if ratio > 0.3:
            signal = "bullish"
            strength = min(1.0, ratio)
        elif ratio > 0.1:
            signal = "slightly_bullish"
            strength = ratio
        elif ratio < -0.3:
            signal = "bearish"
            strength = min(1.0, abs(ratio))
        elif ratio < -0.1:
            signal = "slightly_bearish"
            strength = abs(ratio)
        else:
            signal = "neutral"
            strength = abs(ratio)

    return WhaleMetrics(
        symbol=symbol,
        total_buy_volume_usd=total_buy,
        total_sell_volume_usd=total_sell,
        net_flow_usd=net_flow,
        whale_trade_count=len(trades),
        buy_count=buy_count,
        sell_count=sell_count,
        signal=signal,
        strength=strength,
        last_update=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

_cache: Dict[str, Tuple[WhaleMetrics, float]] = {}
_whale_executor: ThreadPoolExecutor | None = None
_whale_executor_lock = threading.Lock()


def _get_whale_executor() -> ThreadPoolExecutor:
    """Get a shared executor for async->sync bridging in running event loops."""
    global _whale_executor
    with _whale_executor_lock:
        if _whale_executor is None:
            _whale_executor = ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="whale-signal",
            )
        return _whale_executor


def _run_coro_sync(coro_factory):
    """
    Run an async callable from sync code in both sync and async contexts.

    If there is already a running event loop in this thread, execute the
    coroutine in a dedicated worker thread so we never call run_until_complete
    on an active loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())

    executor = _get_whale_executor()
    return executor.submit(lambda: asyncio.run(coro_factory())).result()


async def fetch_whale_alerts(
    symbol: str = "BTC/USDT",
    exchange: Any = None,
    **kwargs,
) -> List[WhaleTrade]:
    """
    Unified function to fetch whale trades.

    Uses:
    1. OKX (if exchange provided)
    2. Binance (fallback)
    """
    trades = []

    # Check config
    try:
        cfg_path = Path("config.json")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        if not cfg.get("enable_whale_alerts", False):
            logger.info(f"Whale alerts disabled via config for {symbol}.")
            return []
    except BEST_EFFORT_EXCEPTIONS:
        pass

    # Try OKX first if exchange available
    if exchange:
        try:
            okx_trades = await fetch_okx_large_trades(exchange, symbol)
            trades.extend(okx_trades)
        except BEST_EFFORT_EXCEPTIONS:
            pass

    # Always add Binance data
    try:
        binance_trades = await fetch_binance_large_trades(symbol)
        trades.extend(binance_trades)
    except BEST_EFFORT_EXCEPTIONS:
        pass

    return trades


async def get_whale_metrics(
    symbol: str,
    exchange: Any = None,
) -> WhaleMetrics:
    """Get whale metrics for a symbol."""
    cache_key = symbol.upper()

    # Check cache
    if cache_key in _cache:
        metrics, ts = _cache[cache_key]
        if time.time() - ts < ADAPTIVE_CACHE_TTL:
            return metrics

    # Fetch trades
    trades = await fetch_whale_alerts(symbol, exchange)

    # Calculate metrics
    metrics = calculate_whale_metrics(trades, symbol)

    # Cache
    _cache[cache_key] = (metrics, time.time())

    return metrics


def get_whale_signal(
    symbol: str,
    exchange: Any = None,
) -> Tuple[str, float]:
    """
    Sync wrapper to get whale signal.

    Returns:
        (signal, strength) - e.g. ("bullish", 0.7)
    """
    try:
        metrics = _run_coro_sync(lambda: get_whale_metrics(symbol, exchange))
        return metrics.signal, metrics.strength
    except BEST_EFFORT_EXCEPTIONS as e:
        logger.debug(f"Whale signal fetch error: {e}")
        return "neutral", 0.0


# =============================================================================
# CONFIDENCE ADJUSTMENT
# =============================================================================


def adjust_confidence_with_whale_data(
    base_confidence: float,
    symbol: str,
    side: str,  # "long" or "short"
    exchange: Any = None,
) -> float:
    """
    Adjust trade confidence based on whale activity.

    Whale buying → boost long confidence, reduce short confidence
    Whale selling → boost short confidence, reduce long confidence
    """
    try:
        signal, strength = get_whale_signal(symbol, exchange)

        if strength < 0.1:
            return base_confidence  # No significant activity

        adjustment = 0.0

        if side.lower() == "long":
            if signal in ["bullish", "slightly_bullish"]:
                adjustment = strength * 0.03  # +3% max boost
            elif signal in ["bearish", "slightly_bearish"]:
                adjustment = -strength * 0.05  # -5% max penalty
        else:  # short
            if signal in ["bearish", "slightly_bearish"]:
                adjustment = strength * 0.03
            elif signal in ["bullish", "slightly_bullish"]:
                adjustment = -strength * 0.05

        new_confidence = max(0.0, min(1.0, base_confidence + adjustment))

        if abs(adjustment) > 0.01:
            logger.info(
                f"[WHALE] {symbol}: {base_confidence:.2f} → {new_confidence:.2f} "
                f"(signal: {signal}, strength: {strength:.1%})"
            )

        return new_confidence

    except BEST_EFFORT_EXCEPTIONS as e:
        logger.debug(f"Whale adjustment error: {e}")
        return base_confidence


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("WHALE TRADE PROVIDER TEST (OKX/Binance)")
    print("=" * 60)

    async def test():
        # Test Binance
        print("\n🐋 Fetching BTC/USDT large trades from Binance...")
        trades = await fetch_binance_large_trades("BTC/USDT")
        print(f"  Found {len(trades)} whale trades")

        if trades:
            metrics = calculate_whale_metrics(trades, "BTC/USDT")
            print(f"\n📊 BTC/USDT Whale Metrics:")
            print(f"  Buy Volume: ${metrics.total_buy_volume_usd:,.0f}")
            print(f"  Sell Volume: ${metrics.total_sell_volume_usd:,.0f}")
            print(f"  Net Flow: ${metrics.net_flow_usd:,.0f}")
            print(f"  Signal: {metrics.signal} ({metrics.strength:.1%} strength)")
            print(f"  Trades: {metrics.buy_count} buys / {metrics.sell_count} sells")

        # Test aggregated
        print("\n📈 Testing 24h ticker stats...")
        stats = await fetch_binance_aggregated_trades("BTC/USDT")
        if stats:
            print(f"  24h Volume: ${stats.get('quote_volume_24h', 0):,.0f}")
            print(f"  Price Change: {stats.get('price_change_pct', 0):.2f}%")
            print(f"  Trade Count: {stats.get('trade_count_24h', 0):,}")

    asyncio.run(test())
