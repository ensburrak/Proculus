"""
orderbook_analyzer.py
----------------------

[FAZA 8.1] Orderbook Imbalance Signal - Gerçek implementasyon.

Fonksiyonlar:
    adjust_confidence_with_imbalance: Master confidence'i orderbook dengesine göre ayarlar
    calculate_imbalance: API'den orderbook derinlik verisini çeker ve imbalance hesaplar
    get_orderbook_signal: bid/ask oranından buying/selling pressure sinyali üretir
    get_orderbook_bias: Fusion'a eklenecek bias skoru hesaplar (ağırlık: %10)
"""
from __future__ import annotations

from core.exceptions import BEST_EFFORT_EXCEPTIONS
import asyncio
import inspect
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from exchanges.endpoint_contracts import build_orderbook_url
from core.symbol_normalization import to_exchange_rest_symbol

log = logging.getLogger(__name__)

# [FAZA 8.1] Orderbook imbalance thresholds
OB_STRONG_RATIO = float(os.getenv("OB_STRONG_RATIO", "1.5"))  # bid/ask > 1.5 = buying pressure
OB_EXTREME_RATIO = float(os.getenv("OB_EXTREME_RATIO", "2.5"))  # bid/ask > 2.5 = extreme pressure
OB_FUSION_WEIGHT = float(os.getenv("OB_FUSION_WEIGHT", "0.10"))  # 10% weight in fusion


@dataclass
class OrderbookSignal:
    """[FAZA 8.1] Structured orderbook signal for decision pipeline."""
    direction: str  # "long", "short", "neutral"
    strength: float  # 0.0 - 1.0
    bid_volume: float
    ask_volume: float
    ratio: float  # bid/ask ratio
    imbalance: float  # -1 to +1 normalized
    reason: str

try:
    import aiohttp  # type: ignore
except BEST_EFFORT_EXCEPTIONS:
    aiohttp = None  # type: ignore

try:
    import requests  # type: ignore
except BEST_EFFORT_EXCEPTIONS:
    requests = None  # type: ignore


async def _await_maybe(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _fetch_orderbook_payload(
    exchange: object,
    symbol: str,
    depth: int,
) -> tuple[str, Any]:
    exchange_id = str(getattr(exchange, "id", "binance")).lower()
    if hasattr(exchange, "fetch_order_book"):
        payload = await _await_maybe(exchange.fetch_order_book(symbol, depth))
        return exchange_id, payload
    if hasattr(exchange, "fetch_orderbook"):
        payload = await _await_maybe(exchange.fetch_orderbook(symbol, depth))
        return exchange_id, payload

    if exchange_id == "okx":
        base_url = os.getenv("OKX_API_BASE_URL", "https://www.okx.com")
        normalized_symbol = to_exchange_rest_symbol(exchange_id, symbol)
        endpoint = build_orderbook_url(exchange_id, base_url=base_url, symbol=normalized_symbol, depth=depth)
    else:
        base_url = os.getenv("BINANCE_FAPI_BASE_URL", "https://fapi.binance.com")
        normalized_symbol = to_exchange_rest_symbol(exchange_id, symbol)
        endpoint = build_orderbook_url("binance", base_url=base_url, symbol=normalized_symbol, depth=depth)

    data = None
    if aiohttp is not None:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(endpoint) as response:
                if response.status == 200:
                    data = await response.json()
    elif requests is not None:
        response = await asyncio.to_thread(requests.get, endpoint, timeout=5)
        if response.status_code == 200:
            data = response.json()
    return exchange_id, data


def _extract_levels(exchange_id: str, data: Any) -> tuple[list, list]:
    if not isinstance(data, dict):
        return [], []
    if exchange_id == "okx":
        book = (data.get("data") or [{}])[0] if isinstance(data.get("data"), list) else {}
        bids = book.get("bids") or []
        asks = book.get("asks") or []
        return list(bids), list(asks)
    bids = data.get("bids") or []
    asks = data.get("asks") or []
    return list(bids), list(asks)


def _sum_side_volume(levels: list[Any], depth: int) -> float:
    total = 0.0
    for entry in levels[:depth]:
        try:
            total += float(entry[1])
        except BEST_EFFORT_EXCEPTIONS:
            continue
    return total


async def fetch_orderbook_features(
    exchange: object,
    symbol: str,
    depth: int = 20,
) -> Optional[Dict[str, Any]]:
    """Fetch the real orderbook snapshot and derived features for decision/runtime use."""
    try:
        depth = max(1, min(int(depth), 100))
        exchange_id, payload = await _fetch_orderbook_payload(exchange, symbol, depth)
        bids, asks = _extract_levels(exchange_id, payload)
        if not bids and not asks:
            return None

        bid_volume = _sum_side_volume(bids, depth)
        ask_volume = _sum_side_volume(asks, depth)
        signal = get_orderbook_signal(bid_volume, ask_volume)
        return {
            "bids": bids[:depth],
            "asks": asks[:depth],
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
            "ratio": signal.ratio,
            "imbalance": signal.imbalance,
            "signal": {
                "direction": signal.direction,
                "strength": signal.strength,
                "reason": signal.reason,
            },
        }
    except BEST_EFFORT_EXCEPTIONS:
        return None


def adjust_confidence_with_imbalance(
    base_conf: float,
    imbalance: Optional[float],
    direction: str,
    scale: float = 0.20,
) -> float:
    """
    Siparis defteri dengesine gore master confidence valueini settinglar.

    Args:
        base_conf (float): Orijinal confidence valuei (0 ile 1 arasinda).
        imbalance (float|None): Orderbook dengesizligi.  Pozitif
            valueler alislarin, negatif valueler satislarin baskin oldugunu
            gosterir.  ``None`` ise hicbir settinglama yapilmaz.
        direction (str): 'long' veya 'short'.  Long icin pozitif dengesizlik
            confidence'i artirir; short icin negatif dengesizlik dikkate alinir.
        scale (float): Denge etkisinin carpani.  Varsayilan 0.20.

    Returns:
        float: Ayarlanmis confidence, 0..1 araliginda kirpilmis.
    """
    try:
        # Baz value 0..1 arasinda olsun
        conf = float(base_conf)
    except BEST_EFFORT_EXCEPTIONS:
        return 0.0
    # Denge verisi yoksa degisiklik yapma
    if imbalance is None:
        return max(0.0, min(1.0, conf))
    try:
        imbal = float(imbalance)
    except BEST_EFFORT_EXCEPTIONS:
        return max(0.0, min(1.0, conf))
    # Yon faktoru
    adjustment = 0.0
    if direction.lower().startswith("long"):
        adjustment = imbal * scale
    elif direction.lower().startswith("short"):
        adjustment = -imbal * scale
    # Yeni confidence'i accountla ve 0..1'e kliple
    new_conf = conf + adjustment
    return max(0.0, min(1.0, new_conf))


async def calculate_imbalance(
    exchange: object,
    symbol: str,
    depth: int = 20,
) -> Optional[float]:
    """
    Siparis defteri alis/satis dengesini accountlar.

    Bu fonksiyon, verilen symbol icin Binance vadeli tradeler (Futures) API'si
    uzerinden siparis defteri derinlik verisini ceker ve ilk ``depth``
    seviyedeki alis ve satis hacimlerini karsilastirarak bir dengesizlik
    olcusu accountlar. Dengesizlik, (toplam alis hacmi - toplam satis hacmi)
    / (toplam alis hacmi + toplam satis hacmi) formulu ile [-1, 1]
    araligina normalize edilir. Sonuc, long tradeler icin pozitif value
    alis baskisini, short tradeler icin negatif value satis baskisini ifade
    eder.  Herhangi bir error statusunda veya veri alinamazsa ``None`` return.

    Args:
        exchange: Borsa API nesnesi (kullanilmiyor, geriye donuk uyum icin)
        symbol (str): Islem symbolu (orn. "BTC/USDT" veya "BTCUSDT")
        depth (int): Orderbook'tan cekilecek seviye sayisi (en fazla 100)

    Returns:
        Optional[float]: Alis/satis dengesizligi veya error halinde ``None``.
    """
    try:
        features = await fetch_orderbook_features(exchange, symbol, depth)
        if not isinstance(features, dict):
            return None
        return float(features.get("imbalance")) if features.get("imbalance") is not None else None
    except BEST_EFFORT_EXCEPTIONS:
        # herhangi bir error statusunda None return
        return None


# =============================================================================
# [FAZA 8.1] Enhanced Orderbook Signal Functions
# =============================================================================

def get_orderbook_signal(
    bid_volume: float,
    ask_volume: float,
) -> OrderbookSignal:
    """[FAZA 8.1] Orderbook imbalance sinyali üret.

    a) bid_volume / ask_volume > 1.5 → buying pressure, long bias
    b) ask_volume / bid_volume > 1.5 → selling pressure, short bias
    c) Aksi halde neutral
    """
    if bid_volume <= 0 and ask_volume <= 0:
        return OrderbookSignal(
            direction="neutral", strength=0.0,
            bid_volume=0, ask_volume=0, ratio=1.0, imbalance=0.0,
            reason="no orderbook data",
        )

    total = bid_volume + ask_volume
    if total <= 0:
        return OrderbookSignal(
            direction="neutral", strength=0.0,
            bid_volume=bid_volume, ask_volume=ask_volume,
            ratio=1.0, imbalance=0.0, reason="zero total volume",
        )

    imbalance = (bid_volume - ask_volume) / total  # -1 to +1

    # Avoid division by zero
    bid_ask_ratio = bid_volume / max(ask_volume, 1e-10)
    ask_bid_ratio = ask_volume / max(bid_volume, 1e-10)

    direction = "neutral"
    strength = 0.0
    reason = "balanced orderbook"

    if bid_ask_ratio >= OB_EXTREME_RATIO:
        direction = "long"
        strength = min(1.0, (bid_ask_ratio - 1.0) / 3.0)
        reason = f"extreme buying pressure (bid/ask={bid_ask_ratio:.2f})"
    elif bid_ask_ratio >= OB_STRONG_RATIO:
        direction = "long"
        strength = min(0.7, (bid_ask_ratio - 1.0) / 2.0)
        reason = f"buying pressure (bid/ask={bid_ask_ratio:.2f})"
    elif ask_bid_ratio >= OB_EXTREME_RATIO:
        direction = "short"
        strength = min(1.0, (ask_bid_ratio - 1.0) / 3.0)
        reason = f"extreme selling pressure (ask/bid={ask_bid_ratio:.2f})"
    elif ask_bid_ratio >= OB_STRONG_RATIO:
        direction = "short"
        strength = min(0.7, (ask_bid_ratio - 1.0) / 2.0)
        reason = f"selling pressure (ask/bid={ask_bid_ratio:.2f})"

    return OrderbookSignal(
        direction=direction,
        strength=strength,
        bid_volume=bid_volume,
        ask_volume=ask_volume,
        ratio=bid_ask_ratio,
        imbalance=imbalance,
        reason=reason,
    )


def get_orderbook_bias(
    imbalance: Optional[float],
    direction: str,
) -> float:
    """[FAZA 8.1] Orderbook bias skoru (fusion entegrasyonu için).

    Returns:
        float: 0.0-1.0 arası skor. 0.5 = neutral.
        Long direction + pozitif imbalance → 0.5-1.0
        Short direction + negatif imbalance → 0.5-1.0
        Ters yönlü imbalance → 0.0-0.5
    """
    if imbalance is None:
        return 0.5  # Neutral

    try:
        imbal = float(imbalance)
    except (TypeError, ValueError):
        return 0.5

    # imbalance: -1 to +1
    # direction alignment check
    if direction.lower() == "long":
        # Pozitif imbalance long'u destekler
        bias = 0.5 + imbal * 0.5  # Maps -1..+1 to 0..1
    elif direction.lower() == "short":
        # Negatif imbalance short'u destekler
        bias = 0.5 - imbal * 0.5  # Maps -1..+1 to 1..0
    else:
        bias = 0.5

    return max(0.0, min(1.0, bias))


async def calculate_imbalance_with_signal(
    exchange: object,
    symbol: str,
    depth: int = 20,
) -> Tuple[Optional[float], Optional[OrderbookSignal]]:
    """[FAZA 8.1] Calculate imbalance and return structured signal.

    Returns:
        (imbalance_float, OrderbookSignal) tuple
    """
    features = await fetch_orderbook_features(exchange, symbol, depth)
    if not isinstance(features, dict):
        return None, None
    signal_payload = features.get("signal") or {}
    signal = OrderbookSignal(
        direction=str(signal_payload.get("direction") or "neutral"),
        strength=float(signal_payload.get("strength") or 0.0),
        bid_volume=float(features.get("bid_volume") or 0.0),
        ask_volume=float(features.get("ask_volume") or 0.0),
        ratio=float(features.get("ratio") or 1.0),
        imbalance=float(features.get("imbalance") or 0.0),
        reason=str(signal_payload.get("reason") or "orderbook snapshot"),
    )
    return signal.imbalance, signal
