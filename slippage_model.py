"""
slippage_model.py
-----------------

[2026-02-19 FAZ1.2] Fixed: return configurable default slippage on error
instead of 0.0%. When orderbook fetch fails, a conservative default (1.5%)
is returned to prevent underestimating execution costs.

Bu modul, tradeler sirasinda olusabilecek slipaji (fiyat kaymasini)
tahmin etmek icin basit bir arayuz saglar.

Fonksiyonlar:
    estimate_slippage(exchange, symbol, amount, side, depth)
        Islem parametrelerine gore slipaj tahmini return.
"""
from __future__ import annotations
from core.exceptions import BEST_EFFORT_EXCEPTIONS
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

import os
try:
    import aiohttp  # type: ignore
except BEST_EFFORT_EXCEPTIONS:
    aiohttp = None  # type: ignore

try:
    import requests  # type: ignore
except BEST_EFFORT_EXCEPTIONS:
    requests = None  # type: ignore

log = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

_DEFAULT_SLIPPAGE_PCT = 0.015  # 1.5% default when orderbook unavailable
_SIZE_REDUCTION_ON_FAILURE = 0.20  # reduce position size by 20% on fetch fail


def _load_slippage_config() -> tuple[float, float]:
    """Load slippage config from config.json if available."""
    try:
        cfg_path = Path("config.json")
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            slip_cfg = cfg.get("slippage", {})
            default_pct = float(slip_cfg.get("default_pct", _DEFAULT_SLIPPAGE_PCT))
            size_reduction = float(slip_cfg.get("size_reduction_on_failure", _SIZE_REDUCTION_ON_FAILURE))
            return default_pct, size_reduction
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        log.debug("[SLIPPAGE] config load fallback: %s", exc)
    return _DEFAULT_SLIPPAGE_PCT, _SIZE_REDUCTION_ON_FAILURE


DEFAULT_SLIPPAGE, SIZE_REDUCTION_FACTOR = _load_slippage_config()


async def estimate_slippage(
    exchange: object,
    symbol: str,
    amount: float | None = None,
    side: str | None = None,
    depth: int = 20,
) -> float:
    """Return the slippage estimate as a float for legacy callers."""
    slippage, _failed = await _estimate_slippage_value(exchange, symbol, amount, side, depth)
    return slippage


async def estimate_slippage_with_status(
    exchange: object,
    symbol: str,
    amount: float | None = None,
    side: str | None = None,
    depth: int = 20,
) -> dict[str, Any]:
    """Return the slippage estimate and orderbook failure state for this call."""
    slippage, failed = await _estimate_slippage_value(exchange, symbol, amount, side, depth)
    return {
        "slippage": slippage,
        "orderbook_fetch_failed": failed,
    }


async def _estimate_slippage_value(
    exchange: object,
    symbol: str,
    amount: float | None = None,
    side: str | None = None,
    depth: int = 20,
) -> tuple[float, bool]:
    """
    Verilen trade parametrelerine gore slipaj (fiyat kaymasi) tahmini return.

    Hata durumunda 0.0 yerine configurable default slippage return.
    Return tuple: ``(slippage, orderbook_fetch_failed)``.

    Args:
        exchange: Borsa API nesnesi (kullanilmiyor)
        symbol (str): Islem symbolu
        amount (float|None): Islem miktari
        side (str|None): 'buy' veya 'sell'
        depth (int): Orderbook derinligi

    Returns:
        Tuple of slippage yuzdesi and this-call orderbook failure status.
    """
    if amount is None or amount <= 0:
        return 0.0, False

    if not side:
        return DEFAULT_SLIPPAGE, False  # No side info -> conservative default

    if requests is None and aiohttp is None:
        log.warning(
            f"[SLIPPAGE] No HTTP library available, returning default {DEFAULT_SLIPPAGE*100:.1f}%"
        )
        return DEFAULT_SLIPPAGE, True

    symbol_normalized = symbol.replace("/", "").replace("-", "").upper()
    # Remove SWAP suffix for Binance API
    if symbol_normalized.endswith("SWAP"):
        symbol_normalized = symbol_normalized[:-4]

    base_url = os.getenv("SLIPPAGE_BOOK_BASE_URL", "https://fapi.binance.com")
    try:
        url = f"{base_url}/fapi/v1/depth"
        params = {"symbol": symbol_normalized, "limit": depth}
        if aiohttp is not None:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        log.warning(
                            f"[SLIPPAGE] Orderbook fetch HTTP {resp.status} for {symbol}, "
                            f"using default {DEFAULT_SLIPPAGE*100:.1f}%"
                        )
                        return DEFAULT_SLIPPAGE, True
                    ob = await resp.json()
        else:
            resp = await asyncio.to_thread(
                requests.get,
                url,
                params=params,
                timeout=5,
            )
            resp.raise_for_status()
            ob = resp.json() or {}

        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        bids = [(float(p), float(q)) for p, q in bids]
        asks = [(float(p), float(q)) for p, q in asks]

        if not bids or not asks:
            log.warning(
                f"[SLIPPAGE] Empty orderbook for {symbol}, "
                f"using default {DEFAULT_SLIPPAGE*100:.1f}%"
            )
            return DEFAULT_SLIPPAGE, True

        side_lower = side.lower()

        if side_lower.startswith("buy"):
            qty_remaining = amount
            cost = 0.0
            total_qty = 0.0
            for price, qty in asks:
                if qty_remaining <= 0:
                    break
                trade_qty = min(qty_remaining, qty)
                cost += trade_qty * price
                total_qty += trade_qty
                qty_remaining -= trade_qty
            if total_qty > 0:
                avg_price = cost / total_qty
                best = asks[0][0]
                slippage = (avg_price - best) / best
                return max(slippage, 0.0), False
            return DEFAULT_SLIPPAGE, True

        elif side_lower.startswith("sell"):
            qty_remaining = amount
            revenue = 0.0
            total_qty = 0.0
            for price, qty in bids:
                if qty_remaining <= 0:
                    break
                trade_qty = min(qty_remaining, qty)
                revenue += trade_qty * price
                total_qty += trade_qty
                qty_remaining -= trade_qty
            if total_qty > 0:
                avg_price = revenue / total_qty
                best = bids[0][0]
                slippage = (best - avg_price) / best
                return max(slippage, 0.0), False
            return DEFAULT_SLIPPAGE, True

        else:
            return DEFAULT_SLIPPAGE, False

    except BEST_EFFORT_EXCEPTIONS as e:
        log.warning(
            f"[SLIPPAGE] Orderbook fetch failed for {symbol}: {e}, "
            f"using default {DEFAULT_SLIPPAGE*100:.1f}%"
        )
        return DEFAULT_SLIPPAGE, True


def orderbook_fetch_failed() -> bool:
    """Deprecated compatibility shim.

    Orderbook fetch status is no longer stored as shared "last call" state.
    Use ``estimate_slippage_with_status`` and read ``orderbook_fetch_failed``
    from that call's return payload.
    """
    return False


# =============================================================================
# [FAZA 5.1] Dynamic Slippage Model
# =============================================================================

def estimate_dynamic_slippage(
    order_notional_usd: float,
    spread_pct: float = 0.0005,
    atr_pct: float = 0.015,
    daily_volume_usd: float = 1_000_000.0,
    impact_factor: float = 0.1,
) -> float:
    """Dinamik slippage tahmini: orderbook + volatilite + boyut etkisi.

    slippage = base_spread + size_impact + volatility_premium

    Args:
        order_notional_usd: Order büyüklüğü (USD)
        spread_pct: Bid-ask spread yüzdesi (default %0.05)
        atr_pct: ATR yüzdesi (volatilite ölçüsü)
        daily_volume_usd: Günlük hacim (USD)
        impact_factor: Boyut etki çarpanı

    Returns:
        Tahmini slippage yüzdesi (ondalık)
    """
    # 1. Base spread — orderbook'taki gerçek spread
    base_spread = max(0.0, spread_pct)

    # 2. Size impact — büyük order'lar daha fazla slippage yaratır
    if daily_volume_usd > 0:
        size_impact = (order_notional_usd / daily_volume_usd) * impact_factor
    else:
        size_impact = 0.005  # Hacim bilinmiyorsa %0.5 varsay

    # 3. Volatility premium — yüksek volatilite = daha fazla slippage
    volatility_premium = atr_pct * 0.1

    total = base_spread + size_impact + volatility_premium
    return max(0.0, min(0.05, total))  # Max %5 cap


async def estimate_slippage_enhanced(
    exchange: object,
    symbol: str,
    amount: float | None = None,
    side: str | None = None,
    order_notional_usd: float = 0.0,
    atr_pct: float = 0.015,
    depth: int = 20,
) -> dict:
    """Geliştirilmiş slippage tahmini: orderbook + dinamik model birleşimi.

    Returns:
        Dict: {
            "orderbook_slippage": float,   # Orderbook bazlı tahmin
            "dynamic_slippage": float,     # Dinamik model tahmini
            "estimated_slippage": float,   # Kullanılacak final tahmin
            "spread_pct": float,           # Hesaplanan spread
            "method": str,                 # Hangi yöntem kullanıldı
        }
    """
    # Orderbook bazlı tahmin
    ob_status = await estimate_slippage_with_status(exchange, symbol, amount, side, depth)
    ob_slippage = float(ob_status["slippage"])
    ob_failed = bool(ob_status["orderbook_fetch_failed"])

    # Spread hesaplama (ticker'dan)
    spread_pct = 0.0005  # default
    daily_volume = 1_000_000.0
    try:
        ticker = await _fetch_ticker_safe(exchange, symbol)
        if ticker:
            bid = float(ticker.get("bid") or 0)
            ask = float(ticker.get("ask") or 0)
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
            if mid > 0:
                spread_pct = (ask - bid) / mid

            # Günlük hacim
            quote_vol = float(ticker.get("quoteVolume") or 0)
            if quote_vol > 0:
                daily_volume = quote_vol
    except BEST_EFFORT_EXCEPTIONS:
        pass

    # Dinamik model tahmini
    dynamic = estimate_dynamic_slippage(
        order_notional_usd=order_notional_usd or (amount or 0) * float(
            (ticker.get("last") or 1) if 'ticker' in dir() else 1
        ),
        spread_pct=spread_pct,
        atr_pct=atr_pct,
        daily_volume_usd=daily_volume,
    )

    # Final: orderbook varsa ortalamasını al, yoksa dinamik kullan
    if ob_failed:
        estimated = dynamic
        method = "dynamic"
    else:
        estimated = (ob_slippage * 0.6 + dynamic * 0.4)  # Ağırlıklı ortalama
        method = "hybrid"

    return {
        "orderbook_slippage": ob_slippage,
        "dynamic_slippage": dynamic,
        "estimated_slippage": estimated,
        "spread_pct": spread_pct,
        "daily_volume_usd": daily_volume,
        "method": method,
    }


async def _fetch_ticker_safe(exchange, symbol: str) -> dict | None:
    """Ticker'ı güvenli şekilde çek."""
    try:
        import inspect
        fn = getattr(exchange, "fetch_ticker", None)
        if fn is None:
            return None
        if inspect.iscoroutinefunction(fn):
            return await fn(symbol)
        import asyncio
        return await asyncio.to_thread(fn, symbol)
    except BEST_EFFORT_EXCEPTIONS:
        return None


def compare_slippage(estimated: float, actual: float) -> dict:
    """Tahmini ve gerçek slippage'ı karşılaştır.

    Args:
        estimated: Tahmini slippage (%)
        actual: Gerçekleşen slippage (%)

    Returns:
        Dict: accuracy metrikleri
    """
    error = actual - estimated
    abs_error = abs(error)
    return {
        "estimated": round(estimated, 6),
        "actual": round(actual, 6),
        "error": round(error, 6),
        "abs_error": round(abs_error, 6),
        "underestimated": error > 0,
        "quality": "good" if abs_error < 0.001 else "fair" if abs_error < 0.003 else "poor",
    }
