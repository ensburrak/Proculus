# -*- coding: utf-8 -*-
"""
spot_hedge.py
=============

[2026-01-16 PROFESSIONAL FIX #49]

Spot-Perpetual Hedge Strategy (Funding Arbitrage).

Strateji:
1. Pozitif funding rate → Spot'ta al + Perpetual'da short
2. Negatif funding rate → Spot'ta sat + Perpetual'da long
3. Her 8 saatte funding geliri al

Risk: ~%0 (delta-neutral)
Getiri: Funding rate'e bağlı (yıllık %20-50)

Kullanım:
    from spot_hedge import FundingArbitrageManager
    
    manager = FundingArbitrageManager(exchange)
    await manager.check_and_execute_opportunities()
"""

from __future__ import annotations


from core.exceptions import BEST_EFFORT_EXCEPTIONS
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from atomic_io import atomic_write_json
from runtime_paths import METRICS_DIR
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

HEDGE_STATE_FILE = METRICS_DIR / "hedge_state.json"
HEDGE_LOG_FILE = METRICS_DIR / "hedge_log.jsonl"

# Minimum requirements
MIN_FUNDING_RATE = 0.0003  # 0.03% minimum funding (3 * 365 = ~33% annual)
MIN_BALANCE_USDT = 1000  # Minimum $1000 for hedge
ALLOCATION_PCT = 0.20  # Use 20% of balance for hedge


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class HedgePosition:
    """Active hedge position."""
    symbol: str
    spot_side: str  # "long" or "short"
    perp_side: str  # opposite of spot
    spot_amount: float
    perp_amount: float
    entry_price: float
    entry_time: str
    accumulated_funding: float = 0.0
    status: str = "active"  # active, closing, closed


@dataclass
class FundingOpportunity:
    """Funding arbitrage opportunity."""
    symbol: str
    funding_rate: float  # Current funding rate (8h)
    annual_yield: float  # Estimated annual yield
    signal: str  # "OPEN_HEDGE" or "CLOSE_HEDGE"
    recommended_size: float  # In base currency


# =============================================================================
# FUNDING ARBITRAGE MANAGER
# =============================================================================

class FundingArbitrageManager:
    """
    Professional funding arbitrage manager.
    
    Strategy:
    1. Monitor funding rates across symbols
    2. When funding > threshold, open hedge (spot long + perp short)
    3. Collect funding payments every 8 hours
    4. Close when funding normalizes or reverses
    """
    
    def __init__(self, exchange: Any = None, spot_exchange: Any = None):
        """
        Initialize manager.
        
        Args:
            exchange: CCXT perpetual/futures exchange
            spot_exchange: CCXT spot exchange (optional, uses same if None)
        """
        self.exchange = exchange
        self.spot_exchange = spot_exchange or exchange
        self.active_hedges: Dict[str, HedgePosition] = {}
        
        # Ensure directories
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load state
        self._load_state()
    
    def _load_state(self):
        """Load saved hedge positions."""
        try:
            if HEDGE_STATE_FILE.exists():
                data = json.loads(HEDGE_STATE_FILE.read_text(encoding="utf-8"))
                for sym, pos_data in data.get("positions", {}).items():
                    self.active_hedges[sym] = HedgePosition(**pos_data)
                logger.info(f"[HEDGE] Loaded {len(self.active_hedges)} active hedges")
        except BEST_EFFORT_EXCEPTIONS as e:
            logger.warning(f"[HEDGE] State load error: {e}")
    
    def _save_state(self):
        """Save hedge positions."""
        try:
            data = {
                "positions": {
                    sym: {
                        "symbol": pos.symbol,
                        "spot_side": pos.spot_side,
                        "perp_side": pos.perp_side,
                        "spot_amount": pos.spot_amount,
                        "perp_amount": pos.perp_amount,
                        "entry_price": pos.entry_price,
                        "entry_time": pos.entry_time,
                        "accumulated_funding": pos.accumulated_funding,
                        "status": pos.status,
                    }
                    for sym, pos in self.active_hedges.items()
                },
                "updated_at": datetime.utcnow().isoformat() + "Z"
            }
            atomic_write_json(HEDGE_STATE_FILE, data)
        except BEST_EFFORT_EXCEPTIONS as e:
            logger.warning(f"[HEDGE] State save error: {e}")
    
    def _log_action(self, action: str, data: Dict):
        """Log hedge action."""
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "action": action,
                **data
            }
            with HEDGE_LOG_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except BEST_EFFORT_EXCEPTIONS:
            pass

    def _rollback_spot_entry(self, symbol: str, size: float, *, reason: str) -> bool:
        """Best-effort unwind when the perp leg fails after the spot leg fills."""
        try:
            rollback_order = self.spot_exchange.create_market_sell_order(symbol, size)
            logger.warning(
                "[HEDGE] Rolled back spot leg for %s after failed hedge open (%s): %s",
                symbol,
                reason,
                rollback_order.get("id") if isinstance(rollback_order, dict) else rollback_order,
            )
            self._log_action(
                "ROLLBACK_SPOT",
                {
                    "symbol": symbol,
                    "size": size,
                    "reason": reason,
                },
            )
            return True
        except BEST_EFFORT_EXCEPTIONS as rollback_exc:
            logger.critical(
                "[HEDGE] Spot rollback failed for %s after failed hedge open (%s): %s",
                symbol,
                reason,
                rollback_exc,
            )
            self._log_action(
                "ROLLBACK_SPOT_FAILED",
                {
                    "symbol": symbol,
                    "size": size,
                    "reason": reason,
                    "error": str(rollback_exc),
                },
            )
            return False
    
    async def get_funding_rate(self, symbol: str) -> Optional[float]:
        """Get current funding rate for symbol."""
        try:
            if not self.exchange:
                return None
            
            # Convert to OKX format
            okx_symbol = symbol.replace("/", "-") + "-SWAP"
            
            try:
                # Try ccxt unified
                if hasattr(self.exchange, 'fetch_funding_rate'):
                    data = self.exchange.fetch_funding_rate(symbol)
                    return float(data.get('fundingRate', 0))
                else:
                    # OKX API
                    response = self.exchange.public_get_public_funding_rate({
                        'instId': okx_symbol
                    })
                    data = response.get('data', [{}])[0]
                    return float(data.get('fundingRate', 0))
            except BEST_EFFORT_EXCEPTIONS:
                return None
                
        except BEST_EFFORT_EXCEPTIONS as e:
            logger.warning(f"[HEDGE] Funding rate error for {symbol}: {e}")
            return None
    
    async def scan_opportunities(
        self, 
        symbols: List[str]
    ) -> List[FundingOpportunity]:
        """Scan for funding arbitrage opportunities."""
        opportunities = []
        
        for symbol in symbols:
            try:
                funding_rate = await self.get_funding_rate(symbol)
                
                if funding_rate is None:
                    continue
                
                # Calculate annual yield (funding * 3 per day * 365)
                annual_yield = abs(funding_rate) * 3 * 365
                
                # Check if opportunity exists
                if abs(funding_rate) >= MIN_FUNDING_RATE:
                    # Positive funding = short perp + long spot
                    # Negative funding = long perp + short spot (needs margin)
                    
                    if funding_rate > 0:
                        # Standard hedge: buy spot, short perp
                        signal = "OPEN_HEDGE"
                    else:
                        # Reverse is risky (shorting spot requires margin)
                        signal = "MONITOR"
                    
                    opportunities.append(FundingOpportunity(
                        symbol=symbol,
                        funding_rate=funding_rate,
                        annual_yield=annual_yield,
                        signal=signal,
                        recommended_size=0.0  # Calculated later
                    ))
                
            except BEST_EFFORT_EXCEPTIONS as e:
                logger.debug(f"[HEDGE] Scan error for {symbol}: {e}")
                continue
        
        # Sort by annual yield
        opportunities.sort(key=lambda x: x.annual_yield, reverse=True)
        
        return opportunities
    
    async def open_hedge(
        self,
        symbol: str,
        amount_usdt: float,
    ) -> Optional[HedgePosition]:
        """
        Open a hedge position.
        
        Steps:
        1. Buy spot
        2. Open short perpetual (same size)
        3. Track position
        """
        try:
            if not self.exchange or not self.spot_exchange:
                logger.warning("[HEDGE] Exchange not configured")
                return None
            
            if symbol in self.active_hedges:
                logger.info(f"[HEDGE] {symbol} hedge already exists")
                return self.active_hedges[symbol]
            
            # Get current price
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                price = float(ticker.get('last', 0))
            except BEST_EFFORT_EXCEPTIONS:
                logger.warning(f"[HEDGE] Cannot get price for {symbol}")
                return None
            
            if price <= 0:
                return None
            
            # Calculate size
            size = amount_usdt / price
            
            logger.info(f"[HEDGE] Opening hedge for {symbol}: ${amount_usdt:.2f} = {size:.6f} @ ${price:.2f}")
            
            # 1. Buy spot
            try:
                spot_order = self.spot_exchange.create_market_buy_order(
                    symbol, size
                )
                logger.info(f"[HEDGE] Spot buy order: {spot_order.get('id')}")
            except BEST_EFFORT_EXCEPTIONS as e:
                logger.error(f"[HEDGE] Spot buy failed: {e}")
                return None
            
            # 2. Short perpetual
            try:
                perp_symbol = symbol  # ccxt should handle format
                perp_order = self.exchange.create_market_sell_order(
                    perp_symbol, size,
                    params={'tdMode': 'isolated', 'reduceOnly': False}
                )
                logger.info(f"[HEDGE] Perp short order: {perp_order.get('id')}")
            except BEST_EFFORT_EXCEPTIONS as e:
                logger.error(f"[HEDGE] Perp short failed: {e}")
                self._rollback_spot_entry(symbol, size, reason="perp_short_failed")
                return None
            
            # Create position
            position = HedgePosition(
                symbol=symbol,
                spot_side="long",
                perp_side="short",
                spot_amount=size,
                perp_amount=size,
                entry_price=price,
                entry_time=datetime.utcnow().isoformat() + "Z",
                accumulated_funding=0.0,
                status="active"
            )
            
            self.active_hedges[symbol] = position
            self._save_state()
            
            # Log
            self._log_action("OPEN_HEDGE", {
                "symbol": symbol,
                "size": size,
                "price": price,
                "amount_usdt": amount_usdt,
            })
            
            # Telegram notification
            try:
                from telegram_notifier import notify_alert_async
                await notify_alert_async(
                    f"🔄 Hedge Açıldı: {symbol}",
                    f"Miktar: {size:.6f}\nGiriş: ${price:,.2f}\nStrateji: Spot Long + Perp Short",
                    level="INFO"
                )
            except BEST_EFFORT_EXCEPTIONS:
                pass
            
            return position
            
        except BEST_EFFORT_EXCEPTIONS as e:
            logger.error(f"[HEDGE] Open hedge error: {e}")
            return None
    
    async def close_hedge(self, symbol: str, reason: str = "Manual") -> bool:
        """
        Close a hedge position.
        
        Steps:
        1. Close perpetual short
        2. Sell spot
        3. Calculate profit
        """
        try:
            if symbol not in self.active_hedges:
                logger.warning(f"[HEDGE] No hedge found for {symbol}")
                return False
            
            position = self.active_hedges[symbol]
            
            if not self.exchange or not self.spot_exchange:
                return False
            
            # Get current price
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                price = float(ticker.get('last', 0))
            except BEST_EFFORT_EXCEPTIONS:
                price = position.entry_price
            
            logger.info(f"[HEDGE] Closing hedge for {symbol}: {position.perp_amount:.6f}")
            
            # 1. Close perpetual (buy to close short)
            try:
                self.exchange.create_market_buy_order(
                    symbol, position.perp_amount,
                    params={'reduceOnly': True}
                )
            except BEST_EFFORT_EXCEPTIONS as e:
                logger.error(f"[HEDGE] Perp close failed: {e}")
            
            # 2. Sell spot
            try:
                self.spot_exchange.create_market_sell_order(
                    symbol, position.spot_amount
                )
            except BEST_EFFORT_EXCEPTIONS as e:
                logger.error(f"[HEDGE] Spot sell failed: {e}")
            
            # Calculate profit (mainly from funding)
            funding_profit = position.accumulated_funding
            
            # Remove from active
            del self.active_hedges[symbol]
            self._save_state()
            
            # Log
            self._log_action("CLOSE_HEDGE", {
                "symbol": symbol,
                "funding_profit": funding_profit,
                "reason": reason,
            })
            
            # Telegram
            try:
                from telegram_notifier import notify_alert_async
                await notify_alert_async(
                    f"🔄 Hedge Kapandı: {symbol}",
                    f"Funding Kar: ${funding_profit:.2f}\nSebep: {reason}",
                    level="INFO"
                )
            except BEST_EFFORT_EXCEPTIONS:
                pass
            
            return True
            
        except BEST_EFFORT_EXCEPTIONS as e:
            logger.error(f"[HEDGE] Close hedge error: {e}")
            return False
    
    async def update_funding_income(self):
        """Update accumulated funding for all positions."""
        for symbol, position in self.active_hedges.items():
            try:
                funding_rate = await self.get_funding_rate(symbol)
                if funding_rate and funding_rate > 0:
                    # Funding income = position_value * funding_rate
                    try:
                        ticker = self.exchange.fetch_ticker(symbol)
                        price = float(ticker.get('last', 0))
                    except BEST_EFFORT_EXCEPTIONS:
                        price = position.entry_price
                    
                    position_value = position.perp_amount * price
                    funding_income = position_value * funding_rate
                    position.accumulated_funding += funding_income
                    
                    logger.info(f"[HEDGE] {symbol} funding income: ${funding_income:.4f}")
                    
            except BEST_EFFORT_EXCEPTIONS as e:
                logger.debug(f"[HEDGE] Funding update error for {symbol}: {e}")
        
        self._save_state()
    
    def get_summary(self) -> Dict:
        """Get hedge positions summary."""
        total_funding = sum(p.accumulated_funding for p in self.active_hedges.values())
        
        return {
            "active_hedges": len(self.active_hedges),
            "symbols": list(self.active_hedges.keys()),
            "total_funding_income": total_funding,
            "positions": {
                sym: {
                    "entry_price": p.entry_price,
                    "amount": p.spot_amount,
                    "funding": p.accumulated_funding,
                    "entry_time": p.entry_time,
                }
                for sym, p in self.active_hedges.items()
            }
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_manager_instance: Optional[FundingArbitrageManager] = None
_manager_instance_lock = None


def get_hedge_manager(exchange: Any = None) -> FundingArbitrageManager:
    """Get global hedge manager."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = FundingArbitrageManager(exchange)
    elif exchange:
        _manager_instance.exchange = exchange
    return _manager_instance


async def check_funding_opportunities(
    exchange: Any,
    symbols: List[str],
    auto_execute: bool = False,
    max_allocation: float = 1000.0,
) -> List[FundingOpportunity]:
    """
    Check for funding arbitrage opportunities.
    
    Args:
        exchange: CCXT exchange instance
        symbols: List of symbols to check
        auto_execute: Automatically open hedge if opportunity found
        max_allocation: Max USDT to allocate
    
    Returns:
        List of opportunities found
    """
    manager = get_hedge_manager(exchange)
    opportunities = await manager.scan_opportunities(symbols)
    
    if auto_execute and opportunities:
        best = opportunities[0]
        if best.signal == "OPEN_HEDGE" and best.symbol not in manager.active_hedges:
            await manager.open_hedge(best.symbol, max_allocation)
    
    return opportunities


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SPOT HEDGE (FUNDING ARBITRAGE) TEST")
    print("=" * 60)
    
    manager = FundingArbitrageManager()
    summary = manager.get_summary()
    
    print(f"\n📊 Active Hedges: {summary['active_hedges']}")
    print(f"💰 Total Funding Income: ${summary['total_funding_income']:.2f}")
    print(f"📌 Symbols: {summary['symbols']}")
