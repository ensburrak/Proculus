# -*- coding: utf-8 -*-
"""
signal_service.py
=================

[2026-01-16 PROFESSIONAL FIX #51]

Profesyonel Copy Trading / Signal Service.

Özellikler:
1. Trade sinyallerini Telegram grubuna broadcast
2. Performans istatistikleri
3. Sinyal geçmişi
4. Abonelik yönetimi (gelecek)

Kullanım:
    from signal_service import SignalService, get_signal_service
    
    service = get_signal_service()
    service.broadcast_signal(signal_data)
"""

from __future__ import annotations


from core.exceptions import BEST_EFFORT_EXCEPTIONS
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import threading

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(__file__).resolve().parent
SIGNALS_LOG = ROOT / "metrics" / "signal_service_log.jsonl"
PERFORMANCE_FILE = ROOT / "metrics" / "signal_service_performance.json"

# Telegram credentials (from .env)
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
# Signal group chat ID (can be different from personal chat)
SIGNAL_CHAT_ID = os.getenv("TELEGRAM_SIGNAL_CHAT_ID", "") or os.getenv("TELEGRAM_CHAT_ID", "")

# =============================================================================
# SIGNAL FORMATTER
# =============================================================================

def format_signal_message(
    signal_type: str,  # "ENTRY" or "EXIT"
    symbol: str,
    side: str,  # "LONG" or "SHORT"
    entry_price: float,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    confidence: Optional[float] = None,
    leverage: Optional[int] = None,
    reason: Optional[str] = None,
) -> str:
    """Format signal message for Telegram broadcast."""
    
    if signal_type == "ENTRY":
        emoji = "🟢" if side.upper() == "LONG" else "🔴"
        direction = "📈 LONG" if side.upper() == "LONG" else "📉 SHORT"
        
        message = f"""
{emoji}{'═'*30}{emoji}
🚀 <b>YENİ SİNYAL</b>
{emoji}{'═'*30}{emoji}

<b>📊 Sembol:</b> {symbol}
<b>📌 Yön:</b> {direction}
<b>💰 Giriş:</b> <code>${entry_price:,.2f}</code>
"""
        
        if stop_loss:
            sl_pct = abs(stop_loss - entry_price) / entry_price * 100
            message += f"<b>🛡 Stop Loss:</b> <code>${stop_loss:,.2f}</code> (<code>-{sl_pct:.1f}%</code>)\n"
        
        if take_profit:
            tp_pct = abs(take_profit - entry_price) / entry_price * 100
            message += f"<b>🎯 Take Profit:</b> <code>${take_profit:,.2f}</code> (<code>+{tp_pct:.1f}%</code>)\n"
        
        if leverage:
            message += f"<b>⚡ Kaldıraç:</b> {leverage}x\n"
        
        if confidence:
            conf_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
            message += f"<b>📊 Güven:</b> [{conf_bar}] {confidence:.0%}\n"
        
        if reason:
            message += f"\n<i>💡 {reason}</i>\n"
        
        message += f"""
{'─'*35}
⏰ <b>{datetime.now(timezone.utc).strftime('%H:%M:%S - %d/%m/%Y')}</b>
🤖 <i>AutoTraderBot Signal Service</i>
"""
    
    else:  # EXIT
        emoji = "💰" if "PROFIT" in str(reason).upper() else "💸"
        
        message = f"""
{emoji}{'═'*30}{emoji}
📤 <b>POZİSYON KAPANDI</b>
{emoji}{'═'*30}{emoji}

<b>📊 Sembol:</b> {symbol}
<b>📌 Yön:</b> {side.upper()}
<b>💰 Çıkış:</b> <code>${entry_price:,.2f}</code>
<b>📝 Sebep:</b> {reason or 'Manual'}

⏰ <b>{datetime.now(timezone.utc).strftime('%H:%M:%S - %d/%m/%Y')}</b>
"""

    return message


# =============================================================================
# SIGNAL SERVICE CLASS
# =============================================================================

class SignalService:
    """
    Professional signal broadcasting service.
    
    Features:
    - Broadcast to Telegram
    - Log all signals
    - Track performance
    """
    
    def __init__(self):
        self.enabled = bool(BOT_TOKEN and SIGNAL_CHAT_ID)
        self.signals_sent = 0
        self.last_signal_time: Optional[datetime] = None
        
        # Ensure directories exist
        SIGNALS_LOG.parent.mkdir(parents=True, exist_ok=True)
        
        # Load performance
        self.performance = self._load_performance()
    
    def _load_performance(self) -> Dict:
        """Load performance statistics."""
        try:
            if PERFORMANCE_FILE.exists():
                return json.loads(PERFORMANCE_FILE.read_text(encoding="utf-8"))
        except BEST_EFFORT_EXCEPTIONS:
            pass
        
        return {
            "total_signals": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl_pct": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "win_rate": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "current_streak": 0,
            "best_streak": 0,
        }
    
    def _save_performance(self):
        """Save performance statistics."""
        try:
            from atomic_io import atomic_write_json
            atomic_write_json(PERFORMANCE_FILE, self.performance)
        except BEST_EFFORT_EXCEPTIONS as e:
            logger.warning(f"Performance save error: {e}")
    
    def broadcast_entry_signal(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        confidence: Optional[float] = None,
        leverage: Optional[int] = None,
        reason: Optional[str] = None,
    ) -> bool:
        """Broadcast entry signal to subscribers."""
        if not self.enabled:
            logger.debug("Signal service not enabled")
            return False
        
        message = format_signal_message(
            signal_type="ENTRY",
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            leverage=leverage,
            reason=reason,
        )
        
        # Log signal
        self._log_signal({
            "type": "ENTRY",
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "confidence": confidence,
            "leverage": leverage,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })
        
        # Broadcast
        success = self._send_telegram(message)
        
        if success:
            self.signals_sent += 1
            self.last_signal_time = datetime.now(timezone.utc)
            self.performance["total_signals"] += 1
            self._save_performance()
        
        return success
    
    def broadcast_exit_signal(
        self,
        symbol: str,
        side: str,
        exit_price: float,
        pnl_pct: float,
        reason: str = "Unknown",
    ) -> bool:
        """Broadcast exit signal."""
        if not self.enabled:
            return False
        
        # Update performance
        if pnl_pct > 0:
            self.performance["wins"] += 1
            self.performance["current_streak"] = max(0, self.performance["current_streak"]) + 1
            self.performance["best_streak"] = max(
                self.performance["best_streak"], 
                self.performance["current_streak"]
            )
            if pnl_pct > self.performance["best_trade"]:
                self.performance["best_trade"] = pnl_pct
        else:
            self.performance["losses"] += 1
            self.performance["current_streak"] = min(0, self.performance["current_streak"]) - 1
            if pnl_pct < self.performance["worst_trade"]:
                self.performance["worst_trade"] = pnl_pct
        
        self.performance["total_pnl_pct"] += pnl_pct
        
        total_trades = self.performance["wins"] + self.performance["losses"]
        if total_trades > 0:
            self.performance["win_rate"] = self.performance["wins"] / total_trades
        
        self._save_performance()
        
        # Format exit message
        emoji = "🎉" if pnl_pct > 0 else "😔"
        result = "KAZANÇ" if pnl_pct > 0 else "KAYIP"
        
        message = f"""
{emoji}{'═'*30}{emoji}
📤 <b>{result} - Pozisyon Kapandı</b>
{emoji}{'═'*30}{emoji}

<b>📊 Sembol:</b> {symbol}
<b>📌 Yön:</b> {side.upper()}
<b>💰 Çıkış:</b> <code>${exit_price:,.2f}</code>
<b>📈 PnL:</b> <code>{'+' if pnl_pct > 0 else ''}{pnl_pct:.2f}%</code>
<b>📝 Sebep:</b> {reason}

{'─'*35}
<b>📊 Performans:</b>
  Win Rate: {self.performance['win_rate']:.0%}
  Toplam: {self.performance['wins']}W / {self.performance['losses']}L
  Net PnL: {'+' if self.performance['total_pnl_pct'] > 0 else ''}{self.performance['total_pnl_pct']:.2f}%

⏰ <b>{datetime.now(timezone.utc).strftime('%H:%M:%S - %d/%m/%Y')}</b>
"""
        
        # Log
        self._log_signal({
            "type": "EXIT",
            "symbol": symbol,
            "side": side,
            "exit_price": exit_price,
            "pnl_pct": pnl_pct,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })
        
        return self._send_telegram(message)
    
    def broadcast_performance_report(self) -> bool:
        """Send performance report."""
        if not self.enabled:
            return False
        
        p = self.performance
        
        message = f"""
📊{'═'*30}📊
<b>PERFORMANS RAPORU</b>
📊{'═'*30}📊

<b>📈 Sinyal İstatistikleri:</b>
  Toplam Sinyal: {p['total_signals']}
  Kazanan: {p['wins']} ✅
  Kaybeden: {p['losses']} ❌
  Win Rate: {p['win_rate']:.0%}

<b>💰 PnL İstatistikleri:</b>
  Toplam PnL: {'+' if p['total_pnl_pct'] > 0 else ''}{p['total_pnl_pct']:.2f}%
  En İyi Trade: +{p['best_trade']:.2f}%
  En Kötü Trade: {p['worst_trade']:.2f}%

<b>🔥 Streak:</b>
  Mevcut: {p['current_streak']}
  En İyi: {p['best_streak']}

{'─'*35}
⏰ <b>{datetime.now(timezone.utc).strftime('%H:%M:%S - %d/%m/%Y')}</b>
🤖 <i>AutoTraderBot Signal Service</i>
"""
        
        return self._send_telegram(message)
    
    def _log_signal(self, signal_data: Dict):
        """Log signal to JSONL file."""
        try:
            with SIGNALS_LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps(signal_data, ensure_ascii=False) + "\n")
        except BEST_EFFORT_EXCEPTIONS as e:
            logger.warning(f"Signal log error: {e}")
    
    def _send_telegram(self, message: str) -> bool:
        """Send message to Telegram."""
        if not requests:
            logger.warning("requests library not available")
            return False
        
        try:
            response = requests.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                json={
                    "chat_id": SIGNAL_CHAT_ID,
                    "text": message,
                    "parse_mode": "HTML",
                },
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Signal broadcast successful")
                return True
            else:
                logger.warning(f"Telegram send failed: {response.status_code}")
                return False
                
        except BEST_EFFORT_EXCEPTIONS as e:
            logger.warning(f"Telegram send error: {e}")
            return False


# =============================================================================
# SINGLETON
# =============================================================================

import threading as _threading_ss

_service_instance: Optional[SignalService] = None
_service_instance_lock = _threading_ss.Lock()


def get_signal_service() -> SignalService:
    """Get global signal service instance (thread-safe)."""
    global _service_instance
    if _service_instance is None:
        with _service_instance_lock:
            if _service_instance is None:
                _service_instance = SignalService()
    return _service_instance


def broadcast_entry(
    symbol: str,
    side: str,
    entry_price: float,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    confidence: Optional[float] = None,
    leverage: Optional[int] = None,
    reason: Optional[str] = None,
) -> bool:
    """Convenience function to broadcast entry signal."""
    service = get_signal_service()
    return service.broadcast_entry_signal(
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        confidence=confidence,
        leverage=leverage,
        reason=reason,
    )


def broadcast_exit(
    symbol: str,
    side: str,
    exit_price: float,
    pnl_pct: float,
    reason: str = "Unknown",
) -> bool:
    """Convenience function to broadcast exit signal."""
    service = get_signal_service()
    return service.broadcast_exit_signal(
        symbol=symbol,
        side=side,
        exit_price=exit_price,
        pnl_pct=pnl_pct,
        reason=reason,
    )


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SIGNAL SERVICE TEST")
    print("=" * 60)
    
    service = get_signal_service()
    
    if not service.enabled:
        print("❌ Signal service not enabled (check TELEGRAM_BOT_TOKEN)")
    else:
        print("✅ Signal service enabled")
        
        # Test broadcast
        print("\nSending test signal...")
        result = service.broadcast_entry_signal(
            symbol="BTC/USDT",
            side="LONG",
            entry_price=105000.0,
            stop_loss=103500.0,
            take_profit=108000.0,
            confidence=0.75,
            leverage=10,
            reason="Test signal - AI Consensus + MTF Alignment",
        )
        
        if result:
            print("✅ Test signal sent!")
        else:
            print("❌ Test signal failed")
