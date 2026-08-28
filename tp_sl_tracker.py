# -*- coding: utf-8 -*-
"""
tp_sl_tracker.py - Enterprise-Level Smart TP/SL Management

[2026-01-20] Created for intelligent TP/SL management
[2026-01-20] v2.0 - Added thread safety, async I/O, comprehensive error handling

Features:
- Thread-safe in-memory operations and async state persistence
- Async file I/O to avoid blocking the trading loop
- Fee-aware calculations for minimum profit
- Profit protection: never moves TP lower or SL in losing direction
- Persistent state across bot restarts
- Comprehensive logging and debugging support
"""
from core.exceptions import BEST_EFFORT_EXCEPTIONS
import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

from atomic_io import atomic_write_json, safe_read_json

try:
    from logger import get_logger
    log = get_logger("tp_sl_tracker")
except ImportError:
    import logging
    log = logging.getLogger("tp_sl_tracker")

# =============================================================================
# CONSTANTS - Centralized configuration
# =============================================================================

# OKX fee rates (configurable for different exchanges)
TAKER_FEE = 0.0005  # 0.05%
MAKER_FEE = 0.0002  # 0.02%
FUNDING_RATE_AVG = 0.0001  # ~0.01% per 8h average

# State file for persistence
STATE_FILE = Path(__file__).resolve().parent / "data" / "tp_sl_state.json"
BACKUP_STATE_FILE = Path(__file__).resolve().parent / "data" / "tp_sl_state.backup.json"

# Minimum change threshold to trigger update (0.1%)
MIN_CHANGE_THRESHOLD = 0.001

# =============================================================================
# THREAD-SAFE TP/SL TRACKER
# =============================================================================

class TPSLTracker:
    """
    Enterprise-level TP/SL order tracker with thread safety and async support.
    
    Key principles:
    1. Thread-safe: In-memory state updates are short locked sections
    2. Non-blocking: Async file I/O runs via asyncio.to_thread
    3. Fault-tolerant: Backup files, graceful error handling
    4. Never widen risk: SL can only move toward profit
    5. Protect profits: TP can only improve for profitable positions
    """
    
    def __init__(self):
        self._async_lock = asyncio.Lock()
        self._lock = threading.Lock()
        self._state: Dict[str, dict] = {}
        self._dirty = False  # Track if state needs saving
        self._save_pending = False  # Debounce rapid saves
        self._load_state()
    
    # =========================================================================
    # STATE PERSISTENCE (Thread-Safe)
    # =========================================================================
    
    def _load_state(self) -> None:
        """Load persisted TP/SL state from disk without holding the state lock."""
        loaded: Dict[str, dict] = {}
        loaded_from = ""
        try:
            payload = safe_read_json(STATE_FILE, default=None)
            if isinstance(payload, dict):
                loaded = payload
                loaded_from = "state"
            else:
                backup = safe_read_json(BACKUP_STATE_FILE, default=None)
                if isinstance(backup, dict):
                    loaded = backup
                    loaded_from = "backup"
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
            log.warning(f"[TP_SL_TRACKER] Failed to load state: {e}")
            loaded = {}

        with self._lock:
            self._state = loaded
        if loaded_from == "state":
            log.info(f"[TP_SL_TRACKER] Loaded state for {len(loaded)} symbols")
        elif loaded_from == "backup":
            log.warning(f"[TP_SL_TRACKER] Loaded from BACKUP for {len(loaded)} symbols")
    
    def _try_load_backup(self) -> None:
        """Attempt to load from backup file."""
        backup = safe_read_json(BACKUP_STATE_FILE, default={})
        with self._lock:
            self._state = backup if isinstance(backup, dict) else {}
        if backup:
            log.warning("[TP_SL_TRACKER] Recovered from backup file")

    def _state_snapshot(self) -> Dict[str, dict]:
        with self._lock:
            return json.loads(json.dumps(self._state, ensure_ascii=False, default=str))

    @staticmethod
    def _write_state_snapshot(snapshot: Dict[str, dict]) -> bool:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        ok = atomic_write_json(STATE_FILE, snapshot)
        if ok:
            atomic_write_json(BACKUP_STATE_FILE, snapshot, warn_on_failure=False)
        return bool(ok)
    
    def _save_state_sync(self) -> bool:
        """
        Synchronously save state to disk (called from thread pool).
        
        Returns:
            True if save successful, False otherwise
        """
        try:
            snapshot = self._state_snapshot()
            if self._write_state_snapshot(snapshot):
                with self._lock:
                    self._dirty = False
                    self._save_pending = False
                return True
            with self._lock:
                self._save_pending = False
            return False
        except (OSError, TypeError, ValueError) as e:
            log.error(f"[TP_SL_TRACKER] Failed to save state: {e}")
            with self._lock:
                self._save_pending = False
            return False

    async def _save_state_to_thread(self) -> bool:
        async with self._async_lock:
            snapshot = self._state_snapshot()
            result = await asyncio.to_thread(self._write_state_snapshot, snapshot)
            if result:
                with self._lock:
                    self._dirty = False
                    self._save_pending = False
            else:
                with self._lock:
                    self._save_pending = False
            return bool(result)

    async def _load_state_async(self) -> None:
        async with self._async_lock:
            payload = await asyncio.to_thread(safe_read_json, STATE_FILE, None)
            if not isinstance(payload, dict):
                payload = await asyncio.to_thread(safe_read_json, BACKUP_STATE_FILE, {})
            with self._lock:
                self._state = payload if isinstance(payload, dict) else {}
                self._dirty = False
    
    async def _save_state_async(self) -> bool:
        """
        Asynchronously save state without blocking the event loop.
        
        Returns:
            True if save successful, False otherwise
        """
        if self._save_pending:
            return True  # Debounce rapid saves
        
        self._save_pending = True
        
        try:
            return await self._save_state_to_thread()
        except RuntimeError:
            # No event loop running, fall back to sync
            return self._save_state_sync()
        except (OSError, TypeError, ValueError) as e:
            log.error(f"[TP_SL_TRACKER] Async save failed: {e}")
            self._save_pending = False
            return False
    
    def _save_state(self) -> None:
        """
        Smart save: uses async if event loop available, else sync.
        Thread-safe with debouncing.
        """
        if not self._dirty:
            return
            
        try:
            loop = asyncio.get_running_loop()
            # Schedule async save without blocking
            asyncio.create_task(self._save_state_async())
        except RuntimeError:
            # No event loop, use sync
            self._save_state_sync()
    
    # =========================================================================
    # CORE READ OPERATIONS (Thread-Safe)
    # =========================================================================
    
    def get_current(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Get current TP/SL for symbol (thread-safe).
        
        Returns:
            (tp_price, sl_price) - Either may be None if not set
        """
        with self._lock:
            entry = self._state.get(symbol, {})
            return entry.get("tp"), entry.get("sl")
    
    def get_entry_info(self, symbol: str) -> Tuple[Optional[float], Optional[str]]:
        """
        Get entry price and side for symbol (thread-safe).
        
        Returns:
            (entry_price, side) - Either may be None if not recorded
        """
        with self._lock:
            entry = self._state.get(symbol, {})
            return entry.get("entry"), entry.get("side")
    
    def get_all_tracked_symbols(self) -> list:
        """Get list of all symbols with active TP/SL tracking (thread-safe)."""
        with self._lock:
            return list(self._state.keys())
    
    def get_state_summary(self) -> dict:
        """Get summary of current tracking state for debugging (thread-safe)."""
        with self._lock:
            summary = {}
            now = time.time()
            for sym, data in self._state.items():
                summary[sym] = {
                    "tp": data.get("tp"),
                    "sl": data.get("sl"),
                    "side": data.get("side"),
                    "entry": data.get("entry"),
                    "age_sec": round(now - data.get("updated_at", 0), 1)
                }
            return summary
    
    # =========================================================================
    # FEE-AWARE CALCULATIONS
    # =========================================================================
    
    def calculate_min_tp_for_profit(
        self, 
        entry_price: float, 
        side: str,
        leverage: int = 1,
        target_profit_pct: float = 0.005  # 0.5% minimum net profit target
    ) -> float:
        """
        Calculate minimum TP price to cover all fees and achieve target profit.
        
        Fee breakdown:
        - Entry taker fee: 0.05%
        - Exit maker fee: 0.02%  
        - Estimated funding (1-2 periods): ~0.02%
        Total: ~0.09%
        
        With leverage, the price move required is: (fees + target) / leverage
        
        Args:
            entry_price: Position entry price
            side: "long" or "short"
            leverage: Position leverage
            target_profit_pct: Target net profit after fees (default 0.5%)
            
        Returns:
            Minimum TP price to achieve target profit
        """
        # Total fee burden
        total_fee_pct = TAKER_FEE + MAKER_FEE + (FUNDING_RATE_AVG * 2)
        
        # Required price movement considering leverage
        required_move_pct = (total_fee_pct + target_profit_pct) / max(1, leverage)
        
        if side == "long":
            min_tp = entry_price * (1 + required_move_pct)
        else:
            min_tp = entry_price * (1 - required_move_pct)
        
        return min_tp
    
    # =========================================================================
    # INTELLIGENT UPDATE DECISIONS (Thread-Safe)
    # =========================================================================
    
    def should_update_tp(
        self, 
        symbol: str, 
        new_tp: float, 
        entry_price: float,
        side: str,
        current_price: float = None
    ) -> Tuple[bool, str]:
        """
        Determine if TP should be updated (thread-safe).
        
        Rules:
        1. No existing TP → Always update
        2. Change < 0.1% → Skip (noise reduction)
        3. TP improvement (higher for long, lower for short) → Update
        4. Position is profitable and new TP is worse → BLOCK (profit protection)
        5. Position is losing and new TP is worse → Allow (adjustment)
        
        Returns:
            (should_update: bool, reason: str)
        """
        with self._lock:
            old_tp = self._state.get(symbol, {}).get("tp")
            
            # Rule 1: No existing TP
            if old_tp is None:
                return True, "No existing TP - creating new"
            
            # Validate new_tp
            if new_tp is None or new_tp <= 0:
                return False, "Invalid new TP value"
            
            # Rule 2: Change threshold
            try:
                change_pct = abs(new_tp - old_tp) / old_tp
                if change_pct < MIN_CHANGE_THRESHOLD:
                    return False, f"Change too small ({change_pct*100:.3f}% < 0.1%)"
            except (ZeroDivisionError, TypeError):
                pass
            
            # Determine if position is profitable
            is_profitable = False
            if current_price is not None and entry_price is not None:
                try:
                    if side == "long":
                        is_profitable = float(current_price) > float(entry_price)
                    else:
                        is_profitable = float(current_price) < float(entry_price)
                except (TypeError, ValueError):
                    pass
            
            # Rule 3 & 4: Direction check with profit protection
            if side == "long":
                if new_tp > old_tp:
                    return True, f"TP raised: {old_tp:.6f} → {new_tp:.6f}"
                elif is_profitable:
                    return False, "Position profitable - won't lower TP (profit protection)"
                else:
                    return True, f"TP adjusted down (losing position): {old_tp:.6f} → {new_tp:.6f}"
            else:  # short
                if new_tp < old_tp:
                    return True, f"TP lowered: {old_tp:.6f} → {new_tp:.6f}"
                elif is_profitable:
                    return False, "Position profitable - won't raise TP (profit protection)"
                else:
                    return True, f"TP adjusted up (losing position): {old_tp:.6f} → {new_tp:.6f}"
    
    def should_update_sl(
        self, 
        symbol: str, 
        new_sl: float, 
        entry_price: float,
        side: str,
        current_price: float = None
    ) -> Tuple[bool, str]:
        """
        Determine if SL should be updated (thread-safe).
        
        Rules:
        1. No existing SL → ALWAYS update (CRITICAL for protection)
        2. Change < 0.1% → Skip (noise reduction)
        3. SL moves toward profit (trailing) → Update
        4. SL moves away from profit (widening risk) → BLOCK
        
        The cardinal rule: NEVER widen risk. SL can only get tighter.
        
        Returns:
            (should_update: bool, reason: str)
        """
        with self._lock:
            old_sl = self._state.get(symbol, {}).get("sl")
            
            # Rule 1: No existing SL - CRITICAL
            if old_sl is None:
                return True, "No existing SL - CRITICAL: creating protection"
            
            # Validate new_sl
            if new_sl is None or new_sl <= 0:
                return False, "Invalid new SL value"
            
            # Rule 2: Change threshold
            try:
                change_pct = abs(new_sl - old_sl) / old_sl
                if change_pct < MIN_CHANGE_THRESHOLD:
                    return False, f"Change too small ({change_pct*100:.3f}% < 0.1%)"
            except (ZeroDivisionError, TypeError):
                pass
            
            # Rule 3 & 4: Direction check - NEVER widen risk
            if side == "long":
                # For long: SL should be below entry and can only move UP (tighter)
                if new_sl >= old_sl:
                    return True, f"SL tightened (trailing): {old_sl:.6f} → {new_sl:.6f}"
                else:
                    return False, f"Won't widen SL for long ({old_sl:.6f} → {new_sl:.6f})"
            else:  # short
                # For short: SL should be above entry and can only move DOWN (tighter)
                if new_sl <= old_sl:
                    return True, f"SL tightened (trailing): {old_sl:.6f} → {new_sl:.6f}"
                else:
                    return False, f"Won't widen SL for short ({old_sl:.6f} → {new_sl:.6f})"
    
    # =========================================================================
    # WRITE OPERATIONS (Thread-Safe with Async Save)
    # =========================================================================
    
    def record_update(
        self, 
        symbol: str, 
        tp: float = None, 
        sl: float = None, 
        entry: float = None, 
        side: str = None
    ) -> None:
        """
        Record a TP/SL update after successful order submission (thread-safe).
        
        Args:
            symbol: Trading pair
            tp: Take-profit price (optional)
            sl: Stop-loss price (optional)
            entry: Entry price (optional)
            side: Position side "long" or "short" (optional)
        """
        with self._lock:
            if symbol not in self._state:
                self._state[symbol] = {}
            
            if tp is not None:
                self._state[symbol]["tp"] = float(tp)
            if sl is not None:
                self._state[symbol]["sl"] = float(sl)
            if entry is not None:
                self._state[symbol]["entry"] = float(entry)
            if side is not None:
                self._state[symbol]["side"] = str(side).lower()
            
            self._state[symbol]["updated_at"] = time.time()
            self._dirty = True
        
        # Save outside lock to minimize lock duration
        self._save_state()
        log.info(f"[TP_SL_TRACKER] {symbol}: recorded TP={tp}, SL={sl}")
    
    def clear_position(self, symbol: str) -> None:
        """
        Clear TP/SL state when position is closed (thread-safe).
        
        Should be called when:
        - Position hits TP or SL
        - Position is manually closed
        - Position is liquidated
        """
        with self._lock:
            if symbol in self._state:
                del self._state[symbol]
                self._dirty = True
        
        # Save outside lock
        self._save_state()
        log.info(f"[TP_SL_TRACKER] {symbol}: cleared state (position closed)")
    
    def clear_all(self) -> None:
        """Clear all tracked positions (thread-safe). Use with caution."""
        with self._lock:
            self._state = {}
            self._dirty = True
        
        self._save_state()
        log.warning("[TP_SL_TRACKER] All positions cleared!")
    
    def sync_active_positions(self, active_symbols: list[str]) -> None:
        """
        [FIX] Clean up ghost orders. Removes any tracked TP/SL state for symbols 
        that are not in the provided active_symbols list. 
        Should be called on bot startup or periodically to prevent API spam.
        """
        removed_count = 0
        with self._lock:
            current_symbols = list(self._state.keys())
            for sym in current_symbols:
                if sym not in active_symbols:
                    del self._state[sym]
                    removed_count += 1
            if removed_count > 0:
                self._dirty = True
        
        if removed_count > 0:
            self._save_state()
            log.info(f"[TP_SL_TRACKER] Synced positions: Removed {removed_count} ghost orders.")
    
    # =========================================================================
    # ASYNC INTERFACE (For async contexts)
    # =========================================================================
    
    async def record_update_async(
        self, 
        symbol: str, 
        tp: float = None, 
        sl: float = None, 
        entry: float = None, 
        side: str = None
    ) -> None:
        """Async version of record_update for use in async contexts."""
        await asyncio.to_thread(
            self.record_update,
            symbol,
            tp=tp,
            sl=sl,
            entry=entry,
            side=side,
        )
    
    async def clear_position_async(self, symbol: str) -> None:
        """Async version of clear_position."""
        await asyncio.to_thread(self.clear_position, symbol)
        
    async def sync_active_positions_async(self, active_symbols: list[str]) -> None:
        """Async version of sync_active_positions."""
        await asyncio.to_thread(self.sync_active_positions, active_symbols)

# =============================================================================
# SINGLETON INSTANCE (Thread-Safe)
# =============================================================================

_tracker: Optional[TPSLTracker] = None
_tracker_lock = threading.Lock()


def get_tp_sl_tracker() -> TPSLTracker:
    """Get the singleton TPSLTracker instance (thread-safe)."""
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            # Double-check locking pattern
            if _tracker is None:
                _tracker = TPSLTracker()
    return _tracker


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def reset_tracker() -> None:
    """Reset the singleton instance (for testing purposes)."""
    global _tracker
    with _tracker_lock:
        _tracker = None


def get_tracker_health() -> dict:
    """Get health status of the tracker for monitoring."""
    try:
        tracker = get_tp_sl_tracker()
        summary = tracker.get_state_summary()
        return {
            "healthy": True,
            "tracked_symbols": len(summary),
            "symbols": list(summary.keys()),
            "state_file_exists": STATE_FILE.exists(),
            "backup_exists": BACKUP_STATE_FILE.exists(),
        }
    except BEST_EFFORT_EXCEPTIONS as e:
        return {
            "healthy": False,
            "error": str(e)
        }
