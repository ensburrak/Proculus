# -*- coding: utf-8 -*-
from __future__ import annotations

"""
smart_exit_manager.py - Trailing Stop + Partial Exit System
=============================================================

[2026-01-15] Kar kaybını önlemek için profesyonel çıkış stratejisi.
[D1] Breakeven + Trailing Hybrid: Kâr arttıkça trailing sıkılaşır.
[D4] ATR-based TP levels: Sabit R yerine ATR bazlı dinamik TP.

Hybrid Trailing Stratejisi (D1):
- 1.0R kâr → SL breakeven'a (risk = 0)
- 1.5R kâr → Trailing başlar (2x ATR mesafe)
- 3.0R kâr → Trailing sıkılaşır (1.5x ATR)
- 5.0R kâr → Trailing çok sıkı (1x ATR)

ATR-Based Exit (D4):
- TP1 = entry + 2*ATR → %25 kapat
- TP2 = entry + 4*ATR → %25 kapat
- TP3 = trailing ile kalan %50
- Trend güçlü (ADX>30): TP1'de sadece %15
- Range (ADX<20): TP1'de %40
- Her partial exit sonrası SL yukarı taşınır

Kullanım:
    from smart_exit_manager import SmartExitManager

    exit_mgr = SmartExitManager()
    exit_mgr.register_position("BTC/USDT", "long", entry=42000, sl=41000, size=0.1)

    # Her fiyat update'inde:
    action = exit_mgr.update_position("BTC/USDT", current_price=43500, atr=500)
    # action = {"action": "partial_exit", "exit_pct": 0.30, "new_sl": 42000}
"""



from core.exceptions import BEST_EFFORT_EXCEPTIONS
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
import logging

try:
    from logger import get_logger
    log = get_logger("exit_mgr")
except ImportError:
    log = logging.getLogger("exit_mgr")
    log.setLevel(logging.INFO)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ExitLevel:
    """Single exit level definition."""
    trigger_r: float  # R multiple to trigger
    exit_pct: float  # Percentage of position to exit
    move_sl_to: str = "breakeven"  # "breakeven", "trailing", or price value
    trailing_enabled: bool = False
    trailing_atr_mult: float = 2.0  # [D1] ATR multiplier for this level


@dataclass
class PositionState:
    """State of a single position."""
    symbol: str
    side: Literal["long", "short"]
    entry_price: float
    initial_sl: float
    current_sl: float
    initial_size: float
    current_size: float
    highest_price: float = 0.0  # For trailing (long)
    lowest_price: float = float("inf")  # For trailing (short)

    # Exit tracking
    exits_completed: List[int] = field(default_factory=list)  # Indices of completed exits
    is_trailing: bool = False
    is_at_breakeven: bool = False

    # [D1] Progressive trailing state
    trailing_atr_mult: float = 2.0  # Mevcut trailing mesafesi
    peak_r: float = 0.0  # Ulaşılan en yüksek R

    # [D2] Time-based holding
    opened_at: str = ""
    time_partial_exits: int = 0
    optimal_hold_minutes: float = 0.0

    # Metrics
    current_r: float = 0.0
    unrealized_pnl: float = 0.0
    last_atr: float = 0.0
    breakeven_cost_bps: float = 0.0

    # Meta
    created_at: str = ""
    last_update: str = ""


@dataclass
class ExitAction:
    """Action to take for a position."""
    action: Literal["hold", "partial_exit", "full_exit", "update_sl", "stop_hit", "time_stop"]
    exit_pct: float = 0.0
    new_sl: float = 0.0
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# [D1] PROGRESSIVE TRAILING ATR SCHEDULE
# Kâr arttıkça trailing distance azalır → kârı daha sıkı korur
# =============================================================================
PROGRESSIVE_TRAILING = [
    # (min_r, trailing_atr_mult)
    (1.0, 2.5),   # 1R'den sonra geniş trailing
    (1.5, 2.0),   # 1.5R → orta trailing
    (3.0, 1.5),   # 3R → sıkı trailing
    (5.0, 1.0),   # 5R → çok sıkı trailing
    (8.0, 0.75),  # 8R → en sıkı (kârı kilitle)
]


def _get_progressive_atr_mult(current_r: float) -> float:
    """[D1] Kâr seviyesine göre trailing ATR çarpanını döndür."""
    mult = 2.5  # varsayılan
    for min_r, atr_mult in PROGRESSIVE_TRAILING:
        if current_r >= min_r:
            mult = atr_mult
    return mult


# =============================================================================
# [D4] ADX-BASED PARTIAL EXIT PERCENTAGES
# Trend gücüne göre TP1'deki çıkış oranı değişir
# =============================================================================

def _adx_adjusted_exit_pct(base_pct: float, adx: Optional[float], level_idx: int) -> float:
    """[D4] ADX'e göre ilk TP seviyesindeki çıkış oranını ayarla.

    - ADX > 30 (güçlü trend): TP1'de az sat → trending devam etsin
    - ADX < 20 (range): TP1'de çok sat → kâr hızlı al
    - Sadece ilk TP seviyesi etkilenir
    """
    if level_idx != 0 or adx is None:
        return base_pct

    if adx > 30:
        # Güçlü trend: TP1'de sadece %15 kapat, kalanı trailing'e bırak
        return 0.15
    elif adx < 20:
        # Range/zayıf: TP1'de %40 kapat
        return 0.40
    return base_pct


# =============================================================================
# SMART EXIT MANAGER
# =============================================================================


class SmartExitManager:
    """
    Professional exit management with:
    - [FAZA 6.6] Regime-based dynamic R/R exit levels
    - [D1] Breakeven + Progressive Trailing Hybrid
    - [D4] ATR-based TP levels with ADX-adjusted partial exits
    - Kademeli partial exits
    - Her partial exit sonrası SL yukarı taşınır
    """

    STATE_PATH = Path("data/exit_positions.json")

    # [FAZA 6.6] Regime-based exit level presets
    # [D1] trailing_atr_mult eklendi → kâr artınca trailing sıkılaşır
    REGIME_EXIT_LEVELS: Dict[str, List[ExitLevel]] = {
        "trending_aligned": [
            ExitLevel(trigger_r=1.0, exit_pct=0.0, move_sl_to="breakeven", trailing_atr_mult=2.5),
            ExitLevel(trigger_r=2.0, exit_pct=0.30, trailing_enabled=True, trailing_atr_mult=2.0),
            ExitLevel(trigger_r=4.0, exit_pct=0.40, trailing_enabled=True, trailing_atr_mult=1.5),
            ExitLevel(trigger_r=7.0, exit_pct=0.30, trailing_enabled=True, trailing_atr_mult=1.0),
        ],
        "trending_counter": [
            ExitLevel(trigger_r=1.0, exit_pct=0.0, move_sl_to="breakeven", trailing_atr_mult=2.0),
            ExitLevel(trigger_r=1.5, exit_pct=0.30, trailing_enabled=True, trailing_atr_mult=1.5),
            ExitLevel(trigger_r=2.5, exit_pct=0.30, trailing_enabled=True, trailing_atr_mult=1.0),
            ExitLevel(trigger_r=4.0, exit_pct=0.40, trailing_enabled=True, trailing_atr_mult=0.75),
        ],
        "sideways": [
            ExitLevel(trigger_r=0.8, exit_pct=0.0, move_sl_to="breakeven", trailing_atr_mult=1.5),
            ExitLevel(trigger_r=1.5, exit_pct=0.40, trailing_enabled=True, trailing_atr_mult=1.2),
            ExitLevel(trigger_r=2.5, exit_pct=0.30, trailing_enabled=True, trailing_atr_mult=1.0),
            ExitLevel(trigger_r=3.5, exit_pct=0.30, trailing_enabled=True, trailing_atr_mult=0.75),
        ],
        "default": [
            ExitLevel(trigger_r=1.0, exit_pct=0.0, move_sl_to="breakeven", trailing_atr_mult=2.0),
            ExitLevel(trigger_r=1.5, exit_pct=0.25, trailing_enabled=True, trailing_atr_mult=2.0),
            ExitLevel(trigger_r=3.0, exit_pct=0.25, trailing_enabled=True, trailing_atr_mult=1.5),
            ExitLevel(trigger_r=5.0, exit_pct=0.50, trailing_enabled=True, trailing_atr_mult=1.0),
        ],
    }

    # [FAZA 6.6] Regime-based trailing ATR multiplier (başlangıç değeri)
    TRAILING_ATR_BY_REGIME: Dict[str, float] = {
        "trending_aligned": 3.0,
        "trending_counter": 2.0,
        "sideways": 1.5,
        "default": 2.5,
    }

    # Legacy default (backward compat)
    DEFAULT_EXIT_LEVELS = REGIME_EXIT_LEVELS["default"]
    TRAILING_ATR_MULT = 2.5

    def __init__(self, exit_levels: Optional[List[ExitLevel]] = None,
                 regime: Optional[str] = None, direction: Optional[str] = None):
        self._regime_key = self._resolve_regime_key(regime, direction)
        if exit_levels is not None:
            self.exit_levels = exit_levels
        else:
            self.exit_levels = self.REGIME_EXIT_LEVELS.get(
                self._regime_key, self.DEFAULT_EXIT_LEVELS
            )
        self.TRAILING_ATR_MULT = self.TRAILING_ATR_BY_REGIME.get(
            self._regime_key, 2.5
        )
        self.positions: Dict[str, PositionState] = {}
        self._load_state()

    @staticmethod
    def _resolve_regime_key(regime: Optional[str], direction: Optional[str]) -> str:
        """[FAZA 6.6] Regime + direction -> exit preset key."""
        if regime is None:
            return "default"
        r = regime.upper()
        d = (direction or "").lower()
        if r == "SIDEWAYS":
            return "sideways"
        if r in ("BULL", "STRONG_BULL", "WEAK_BULL"):
            return "trending_aligned" if d == "long" else "trending_counter"
        if r in ("BEAR", "STRONG_BEAR", "WEAK_BEAR"):
            return "trending_aligned" if d == "short" else "trending_counter"
        return "default"

    def update_regime(self, regime: str, direction: str) -> None:
        """[FAZA 6.6] Regime degistiginde exit seviyelerini guncelle."""
        new_key = self._resolve_regime_key(regime, direction)
        if new_key != self._regime_key:
            self._regime_key = new_key
            self.exit_levels = self.REGIME_EXIT_LEVELS.get(new_key, self.DEFAULT_EXIT_LEVELS)
            self.TRAILING_ATR_MULT = self.TRAILING_ATR_BY_REGIME.get(new_key, 2.5)
            log.info("[EXIT] Regime updated -> %s: levels=%s, trailing_atr=%.1f",
                     new_key,
                     [f"{l.trigger_r}R/{l.exit_pct:.0%}" for l in self.exit_levels],
                     self.TRAILING_ATR_MULT)
    
    def register_position(
        self,
        symbol: str,
        side: Literal["long", "short"],
        entry_price: float,
        stop_loss: float,
        size: float,
        atr: float = 0.0,
        breakeven_cost_bps: float = 0.0,
    ) -> PositionState:
        """
        Register a new position for exit management.

        Returns:
            PositionState
        """
        now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        position = PositionState(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            initial_sl=stop_loss,
            current_sl=stop_loss,
            initial_size=size,
            current_size=size,
            highest_price=entry_price if side == "long" else float("inf"),
            lowest_price=entry_price if side == "short" else 0.0,
            trailing_atr_mult=self.TRAILING_ATR_MULT,
            opened_at=now_iso,
            optimal_hold_minutes=0.0,
            last_atr=max(0.0, float(atr or 0.0)),
            breakeven_cost_bps=max(0.0, float(breakeven_cost_bps or 0.0)),
            created_at=now_iso,
            last_update=now_iso,
        )

        self.positions[symbol] = position
        self._save_state()

        log.info(
            "[EXIT] Registered %s %s @ %.2f | SL: %.2f | Size: %.4f",
            symbol, side.upper(), entry_price, stop_loss, size,
        )

        return position
    
    def update_position(
        self,
        symbol: str,
        current_price: float,
        atr: float = None,
        adx: float = None,
    ) -> ExitAction:
        """
        Update position and check for exit actions.

        Args:
            symbol: Sembol adı.
            current_price: Güncel fiyat.
            atr: Current ATR değeri (trailing stop için).
            adx: Current ADX değeri ([D4] trend gücüne göre TP ayarı).

        Returns:
            ExitAction with what to do
        """
        if symbol not in self.positions:
            return ExitAction(action="hold", reason="Position not found")

        pos = self.positions[symbol]
        pos.last_update = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        if atr is not None and atr > 0:
            pos.last_atr = float(atr)

        # Update high/low
        if pos.side == "long":
            if current_price > pos.highest_price:
                pos.highest_price = current_price
        else:
            if current_price < pos.lowest_price:
                pos.lowest_price = current_price

        # Calculate current R
        risk_per_unit = abs(pos.entry_price - pos.initial_sl)
        if risk_per_unit > 0:
            if pos.side == "long":
                pnl_per_unit = current_price - pos.entry_price
            else:
                pnl_per_unit = pos.entry_price - current_price
            pos.current_r = pnl_per_unit / risk_per_unit
        else:
            pos.current_r = 0.0

        # [D1] Track peak R for progressive trailing
        if pos.current_r > pos.peak_r:
            pos.peak_r = pos.current_r

        # Check stop loss hit
        if pos.side == "long" and current_price <= pos.current_sl:
            log.warning("[EXIT] %s STOP HIT @ %.2f (SL: %.2f)", symbol, current_price, pos.current_sl)
            self._save_state()
            action = ExitAction(
                action="stop_hit",
                exit_pct=1.0,
                reason=f"Stop loss hit at {current_price}",
                details={
                    "final_r": round(pos.current_r, 2),
                    "was_trailing": pos.is_trailing,
                    "commit": {"remove_position": True},
                },
            )
            return action

        if pos.side == "short" and current_price >= pos.current_sl:
            log.warning("[EXIT] %s STOP HIT @ %.2f (SL: %.2f)", symbol, current_price, pos.current_sl)
            self._save_state()
            action = ExitAction(
                action="stop_hit",
                exit_pct=1.0,
                reason=f"Stop loss hit at {current_price}",
                details={
                    "final_r": round(pos.current_r, 2),
                    "was_trailing": pos.is_trailing,
                    "commit": {"remove_position": True},
                },
            )
            return action

        # [D2] Time stop: pozisyon çok uzun süredir açık ve kârsızsa kapat
        time_stop_action = self._check_time_stop(pos, current_price, atr=atr)
        if time_stop_action is not None:
            self._save_state()
            return time_stop_action

        # Fee adjustment in R-multiple terms (entry + exit round-trip)
        _fee_r_adj = 0.0
        try:
            _cost_bps = float(getattr(pos, "breakeven_cost_bps", 0) or 0)
            _risk_per_unit = abs(float(pos.entry_price or 0) - float(pos.initial_sl or 0))
            if _cost_bps > 0 and _risk_per_unit > 0 and float(pos.entry_price or 0) > 0:
                _fee_r_adj = (float(pos.entry_price) * _cost_bps / 10_000.0) / _risk_per_unit
        except (TypeError, ValueError, ZeroDivisionError):
            _fee_r_adj = 0.0

        # Check exit levels
        for i, level in enumerate(self.exit_levels):
            if i in pos.exits_completed:
                continue  # Already executed this level

            # Apply fee adjustment only to profit-taking levels, not breakeven/SL-only
            effective_trigger = level.trigger_r + (_fee_r_adj if level.exit_pct > 0 else 0.0)
            if pos.current_r >= effective_trigger:
                # [D4] ADX-adjusted exit percentage
                actual_exit_pct = _adx_adjusted_exit_pct(level.exit_pct, adx, i)

                # Calculate new SL
                new_sl = pos.current_sl
                is_at_breakeven_after = pos.is_at_breakeven
                is_trailing_after = pos.is_trailing
                trailing_atr_mult_after = pos.trailing_atr_mult
                if level.move_sl_to == "breakeven" and not pos.is_at_breakeven:
                    new_sl = self._breakeven_stop_price(pos)
                    is_at_breakeven_after = True
                    log.info("[EXIT] %s SL → breakeven @ %.2f (risk=0)", symbol, new_sl)

                if level.trailing_enabled:
                    is_trailing_after = True
                    trailing_atr_mult_after = level.trailing_atr_mult

                # [D4] Her partial exit sonrası SL'yi yukarı taşı
                if actual_exit_pct > 0 and is_at_breakeven_after and atr and atr > 0:
                    # SL'yi mevcut kârın yarısına taşı
                    if pos.side == "long":
                        profit_sl = pos.entry_price + (current_price - pos.entry_price) * 0.5
                        new_sl = max(new_sl, profit_sl)
                    else:
                        profit_sl = pos.entry_price - (pos.entry_price - current_price) * 0.5
                        new_sl = min(new_sl, profit_sl)

                if actual_exit_pct > 0:
                    log.info(
                        "[EXIT] %s HIT %.1fR | Exit %.0f%% | R=%.2f | trailing_atr=%.1f",
                        symbol, level.trigger_r, actual_exit_pct * 100,
                        pos.current_r, pos.trailing_atr_mult,
                    )
                else:
                    log.info(
                        "[EXIT] %s HIT %.1fR | SL→breakeven | R=%.2f",
                        symbol, level.trigger_r, pos.current_r,
                    )

                self._save_state()

                action_type = "partial_exit" if actual_exit_pct > 0 else "update_sl"
                return ExitAction(
                    action=action_type,
                    exit_pct=actual_exit_pct,
                    new_sl=new_sl,
                    reason=f"Hit {level.trigger_r}R target",
                    details={
                        "trailing_atr_mult": trailing_atr_mult_after,
                        "adx": adx,
                        "peak_r": round(pos.peak_r, 2),
                        "exit_level_index": i,
                        "effective_trigger_r": round(float(effective_trigger), 6),
                        "commit": {
                            "completed_exit_index": i,
                            "is_at_breakeven": is_at_breakeven_after,
                            "is_trailing": is_trailing_after,
                            "trailing_atr_mult": trailing_atr_mult_after,
                            "current_sl": float(new_sl),
                            "remaining_size": max(0.0, float(pos.current_size) - (float(pos.initial_size) * actual_exit_pct)),
                        },
                    },
                )

        # [D1] Progressive trailing: Update trailing stop with R-based ATR tightening
        if pos.is_trailing and atr and atr > 0:
            # Progressive ATR mult: kâr arttıkça mesafe azalır
            progressive_mult = min(pos.trailing_atr_mult, _get_progressive_atr_mult(pos.current_r))

            shadow_pos = PositionState(**{**pos.__dict__, "trailing_atr_mult": progressive_mult})
            new_trailing_sl = self._calculate_trailing_sl(shadow_pos, current_price, atr)

            if new_trailing_sl != pos.current_sl:
                log.info(
                    "[EXIT] %s Trailing SL: %.2f → %.2f (ATR mult=%.2f, R=%.1f)",
                    symbol, pos.current_sl, new_trailing_sl,
                    progressive_mult, pos.current_r,
                )
                self._save_state()

                return ExitAction(
                    action="update_sl",
                    new_sl=new_trailing_sl,
                    reason=f"Trailing stop updated (ATR×{progressive_mult:.1f})",
                    details={
                        "trailing_atr_mult": progressive_mult,
                        "peak_r": round(pos.peak_r, 2),
                        "commit": {
                            "current_sl": float(new_trailing_sl),
                            "trailing_atr_mult": float(progressive_mult),
                            "is_trailing": True,
                        },
                    },
                )

        self._save_state()
        return ExitAction(action="hold", reason=f"R={pos.current_r:.2f}")

    def commit_action(
        self,
        symbol: str,
        action: ExitAction,
        *,
        executed_size: float | None = None,
    ) -> Optional[PositionState]:
        """Persist an executable exit decision only after exchange-side confirmation."""
        pos = self.positions.get(symbol)
        if pos is None:
            return None

        details = action.details if isinstance(action.details, dict) else {}
        commit = details.get("commit") if isinstance(details.get("commit"), dict) else {}
        if bool(commit.get("remove_position")) or action.action in {"full_exit", "stop_hit", "time_stop"}:
            del self.positions[symbol]
            self._save_state()
            return None

        exit_level_index = commit.get("completed_exit_index")
        if exit_level_index is not None:
            try:
                idx = int(exit_level_index)
                if idx not in pos.exits_completed:
                    pos.exits_completed.append(idx)
            except (TypeError, ValueError):
                pass

        for attr_name in ("is_at_breakeven", "is_trailing"):
            if attr_name in commit:
                setattr(pos, attr_name, bool(commit.get(attr_name)))
        if "trailing_atr_mult" in commit:
            try:
                pos.trailing_atr_mult = float(commit.get("trailing_atr_mult") or pos.trailing_atr_mult)
            except (TypeError, ValueError):
                pass
        if "current_sl" in commit:
            try:
                pos.current_sl = float(commit.get("current_sl") or pos.current_sl)
            except (TypeError, ValueError):
                pass

        if action.action == "partial_exit":
            qty = executed_size
            if qty is None:
                try:
                    qty = float(pos.initial_size) * float(action.exit_pct or 0.0)
                except (TypeError, ValueError):
                    qty = 0.0
            try:
                pos.current_size = max(0.0, float(pos.current_size) - float(qty or 0.0))
            except (TypeError, ValueError):
                pass
        elif "remaining_size" in commit:
            try:
                pos.current_size = max(0.0, float(commit.get("remaining_size") or 0.0))
            except (TypeError, ValueError):
                pass

        if "time_partial_exits_increment" in commit:
            try:
                pos.time_partial_exits += max(0, int(commit.get("time_partial_exits_increment") or 0))
            except (TypeError, ValueError):
                pass

        pos.last_update = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        if pos.current_size <= 1e-12:
            del self.positions[symbol]
            self._save_state()
            return None
        self._save_state()
        return pos
    
    def close_position(self, symbol: str, reason: str = "Manual close"):
        """Remove position from tracking."""
        if symbol in self.positions:
            del self.positions[symbol]
            self._save_state()
            log.info(f"[EXIT] {symbol} closed: {reason}")
    
    def get_position(self, symbol: str) -> Optional[PositionState]:
        """Get current position state."""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, PositionState]:
        """Get all tracked positions."""
        return self.positions
    
    def get_position_status(self, symbol: str) -> Dict:
        """Get position status as dict."""
        pos = self.positions.get(symbol)
        if not pos:
            return {}
        
        return {
            "symbol": pos.symbol,
            "side": pos.side,
            "entry_price": pos.entry_price,
            "current_sl": pos.current_sl,
            "initial_size": pos.initial_size,
            "current_size": pos.current_size,
            "current_r": round(pos.current_r, 2),
            "is_trailing": pos.is_trailing,
            "is_at_breakeven": pos.is_at_breakeven,
            "exits_completed": len(pos.exits_completed),
            "time_partial_exits": pos.time_partial_exits,
            "optimal_hold_minutes": round(pos.optimal_hold_minutes, 2),
        }
    
    # -------------------------------------------------------------------------
    # Private methods
    # -------------------------------------------------------------------------
    
    def _calculate_trailing_sl(
        self,
        pos: PositionState,
        current_price: float,
        atr: float
    ) -> float:
        """[D1] Calculate trailing stop with progressive tightening.

        Kâr arttıkça trailing_atr_mult azalır → kâr daha sıkı korunur.
        """
        # [D1] Pozisyonun kendi trailing_atr_mult değerini kullan
        trailing_distance = atr * pos.trailing_atr_mult

        if pos.side == "long":
            new_sl = pos.highest_price - trailing_distance
            return max(pos.current_sl, new_sl)
        else:
            new_sl = pos.lowest_price + trailing_distance
            return min(pos.current_sl, new_sl)

    @staticmethod
    def _breakeven_stop_price(pos: PositionState) -> float:
        try:
            entry = float(pos.entry_price or 0.0)
            cost_bps = max(0.0, float(pos.breakeven_cost_bps or 0.0))
        except (TypeError, ValueError):
            return float(getattr(pos, "entry_price", 0.0) or 0.0)
        if entry <= 0.0 or cost_bps <= 0.0:
            return entry
        adjustment = entry * cost_bps / 10000.0
        return entry + adjustment if pos.side == "long" else entry - adjustment

    def _check_time_stop(
        self,
        pos: PositionState,
        current_price: float,
        atr: float = None,
        max_hold_hours: float = 48.0,
    ) -> Optional[ExitAction]:
        """[D2] Time stop: uzun süredir açık ve kârsız pozisyonu kapat.

        - 48 saatten uzun açık + kârsız → kapat
        - Kârlı pozisyon etkilenmez

        Returns:
            ExitAction if time stop triggered, None otherwise.
        """
        if not pos.opened_at:
            return None

        try:
            opened = datetime.fromisoformat(
                pos.opened_at.replace("Z", "+00:00")
            ).astimezone(timezone.utc)
            age_hours = (datetime.now(timezone.utc) - opened).total_seconds() / 3600
        except (ValueError, TypeError):
            return None

        cfg = self._load_time_stop_config()
        if not cfg["enabled"]:
            return None

        optimal_hold_minutes, profile = self._resolve_optimal_holding(pos.symbol, cfg)
        pos.optimal_hold_minutes = optimal_hold_minutes
        optimal_hold_hours = max(0.25, optimal_hold_minutes / 60.0)

        # [D2] Volatility-adjusted holding time:
        # Yüksek volatilitede daha kısa tut, düşük volatilitede daha uzun tut.
        # ATR / entry_price oranı ile normalize edilir.
        if atr is not None and atr > 0 and pos.entry_price > 0:
            atr_pct = atr / pos.entry_price  # ATR as percentage of price
            # Baseline ATR: ~1% typical for crypto on 15m
            baseline_atr_pct = 0.01
            vol_ratio = atr_pct / baseline_atr_pct if baseline_atr_pct > 0 else 1.0
            # vol_ratio > 1 → high vol → reduce holding; < 1 → low vol → extend
            # Clamp adjustment between 0.5x and 2.0x
            vol_adjustment = max(0.5, min(2.0, 1.0 / vol_ratio))
            optimal_hold_hours = max(0.25, optimal_hold_hours * vol_adjustment)

        hard_stop_hours = max(float(cfg["hard_max_hold_hours"]), max_hold_hours)
        min_r_to_keep = float(cfg["min_r_to_keep"])
        partial_exit_pct = float(cfg["partial_exit_pct"])

        if age_hours < optimal_hold_hours:
            return None

        if pos.current_r > min_r_to_keep and pos.time_partial_exits == 0:
            hazard_rate = float(profile.get("hazard_rate", 0.0) or 0.0)
            survival_probability = float(profile.get("survival_probability", 0.0) or 0.0)
            risk_pressure = max(0.0, hazard_rate - survival_probability)
            effective_atr = float(atr or pos.last_atr or 0.0)
            atr_mult = max(0.75, 1.75 - min(0.75, risk_pressure * 1.5))
            if effective_atr > 0.0:
                if pos.side == "long":
                    profit_floor = pos.entry_price + max(0.0, current_price - pos.entry_price) * 0.35
                    new_sl = max(pos.current_sl, profit_floor, current_price - (effective_atr * atr_mult))
                else:
                    profit_floor = pos.entry_price - max(0.0, pos.entry_price - current_price) * 0.35
                    new_sl = min(pos.current_sl, profit_floor, current_price + (effective_atr * atr_mult))
            elif pos.side == "long":
                new_sl = max(pos.current_sl, pos.entry_price + max(0.0, current_price - pos.entry_price) * 0.35)
            else:
                new_sl = min(pos.current_sl, pos.entry_price - max(0.0, pos.entry_price - current_price) * 0.35)
            self._save_state()
            return ExitAction(
                action="partial_exit",
                exit_pct=partial_exit_pct,
                new_sl=new_sl,
                reason=f"Survival partial exit at {age_hours:.1f}h (optimal {optimal_hold_hours:.1f}h)",
                details={
                    "age_hours": round(age_hours, 2),
                    "optimal_hold_hours": round(optimal_hold_hours, 2),
                    "hazard_rate": round(hazard_rate, 4),
                    "survival_probability": round(survival_probability, 4),
                    "atr_used": round(effective_atr, 6),
                    "atr_stop_mult": round(atr_mult, 4),
                    "commit": {
                        "current_sl": float(new_sl),
                        "remaining_size": max(0.0, float(pos.current_size) - (float(pos.initial_size) * partial_exit_pct)),
                        "time_partial_exits_increment": 1,
                    },
                },
            )

        if age_hours < hard_stop_hours and pos.current_r > min_r_to_keep:
            return None

        log.warning(
            "[EXIT] %s TIME STOP: açık %.1f saat, R=%.2f (optimal=%.1fh)",
            pos.symbol, age_hours, pos.current_r, optimal_hold_hours,
        )

        return ExitAction(
            action="time_stop",
            exit_pct=1.0,
            reason=f"Time stop: {age_hours:.0f}h açık, optimal={optimal_hold_hours:.0f}h, R={pos.current_r:.2f}",
            details={
                "age_hours": round(age_hours, 1),
                "final_r": round(pos.current_r, 2),
                "optimal_hold_hours": round(optimal_hold_hours, 2),
                "hazard_rate": round(float(profile.get("hazard_rate", 0.0) or 0.0), 4),
                "commit": {"remove_position": True},
            },
        )

    @staticmethod
    def _load_time_stop_config() -> Dict[str, Any]:
        try:
            from core.config_loader import load_config

            cfg = load_config()
        except BEST_EFFORT_EXCEPTIONS:
            cfg = {}
        trade_mgmt = cfg.get("trade_management", {}) if isinstance(cfg, dict) else {}
        time_stop = trade_mgmt.get("time_stop") if isinstance(trade_mgmt, dict) else {}
        portfolio = cfg.get("portfolio_optimization", {}) if isinstance(cfg, dict) else {}
        survival = portfolio.get("survival_analysis") if isinstance(portfolio, dict) else {}
        if not isinstance(time_stop, dict):
            time_stop = {}
        if not isinstance(survival, dict):
            survival = {}
        return {
            "enabled": bool(time_stop.get("enabled", True) and survival.get("enabled", False)),
            "max_hold_hours": float(time_stop.get("max_hold_hours", 48.0) or 48.0),
            "min_r_to_keep": float(time_stop.get("min_r_to_keep", 0.5) or 0.5),
            "partial_exit_pct": float(survival.get("partial_exit_pct", 0.25) or 0.25),
            "hard_max_hold_hours": float(survival.get("hard_max_hold_hours", time_stop.get("max_hold_hours", 48.0)) or 48.0),
        }

    @staticmethod
    def _resolve_optimal_holding(symbol: str, cfg: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        try:
            from risk.survival import build_holding_profile

            profile = build_holding_profile(symbol)
        except BEST_EFFORT_EXCEPTIONS:
            profile = {}
        optimal_hold_minutes = float(profile.get("optimal_hold_minutes", float(cfg["max_hold_hours"]) * 60.0) or float(cfg["max_hold_hours"]) * 60.0)
        return optimal_hold_minutes, profile
    
    def _load_state(self):
        """Load positions from file."""
        try:
            if self.STATE_PATH.exists():
                with open(self.STATE_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                for symbol, pos_data in data.items():
                    self.positions[symbol] = PositionState(
                        symbol=pos_data.get("symbol", symbol),
                        side=pos_data.get("side", "long"),
                        entry_price=pos_data.get("entry_price", 0),
                        initial_sl=pos_data.get("initial_sl", 0),
                        current_sl=pos_data.get("current_sl", 0),
                        initial_size=pos_data.get("initial_size", 0),
                        current_size=pos_data.get("current_size", 0),
                        highest_price=pos_data.get("highest_price", 0),
                        lowest_price=pos_data.get("lowest_price", float("inf")),
                        exits_completed=pos_data.get("exits_completed", []),
                        is_trailing=pos_data.get("is_trailing", False),
                        is_at_breakeven=pos_data.get("is_at_breakeven", False),
                        trailing_atr_mult=pos_data.get("trailing_atr_mult", 2.0),
                        peak_r=pos_data.get("peak_r", 0.0),
                        opened_at=pos_data.get("opened_at", pos_data.get("created_at", "")),
                        time_partial_exits=pos_data.get("time_partial_exits", 0),
                        optimal_hold_minutes=pos_data.get("optimal_hold_minutes", 0.0),
                        last_atr=pos_data.get("last_atr", 0.0),
                        breakeven_cost_bps=pos_data.get("breakeven_cost_bps", 0.0),
                        created_at=pos_data.get("created_at", ""),
                        last_update=pos_data.get("last_update", ""),
                    )
        except BEST_EFFORT_EXCEPTIONS as e:
            log.warning(f"Failed to load exit positions: {e}")
    
    def _save_state(self):
        """Save positions to file."""
        try:
            self.STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            for symbol, pos in self.positions.items():
                data[symbol] = {
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "entry_price": pos.entry_price,
                    "initial_sl": pos.initial_sl,
                    "current_sl": pos.current_sl,
                    "initial_size": pos.initial_size,
                    "current_size": pos.current_size,
                    "highest_price": pos.highest_price,
                    "lowest_price": pos.lowest_price,
                    "exits_completed": pos.exits_completed,
                    "is_trailing": pos.is_trailing,
                    "is_at_breakeven": pos.is_at_breakeven,
                    "trailing_atr_mult": pos.trailing_atr_mult,
                    "peak_r": pos.peak_r,
                    "opened_at": pos.opened_at,
                    "time_partial_exits": pos.time_partial_exits,
                    "optimal_hold_minutes": pos.optimal_hold_minutes,
                    "current_r": pos.current_r,
                    "last_atr": pos.last_atr,
                    "breakeven_cost_bps": pos.breakeven_cost_bps,
                    "created_at": pos.created_at,
                    "last_update": pos.last_update
                }
            with open(self.STATE_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except BEST_EFFORT_EXCEPTIONS as e:
            log.warning(f"Failed to save exit positions: {e}")




# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

import threading as _threading_sem

_manager_instance: Optional[SmartExitManager] = None
_manager_instance_lock = _threading_sem.Lock()


def get_exit_manager(regime: Optional[str] = None,
                     direction: Optional[str] = None) -> SmartExitManager:
    """Get global exit manager instance (thread-safe).

    [FAZA 6.6] regime/direction parametreleri ile regime-aware exit seviyeleri.
    """
    global _manager_instance
    if _manager_instance is None:
        with _manager_instance_lock:
            if _manager_instance is None:
                _manager_instance = SmartExitManager(regime=regime, direction=direction)
    elif regime is not None and direction is not None:
        _manager_instance.update_regime(regime, direction)
    return _manager_instance


def register_position(
    symbol: str,
    side: str,
    entry: float,
    stop_loss: float,
    size: float,
    atr: float = 0.0,
    breakeven_cost_bps: float = 0.0,
) -> PositionState:
    """Quick register function."""
    mgr = get_exit_manager()
    return mgr.register_position(
        symbol,
        side,
        entry,
        stop_loss,
        size,
        atr=atr,
        breakeven_cost_bps=breakeven_cost_bps,
    )


def check_exit(symbol: str, current_price: float, atr: float = None) -> Dict:
    """Quick check function."""
    mgr = get_exit_manager()
    action = mgr.update_position(symbol, current_price, atr)
    return {
        "action": action.action,
        "exit_pct": action.exit_pct,
        "new_sl": action.new_sl,
        "reason": action.reason
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Smart Exit Manager Test")
    print("=" * 50)
    
    mgr = SmartExitManager()
    
    # Register a test position
    mgr.register_position(
        symbol="BTC/USDT",
        side="long",
        entry_price=42000,
        stop_loss=41000,  # 1000 risk
        size=0.1
    )
    
    # Simulate price movements
    print("\n--- Price at 43000 (1R) ---")
    action = mgr.update_position("BTC/USDT", 43000, atr=500)
    print(f"Action: {action}")
    
    print("\n--- Price at 44000 (2R) ---")
    action = mgr.update_position("BTC/USDT", 44000, atr=500)
    print(f"Action: {action}")
    
    print("\n--- Price at 45000 (3R) ---")
    action = mgr.update_position("BTC/USDT", 45000, atr=500)
    print(f"Action: {action}")
    
    print("\n--- Status ---")
    print(mgr.get_position_status("BTC/USDT"))

