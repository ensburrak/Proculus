# -*- coding: utf-8 -*-
"""Lightweight runtime performance hooks used by Phase 3 protections."""

from __future__ import annotations

from typing import Any, Dict

from risk.cooldowns import record_trade_outcome
from risk.daily_limits import record_pnl
from state_manager import get_metric, set_metric


def record_closed_trade(symbol: str, *, pnl_abs: float | None = None, pnl_pct: float | None = None) -> Dict[str, Any]:
    """Update risk subsystems after a realized trade close."""
    if pnl_abs is not None:
        try:
            record_pnl(float(pnl_abs))
            current_daily = float(get_metric("daily_realized_pnl", 0.0) or 0.0)
            set_metric("daily_realized_pnl", current_daily + float(pnl_abs))
        except (TypeError, ValueError):
            pass
    return record_trade_outcome(symbol, pnl_pct=pnl_pct, pnl_abs=pnl_abs)


def analyze_and_update_cooldowns() -> Dict[str, Any]:
    """Refresh persisted performance aggregates and return the latest summary."""
    from execution.trade_logger import refresh_performance_aggregates

    return refresh_performance_aggregates()


__all__ = ["analyze_and_update_cooldowns", "record_closed_trade"]
