# -*- coding: utf-8 -*-
"""
circuit_breaker.py
===================

Provides a simple circuit breaker for the trading bot.  When network
errors, data anomalies or large drawdowns occur, new positions
should be halted to prevent compounding losses or exacerbating
instability.  This module keeps a running count of recent errors and
anomalies and enforces a cooldown period when thresholds are exceeded.

Key functions:

- ``record_error()``: call when a non‑rate‑limit exception occurs
  during API calls or critical operations.  Consecutive errors will
  increment the internal counter.  Once the number of errors within
  the measurement window crosses ``ERROR_THRESHOLD``, the breaker
  triggers and trading halts for ``COOLDOWN_SECONDS``.

- ``record_anomaly()``: call when the anomaly detector flags a price
  anomaly.  Similar to ``record_error``, repeated anomalies will
  trigger the breaker.

- ``update_outcome(pnl_frac)``: inform the breaker of closed trades.
  Positive outcomes reduce the error counter, while very negative
  outcomes can contribute to triggering the breaker.  This keeps the
  breaker adaptive to strategy performance.

- ``circuit_should_halt(balance)``: returns ``True`` if the breaker is
  currently active and new trades should be suppressed.  The
  ``balance`` parameter is accepted for future enhancements (e.g.,
  using exposure or account drawdown).

This circuit breaker is intentionally conservative—thresholds can be
adjusted to fit your risk appetite.

CHANGELOG:
- v1.1: ERROR_THRESHOLD increased from 3 to 5 for better stability
- v1.2: Added get_state() and reset() functions for dashboard control
- v1.3: Added config.json support for dynamic threshold configuration
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration Loading
# ---------------------------------------------------------------------------
def _load_config() -> Dict[str, Any]:
    """Load circuit breaker config from config.json if available."""
    try:
        cfg_path = Path("config.json")
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            return cfg.get("circuit_breaker", {})
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        log.warning("Circuit breaker config load failed: %s", exc)
    return {}

_config = _load_config()

# ---------------------------------------------------------------------------
# Thresholds and cooldown period (can be overridden via config.json)
# ---------------------------------------------------------------------------
# FIXED: Increased from 3 to 5 for better stability
ERROR_THRESHOLD: int = int(_config.get("error_threshold", 5))
ANOMALY_THRESHOLD: int = int(_config.get("anomaly_threshold", 5))
COOLDOWN_SECONDS: int = int(_config.get("cooldown_seconds", 15 * 60))  # 15 minutes
LARGE_LOSS_THRESHOLD: float = float(_config.get("large_loss_threshold", -0.05))  # -5%
EVENT_WINDOW_SECONDS: int = int(_config.get("event_window_seconds", 60))

# Internal state. Circuit breaker calls are synchronous and intentionally avoid
# blocking event-loop threads with a threading lock; deque/state updates are kept
# short and deterministic.
_state = {
    "error_count": 0,
    "anomaly_count": 0,
    "triggered_until": None,  # type: Optional[float]
    "trigger_reason": None,   # NEW: Track why breaker was triggered
    "total_triggers": 0,      # NEW: Track total number of triggers
}
_error_events: deque[float] = deque()
_anomaly_events: deque[float] = deque()


def _now() -> float:
    """Return current epoch seconds."""
    return time.time()


def reload_config() -> None:
    """Reload configuration from config.json (for dashboard hot-reload)."""
    global _config, ERROR_THRESHOLD, ANOMALY_THRESHOLD, COOLDOWN_SECONDS, LARGE_LOSS_THRESHOLD, EVENT_WINDOW_SECONDS
    _config = _load_config()
    ERROR_THRESHOLD = int(_config.get("error_threshold", 5))
    ANOMALY_THRESHOLD = int(_config.get("anomaly_threshold", 5))
    COOLDOWN_SECONDS = int(_config.get("cooldown_seconds", 15 * 60))
    LARGE_LOSS_THRESHOLD = float(_config.get("large_loss_threshold", -0.05))
    EVENT_WINDOW_SECONDS = int(_config.get("event_window_seconds", 60))


def _prune_events(events: deque[float], now: float) -> None:
    while events and now - float(events[0]) > EVENT_WINDOW_SECONDS:
        events.popleft()


def _trigger_locked(reason: str) -> None:
    _state["triggered_until"] = _now() + COOLDOWN_SECONDS
    _state["trigger_reason"] = reason
    _state["total_triggers"] = int(_state.get("total_triggers", 0) or 0) + 1
    _state["error_count"] = 0
    _state["anomaly_count"] = 0
    _error_events.clear()
    _anomaly_events.clear()


def get_state() -> Dict[str, Any]:
    """
    Return current circuit breaker state for dashboard display.
    """
    until = _state.get("triggered_until")
    now = _now()
    is_triggered = until is not None and now < float(until)
    remaining = max(0, float(until) - now) if is_triggered else 0

    return {
        "error_count": len(_error_events),
        "anomaly_count": len(_anomaly_events),
        "is_triggered": is_triggered,
        "remaining_cooldown_sec": int(remaining),
        "trigger_reason": _state.get("trigger_reason"),
        "total_triggers": _state.get("total_triggers", 0),
        "thresholds": {
            "error": ERROR_THRESHOLD,
            "anomaly": ANOMALY_THRESHOLD,
            "cooldown_sec": COOLDOWN_SECONDS,
            "large_loss": LARGE_LOSS_THRESHOLD,
        }
    }


def reset() -> None:
    """
    Manually reset the circuit breaker (for dashboard control).
    Clears all counters and deactivates the breaker.
    """
    _state["error_count"] = 0
    _state["anomaly_count"] = 0
    _state["triggered_until"] = None
    _state["trigger_reason"] = None
    _error_events.clear()
    _anomaly_events.clear()


def record_error(reason: str = "api_error") -> None:
    """
    Increment the error counter and potentially trigger the breaker.

    Each time a non‑rate‑limit error occurs during an exchange call,
    this function should be invoked.  When the error counter reaches
    ``ERROR_THRESHOLD`` the breaker is set for ``COOLDOWN_SECONDS``.
    
    Args:
        reason: Description of the error (for logging/dashboard)
    """
    try:
        now = _now()
        _error_events.append(now)
        _prune_events(_error_events, now)
        cnt = len(_error_events)
        _state["error_count"] = cnt
        if cnt >= ERROR_THRESHOLD:
            _trigger_locked(f"errors ({cnt} >= {ERROR_THRESHOLD}) in {EVENT_WINDOW_SECONDS}s: {reason}")
    except (TypeError, ValueError) as exc:
        log.exception("Circuit breaker record_error failed: %s", exc)


def record_anomaly(reason: str = "price_anomaly") -> None:
    """
    Increment the anomaly counter and potentially trigger the breaker.
    Use this when price jumps beyond ATR bounds or other anomalies are
    detected in ``main_bot_async``.
    
    Args:
        reason: Description of the anomaly (for logging/dashboard)
    """
    try:
        now = _now()
        _anomaly_events.append(now)
        _prune_events(_anomaly_events, now)
        cnt = len(_anomaly_events)
        _state["anomaly_count"] = cnt
        if cnt >= ANOMALY_THRESHOLD:
            _trigger_locked(f"anomalies ({cnt} >= {ANOMALY_THRESHOLD}) in {EVENT_WINDOW_SECONDS}s: {reason}")
    except (TypeError, ValueError) as exc:
        log.exception("Circuit breaker record_anomaly failed: %s", exc)


def update_outcome(pnl_frac: float) -> None:
    """
    Incorporate the outcome of a closed trade into the breaker state.

    Args:
        pnl_frac: Realized return as a fraction (e.g. 0.02 for +2%, -0.03
            for -3%).  Positive values reduce the error counter; large
            losses increase it.

    A very negative trade (less than ``LARGE_LOSS_THRESHOLD``) counts as
    an error and may trip the breaker.  Profitable trades reduce the
    error counter, allowing recovery over time.
    """
    try:
        if pnl_frac is None:
            return
        pnl = float(pnl_frac)
        if pnl > 0.0 and _error_events:
            _error_events.pop()
            _state["error_count"] = len(_error_events)
            return
        if pnl <= LARGE_LOSS_THRESHOLD:
            record_error(f"large_loss (pnl={pnl:.2%})")
    except (TypeError, ValueError) as exc:
        log.exception("Circuit breaker update_outcome failed: %s", exc)


def circuit_should_halt(balance: float | None = None) -> bool:
    """
    Check whether the circuit breaker is active.

    Args:
        balance: Current account balance (unused currently but kept for
            potential future use).  Could be used to scale thresholds
            depending on account equity.
    Returns:
        True if trading should be halted, False otherwise.
    """
    try:
        if balance is not None and float(balance) <= 0.0:
            log.warning("Circuit breaker fail-safe halt: invalid balance=%s", balance)
            return True
        until = _state.get("triggered_until")
        if until is not None and _now() < float(until):
            return True
    except (TypeError, ValueError) as exc:
        log.exception("Circuit breaker halt check failed: %s", exc)
        return True
    return False
