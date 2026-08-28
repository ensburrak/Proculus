"""
runtime_status.py
------------------

This helper module centralises runtime status and simple dashboard
information for the trading bot.  During each trading loop, the bot can
call ``update_status`` with metrics such as the number of open trades,
daily PnL, kill switch state and recent activity counts.  The status is
persisted to ``metrics/runtime_status.json`` and duplicated to
``metrics/dashboard.json`` for compatibility with external dashboards.
Writes are atomic and safe under concurrent access via the
``atomic_io.safe_write_json`` helper.  Any exceptions are suppressed so
status updates do not affect trading logic.

Example usage::

    from .runtime_status import update_status
    update_status(open_trades=3, daily_pnl=5.0, kill_switch="OFF")

Additional keyword arguments may be passed to record arbitrary fields.
All values should be JSON serialisable.  A timestamp of the last update
is automatically included (UNIX epoch seconds).
"""

from __future__ import annotations


from core.exceptions import BEST_EFFORT_EXCEPTIONS
import logging
import json
import time
from pathlib import Path
from typing import Any, Dict


from atomic_io import safe_write_json
from runtime_paths import METRICS_DIR

ROOT_DIR = Path(__file__).resolve().parent
RUNTIME_STATUS_FILE = METRICS_DIR / "runtime_status.json"
DASHBOARD_FILE = METRICS_DIR / "dashboard.json"

# Ensure metrics directory exists
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def update_status(**kwargs: Any) -> None:
    """
    Update runtime status with arbitrary keyword arguments.  Existing
    status values are preserved unless overwritten.  A `last_update_ts`
    field containing the current UNIX timestamp is automatically added.

    Args:
        **kwargs: Arbitrary key/value pairs to record in the status.
    """
    try:
        # Load current status if exists
        data: Dict[str, Any] = {}
        if RUNTIME_STATUS_FILE.exists():
            try:
                text = RUNTIME_STATUS_FILE.read_text(encoding="utf-8")
                if text:
                    data = json.loads(text)
            except BEST_EFFORT_EXCEPTIONS:
                data = {}
        # Merge updates
        data.update(kwargs)
        data.setdefault("decision_degradation", "unknown")
        data.setdefault("fallback_reason", None)
        data.setdefault("provider_status", {})
        data.setdefault("model_source", "unknown")
        data["last_update_ts"] = int(time.time())
        # Persist to both runtime_status.json and dashboard.json atomically
        safe_write_json(RUNTIME_STATUS_FILE, data)
        safe_write_json(DASHBOARD_FILE, data)
    except BEST_EFFORT_EXCEPTIONS:
        # Do not propagate exceptions
        pass
