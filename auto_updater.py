# -*- coding: utf-8 -*-
"""Compatibility wrapper for the isolated auto-updater CLI.

The trading runtime must not run updater jobs in-process. The actual updater
implementation lives in tools.auto_updater_cli and is started only through
runtime.auto_updater_service.
"""
from __future__ import annotations

from tools import auto_updater_cli as _impl

TASKS = _impl.TASKS
UPDATE_PLAN = _impl.UPDATE_PLAN
run_update_cycle = _impl.run_update_cycle
start_background = _impl.start_background
threading = _impl.threading

__all__ = ["TASKS", "UPDATE_PLAN", "run_update_cycle", "start_background", "threading"]

if __name__ == "__main__":
    raise SystemExit(
        "Direct auto_updater.py execution is disabled. Use `python -m runtime.auto_updater_service`."
    )
