# -*- coding: utf-8 -*-
"""
controller_decision_services.py
===============================
Delegates decide_batch calls to the official decision pipeline wrapper.
This module was extracted during the Jan 22 modularization.
"""
from __future__ import annotations


from core.exceptions import BEST_EFFORT_EXCEPTIONS
import logging
from typing import Any, Dict, List

log = logging.getLogger(__name__)


async def decide_batch_service(symbol_inputs: List[Dict[str, Any]]) -> Dict[str, Dict]:
    """
    Batch decision service — delegates to controller_async.decide_batch.

    Args:
        symbol_inputs: List of dicts with keys:
            symbol, tf, price, ta_pack, senti

    Returns:
        {symbol: {action, master_confidence, lev, parts, reason, base_decision}}
    """
    try:
        from core.decision_pipeline import DecisionPipeline

        return await DecisionPipeline(exchange=None).decide_batch(symbol_inputs)
    except ImportError:
        log.warning("[DECIDE_SVC] official decision pipeline not available")
        results = {}
        for inp in (symbol_inputs or []):
            sym = inp.get("symbol", "?")
            results[sym] = {
                "action": "hold",
                "master_confidence": 0.0,
                "lev": 0,
                "parts": {},
                "reason": "official decision pipeline unavailable",
                "base_decision": "hold",
            }
        return results
    except BEST_EFFORT_EXCEPTIONS as exc:
        log.error("[DECIDE_SVC] official decision pipeline failed: %s", exc)
        return {}
