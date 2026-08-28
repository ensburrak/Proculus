"""
onchain_analytics.py
--------------------

This module computes an on‑chain sentiment score from pseudo on‑chain
metrics stored in ``data/onchain_metrics.json``.  These metrics are
produced by ``onchain_data_updater.py`` and are derived from free
exchange data rather than premium on‑chain providers.  Each metric is
expected to lie in the range ``[-1, 1]``.  The sentiment score is
obtained by normalising each metric to ``[0, 1]`` and averaging them.

If no metrics are available for a symbol, the sentiment defaults to
``0.5`` (neutral).  All sentiment scores returned by this module are
floats between ``0.0`` and ``1.0``.
"""

from __future__ import annotations


from core.exceptions import BEST_EFFORT_EXCEPTIONS
import json
import logging
from pathlib import Path
from typing import Any, Dict

_cache: Dict[str, Any] | None = None
logger = logging.getLogger(__name__)


def _load_metrics() -> Dict[str, Any]:
    """Load the pseudo on‑chain metrics from disk and cache them.

    The metrics file may either be a dict of symbol → metric dict
    (legacy format) or a dict with a ``"metrics"`` key containing
    symbol → metric dict.  This function normalises both formats to
    return just the mapping of symbols to metrics.

    Returns
    -------
    dict
        Mapping of uppercase base symbols to metric dicts.
    """
    global _cache
    if _cache is not None:
        return _cache
    try:
        path = Path(__file__).resolve().parent / "data" / "onchain_metrics.json"
        if path.exists():
            txt = path.read_text(encoding="utf-8").strip()
            if txt:
                data = json.loads(txt)
                if isinstance(data, dict):
                    raw_metrics = data.get("metrics")
                    metrics = raw_metrics if isinstance(raw_metrics, dict) else data
                    _cache = {str(k): v for k, v in metrics.items()}
                    return _cache
    except BEST_EFFORT_EXCEPTIONS as exc:
        logger.warning("Failed to load on-chain metrics: %s", exc)
    _cache = {}
    return _cache


def get_onchain_sentiment(symbol: str) -> float:
    """Return a sentiment score for the given symbol based on pseudo on‑chain metrics.

    The score is the average of all available metrics after mapping
    their values from ``[-1, 1]`` to ``[0, 1]``.  If no metrics are
    available for the symbol, 0.5 (neutral) is returned.

    Parameters
    ----------
    symbol : str
        Symbol to query.  Accepts either ``"BTC/USDT"`` or ``"BTC"``.

    Returns
    -------
    float
        A value between 0.0 and 1.0, where 0.0 is very bearish,
        1.0 is very bullish and 0.5 is neutral.
    """
    sym = symbol.split("/")[0] if isinstance(symbol, str) else str(symbol)
    metrics = _load_metrics()
    if not metrics:
        return 0.5
    m = metrics.get(sym.upper())
    if not isinstance(m, dict):
        return 0.5
    vals = []
    for v in m.values():
        if isinstance(v, (int, float)):
            # Convert from [-1, 1] → [0, 1]
            val = (v + 1.0) / 2.0
            if val < 0.0:
                val = 0.0
            if val > 1.0:
                val = 1.0
            vals.append(val)
    if not vals:
        return 0.5
    return sum(vals) / len(vals)
