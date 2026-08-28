# -*- coding: utf-8 -*-
"""
social_scanner.py
==================

This module provides helper functions for scanning social data and
trending coins from various sources.  It exposes two primary entry
points:

``get_social_score(symbol)``  —  Computes a normalised social sentiment score
for a given trading symbol using a simple combination of the number
of mentions and an associated sentiment value.

``update_social_trends()``  —  Periodically refreshes the ``data/social_trends.json``
file by pulling trending coins and associated sentiment from external
APIs (CoinMarketCap trending/most viewed, LunarCrush galaxy scores).
If API keys are unavailable or network connectivity is restricted,
the function safely falls back to the last known trends.  This method
is designed to be scheduled by ``auto_updater.py``.

Data format (social_trends.json)::

    {
      "BTC/USDT": {"mentions": 1200, "sentiment": 0.65},
      "ETH/USDT": {"mentions": 800,  "sentiment": 0.55},
      ...
    }

Fields:
  * ``mentions`` – the relative popularity of a coin (e.g. number of
    mentions on social platforms or a trending rank)
  * ``sentiment`` – a normalised sentiment score in the range 0–1

The default implementation uses a local JSON file for trends but can be
extended to call third‑party APIs via the helper functions provided
below.  These functions are decorated with ``file_cache`` to reduce
duplicate network requests and ``retry`` to handle transient failures.

Note: This module does not directly drive trading decisions but can
inform signal generation with an additional layer of sentiment
confidence.
"""
from __future__ import annotations


from core.exceptions import BEST_EFFORT_EXCEPTIONS
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from atomic_io import atomic_write_json, safe_read_json

# In‑memory cache for social trends to avoid repeated disk I/O
_cache: Dict[str, Any] | None = None

def _load_trends(path: Path | None = None) -> Dict[str, Any]:
    """Load social trends from JSON file, caching the result.

    The trends file lives under ``data/social_trends.json`` relative to
    this module.  If the file cannot be read or parsed, returns an
    empty dict.
    """
    global _cache
    if _cache is not None:
        return _cache
    if path is None:
        path = Path(__file__).resolve().parent / "data" / "social_trends.json"
    try:
        if path.exists():
            data = safe_read_json(path, default={})
            if isinstance(data, dict):
                _cache = data  # type: ignore
                return _cache
    except BEST_EFFORT_EXCEPTIONS:
        pass
    _cache = {}
    return _cache  # type: ignore

def get_social_score(symbol: str) -> float:
    """Return a normalised social score for the given trading symbol.

    The score is computed by combining the relative mentions count and
    the sentiment value for that symbol.  Both quantities are scaled to
    the range [0, 1] and averaged.  If no information is available,
    returns 0.5 (neutral).

    Args:
        symbol: e.g. "BTC/USDT"

    Returns:
        A float in the range [0, 1].
    """
    data = _load_trends()
    try:
        info = data.get(symbol)
        if not isinstance(info, dict):
            info = None
        # Mentions and basic sentiment from trending data
        mentions = float(info.get("mentions", 0.0)) if info else 0.0
        trend_sent = float(info.get("sentiment", 0.5)) if info else 0.5
        # Load per‑symbol sentiment from metrics file if available
        per_sent_score: float | None = None
        try:
            sent_path = Path(__file__).resolve().parent / "metrics" / "social_sentiment.json"
            if sent_path.exists():
                per_data = safe_read_json(sent_path, default={})
                # Key may be stored as e.g. "BTC/USDT"; ensure same format
                per_key = symbol
                val = per_data.get(per_key) if isinstance(per_data, dict) else None
                if isinstance(val, (int, float)):
                    # Map from [‑1,1] to [0,1]
                    per_sent_score = (float(val) + 1.0) / 2.0
        except BEST_EFFORT_EXCEPTIONS:
            per_sent_score = None
        # Normalise mentions relative to max across all coins
        try:
            max_mentions = max((float(item.get("mentions", 1.0)) for item in data.values() if isinstance(item, dict)), default=1.0)
        except BEST_EFFORT_EXCEPTIONS:
            max_mentions = 1.0
        if max_mentions <= 0:
            max_mentions = 1.0
        norm_mentions = min(1.0, mentions / max_mentions) if max_mentions else 0.0
        # Normalise trend sentiment [0,1]
        norm_trend_sent = max(0.0, min(1.0, trend_sent))
        # Combine scores: average of available components (mentions, trend sentiment, per symbol)
        components: List[float] = []
        components.append(norm_mentions)
        components.append(norm_trend_sent)
        if per_sent_score is not None:
            components.append(max(0.0, min(1.0, per_sent_score)))
        if not components:
            return 0.5
        return sum(components) / len(components)
    except BEST_EFFORT_EXCEPTIONS:
        return 0.5

# ----------------------------------------------------------------------
# Trending coin discovery helpers
# ----------------------------------------------------------------------

try:
    from retry_utils import retry  # type: ignore
except BEST_EFFORT_EXCEPTIONS:
    # fallback no‑retry decorator
    def retry(exceptions, tries=3, base_delay=0.5, max_delay=4.0):
        def decorator(fn):
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)
            return wrapper
        return decorator

from cache_manager import file_cache  # reuse for caching API responses
import logging
logger = logging.getLogger(__name__)
import os
import requests

@file_cache("cmc_trending.json", ttl=3600)
@retry((Exception,), tries=2, base_delay=1.0, max_delay=4.0)
def _fetch_coinmarketcap_trending(limit: int = 5) -> List[Tuple[str, float]]:
    """Fetch the top trending coins using the CoinMarketCap API.

    Requires a valid API key in the ``CMC_API_KEY`` or
    ``COINMARKETCAP_API_KEY`` environment variable.  Returns a list of
    ``(symbol, rank)`` tuples where lower ranks indicate higher trending
    status.  If the API key is missing or the request fails, returns
    an empty list.

    The endpoint ``/cryptocurrency/trending/latest`` is used for the
    "Most Viewed" list.  Consult the CoinMarketCap API docs for other
    options.
    """
    key = os.getenv("CMC_API_KEY") or os.getenv("COINMARKETCAP_API_KEY")
    if not key:
        return []
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/trending/latest"
    params = {"limit": limit}
    headers = {"X-CMC_PRO_API_KEY": key}
    # Perform the request and handle common HTTP errors gracefully.  In particular,
    # CoinMarketCap may return a 403 Forbidden if the API key is invalid or the plan
    # does not include the trending endpoint.  In that case we attempt to fall back
    # to a more widely accessible endpoint (``listings/latest``).  This fallback
    # returns the top ``limit`` coins by market capitalisation.  If the fallback
    # also fails, we log the error and return an empty list.
    resp = requests.get(url, params=params, headers=headers, timeout=6)
    # Check for explicit permission or not-found errors before calling raise_for_status.
    # Some plans do not provide access to the trending endpoint (403) and some setups
    # (e.g. enterprise vs. pro) may return 404.  In these cases we degrade gracefully.
    if resp.status_code in (403, 404):
        logger.warning("CMC trending endpoint unavailable (status %s). Falling back to listings/latest", resp.status_code)
        # Attempt fallback to listings/latest: this endpoint lists coins by CMC rank.
        # We request only the top ``limit`` results.  This endpoint is generally
        # available on the free tier and should not return a permission error.
        fallback_url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
        fallback_params = {"limit": limit, "convert": "USD"}
        try:
            r2 = requests.get(fallback_url, params=fallback_params, headers=headers, timeout=6)
            # If the fallback call also returns 403/404, treat as no data.
            if r2.status_code in (403, 404):
                logger.warning("CMC listings fallback failed: status %s", r2.status_code)
                return []
            r2.raise_for_status()
            data2 = r2.json()
            out: List[Tuple[str, float]] = []
            for idx, item in enumerate(data2.get("data", [])):
                sym = item.get("symbol")
                if isinstance(sym, str):
                    out.append((sym, float(idx + 1)))
            return out
        except BEST_EFFORT_EXCEPTIONS as exc:
            # Fallback failed; log and return empty list
            logger.warning("CMC listings fallback error: %s", exc)
            return []
    # For other status codes, raise errors to be handled by the caller (which
    # may trigger retries or propagate exceptions).
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as exc:  # type: ignore[attr-defined]
        raise
    data = resp.json()
    out: List[Tuple[str, float]] = []
    for idx, item in enumerate(data.get("data", [])):
        symbol = item.get("symbol")
        if isinstance(symbol, str):
            # rank is simply the index + 1
            out.append((symbol, float(idx + 1)))
    return out

@file_cache("lunarcrush_trending.json", ttl=3600)
@retry((Exception,), tries=2, base_delay=1.0, max_delay=4.0)
def _fetch_lunarcrush_trending(limit: int = 5) -> List[Tuple[str, float]]:
    """Fetch trending coins using the LunarCrush API.

    Requires a valid ``LUNARCRUSH_API_KEY`` environment variable.  The
    API returns a list of assets with a ``galaxy_score``.  We sort
    assets by descending galaxy score and return the top ``limit``.
    If the API key is missing or the request fails, returns an empty
    list.
    """
    key = os.getenv("LUNARCRUSH_API_KEY") or os.getenv("LUNARCRUSH_API")
    if not key:
        return []
    # Use the updated LunarCrush v4 API with bearer authentication.  Trending coins
    # can be obtained via the ``/public/coins/list/v1`` endpoint.  The previous
    # implementation used ``?data=assets`` which returned 404.  Here we sort by
    # Galaxy Score descending and limit the number of results.  See the API docs
    # for details: https://lunarcrush.com/developers/api/public/coins/list/v1
    url = "https://lunarcrush.com/api4/public/coins/list/v1"
    params = {
        "limit": limit,
        # Sort by galaxy_score; other options include market_cap_rank, social_volume, etc.
        "sort": "galaxy_score",
        # Descending order
        "desc": True,
    }
    headers = {"Authorization": f"Bearer {key}"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=6)
        # Handle Forbidden or not found errors gracefully; LunarCrush may
        # return 403 when the API key is invalid or rate limited, or 404 when
        # the endpoint is unavailable for the current plan.  In these cases we
        # log the error and return an empty list so the caller can continue
        # without trending data.
        if resp.status_code in (402, 403, 404, 429):
            # 402: Payment Required (subscription needed), 403: Forbidden (invalid key),
            # 404: Not Found (endpoint not available), 429: Too Many Requests (rate limit).
            logger.warning("LunarCrush trending fetch failed: %s", resp.status_code)
            return []
        resp.raise_for_status()
        js = resp.json()
    except BEST_EFFORT_EXCEPTIONS as exc:
        # Any exception (DNS failure, connection reset, invalid JSON, etc.) is
        # logged via the retry decorator; sanitize any API key from the exception
        # message before logging.  Returning an empty list prevents upstream
        # tasks from being disabled for an extended period.
        exc_str = str(exc)
        # Mask the API key if it appears in the error message
        if key:
            exc_str = exc_str.replace(key, "[REDACTED]")
        logger.warning("LunarCrush trending fetch failed: %s", exc_str)
        return []
    out: List[Tuple[str, float]] = []
    for item in js.get("data", []):
        sym = item.get("symbol")
        gscore = item.get("galaxy_score")
        if isinstance(sym, str) and isinstance(gscore, (int, float)):
            out.append((sym, float(gscore)))
    # Higher galaxy score implies stronger trending
    return out

def update_social_trends(path: Path | None = None, limit: int = 10) -> List[str]:
    """Refresh the social trends file by fetching trending coins.

    This helper merges multiple sources (CoinMarketCap, LunarCrush) into
    a single ranking.  Coins appearing in both lists are weighted by
    their rank/score; duplicates are de‑duplicated by taking the best
    score.  The resulting dictionary is written to
    ``data/social_trends.json`` with default sentiment values and
    mention counts proportional to rank.  Returns a list of the top
    trending symbols.

    Args:
        path: Optional override for where to write the trends file.
        limit: Maximum number of coins to include in the final list.

    Returns:
        A list of trending symbol strings.
    """
    # Fetch trending lists from available sources
    try:
        cmc = _fetch_coinmarketcap_trending(limit=limit)
    except BEST_EFFORT_EXCEPTIONS as exc:
        logger.warning("CMC trending fetch failed: %s", exc)
        cmc = []
    try:
        lc = _fetch_lunarcrush_trending(limit=limit)
    except BEST_EFFORT_EXCEPTIONS as exc:
        logger.warning("LunarCrush trending fetch failed: %s", exc)
        lc = []
    # Combine into a dict: lower scores (rank) or higher scores (galaxy) -> better
    combined: Dict[str, float] = {}
    for sym, rank in cmc:
        # Lower rank is better; invert to get higher score
        combined[sym] = combined.get(sym, 0.0) + (1.0 / (rank + 1.0))
    for sym, score in lc:
        combined[sym] = combined.get(sym, 0.0) + float(score)
    if not combined:
        # If no trending data, do not modify existing file; return empty list
        return []
    # Sort by combined score descending
    sorted_syms = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
    top_syms = [sym for sym, _ in sorted_syms[:limit]]
    # Build JSON data for trends file with default sentiment and mention count
    trends: Dict[str, Dict[str, float]] = {}
    for rank, sym in enumerate(top_syms, start=1):
        # Convert to trading pair format with /USDT if missing slash
        if "/" not in sym:
            pair = f"{sym.upper()}/USDT"
        else:
            pair = sym.upper()
        # Higher rank gets more mentions (inverse of rank) scaled to 1.0
        mentions_score = (limit - rank + 1) / float(limit)
        trends[pair] = {"mentions": mentions_score, "sentiment": 0.5}
    # Write to file
    dest = path or (Path(__file__).resolve().parent / "data" / "social_trends.json")
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(dest, trends)
        # Update in‑memory cache
        global _cache
        _cache = trends  # type: ignore
        logger.info("Social trends updated: %s", list(trends.keys()))
    except BEST_EFFORT_EXCEPTIONS as exc:
        logger.error("Failed to write social trends: %s", exc)
    return top_syms
