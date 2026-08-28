# -*- coding: utf-8 -*-
"""
llm_cache.py
============
[2026-02-09] Cost Optimization Module

Caches LLM (ChatGPT/DeepSeek) API responses to reduce duplicate calls
and costs.  Implements a disk-backed TTL cache with LRU eviction.
"""
from __future__ import annotations


from core.exceptions import BEST_EFFORT_EXCEPTIONS
import hashlib
import json
import logging
import time
import threading
from pathlib import Path
from atomic_io import atomic_write_json
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
CACHE_FILE = ROOT / "data" / "llm_cache.json"
DEFAULT_TTL = 300  # 5 minutes default
MAX_ENTRIES = 500


_cache_lock = threading.Lock()

def _load_cache() -> Dict[str, Any]:
    if not CACHE_FILE.exists():
        return {}
    try:
        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except BEST_EFFORT_EXCEPTIONS:
        return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(CACHE_FILE, cache)


def _make_key(prompt: str, model: str = "") -> str:
    raw = f"{model}:{prompt}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get_dynamic_ttl(volatility: str = "normal") -> int:
    """Return TTL based on market volatility level."""
    ttl_map = {
        "low": DEFAULT_TTL,
        "normal": DEFAULT_TTL,
        "high": 60,
        "extreme": 0,
    }
    return ttl_map.get(volatility, DEFAULT_TTL)


def get_cached(prompt: str, model: str = "", ttl: int = DEFAULT_TTL) -> Optional[Any]:
    """Return cached response if available and not expired."""
    if ttl <= 0:
        return None
    key = _make_key(prompt, model)
    with _cache_lock:
        cache = _load_cache()
        entry = cache.get(key)
        if entry is None:
            return None
        if time.time() - entry.get("ts", 0) > ttl:
            return None
        log.debug("[LLM_CACHE] Hit for %s (model=%s)", key, model)
        return entry.get("response")


def set_cached(prompt: str, response: Any, model: str = "") -> None:
    """Store a response in the cache."""
    key = _make_key(prompt, model)
    with _cache_lock:
        cache = _load_cache()
        cache[key] = {"response": response, "ts": time.time(), "model": model}

        # Evict old entries if over limit
        if len(cache) > MAX_ENTRIES:
            sorted_keys = sorted(cache.keys(), key=lambda k: cache[k].get("ts", 0))
            for k in sorted_keys[:len(cache) - MAX_ENTRIES]:
                del cache[k]

        _save_cache(cache)
    log.debug("[LLM_CACHE] Stored for %s", key)


def clear_cache() -> int:
    """Clear entire cache. Returns number of entries removed."""
    with _cache_lock:
        cache = _load_cache()
        count = len(cache)
        _save_cache({})
    return count
