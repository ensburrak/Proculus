"""
cache_manager.py
----------------

This module centralises caching logic for expensive API calls and slow
computations.  It provides simple decorators and helper functions to
memoise function results both in memory and on disk with a time-to-live
(TTL).  By caching results across multiple calls, the bot can reduce
latency and API usage costs.

Two primary approaches are offered:

``@memoize``
    Caches results in memory for the lifetime of the process.  Suitable
    for fast functions where persistence across bot restarts is not
    required.

``file_cache``
    Stores the result of a function call in a JSON file along with a
    timestamp.  Subsequent calls within ``ttl`` seconds return the
    cached value.  This is particularly useful for API calls where
    repeated queries would otherwise consume rate limits.

Example usage::

    from cache_manager import memoize, file_cache

    @memoize
    def fibonacci(n):
        return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)

    @file_cache("twitter_sentiment.json", ttl=900)
    def fetch_twitter_sentiment(query):
        # perform API call ...
        return result

The ``cache_manager`` does not impose any particular caching backend,
allowing developers to replace the file-based cache with Redis or other
systems by adapting the helper functions.
"""

from __future__ import annotations


from core.exceptions import BEST_EFFORT_EXCEPTIONS
import functools
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def memoize(func: F) -> F:
    """Simple in-memory memoization decorator.

    Stores results keyed by arguments for the lifetime of the process.
    Not suitable for functions with unhashable arguments.
    """

    cache: Dict[Tuple[Any, ...], Any] = {}

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        key = args + tuple(sorted(kwargs.items()))
        if key in cache:
            return cache[key]
        result = func(*args, **kwargs)
        cache[key] = result
        return result

    return wrapper  # type: ignore


def file_cache(cache_name: str, ttl: int = 3600, watch_paths: Optional[list[str]] = None) -> Callable[[F], F]:
    """Decorator to cache function results in a JSON file on disk.

    In addition to a simple time‑to‑live (TTL), this decorator can
    invalidate the cached value when any file in ``watch_paths`` has been
    modified since the cache was written.  This is particularly useful
    for cases where upstream data sets or configuration files change and
    cached values should not be reused.

    Args:
        cache_name: Filename (within ``data/cache``) used to store the
            cached result.
        ttl: Time‑to‑live in seconds; cached values older than this are
            ignored and recomputed.
        watch_paths: Optional list of file paths to monitor.  When any
            of these files has a modification time newer than the
            cached value, the cache is invalidated.

    Returns:
        A decorator that wraps a function, adding caching behaviour.
    """

    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / cache_name
    # Normalise watch paths to Path objects
    watch_files: list[Path] = []
    if watch_paths:
        for p in watch_paths:
            try:
                watch_files.append(Path(p))
            except BEST_EFFORT_EXCEPTIONS:
                continue

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            # Try to load from cache if present and not expired
            if cache_path.exists():
                try:
                    with cache_path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    timestamp = float(data.get("timestamp", 0))
                    # If TTL has expired, ignore cache
                    if time.time() - timestamp < ttl:
                        # If watching files, ensure none have been modified since cache write
                        invalidate = False
                        if watch_files:
                            for wf in watch_files:
                                try:
                                    if wf.exists():
                                        mtime = wf.stat().st_mtime
                                        if mtime > timestamp:
                                            invalidate = True
                                            break
                                except BEST_EFFORT_EXCEPTIONS:
                                    continue
                        if not invalidate:
                            return data.get("value")
                except BEST_EFFORT_EXCEPTIONS as exc:
                    logger.debug("Failed to read cache %s: %s", cache_path, exc)
            # Compute fresh result
            result = func(*args, **kwargs)
            # Write to cache with timestamp
            try:
                from atomic_io import atomic_write_json
                atomic_write_json(cache_path, {"timestamp": time.time(), "value": result})
            except BEST_EFFORT_EXCEPTIONS as exc:
                logger.debug("Failed to write cache %s: %s", cache_path, exc)
            return result

        return wrapped  # type: ignore

    return decorator


def clear_cache(cache_name: Optional[str] = None) -> None:
    """Remove cached file(s).

    :param cache_name: specific cache file to remove; if None, remove all
    """
    cache_dir = Path("data/cache")
    if cache_name:
        path = cache_dir / cache_name
        try:
            path.unlink(missing_ok=True)
        except BEST_EFFORT_EXCEPTIONS as _exc:
            logging.getLogger(__name__).debug("Suppressed in clear_cache: %s", _exc)
    else:
        for file in cache_dir.glob("*.json"):
            try:
                file.unlink()
            except BEST_EFFORT_EXCEPTIONS as _exc:
                logging.getLogger(__name__).debug("Suppressed in clear_cache: %s", _exc)
