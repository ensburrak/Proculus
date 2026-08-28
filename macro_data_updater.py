"""
macro_data_updater.py
---------------------

This module refreshes the macro events file used by the macro risk filter.
It uses the free economic calendar endpoint provided by Financial Modeling
Prep (FMP) to fetch upcoming macroeconomic events (e.g. CPI releases,
employment data, FOMC meetings). The fetched events are normalised into
the format consumed by ``macro_filter.py`` and written to
``data/macro_events.json``.  The FMP API requires an API key but offers
a generous free tier.  Events are pulled for the next ``days_ahead``
days (default 7).

Environment variables:

* ``MACRO_API_URL`` – Base URL of the economic calendar API.  Defaults
  to FMP's calendar endpoint ``https://financialmodelingprep.com/api/v3/economic_calendar``.
* ``MACRO_API_KEY`` – Your Financial Modeling Prep API key.  Required
  to call the API.  Without this the existing macro events file will be
  used and a warning is logged.

The module can be run as a script to manually refresh events:

    python -m macro_data_updater

This will download upcoming events and print how many were written.
"""

from __future__ import annotations


from core.exceptions import BEST_EFFORT_EXCEPTIONS
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

# .env dosyasını yüklemek için eklenen kısım.  Use find_dotenv to locate
# the nearest .env file in the directory hierarchy.  If dotenv is not
# installed, simply ignore and rely on OS environment variables.
try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore
    load_dotenv(find_dotenv())
except BEST_EFFORT_EXCEPTIONS:
    pass

logger = logging.getLogger(__name__)

# Optional dependencies handling
try:
    from retry_utils import retry  # type: ignore
except BEST_EFFORT_EXCEPTIONS:  # pragma: no cover
    def retry(*args, **kwargs):  # type: ignore
        def decorator(func): return func
        return decorator

try:
    from cache_manager import file_cache  # type: ignore
except BEST_EFFORT_EXCEPTIONS:  # pragma: no cover
    def file_cache(*args, **kwargs):  # type: ignore
        def decorator(func): return func
        return decorator


def _impact_to_multiplier(impact: str) -> float:
    """Convert FMP ``impact`` field into a risk multiplier.

    High impact events reduce risk more strongly than medium or low
    impact events.  The returned multiplier should be between 0 and 1.

    Parameters
    ----------
    impact : str
        Impact level string (case insensitive).  Common values are
        "High", "Medium" and "Low".

    Returns
    -------
    float
        Multiplier to apply to the master confidence during the event.
    """
    impact = (impact or "").lower()
    if impact == "high":
        return 0.5
    if impact == "medium":
        return 0.7
    if impact == "low":
        return 0.9
    return 1.0

# ---------------------------------------------------------------------------
# FRED API integration

def _fetch_fred_series(series_id: str, api_key: Optional[str],
                       start: Optional[str] = None, end: Optional[str] = None,
                       timeout: float = 10.0) -> List[Dict[str, Any]]:
    """
    Fetch a time series from the Federal Reserve Economic Data (FRED) API.

    Parameters
    ----------
    series_id : str
        FRED series identifier (e.g. "CPIAUCSL" for the Consumer Price Index).
    api_key : str or None
        Your FRED API key.  If not provided, the ``FRED_API_KEY``
        environment variable will be used.
    start : str, optional
        Optional observation start date in YYYY-MM-DD format.  If omitted,
        FRED defaults to the earliest available observation.
    end : str, optional
        Optional observation end date in YYYY-MM-DD format.  If omitted,
        FRED defaults to the latest available observation.
    timeout : float, optional
        HTTP request timeout in seconds.  Defaults to 10 seconds.

    Returns
    -------
    List of dictionaries containing observation dates and values.  An empty
    list is returned on error.
    """
    api_key = api_key or os.environ.get("FRED_API_KEY")
    if not api_key:
        logger.debug("No FRED API key set; skipping FRED series fetch for %s", series_id)
        return []
    url = "https://api.stlouisfed.org/fred/series/observations"
    params: Dict[str, Any] = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }
    if start:
        params["observation_start"] = start
    if end:
        params["observation_end"] = end
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        observations = data.get("observations", [])
        # Each observation has a date and a value string; convert value to float when possible
        cleaned: List[Dict[str, Any]] = []
        for obs in observations:
            date_str = obs.get("date")
            val_str = obs.get("value")
            try:
                val = float(val_str) if val_str not in (None, ".", "") else None
            except BEST_EFFORT_EXCEPTIONS:
                val = None
            cleaned.append({"date": date_str, "value": val})
        return cleaned
    except BEST_EFFORT_EXCEPTIONS as e:
        logger.warning("FRED series fetch failed for %s: %s", series_id, e)
        return []


@retry(tries=3, base_delay=5, max_delay=30)
@file_cache("fred_indicators_cache.json", ttl=86400)
def update_fred_indicators(series_ids: List[str] | None = None,
                           api_key: Optional[str] = None,
                           out_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch selected macroeconomic indicators from FRED and write them to disk.

    This helper is separate from the FMP macro calendar; it provides
    continuous time-series indicators (e.g. inflation, unemployment) used
    for advanced macro risk models.  It is not scheduled by default.

    Parameters
    ----------
    series_ids : list of str, optional
        A list of FRED series identifiers to fetch.  Defaults to a
        sensible set of popular indicators (CPI, unemployment rate,
        effective federal funds rate).
    api_key : str, optional
        FRED API key.  If not provided, ``FRED_API_KEY`` from the
        environment is used.
    out_file : str, optional
        Path to write the resulting JSON.  Defaults to
        ``data/fred_indicators.json`` relative to the project root.

    Returns
    -------
    dict
        A dictionary mapping series IDs to lists of observations with
        ``date`` and ``value`` keys.  Also includes an ``updated_at``
        timestamp.
    """
    root = Path(__file__).resolve().parent
    out_path = Path(out_file) if out_file else root / "data" / "fred_indicators.json"
    api_key = api_key or os.environ.get("FRED_API_KEY")
    if not series_ids:
        # Default indicators: CPI, unemployment rate, Fed funds rate
        series_ids = ["CPIAUCSL", "UNRATE", "FEDFUNDS"]
    result: Dict[str, Any] = {"updated_at": datetime.now(timezone.utc).isoformat()}
    for sid in series_ids:
        observations = _fetch_fred_series(sid, api_key)
        result[sid] = observations
    # Write JSON output
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info("Wrote FRED indicators to %s", out_path)
    except BEST_EFFORT_EXCEPTIONS as e:
        logger.warning("Failed to write FRED indicators file: %s", e)
    return result


def _normalise_event(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalise an event from the FMP API into the macro event format.

    Parameters
    ----------
    raw : dict
        Raw event data from FMP.  Expected keys include ``event``,
        ``date``, ``time``, ``impact`` and optionally ``country``.

    Returns
    -------
    dict or None
        The normalised event dictionary or None if required fields are
        missing or cannot be parsed.
    """
    try:
        event_name = str(raw.get("event") or "").strip()
        if not event_name:
            return None
        country = str(raw.get("country") or "").strip()
        name = f"{event_name} ({country})" if country else event_name
        date_str = str(raw.get("date") or "").strip()
        time_str = str(raw.get("time") or "").strip()
        if not date_str:
            return None
        if time_str:
            dt_str = f"{date_str}T{time_str}"
        else:
            dt_str = f"{date_str}T00:00:00"
        # Parse the timestamp, assume ISO format; fallback to date-only.
        try:
            dt = datetime.fromisoformat(dt_str)
        except BEST_EFFORT_EXCEPTIONS:
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
            except BEST_EFFORT_EXCEPTIONS:
                return None
        # Convert to UTC, drop timezone if naive
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        start = dt
        end = start + timedelta(hours=1)
        impact = str(raw.get("impact") or "")
        multiplier = _impact_to_multiplier(impact)
        impact_key = impact.lower()
        if impact_key not in {"low", "medium", "high"}:
            impact_key = "medium"
        pre_minutes = 60
        return {
            "name": name,
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "datetime": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "impact": impact_key.upper(),
            "multiplier": float(multiplier),
            "pre_minutes": int(pre_minutes),
            "source": "fmp",
        }
    except BEST_EFFORT_EXCEPTIONS:
        return None


@retry(tries=3, base_delay=5, max_delay=30)
@file_cache("macro_events_cache.json", ttl=3600)
def update_macro_events(
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    days_ahead: int = 7,
) -> List[Dict[str, Any]]:
    """Update the macro events file using the FMP economic calendar API.

    Parameters
    ----------
    api_url : str, optional
        The economic calendar API endpoint.  Defaults to the FMP
        endpoint if not provided.  The environment variable
        ``MACRO_API_URL`` can override this.
    api_key : str, optional
        The API key for FMP.  If not provided, the environment
        variable ``MACRO_API_KEY`` will be used.  If no key is
        available, the existing macro events file will be returned
        unchanged.
    days_ahead : int, optional
        Number of days ahead to fetch events for.  Defaults to 7.

    Returns
    -------
    list
        List of normalised events that were written to the JSON file.
    """
    root = Path(__file__).resolve().parent
    data_file = root / "data" / "macro_events.json"
    
    # Resolve config from environment if not explicitly provided
    api_url = api_url or os.environ.get("MACRO_API_URL")
    api_key = api_key or os.environ.get("MACRO_API_KEY")

    # Check config for enabling macro events
    try:
        cfg_path = root / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        macro_cfg = cfg.get("macro_events") if isinstance(cfg.get("macro_events"), dict) else {}
        enabled_flag = cfg.get("enable_macro_events", macro_cfg.get("enabled"))
        if enabled_flag is False:
            logger.info("Macro events are disabled via config (enable_macro_events=False).")
            return []
    except BEST_EFFORT_EXCEPTIONS:
        pass
    
    # Default to FMP endpoint
    if not api_url:
        api_url = "https://financialmodelingprep.com/api/v3/economic_calendar"
    
    if not api_key:
        logger.warning("Macro API credentials missing; attempting to load manual macro events.")
        # Try to load a manual events file (data/manual_macro_events.json) if present.
        manual_path = root / "data" / "manual_macro_events.json"
        try:
            if manual_path.exists():
                manual_text = manual_path.read_text(encoding="utf-8").strip()
                if manual_text:
                    events = json.loads(manual_text)
                    if isinstance(events, list):
                        logger.info(f"Loaded {len(events)} manual macro events from {manual_path}")
                        # Overwrite existing events file for consistency
                        try:
                            data_file.parent.mkdir(parents=True, exist_ok=True)
                            with data_file.open("w", encoding="utf-8") as f:
                                json.dump(events, f, indent=2)
                        except BEST_EFFORT_EXCEPTIONS:
                            pass
                        return events
        except BEST_EFFORT_EXCEPTIONS as exc:
            logger.warning(f"Failed to read manual macro events: {exc}")
        
        # Fallback to existing macro_events.json if available
        try:
            if data_file.exists():
                existing_text = data_file.read_text(encoding="utf-8").strip()
                if existing_text:
                    events = json.loads(existing_text)
                    if isinstance(events, list):
                        logger.info(f"Using {len(events)} cached macro events from {data_file}")
                        return events
        except BEST_EFFORT_EXCEPTIONS:
            pass
        # No manual or cached events found; return empty list
        return []

    # Determine date range
    today = datetime.now(timezone.utc).date()
    start_date = today.strftime("%Y-%m-%d")
    end_date = (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    params = {
        "from": start_date,
        "to": end_date,
        "apikey": api_key,
    }
    headers = {
        "User-Agent": "CryptoBot/1.0",
        "Accept": "application/json",
    }
    events: List[Dict[str, Any]] = []
    
    try:
        resp = requests.get(api_url, headers=headers, params=params, timeout=15)
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as http_err:  # type: ignore[attr-defined]
            # Handle specific HTTP status codes
            status = getattr(http_err.response, "status_code", None)
            if status == 403:
                # Forbidden – likely invalid API key or quota exceeded.
                logger.warning("Macro API request failed: 403 Forbidden. Please check your MACRO_API_KEY in .env")
                try:
                    if data_file.exists():
                        return json.loads(data_file.read_text(encoding="utf-8"))
                except BEST_EFFORT_EXCEPTIONS:
                    return []
                return []
            # For other HTTP errors re‑raise to be caught below
            raise
        
        data = resp.json()
        if isinstance(data, dict):
            raw_events = data.get("data") or data.get("events") or []
        elif isinstance(data, list):
            raw_events = data
        else:
            raw_events = []
        for item in raw_events:
            norm = _normalise_event(item)
            if norm:
                events.append(norm)
                
    except BEST_EFFORT_EXCEPTIONS as e:
        # Generic error: log and fall back to cached events if available
        logger.warning(f"Macro API request failed: {e}")
        try:
            if data_file.exists():
                return json.loads(data_file.read_text(encoding="utf-8"))
        except BEST_EFFORT_EXCEPTIONS:
            return []
        return []

    # Write out the events
    try:
        data_file.parent.mkdir(parents=True, exist_ok=True)
        with data_file.open("w", encoding="utf-8") as f:
            json.dump(events, f, indent=2)
        logger.info(f"Wrote {len(events)} macro events to {data_file}")
    except BEST_EFFORT_EXCEPTIONS as e:
        logger.error(f"Failed to write macro events: {e}")
    return events

import asyncio
async def update_macro_events_async(*args, **kwargs) -> List[Dict[str, Any]]:
    """Async wrapper that runs the synchronous fetching in a separate thread to prevent Event Loop blocking."""
    return await asyncio.to_thread(update_macro_events, *args, **kwargs)
if __name__ == "__main__":  # pragma: no cover
    # Script olarak çalıştırıldığında .env'in yüklendiğinden emin oluyoruz (yukarıdaki import sayesinde)
    events = update_macro_events()
    print(f"Updated {len(events)} macro events.")
