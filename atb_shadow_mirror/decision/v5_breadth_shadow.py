from __future__ import annotations

import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from analysis.indicator_math import wilder_dmi_adx
from atomic_io import enqueue_append_jsonl
from research.cross_market_v5 import (
    BEAR_BREADTH,
    BULL_BREADTH,
    DONCHIAN_LENGTH,
    FAMILY,
    MIN_MEDIAN_ADX,
    PREREGISTERED_SYMBOLS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE_PATH = PROJECT_ROOT / "reports" / "v5_breadth_shadow_events.jsonl"
MIN_COMPLETED_4H_BARS = 211
BREAKOUT_BUFFER = 0.001

_RECORD_LOCK = threading.Lock()
_LAST_RECORDED_KEY: str | None = None


def _canonical_symbol(value: Any) -> str:
    raw = str(value or "").strip().upper()
    if not raw:
        return ""
    base = raw.split("/", 1)[0].split("-", 1)[0].strip()
    return f"{base}/USDT:USDT" if base else ""


def _column(frame: pd.DataFrame, base: str) -> str | None:
    candidates = (
        f"{base}_4h",
        base,
        base.lower(),
        base.upper(),
        base.capitalize(),
    )
    columns = set(str(column) for column in frame.columns)
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _runtime_4h_frame(item: dict[str, Any], *, now: datetime) -> pd.DataFrame | None:
    multi = item.get("mtf_data")
    if not isinstance(multi, dict):
        return None
    raw = multi.get("4h")
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        return None

    timestamp_col = _column(raw, "timestamp")
    open_col = _column(raw, "open")
    high_col = _column(raw, "high")
    low_col = _column(raw, "low")
    close_col = _column(raw, "close")
    volume_col = _column(raw, "volume")
    required = (timestamp_col, open_col, high_col, low_col, close_col, volume_col)
    if any(column is None for column in required):
        return None

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(raw[timestamp_col], utc=True, errors="coerce"),
            "open": pd.to_numeric(raw[open_col], errors="coerce"),
            "high": pd.to_numeric(raw[high_col], errors="coerce"),
            "low": pd.to_numeric(raw[low_col], errors="coerce"),
            "close": pd.to_numeric(raw[close_col], errors="coerce"),
            "volume": pd.to_numeric(raw[volume_col], errors="coerce"),
        }
    )
    frame = (
        frame.dropna(subset=["timestamp", "open", "high", "low", "close"])
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    if frame.empty:
        return None

    now_ts = pd.Timestamp(now)
    now_ts = now_ts.tz_localize("UTC") if now_ts.tzinfo is None else now_ts.tz_convert("UTC")
    completed = frame[(frame["timestamp"] + pd.Timedelta(hours=4)) <= now_ts].copy()
    return completed.reset_index(drop=True)


def _decorate(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    out["ema50"] = close.ewm(span=50, adjust=False, min_periods=50).mean()
    out["ema200"] = close.ewm(span=200, adjust=False, min_periods=200).mean()
    dmi = wilder_dmi_adx(out["high"], out["low"], close, period=14)
    out["atr"] = dmi["atr"]
    out["adx"] = dmi["adx"]
    return out


def _not_ready(
    reason: str,
    *,
    now: datetime,
    missing: Iterable[str] = (),
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": "autotraderbot-v5-breadth-shadow-v1",
        "family": FAMILY,
        "shadow_only": True,
        "execution_authority": False,
        "ready": False,
        "reason": reason,
        "observed_at": now.astimezone(UTC).isoformat(),
        "missing_symbols": list(missing),
        "details": dict(details or {}),
        "candidates": [],
    }


def evaluate_v5_breadth_shadow(
    symbol_inputs: list[dict[str, Any]],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    observed_at = now or datetime.now(UTC)
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=UTC)
    else:
        observed_at = observed_at.astimezone(UTC)

    by_symbol: dict[str, dict[str, Any]] = {}
    for item in symbol_inputs:
        if not isinstance(item, dict):
            continue
        canonical = _canonical_symbol(item.get("symbol"))
        if canonical in PREREGISTERED_SYMBOLS and canonical not in by_symbol:
            by_symbol[canonical] = item

    missing = [symbol for symbol in PREREGISTERED_SYMBOLS if symbol not in by_symbol]
    if missing:
        return _not_ready(
            "missing_preregistered_symbols",
            now=observed_at,
            missing=missing,
            details={"present": sorted(by_symbol)},
        )

    frames: dict[str, pd.DataFrame] = {}
    short: dict[str, int] = {}
    for symbol in PREREGISTERED_SYMBOLS:
        frame = _runtime_4h_frame(by_symbol[symbol], now=observed_at)
        count = 0 if frame is None else len(frame)
        if frame is None or count < MIN_COMPLETED_4H_BARS:
            short[symbol] = count
            continue
        frames[symbol] = _decorate(frame)

    if short:
        return _not_ready(
            "insufficient_completed_4h_history",
            now=observed_at,
            details={
                "minimum_completed_4h_bars": MIN_COMPLETED_4H_BARS,
                "completed_bars": short,
            },
        )

    latest_times = {
        symbol: pd.Timestamp(frame.iloc[-1]["timestamp"])
        for symbol, frame in frames.items()
    }
    common_latest = set(latest_times.values())
    if len(common_latest) != 1:
        return _not_ready(
            "unaligned_completed_4h_bars",
            now=observed_at,
            details={
                "latest_completed_4h": {
                    symbol: timestamp.isoformat()
                    for symbol, timestamp in latest_times.items()
                }
            },
        )

    bar_time = next(iter(common_latest))
    latest_rows: dict[str, pd.Series] = {}
    above: list[bool] = []
    adx_values: list[float] = []

    for symbol in PREREGISTERED_SYMBOLS:
        frame = frames[symbol]
        row = frame.iloc[-1]
        close = float(row["close"])
        ema200 = float(row["ema200"])
        adx = float(row["adx"])
        if not all(np.isfinite(value) for value in (close, ema200, adx)):
            return _not_ready(
                "indicator_warmup_incomplete",
                now=observed_at,
                details={"symbol": symbol},
            )
        latest_rows[symbol] = row
        above.append(close > ema200)
        adx_values.append(adx)

    breadth = float(sum(above)) / float(len(above))
    median_adx = float(np.median(adx_values))
    if median_adx < MIN_MEDIAN_ADX:
        regime = "neutral"
    elif breadth >= BULL_BREADTH:
        regime = "bull"
    elif breadth <= BEAR_BREADTH:
        regime = "bear"
    else:
        regime = "neutral"

    candidates: list[dict[str, Any]] = []
    if regime in {"bull", "bear"}:
        for symbol in PREREGISTERED_SYMBOLS:
            frame = frames[symbol]
            row = latest_rows[symbol]
            close = float(row["close"])
            ema50 = float(row["ema50"])
            ema200 = float(row["ema200"])
            atr = float(row["atr"])
            if not all(np.isfinite(value) for value in (close, ema50, ema200, atr)) or atr <= 0:
                continue

            prior = frame.iloc[-(DONCHIAN_LENGTH + 1) : -1]
            if len(prior) < DONCHIAN_LENGTH:
                continue
            prior_high = float(pd.to_numeric(prior["high"], errors="coerce").max())
            prior_low = float(pd.to_numeric(prior["low"], errors="coerce").min())

            side: str | None = None
            if regime == "bull" and ema50 > ema200 and close > prior_high * (1.0 + BREAKOUT_BUFFER):
                side = "long"
            elif regime == "bear" and ema50 < ema200 and close < prior_low * (1.0 - BREAKOUT_BUFFER):
                side = "short"
            if side is None:
                continue

            candidates.append(
                {
                    "family": FAMILY,
                    "setup_id": f"{FAMILY}.entry.{side}.4h.v1",
                    "symbol": symbol,
                    "side": side,
                    "bar_time": bar_time.isoformat(),
                    "reference_price": close,
                    "atr_4h": atr,
                    "stop_atr_mult": 2.0,
                    "trail_atr_mult": 3.0,
                    "breakeven_after_r": 0.5,
                    "trail_after_r": 1.0,
                    "decay_bars_15m": 192,
                    "max_hold_bars_15m": 960,
                    "hypothetical_risk_per_trade_pct": 0.0025,
                    "hypothetical_wallet_cap_pct": 0.20,
                    "hypothetical_leverage": 1.0,
                    "shadow_only": True,
                    "execution_authority": False,
                }
            )

    return {
        "schema": "autotraderbot-v5-breadth-shadow-v1",
        "family": FAMILY,
        "shadow_only": True,
        "execution_authority": False,
        "ready": True,
        "reason": "ready",
        "observed_at": observed_at.isoformat(),
        "bar_time": bar_time.isoformat(),
        "symbols": list(PREREGISTERED_SYMBOLS),
        "global_state": {
            "regime": regime,
            "breadth_above_ema200": round(breadth, 6),
            "median_adx": round(median_adx, 6),
            "bull_threshold": BULL_BREADTH,
            "bear_threshold": BEAR_BREADTH,
            "min_median_adx": MIN_MEDIAN_ADX,
        },
        "candidates": candidates,
    }


def _event_key(payload: dict[str, Any]) -> str:
    if bool(payload.get("ready")):
        return f"ready:{payload.get('bar_time')}"
    return "not_ready:" + str(payload.get("reason") or "unknown")


def record_v5_shadow_evidence(
    payload: dict[str, Any],
    *,
    path: Path | str = DEFAULT_EVIDENCE_PATH,
) -> bool:
    global _LAST_RECORDED_KEY
    key = _event_key(payload)
    with _RECORD_LOCK:
        if key == _LAST_RECORDED_KEY:
            return True
        queued = enqueue_append_jsonl(path, payload)
        if queued:
            _LAST_RECORDED_KEY = key
        return queued


def v5_breadth_shadow_enabled(config: dict[str, Any] | None = None) -> bool:
    """Return the explicit runtime shadow flag.

    This is deliberately default-off. Enabling it only records hypothetical
    V5 candidates; it never grants execution authority.
    """
    enabled: Any = None
    if isinstance(config, dict):
        research = config.get("research_runtime")
        if isinstance(research, dict):
            shadow_cfg = research.get("v5_breadth_shadow")
            if isinstance(shadow_cfg, dict) and "enabled" in shadow_cfg:
                enabled = shadow_cfg.get("enabled")
    if enabled is None:
        enabled = os.getenv("ATB_V5_BREADTH_SHADOW_ENABLED", "")
    if isinstance(enabled, bool):
        return enabled
    return str(enabled or "").strip().lower() in {"1", "true", "yes", "on", "enabled"}


def run_v5_breadth_shadow_cycle(
    symbol_inputs: list[dict[str, Any]],
    *,
    config: dict[str, Any] | None = None,
    now: datetime | None = None,
    path: Path | str | None = None,
) -> dict[str, Any]:
    """Evaluate and persist one research-only V5 runtime shadow cycle."""
    if not v5_breadth_shadow_enabled(config):
        return {
            "schema": "autotraderbot-v5-breadth-shadow-v1",
            "family": FAMILY,
            "feature_flag_enabled": False,
            "shadow_only": True,
            "execution_authority": False,
            "ready": False,
            "reason": "feature_flag_disabled",
            "candidates": [],
            "evidence_recorded": False,
        }

    payload = evaluate_v5_breadth_shadow(symbol_inputs, now=now)
    evidence_path: Path | str = path or DEFAULT_EVIDENCE_PATH
    if path is None and isinstance(config, dict):
        research = config.get("research_runtime")
        if isinstance(research, dict):
            shadow_cfg = research.get("v5_breadth_shadow")
            if isinstance(shadow_cfg, dict):
                configured = shadow_cfg.get("evidence_path")
                if isinstance(configured, str) and configured.strip():
                    evidence_path = configured.strip()

    recorded = record_v5_shadow_evidence(payload, path=evidence_path)
    return {
        **payload,
        "feature_flag_enabled": True,
        "shadow_only": True,
        "execution_authority": False,
        "evidence_recorded": bool(recorded),
        "evidence_path": str(evidence_path),
    }


__all__ = [
    "DEFAULT_EVIDENCE_PATH",
    "MIN_COMPLETED_4H_BARS",
    "PREREGISTERED_SYMBOLS",
    "evaluate_v5_breadth_shadow",
    "record_v5_shadow_evidence",
    "run_v5_breadth_shadow_cycle",
    "v5_breadth_shadow_enabled",
]
