#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atomic_io import atomic_write_json, file_lock, flush_async_io_queue
from decision.v5_breadth_shadow import (
    PREREGISTERED_SYMBOLS,
    evaluate_v5_breadth_shadow,
    record_v5_shadow_evidence,
)
from tools.v5_shadow_outcome_report import _read_jsonl, build_outcome_report

OKX_BASE = "https://www.okx.com"
DEFAULT_EVIDENCE = ROOT / "reports" / "v5_breadth_shadow_events.jsonl"
DEFAULT_DATA_DIR = ROOT / "data" / "fresh_v5_shadow"
DEFAULT_POLICY = ROOT / "config" / "v5_prospective_shadow_policy.json"
DEFAULT_STATUS = ROOT / "reports" / "v5_public_shadow_observer_status.json"
DEFAULT_OUTCOMES = ROOT / "reports" / "v5_prospective_shadow_outcomes.json"
FOUR_HOUR_MINIMUM = 211
FIFTEEN_MINUTE_CACHE_BARS = 1200
REQUEST_TIMEOUT_SECONDS = 20.0
USER_AGENT = "AutoTraderBot-V5-Public-Shadow/20261008"


def _inst_id(symbol: str) -> str:
    base = symbol.split("/", 1)[0].strip().upper()
    return f"{base}-USDT-SWAP"


def _data_path(data_dir: Path, symbol: str) -> Path:
    base = symbol.split("/", 1)[0].strip().upper()
    return data_dir / f"{base}_USDT_USDT_15m.parquet"


def _get_json(
    session: requests.Session,
    path: str,
    *,
    params: dict[str, str] | None = None,
    attempts: int = 5,
) -> dict[str, Any]:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            response = session.get(
                OKX_BASE + path,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
                headers={"User-Agent": USER_AGENT},
            )
            if response.status_code == 429:
                time.sleep(0.5 + attempt * 0.75)
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("OKX response is not an object")
            if str(payload.get("code", "0")) != "0":
                raise RuntimeError(
                    f"OKX error code={payload.get('code')} msg={payload.get('msg')}"
                )
            return payload
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(0.5 + attempt * 0.75)
    raise RuntimeError(f"OKX public request failed path={path} params={params}: {last}")


def parse_okx_candles(rows: Any) -> pd.DataFrame:
    prepared: list[tuple[pd.Timestamp, float, float, float, float, float]] = []
    if not isinstance(rows, list):
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
    for raw in rows:
        if not isinstance(raw, list) or len(raw) < 6:
            continue
        try:
            confirm = str(raw[8]) if len(raw) >= 9 else str(raw[-1])
            if confirm != "1":
                continue
            prepared.append(
                (
                    pd.to_datetime(int(raw[0]), unit="ms", utc=True),
                    float(raw[1]),
                    float(raw[2]),
                    float(raw[3]),
                    float(raw[4]),
                    float(raw[5]),
                )
            )
        except (TypeError, ValueError, OverflowError):
            continue
    if not prepared:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
    return (
        pd.DataFrame(
            prepared,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def validate_candle_cadence(
    frame: pd.DataFrame,
    *,
    expected: pd.Timedelta,
    minimum_bars: int,
) -> None:
    if len(frame) < int(minimum_bars):
        raise ValueError(
            f"insufficient confirmed candles: got={len(frame)} required={minimum_bars}"
        )
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
    if timestamps.has_duplicates:
        raise ValueError("duplicate candle timestamps")
    deltas = timestamps[1:] - timestamps[:-1]
    if bool((deltas != expected).any()):
        position = int((deltas != expected).argmax())
        raise ValueError(
            "cadence gap: "
            f"{timestamps[position].isoformat()} -> "
            f"{timestamps[position + 1].isoformat()} "
            f"delta={deltas[position]}"
        )


def merge_candle_cache(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    *,
    max_bars: int,
) -> pd.DataFrame:
    frames = [frame for frame in (existing, incoming) if not frame.empty]
    if not frames:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
    merged = pd.concat(frames, ignore_index=True)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    merged = (
        merged.dropna(subset=["timestamp", "open", "high", "low", "close"])
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    if max_bars > 0 and len(merged) > max_bars:
        merged = merged.iloc[-max_bars:].reset_index(drop=True)
    return merged


def _write_parquet_locked(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path, timeout=10.0):
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        try:
            frame.to_parquet(tmp, index=False)
            os.replace(tmp, path)
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass


def _load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError, ImportError) as exc:
        raise RuntimeError(f"unable to read existing cache {path}: {exc}") from exc


def _server_time(session: requests.Session) -> datetime:
    payload = _get_json(session, "/api/v5/public/time")
    rows = payload.get("data")
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        raise RuntimeError("OKX public time response missing data")
    try:
        ts_ms = int(rows[0]["ts"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("OKX public time response has invalid timestamp") from exc
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC)


def _fetch_candles(
    session: requests.Session,
    symbol: str,
    *,
    bar: str,
    limit: int = 300,
) -> pd.DataFrame:
    payload = _get_json(
        session,
        "/api/v5/market/candles",
        params={
            "instId": _inst_id(symbol),
            "bar": str(bar),
            "limit": str(limit),
        },
    )
    return parse_okx_candles(payload.get("data"))


def _runtime_item(symbol: str, four_hour: pd.DataFrame) -> dict[str, Any]:
    runtime_4h = four_hour.rename(
        columns={
            "timestamp": "timestamp_4h",
            "open": "open_4h",
            "high": "high_4h",
            "low": "low_4h",
            "close": "close_4h",
            "volume": "volume_4h",
        }
    )
    price = float(four_hour.iloc[-1]["close"])
    return {
        "symbol": symbol,
        "price": price,
        "current_price": price,
        "mtf_data": {"4h": runtime_4h},
        "research_observer_source": "okx_public_mainnet_rest",
    }


def collect_once(
    *,
    evidence_path: Path,
    data_dir: Path,
    policy_path: Path,
    status_path: Path,
    outcome_path: Path,
) -> dict[str, Any]:
    session = requests.Session()
    now = _server_time(session)
    symbol_inputs: list[dict[str, Any]] = []
    integrity: dict[str, Any] = {}

    for symbol in PREREGISTERED_SYMBOLS:
        four_hour = _fetch_candles(session, symbol, bar="4H")
        validate_candle_cadence(
            four_hour,
            expected=pd.Timedelta(hours=4),
            minimum_bars=FOUR_HOUR_MINIMUM,
        )
        symbol_inputs.append(_runtime_item(symbol, four_hour))

        fifteen = _fetch_candles(session, symbol, bar="15m")
        if len(fifteen) < 2:
            raise RuntimeError(f"insufficient public 15m candles for {symbol}")
        validate_candle_cadence(
            fifteen,
            expected=pd.Timedelta(minutes=15),
            minimum_bars=2,
        )
        path = _data_path(data_dir, symbol)
        existing = _load_existing(path)
        merged = merge_candle_cache(
            existing,
            fifteen,
            max_bars=FIFTEEN_MINUTE_CACHE_BARS,
        )
        # A gap in the retained cache makes an affected candidate unscorable.
        # Refuse to silently stitch across missing market observations.
        if len(merged) >= 2:
            validate_candle_cadence(
                merged,
                expected=pd.Timedelta(minutes=15),
                minimum_bars=2,
            )
        _write_parquet_locked(path, merged)
        integrity[symbol] = {
            "4h_bars": len(four_hour),
            "4h_first": pd.Timestamp(four_hour.iloc[0]["timestamp"]).isoformat(),
            "4h_last": pd.Timestamp(four_hour.iloc[-1]["timestamp"]).isoformat(),
            "15m_cache_bars": len(merged),
            "15m_cache_first": pd.Timestamp(merged.iloc[0]["timestamp"]).isoformat(),
            "15m_cache_last": pd.Timestamp(merged.iloc[-1]["timestamp"]).isoformat(),
        }

    payload = dict(evaluate_v5_breadth_shadow(symbol_inputs, now=now))
    payload["evidence_source"] = {
        "venue": "okx",
        "market_data": "public_mainnet_rest",
        "authenticated": False,
        "order_endpoints_used": False,
        "okx_server_time": now.isoformat(),
    }
    if payload.get("execution_authority") is not False:
        raise RuntimeError("V5 observer evaluator violated execution_authority=false")
    if not record_v5_shadow_evidence(payload, path=evidence_path):
        raise RuntimeError("failed to record V5 public shadow evidence")
    if not flush_async_io_queue(timeout=5.0):
        raise RuntimeError("timed out flushing V5 shadow evidence")

    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        if not isinstance(policy, dict):
            raise ValueError("policy must be a JSON object")
        records = _read_jsonl(evidence_path)
        outcomes = build_outcome_report(records, policy, data_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"unable to build V5 forward outcome report: {exc}") from exc

    outcomes["events_path"] = str(evidence_path)
    outcomes["policy_path"] = str(policy_path)
    outcomes["data_dir"] = str(data_dir)
    outcomes["record_count"] = len(records)
    if not atomic_write_json(outcome_path, outcomes):
        raise RuntimeError(f"failed to write V5 outcome report {outcome_path}")

    status = {
        "schema": "autotraderbot-v5-public-shadow-observer-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "okx_server_time": now.isoformat(),
        "public_mainnet_only": True,
        "authenticated_requests": False,
        "order_endpoints_used": False,
        "live_mode_used": False,
        "execution_authority": False,
        "promotion_authority": False,
        "symbols": list(PREREGISTERED_SYMBOLS),
        "integrity": integrity,
        "shadow_event": payload,
        "outcome_status": {
            "scoring_status": outcomes.get("scoring_status"),
            "observation_gate_passed": bool(
                (outcomes.get("observation_gate") or {}).get("passed")
            ),
            "outcome_gate_passed": bool(
                (outcomes.get("outcome_gate") or {}).get("passed")
            ),
            "promotion_review_ready": bool(outcomes.get("promotion_review_ready")),
        },
    }
    if not atomic_write_json(status_path, status):
        raise RuntimeError(f"failed to write V5 observer status {status_path}")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collect V5 prospective shadow evidence from unauthenticated "
            "OKX public mainnet market data only."
        )
    )
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    args = parser.parse_args()

    status = collect_once(
        evidence_path=args.evidence,
        data_dir=args.data_dir,
        policy_path=args.policy,
        status_path=args.status,
        outcome_path=args.outcomes,
    )
    print("V5_PUBLIC_SHADOW=" + json.dumps(status, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
