# -*- coding: utf-8 -*-
"""
signal_labeler.py
=================

Leakage-safe signal labeling utilities.

Goals:
- avoid row-order matching
- support richer labels than binary win/loss
- keep backward-compatible ``label`` for existing calibration consumers
"""

from __future__ import annotations


from core.exceptions import BEST_EFFORT_EXCEPTIONS
import csv
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from atomic_io import file_lock
from runtime_paths import get_trade_log_path

logger = logging.getLogger(__name__)
_labeler_lock = threading.RLock()

ROOT = Path(__file__).resolve().parent
SIGNAL_DATASET = ROOT / "data" / "signal_dataset.csv"
TRADE_LOG = get_trade_log_path()
LABELED_SIGNALS_LOG = ROOT / "metrics" / "labeled_signals.jsonl"

BASE_COLUMNS = [
    "signal_id",
    "timestamp",
    "symbol",
    "tf",
    "master_conf_raw",
    "ai_score",
    "tech_score",
    "sent_score",
    "rl_score",
    "base_decision",
    "price_entry_planned",
    "regime",
    "label",
]
LABEL_COLUMNS = [
    "label_class",
    "normalized_pnl",
    "sharpe_per_trade",
    "label_timestamp",
    "label_entry_time",
    "label_exit_time",
]
TARGET_MODE_TO_COLUMN = {
    "binary": "label",
    "multiclass": "label_class",
    "regression": "normalized_pnl",
}


def _ensure_dirs() -> None:
    SIGNAL_DATASET.parent.mkdir(parents=True, exist_ok=True)
    LABELED_SIGNALS_LOG.parent.mkdir(parents=True, exist_ok=True)


def _iso_or_none(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    try:
        return value.astimezone(timezone.utc).isoformat()
    except BEST_EFFORT_EXCEPTIONS:
        return value.isoformat()


def _parse_dt(raw: Any) -> Optional[datetime]:
    if raw is None or raw == "":
        return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _normalize_signal_id(signal_id: Optional[str], symbol: str, entry_time: Optional[datetime]) -> Optional[str]:
    if signal_id:
        return str(signal_id)
    if entry_time is None:
        return None
    ts_iso = entry_time.astimezone(timezone.utc).replace(microsecond=0).isoformat()
    return f"{ts_iso}_{symbol}"


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _classify_outcome(normalized_pnl: Optional[float], pnl: float, sharpe_per_trade: Optional[float]) -> str:
    score = normalized_pnl
    if score is None:
        score = sharpe_per_trade
    if score is None:
        score = pnl
    if score >= 1.0:
        return "buyuk_kazanc"
    if score >= 0.1:
        return "kucuk_kazanc"
    if score <= -1.0:
        return "buyuk_kayip"
    if score <= -0.1:
        return "kucuk_kayip"
    return "notr"


def _label_payload(
    *,
    pnl: float,
    atr: Optional[float],
    volatility: Optional[float],
    entry_time: Optional[datetime],
    exit_time: Optional[datetime],
) -> dict[str, Any]:
    normalized_pnl = None
    if atr is not None and abs(atr) > 1e-12:
        normalized_pnl = pnl / atr
    sharpe_per_trade = None
    if volatility is not None and abs(volatility) > 1e-12:
        sharpe_per_trade = pnl / volatility
    label_class = _classify_outcome(normalized_pnl, pnl, sharpe_per_trade)
    return {
        "label": 1 if pnl > 0 else 0,
        "label_class": label_class,
        "normalized_pnl": round(normalized_pnl, 6) if normalized_pnl is not None else None,
        "sharpe_per_trade": round(sharpe_per_trade, 6) if sharpe_per_trade is not None else None,
        "label_timestamp": datetime.now(timezone.utc).isoformat(),
        "label_entry_time": _iso_or_none(entry_time),
        "label_exit_time": _iso_or_none(exit_time),
    }


def label_trade_result(
    symbol: str,
    pnl: float,
    entry_time: Optional[datetime] = None,
    exit_time: Optional[datetime] = None,
    *,
    signal_id: Optional[str] = None,
    atr: Optional[float] = None,
    volatility: Optional[float] = None,
) -> bool:
    """
    Update a signal row using deterministic trade-to-signal matching.

    Matching priority:
    1. signal_id
    2. symbol + entry timestamp
    3. derived signal_id from ``entry_time`` and ``symbol``
    """
    _ensure_dirs()
    payload = _label_payload(
        pnl=float(pnl),
        atr=_safe_float(atr),
        volatility=_safe_float(volatility),
        entry_time=entry_time,
        exit_time=exit_time,
    )
    resolved_signal_id = _normalize_signal_id(signal_id, symbol, entry_time)
    updated = _update_signal_dataset(
        symbol=symbol,
        label_payload=payload,
        signal_id=resolved_signal_id,
        entry_time=entry_time,
    )
    _log_labeled_signal(
        symbol=symbol,
        pnl=float(pnl),
        signal_id=resolved_signal_id,
        payload=payload,
        entry_time=entry_time,
        exit_time=exit_time,
    )
    if updated:
        logger.info(
            "[SIGNAL_LABEL] %s: label=%s class=%s pnl=%.4f signal_id=%s",
            symbol,
            payload["label"],
            payload["label_class"],
            float(pnl),
            resolved_signal_id,
        )
        try:
            from decision.master_score_calibration import maybe_recalibrate_master_score

            maybe_recalibrate_master_score(source_hint=SIGNAL_DATASET)
        except BEST_EFFORT_EXCEPTIONS:
            pass
    return updated


def _ensure_fieldnames(headers: list[str]) -> list[str]:
    merged = list(headers or BASE_COLUMNS)
    for col in LABEL_COLUMNS:
        if col not in merged:
            merged.append(col)
    return merged


def _rewrite_signal_dataset(headers: list[str], rows: list[dict[str, Any]]) -> None:
    tmp_path = SIGNAL_DATASET.with_suffix(SIGNAL_DATASET.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in headers})
        f.flush()
        os.fsync(f.fileno())
    last_error: Exception | None = None
    for _ in range(5):
        try:
            os.replace(tmp_path, SIGNAL_DATASET)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.05)
    if last_error is not None:
        raise last_error
    os.replace(tmp_path, SIGNAL_DATASET)


def _find_matching_row_index(
    rows: list[dict[str, Any]],
    *,
    symbol: str,
    signal_id: Optional[str],
    entry_time: Optional[datetime],
) -> Optional[int]:
    entry_iso = _iso_or_none(entry_time)
    for idx, row in enumerate(rows):
        if str(row.get("symbol") or "") != symbol:
            continue
        if signal_id and str(row.get("signal_id") or "") == signal_id:
            return idx
    if entry_iso is not None:
        for idx, row in enumerate(rows):
            if str(row.get("symbol") or "") != symbol:
                continue
            if str(row.get("timestamp") or "") == entry_iso:
                return idx
    return None


def _update_signal_dataset(
    *,
    symbol: str,
    label_payload: dict[str, Any],
    signal_id: Optional[str],
    entry_time: Optional[datetime],
) -> bool:
    with _labeler_lock:
        if not SIGNAL_DATASET.exists():
            logger.debug("signal_dataset.csv not found")
            return False
        try:
            with file_lock(SIGNAL_DATASET, timeout=10):
                with SIGNAL_DATASET.open("r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    headers = _ensure_fieldnames(list(reader.fieldnames or []))
                if not rows:
                    return False

                match_idx = _find_matching_row_index(
                    rows,
                    symbol=symbol,
                    signal_id=signal_id,
                    entry_time=entry_time,
                )
                if match_idx is None:
                    logger.debug("No signal row matched symbol=%s signal_id=%s", symbol, signal_id)
                    return False

                updated_row = dict(rows[match_idx])
                for key, value in label_payload.items():
                    updated_row[key] = "" if value is None else str(value)
                rows[match_idx] = updated_row
                _rewrite_signal_dataset(headers, rows)
                return True
        except BEST_EFFORT_EXCEPTIONS as e:
            logger.warning("signal_dataset update error: %s", e)
            return False


def _log_labeled_signal(
    *,
    symbol: str,
    pnl: float,
    signal_id: Optional[str],
    payload: dict[str, Any],
    entry_time: Optional[datetime],
    exit_time: Optional[datetime],
) -> None:
    with _labeler_lock:
        try:
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "signal_id": signal_id,
                "pnl": round(float(pnl), 6),
                "label": payload.get("label"),
                "label_class": payload.get("label_class"),
                "normalized_pnl": payload.get("normalized_pnl"),
                "sharpe_per_trade": payload.get("sharpe_per_trade"),
                "entry_time": _iso_or_none(entry_time),
                "exit_time": _iso_or_none(exit_time),
            }
            with LABELED_SIGNALS_LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except BEST_EFFORT_EXCEPTIONS as e:
            logger.debug("labeled_signals log error: %s", e)


def batch_label_from_trade_log() -> int:
    """Read all trades from trade log and assign richer labels deterministically."""
    if not TRADE_LOG.exists():
        return 0
    try:
        data = json.loads(TRADE_LOG.read_text(encoding="utf-8"))
        trades = data if isinstance(data, list) else data.get("rows", []) if isinstance(data, dict) else []
        labeled_count = 0
        for trade in trades:
            symbol = trade.get("symbol")
            pnl = _safe_float(trade.get("pnl_pct"))
            if pnl is None:
                pnl = _safe_float(trade.get("pnl_abs"))
            if not symbol or pnl is None:
                continue
            if label_trade_result(
                symbol=str(symbol),
                pnl=float(pnl),
                entry_time=_parse_dt(trade.get("timestamp_open") or trade.get("entry_time")),
                exit_time=_parse_dt(trade.get("timestamp_close") or trade.get("exit_time")),
                signal_id=trade.get("signal_id") or trade.get("decision_id"),
                atr=_safe_float(trade.get("atr") or trade.get("atr14") or trade.get("atr14_pct")),
                volatility=_safe_float(trade.get("volatility") or trade.get("max_drawdown_pct")),
            ):
                labeled_count += 1
        logger.info("[SIGNAL_LABEL] Batch labeling completed: %d trades", labeled_count)
        return labeled_count
    except BEST_EFFORT_EXCEPTIONS as e:
        logger.warning("Batch labeling error: %s", e)
        return 0


def get_labeling_stats() -> dict:
    stats = {
        "total_signals": 0,
        "labeled_signals": 0,
        "unlabeled_signals": 0,
        "win_count": 0,
        "loss_count": 0,
        "neutral_count": 0,
        "win_rate": 0.0,
        "class_counts": {
            "buyuk_kazanc": 0,
            "kucuk_kazanc": 0,
            "notr": 0,
            "kucuk_kayip": 0,
            "buyuk_kayip": 0,
        },
    }
    with _labeler_lock:
        if not SIGNAL_DATASET.exists():
            return stats
        try:
            with SIGNAL_DATASET.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    stats["total_signals"] += 1
                    label = str(row.get("label") or "").strip()
                    label_class = str(row.get("label_class") or "").strip()
                    if label in {"0", "1"}:
                        stats["labeled_signals"] += 1
                        if label == "1":
                            stats["win_count"] += 1
                        else:
                            stats["loss_count"] += 1
                    else:
                        stats["unlabeled_signals"] += 1
                    if label_class in stats["class_counts"]:
                        stats["class_counts"][label_class] += 1
                        if label_class == "notr":
                            stats["neutral_count"] += 1
            if stats["labeled_signals"] > 0:
                stats["win_rate"] = round(stats["win_count"] / stats["labeled_signals"], 4)
        except BEST_EFFORT_EXCEPTIONS as exc:
            logger.debug("Suppressed in get_labeling_stats: %s", exc)
    return stats


def load_training_targets(
    *,
    dataset_path: Optional[Path] = None,
    target_mode: str = "regression",
) -> list[dict[str, Any]]:
    """Load canonical signal targets for downstream training/evaluation code."""
    target_column = TARGET_MODE_TO_COLUMN.get(str(target_mode).lower())
    if target_column is None:
        raise ValueError(f"Unsupported target_mode: {target_mode}")
    path = dataset_path or SIGNAL_DATASET
    rows_out: list[dict[str, Any]] = []
    with _labeler_lock:
        if not path.exists():
            return rows_out
        with file_lock(path, timeout=10):
            with path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    target = row.get(target_column)
                    if target in (None, ""):
                        continue
                    payload: dict[str, Any] = {
                        "signal_id": row.get("signal_id"),
                        "timestamp": row.get("timestamp"),
                        "symbol": row.get("symbol"),
                        "target_mode": str(target_mode).lower(),
                        "target": target,
                    }
                    if target_column == "label":
                        payload["target"] = int(float(target))
                    elif target_column == "normalized_pnl":
                        payload["target"] = float(target)
                    else:
                        payload["target"] = str(target)
                    rows_out.append(payload)
    return rows_out


if __name__ == "__main__":
    stats = get_labeling_stats()
    print("=" * 60)
    print("SIGNAL LABELER TEST")
    print("=" * 60)
    print(f"  Total signals: {stats['total_signals']}")
    print(f"  Etiketli: {stats['labeled_signals']}")
    print(f"  Etiketsiz: {stats['unlabeled_signals']}")
    print(f"  Win rate: {stats['win_rate']:.1%}")
    print(f"  Class counts: {stats['class_counts']}")
