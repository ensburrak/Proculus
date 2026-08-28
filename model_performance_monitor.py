"""
model_performance_monitor.py
----------------------------

Runtime monitoring for AI model quality and attribution.

The module reads two sources:
  * ``trade_log.json`` / SQLite-backed trade log via ``runtime_paths``
  * ``metrics/ai_predictions.json`` containing prediction events

Prediction events are resolved at trade close, so the report can compare
predicted direction vs actual outcome and compute per-model attributed PnL.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from atomic_io import atomic_write_json
from runtime_paths import METRICS_DIR, get_trade_log_path
from utils.math_helpers import clamp

logger = logging.getLogger(__name__)
LIVE_DRIFT_HISTORY_PATH = METRICS_DIR / "live_drift_weekly.json"
LIVE_DRIFT_LATEST_PATH = METRICS_DIR / "live_drift_latest.json"
JSON_IO_EXCEPTIONS = (OSError, json.JSONDecodeError, TypeError, ValueError)


def _load_json(path: Path) -> Any:
    """Load JSON if the file exists; return None otherwise."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except JSON_IO_EXCEPTIONS as exc:
        logger.debug("Failed to load JSON from %s: %s", path, exc)
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _sign_bucket(value: float | None) -> int:
    if value is None:
        return 0
    if value > 1e-12:
        return 1
    if value < -1e-12:
        return -1
    return 0


def _prediction_actual_value(record: Dict[str, Any]) -> float | None:
    actual = _safe_float(record.get("actual"))
    if actual is not None:
        return actual
    pnl_abs = _safe_float(record.get("pnl_abs"))
    if pnl_abs is not None:
        return float(_sign_bucket(pnl_abs))
    pnl_frac = _safe_float(record.get("pnl"))
    if pnl_frac is not None:
        return float(_sign_bucket(pnl_frac))
    outcome = str(record.get("outcome", "")).strip().lower()
    if outcome == "win":
        return 1.0
    if outcome == "loss":
        return -1.0
    if outcome == "flat":
        return 0.0
    return None


def _is_prediction_resolved(record: Dict[str, Any]) -> bool:
    return _prediction_actual_value(record) is not None


def _compute_sign_accuracy(predictions: List[Dict[str, Any]]) -> Tuple[int, int, float]:
    """Compute sign-based accuracy for resolved predictions."""
    correct = 0
    total = 0
    for record in predictions:
        pred = _safe_float(record.get("prediction"))
        actual = _prediction_actual_value(record)
        if pred is None or actual is None:
            continue
        if _sign_bucket(pred) == _sign_bucket(actual):
            correct += 1
        total += 1
    acc = (correct / total) if total > 0 else 0.0
    return correct, total, acc


def _summarise_trade_log(trade_log: Iterable[Dict[str, Any]]) -> tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    total_pnl = 0.0
    total_trades = 0
    model_totals: Dict[str, Dict[str, float]] = {}
    for trade in trade_log:
        pnl_abs = _safe_float(trade.get("pnl_abs"))
        if pnl_abs is None:
            pnl_abs = _safe_float(trade.get("pnl_usd")) or 0.0
        total_pnl += pnl_abs
        total_trades += 1
        model_name = str(trade.get("model") or trade.get("source") or "unknown").lower()
        stats = model_totals.setdefault(model_name, {"num_trades": 0.0, "total_pnl": 0.0})
        stats["num_trades"] += 1.0
        stats["total_pnl"] += pnl_abs

    overall = {
        "num_trades": total_trades,
        "total_pnl": total_pnl,
        "avg_pnl_per_trade": (total_pnl / total_trades) if total_trades else None,
    }
    return overall, model_totals


def _aggregate_rl_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    rl_entries = []
    for model_name, metrics in report.items():
        if not isinstance(metrics, dict):
            continue
        if model_name == "rl" or model_name.startswith("rl") or "ppo" in model_name:
            rl_entries.append(metrics)
    if not rl_entries:
        return {}

    total_predictions = 0
    resolved_predictions = 0
    num_correct = 0
    total_pnl = 0.0
    win_count = 0
    for metrics in rl_entries:
        total_predictions += int(metrics.get("num_predictions", 0) or 0)
        resolved_predictions += int(metrics.get("resolved_predictions", 0) or 0)
        num_correct += int(metrics.get("num_correct", 0) or 0)
        total_pnl += float(metrics.get("total_pnl", 0.0) or 0.0)
        win_count += int(metrics.get("win_count", 0) or 0)

    sign_accuracy = (num_correct / resolved_predictions) if resolved_predictions else 0.0
    win_rate = (win_count / resolved_predictions) if resolved_predictions else None
    avg_pnl = (total_pnl / resolved_predictions) if resolved_predictions else None
    return {
        "num_predictions": total_predictions,
        "resolved_predictions": resolved_predictions,
        "pending_predictions": max(total_predictions - resolved_predictions, 0),
        "num_correct": num_correct,
        "sign_accuracy": sign_accuracy,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "win_rate": win_rate,
        "win_count": win_count,
    }


def _summarise_predictions(
    predictions: Iterable[Dict[str, Any]],
    overall_total_pnl: float,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for rec in predictions:
        model_name = str(rec.get("model", "unknown")).lower()
        by_model.setdefault(model_name, []).append(rec)

    summary: Dict[str, Dict[str, Any]] = {}
    monitoring = {
        "num_predictions": 0,
        "resolved_predictions": 0,
        "pending_predictions": 0,
        "coverage_ratio": 0.0,
    }

    for model_name, rows in by_model.items():
        monitoring["num_predictions"] += len(rows)
        resolved = [row for row in rows if _is_prediction_resolved(row)]
        pending = max(len(rows) - len(resolved), 0)
        correct, total, acc = _compute_sign_accuracy(resolved)
        monitoring["resolved_predictions"] += len(resolved)
        monitoring["pending_predictions"] += pending

        raw_pnl_values: list[float | None] = [_safe_float(row.get("pnl_abs")) for row in resolved]
        pnl_values: list[float] = [v for v in raw_pnl_values if v is not None]
        raw_return_values: list[float | None] = [_safe_float(row.get("actual_return")) for row in resolved]
        return_values: list[float] = [v for v in raw_return_values if v is not None]
        raw_confidences: list[float | None] = [_safe_float(row.get("confidence")) for row in rows]
        confidences: list[float] = [v for v in raw_confidences if v is not None]
        win_count = sum(1 for row in resolved if str(row.get("outcome", "")).lower() == "win")

        model_summary: Dict[str, Any] = {
            "num_predictions": len(rows),
            "resolved_predictions": len(resolved),
            "pending_predictions": pending,
            "num_correct": correct,
            "sign_accuracy": acc,
            "win_count": win_count,
            "win_rate": (win_count / len(resolved)) if resolved else None,
            "avg_confidence": (sum(confidences) / len(confidences)) if confidences else None,
        }
        if pnl_values:
            total_pnl = sum(pnl_values)
            model_summary["total_pnl"] = total_pnl
            model_summary["avg_pnl"] = total_pnl / len(pnl_values)
            if abs(overall_total_pnl) > 1e-12:
                model_summary["pnl_attribution_share"] = total_pnl / overall_total_pnl
        if return_values:
            model_summary["avg_actual_return"] = sum(return_values) / len(return_values)
        summary[model_name] = model_summary

    if monitoring["num_predictions"] > 0:
        monitoring["coverage_ratio"] = monitoring["resolved_predictions"] / monitoring["num_predictions"]
    return summary, monitoring


def evaluate_models(
    trade_log_path: Path = get_trade_log_path(),
    ai_pred_path: Path = METRICS_DIR / "ai_predictions.json",
) -> Dict[str, Any]:
    """Compute a performance report for each AI component."""
    report: Dict[str, Any] = {
        "overall": {},
        "monitoring": {},
        "rl": {},
        "chatgpt": {},
        "deepseek": {},
        "transformer": {},
        "hybrid": {},
    }

    trade_log = _load_json(trade_log_path)
    overall_total_pnl = 0.0
    if isinstance(trade_log, list):
        overall, trade_by_model = _summarise_trade_log(trade_log)
        report["overall"] = overall
        overall_total_pnl = float(overall.get("total_pnl") or 0.0)
        for model_name, stats in trade_by_model.items():
            num_trades = int(stats.get("num_trades", 0.0))
            total_pnl = float(stats.get("total_pnl", 0.0))
            report.setdefault(model_name, {}).update(
                {
                    "num_trades": num_trades,
                    "trade_total_pnl": total_pnl,
                    "trade_avg_pnl": (total_pnl / num_trades) if num_trades else None,
                }
            )
    else:
        report["overall"] = None

    ai_preds = _load_json(ai_pred_path)
    if isinstance(ai_preds, list):
        prediction_summary, monitoring = _summarise_predictions(ai_preds, overall_total_pnl)
        report["monitoring"] = monitoring
        for model_name, stats in prediction_summary.items():
            report.setdefault(model_name, {}).update(stats)

    rl_metrics = _aggregate_rl_metrics(report)
    if rl_metrics:
        report["rl"] = rl_metrics

    return report


def build_live_drift_snapshot(
    report: Dict[str, Any] | None = None,
    *,
    generated_at: str | None = None,
) -> Dict[str, Any]:
    """Build a compact weekly live-drift snapshot from the latest monitoring report."""
    report = report or evaluate_models()
    overall = report.get("overall") or {}
    monitoring = report.get("monitoring") or {}

    snapshot_models: Dict[str, Dict[str, Any]] = {}
    for model_name in ("hybrid", "chatgpt", "deepseek", "transformer", "rl"):
        metrics = report.get(model_name) or {}
        if not isinstance(metrics, dict) or not metrics:
            continue
        snapshot_models[model_name] = {
            "num_predictions": int(metrics.get("num_predictions", 0) or 0),
            "resolved_predictions": int(metrics.get("resolved_predictions", 0) or 0),
            "sign_accuracy": float(metrics.get("sign_accuracy", 0.0) or 0.0),
            "win_rate": float(metrics.get("win_rate", 0.0) or 0.0),
            "avg_confidence": float(metrics.get("avg_confidence", 0.0) or 0.0),
            "total_pnl": float(metrics.get("total_pnl", 0.0) or 0.0),
            "pnl_attribution_share": float(metrics.get("pnl_attribution_share", 0.0) or 0.0),
        }

    return {
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "window": "weekly",
        "overall": {
            "num_trades": int(overall.get("num_trades", 0) or 0),
            "trade_total_pnl": float(overall.get("total_pnl", 0.0) or 0.0),
            "trade_avg_pnl": float(overall.get("avg_pnl_per_trade", 0.0) or 0.0),
            "num_predictions": int(monitoring.get("num_predictions", 0) or 0),
            "resolved_predictions": int(monitoring.get("resolved_predictions", 0) or 0),
            "pending_predictions": int(monitoring.get("pending_predictions", 0) or 0),
            "coverage_ratio": float(monitoring.get("coverage_ratio", 0.0) or 0.0),
        },
        "models": snapshot_models,
    }


def _snapshot_delta(current: Dict[str, Any], previous: Dict[str, Any] | None) -> Dict[str, Any]:
    if not previous:
        return {}

    delta: Dict[str, Any] = {}
    for key, value in current.items():
        prev_value = previous.get(key)
        if isinstance(value, dict) and isinstance(prev_value, dict):
            nested = _snapshot_delta(value, prev_value)
            if nested:
                delta[key] = nested
        elif isinstance(value, (int, float)) and isinstance(prev_value, (int, float)):
            delta[key] = round(float(value) - float(prev_value), 10)
    return delta


def persist_live_drift_snapshot(
    report: Dict[str, Any] | None = None,
    *,
    history_path: Path = LIVE_DRIFT_HISTORY_PATH,
    latest_path: Path = LIVE_DRIFT_LATEST_PATH,
    keep: int = 52,
) -> Dict[str, Any]:
    """Persist the weekly live-drift snapshot and return the stored payload."""
    snapshot = build_live_drift_snapshot(report)
    history_raw = _load_json(history_path)
    history: List[Dict[str, Any]] = history_raw if isinstance(history_raw, list) else []
    previous = history[-1] if history else None
    snapshot["delta"] = _snapshot_delta(snapshot, previous)

    history_path.parent.mkdir(parents=True, exist_ok=True)
    history.append(snapshot)
    if keep > 0:
        history = history[-keep:]

    atomic_write_json(history_path, history)
    atomic_write_json(latest_path, snapshot)
    return snapshot


def persist_live_monitor_snapshot(
    report: Dict[str, Any] | None = None,
    *,
    latest_path: Path = LIVE_DRIFT_LATEST_PATH,
) -> Dict[str, Any]:
    """Persist the latest runtime monitoring snapshot without appending history."""
    snapshot = build_live_drift_snapshot(report)
    snapshot["window"] = "current"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(latest_path, snapshot)
    return snapshot


def adjust_weights(report: Dict[str, Any], config_path: Path = Path("config.json")) -> bool:
    """Adjust model weights in the configuration based on performance."""
    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except JSON_IO_EXCEPTIONS as exc:
        logger.warning("Could not read config: %s", exc)
        return False

    weights = config.setdefault("weights", {})
    modified = False

    rl_metrics = report.get("rl")
    if isinstance(rl_metrics, dict):
        avg_pnl = rl_metrics.get("avg_pnl")
        if isinstance(avg_pnl, (float, int)) and avg_pnl < 0:
            old = float(weights.get("rl", 1.0))
            new = clamp(old * 0.9, 0.0, max(old, 1.0))
            if new != old:
                weights["rl"] = new
                logger.info("Reduced RL weight from %.3f to %.3f due to negative average PnL", old, new)
                modified = True

    tf_metrics = report.get("transformer")
    if isinstance(tf_metrics, dict):
        acc = tf_metrics.get("sign_accuracy")
        if isinstance(acc, (float, int)) and acc < 0.5:
            old = float(weights.get("transformer", 1.0))
            new = clamp(old * 0.85, 0.0, max(old, 1.0))
            if new != old:
                weights["transformer"] = new
                logger.info("Reduced Transformer weight from %.3f to %.3f due to low accuracy", old, new)
                modified = True

    for model_name in ("chatgpt", "deepseek"):
        model_metrics = report.get(model_name)
        if not isinstance(model_metrics, dict):
            continue
        acc = model_metrics.get("sign_accuracy")
        if isinstance(acc, (float, int)) and acc < 0.5:
            old = float(weights.get(model_name, 1.0))
            new = clamp(old * 0.9, 0.0, max(old, 1.0))
            if new != old:
                weights[model_name] = new
                logger.info("Reduced %s weight from %.3f to %.3f due to low accuracy", model_name, old, new)
                modified = True

    if modified:
        try:
            atomic_write_json(config_path, config)
            return True
        except JSON_IO_EXCEPTIONS as exc:
            logger.warning("Failed to write updated config: %s", exc)
            return False
    return False


def deactivate_underperforming_models(
    report: Dict[str, Any],
    config_path: Path = Path("config.json"),
    pnl_threshold: float = -0.01,
    accuracy_threshold: float = 0.3,
) -> bool:
    """Disable models that persistently underperform by setting their weights to zero."""
    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except JSON_IO_EXCEPTIONS as exc:
        logger.warning("Could not read config for deactivation: %s", exc)
        return False

    weights = config.setdefault("weights", {})
    disabled = config.setdefault("disabled_models", [])
    modified = False

    rl_metrics = report.get("rl")
    if isinstance(rl_metrics, dict):
        avg_pnl = rl_metrics.get("avg_pnl")
        if isinstance(avg_pnl, (float, int)) and avg_pnl < pnl_threshold:
            if float(weights.get("rl", 0.0)) > 0.0:
                weights["rl"] = 0.0
                if "rl" not in disabled:
                    disabled.append("rl")
                logger.info(
                    "Deactivating RL model due to low average PnL %.4f (threshold %.4f)",
                    avg_pnl,
                    pnl_threshold,
                )
                modified = True

    for model_name in ("chatgpt", "deepseek"):
        model_metrics = report.get(model_name)
        if not isinstance(model_metrics, dict):
            continue
        acc = model_metrics.get("sign_accuracy")
        if isinstance(acc, (float, int)) and acc < accuracy_threshold:
            if float(weights.get(model_name, 0.0)) > 0.0:
                weights[model_name] = 0.0
                if model_name not in disabled:
                    disabled.append(model_name)
                logger.info(
                    "Deactivating %s model due to low accuracy %.4f (threshold %.4f)",
                    model_name,
                    acc,
                    accuracy_threshold,
                )
                modified = True

    if modified:
        try:
            atomic_write_json(config_path, config)
            return True
        except JSON_IO_EXCEPTIONS as exc:
            logger.warning("Failed to write updated config during deactivation: %s", exc)
            return False
    return False


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    rep = evaluate_models()
    logger.info("Model performance report:\n%s", json.dumps(rep, indent=2))
    adjust_weights(rep)


auto_adjust_weights = adjust_weights
get_performance_summary = evaluate_models
