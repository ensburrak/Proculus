# -*- coding: utf-8 -*-
"""
calibrate.py
============

[O1 FIX] Unified Calibration Script
Combines functionality of:
- calibrate_confidence.py
- calibrate_master_platt.py
- calibrate_signal_scores.py

Tasks:
1. Platt Scaling Calibration (calibration.json):
   Maps raw master confidence to actual win probability using Logistic Regression (Platt scaling).
2. Weight Learning (logistic_weights.json):
   Learns weights for ai_score, tech_score, and sent_score.
3. Risk Schedule (risk_schedule.json):
   Computes risk-reward ratio for confidence bins to determine leverage levels.
"""

from __future__ import annotations


from core.exceptions import BEST_EFFORT_EXCEPTIONS
import argparse
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from atomic_io import atomic_write_json
from runtime_paths import get_artifact_path

log = logging.getLogger(__name__)
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from utils.math_helpers import clamp


LOGISTIC_WEIGHTS_SCHEMA = "meta-logistic-weights-v1"
LOGISTIC_WEIGHT_FEATURE_COLUMNS = ["ai_score", "tech_score", "sent_score"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_weight_status(path: Path, payload: dict[str, Any]) -> None:
    status = {"schema": "logistic-weights-status-v1", "created_at_utc": _utc_now(), **payload}
    atomic_write_json(path, status)


def _build_logistic_weights_payload(
    *,
    data_path: Path,
    trained_on: int,
    label_classes: list[int],
    w0: float,
    w_ai: float,
    w_tech: float,
    w_sent: float,
    cv_score: float | None = None,
    cv_folds: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": LOGISTIC_WEIGHTS_SCHEMA,
        "source": data_path.name,
        "created_at_utc": _utc_now(),
        "trained_on": int(trained_on),
        "label_classes": list(label_classes),
        "feature_columns": list(LOGISTIC_WEIGHT_FEATURE_COLUMNS),
        "w0": w0,
        "w_ai": w_ai,
        "w_tech": w_tech,
        "w_sent": w_sent,
    }
    if cv_score is not None:
        payload["cv_score"] = round(float(cv_score), 4)
    if cv_folds is not None:
        payload["cv_folds"] = int(cv_folds)
    return payload


def _sigmoid(z: float) -> float:
    z = max(-20.0, min(20.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def _logit(p: float) -> float:
    p = max(1e-6, min(1.0 - 1e-6, p))
    return math.log(p / (1.0 - p))


def _calculate_ece_raw(
    predicted_probs: List[float] | np.ndarray,
    labels: List[int],
    n_bins: int = 10,
) -> float:
    """Calculate Expected Calibration Error from raw predicted probabilities."""
    probs = np.array(predicted_probs, dtype=float)
    labs = np.array(labels, dtype=float)
    if len(probs) == 0:
        return 0.0
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])
        count = mask.sum()
        if count == 0:
            continue
        avg_conf = probs[mask].mean()
        avg_acc = labs[mask].mean()
        ece += (count / len(probs)) * abs(avg_conf - avg_acc)
    return float(ece)


def _calculate_ece(
    raw_confs: List[float],
    labels: List[int],
    a: float,
    b: float,
    n_bins: int = 10,
) -> float:
    """Calculate ECE after applying Platt scaling parameters."""
    calibrated = [_sigmoid(a * _logit(p) + b) for p in raw_confs]
    return _calculate_ece_raw(calibrated, labels, n_bins)


def _fit_platt(xs: List[float], ys: List[int]) -> Tuple[float, float]:
    """Fit a and b by Newton-Raphson on logistic loss."""
    if len(xs) < 50:
        return 1.0, 0.0

    a, b = 1.0, 0.0
    lam = 1e-2  # L2 damping

    for _ in range(40):
        ga, gb = 0.0, 0.0
        haa, hbb, hab = lam, lam, 0.0

        for x, y in zip(xs, ys):
            z = a * x + b
            p = _sigmoid(z)
            err = p - y
            ga += err * x
            gb += err
            w = p * (1.0 - p)
            haa += w * x * x
            hab += w * x
            hbb += w

        det = haa * hbb - hab * hab
        if abs(det) < 1e-9:
            break

        da = (hbb * ga - hab * gb) / det
        db = (-hab * ga + haa * gb) / det

        step = max(0.1, min(1.0, 1.0 / (1.0 + abs(da) + abs(db))))
        a -= step * da
        b -= step * db

        if abs(da) < 1e-6 and abs(db) < 1e-6:
            break

    a = max(0.25, min(4.0, float(a)))
    b = max(-2.0, min(2.0, float(b)))
    return a, b


def _build_reliability_report(
    raw_confs: np.ndarray,
    labels: np.ndarray,
    a: float,
    b: float,
    metrics_dir: Path,
    *,
    n_bins: int = 10,
) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    calibrated = np.array([_sigmoid(a * _logit(float(p)) + b) for p in raw_confs], dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, float | int]] = []
    for idx in range(n_bins):
        if idx == n_bins - 1:
            mask = (calibrated >= bin_edges[idx]) & (calibrated <= bin_edges[idx + 1])
        else:
            mask = (calibrated >= bin_edges[idx]) & (calibrated < bin_edges[idx + 1])
        count = int(mask.sum())
        if count == 0:
            bins.append(
                {
                    "bin_start": round(float(bin_edges[idx]), 4),
                    "bin_end": round(float(bin_edges[idx + 1]), 4),
                    "count": 0,
                    "predicted": 0.0,
                    "actual": 0.0,
                }
            )
            continue
        bins.append(
            {
                "bin_start": round(float(bin_edges[idx]), 4),
                "bin_end": round(float(bin_edges[idx + 1]), 4),
                "count": count,
                "predicted": round(float(calibrated[mask].mean()), 6),
                "actual": round(float(labels[mask].mean()), 6),
            }
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ece": round(_calculate_ece_raw(calibrated, labels.tolist(), n_bins=n_bins), 6),
        "bins": bins,
    }
    atomic_write_json(metrics_dir / "reliability_diagram.json", report)

    try:
        import matplotlib.pyplot as plt

        xs = [float(item["predicted"]) for item in bins if int(item["count"]) > 0]
        ys = [float(item["actual"]) for item in bins if int(item["count"]) > 0]
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        if xs and ys:
            plt.plot(xs, ys, marker="o", linewidth=2)
        plt.title("Reliability Diagram")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Actual Win Rate")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(metrics_dir / "reliability_diagram.png")
        plt.close()
    except BEST_EFFORT_EXCEPTIONS:
        pass


def calibrate_all(data_path: Path, output_dir: Path, *, reports_only: bool = False) -> None:
    if not data_path.exists():
        log.warning("Dataset not found: %s", data_path)
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    try:
        if data_path.suffix == ".jsonl":
            with data_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rows.append(json.loads(line))
                        except BEST_EFFORT_EXCEPTIONS:
                            continue
            df = pd.DataFrame(rows)
        else:
            df = pd.read_csv(data_path)
    except BEST_EFFORT_EXCEPTIONS as e:
        log.error("Error reading data: %s", e)
        return

    if df.empty:
        log.warning("No data.")
        return

    # Derive label
    if "label" not in df.columns:
        if "win" in df.columns:
            df["label"] = df["win"]
        elif "realized_pnl" in df.columns:
            df["label"] = df["realized_pnl"].apply(lambda x: 1 if float(x) > 0 else 0)
        else:
            log.warning("No label column found.")
            return

    df["label"] = df["label"].astype(int)

    # Prepare paths
    cal_path = output_dir / "calibration.json"
    weights_path = output_dir / "logistic_weights.json"
    weights_status_path = output_dir / "logistic_weights_status.json"
    schedule_path = output_dir / "risk_schedule.json"
    target_summary_path = output_dir / "label_target_summary.json"
    metrics_dir = output_dir / "metrics"

    # 0. Rich target summary - keep binary compatibility but treat richer labels as canonical training targets.
    target_summary: Dict[str, Any] = {"source": data_path.name}
    if data_path.suffix.lower() == ".csv":
        try:
            from signal_labeler import load_training_targets

            regression_targets = load_training_targets(dataset_path=data_path, target_mode="regression")
            multiclass_targets = load_training_targets(dataset_path=data_path, target_mode="multiclass")
            binary_targets = load_training_targets(dataset_path=data_path, target_mode="binary")
            target_summary["binary_rows"] = len(binary_targets)
            target_summary["multiclass_rows"] = len(multiclass_targets)
            target_summary["regression_rows"] = len(regression_targets)
        except BEST_EFFORT_EXCEPTIONS:
            regression_targets = []
            multiclass_targets = []
            binary_targets = []

    if "label_class" in df.columns:
        class_counts = (
            df["label_class"]
            .fillna("")
            .astype(str)
            .value_counts()
            .to_dict()
        )
        target_summary["class_counts"] = class_counts
    if "normalized_pnl" in df.columns:
        reg_series = pd.to_numeric(df["normalized_pnl"], errors="coerce").dropna()
        if not reg_series.empty:
            target_summary["normalized_pnl"] = {
                "mean": round(float(reg_series.mean()), 6),
                "median": round(float(reg_series.median()), 6),
                "std": round(float(reg_series.std(ddof=0)), 6),
                "rows": int(reg_series.shape[0]),
            }
    if target_summary:
        atomic_write_json(target_summary_path, target_summary)

    # 1. Platt Scaling Calibration — [FAZA 4.4] Temporal split + ECE
    log.info("--- 1. Platt Scaling Calibration (temporal split) ---")
    conf_candidates = [
        "master_raw",
        "master_conf_before",
        "master_conf_raw",
        "master_confidence",
    ]
    conf_col = next((name for name in conf_candidates if name in df.columns), None)
    if conf_col in df.columns:
        sub_df = df.dropna(subset=[conf_col, "label"]).copy()
        if len(sub_df) >= 50:
            # [FAZA 4.4] Temporal split: son %20 sadece test — overfit önleme
            split_idx = int(len(sub_df) * 0.80)
            train_df = sub_df.iloc[:split_idx]
            test_df = sub_df.iloc[split_idx:]

            xs_train = [_logit(float(p)) for p in train_df[conf_col]]
            ys_train = list(train_df["label"])
            a, b = _fit_platt(xs_train, ys_train)

            # ECE (Expected Calibration Error) hesaplama
            ece = _calculate_ece(
                [float(p) for p in test_df[conf_col]],
                list(test_df["label"]),
                a, b, n_bins=10,
            )
            overconfidence = max(0.0, ece - 0.05)
            penalty_multiplier = round(max(0.65, 1.0 - overconfidence * 2.0), 4)

            # Isotonic regression ile karşılaştırma
            iso_ece = None
            try:
                from sklearn.isotonic import IsotonicRegression
                iso_model = IsotonicRegression(out_of_bounds="clip")
                iso_model.fit(
                    train_df[conf_col].astype(float).values,
                    train_df["label"].values,
                )
                iso_preds = iso_model.predict(test_df[conf_col].astype(float).values)
                iso_ece = _calculate_ece_raw(iso_preds, list(test_df["label"]), n_bins=10)
                log.info("Isotonic ECE=%.4f vs Platt ECE=%.4f", iso_ece, ece)
            except ImportError:
                log.debug("sklearn.isotonic not available, skipping isotonic comparison")

            # Overconfidence penalty: ECE > 0.05 ise uyar
            cal_data = {
                "type": "logistic",
                "a": a,
                "b": b,
                "trained_on": len(xs_train),
                "test_size": len(test_df),
                "ece": round(ece, 4),
                "iso_ece": round(iso_ece, 4) if iso_ece is not None else None,
                "overconfident": ece > 0.05,
                "penalty_multiplier": penalty_multiplier,
                "source": data_path.name,
            }
            atomic_write_json(cal_path, cal_data)
            _build_reliability_report(
                test_df[conf_col].astype(float).to_numpy(),
                test_df["label"].astype(int).to_numpy(),
                a,
                b,
                metrics_dir,
            )
            log.info("calibration.json updated: a=%.4f, b=%.4f ECE=%.4f (train=%d, test=%d)",
                     a, b, ece, len(train_df), len(test_df))
            if ece > 0.05:
                log.warning("ECE=%.4f > 0.05 — model may be overconfident!", ece)
        else:
            if cal_path.exists():
                log.warning("Not enough data for Platt scaling (<50 rows). Preserving existing calibration.")
            else:
                log.warning("Not enough data for Platt scaling (<50 rows). Writing default calibration.")
                atomic_write_json(
                    cal_path,
                    {
                        "type": "logistic",
                        "a": 1.0,
                        "b": 0.0,
                        "ece": None,
                        "iso_ece": None,
                        "overconfident": False,
                        "penalty_multiplier": 1.0,
                        "source": data_path.name,
                        "trained_on": len(sub_df),
                        "test_size": 0,
                    },
                )
    else:
        log.warning("Missing confidence column for Platt scaling.")

    if reports_only:
        return

    # 2. Weight Learning — [FAZA 4.3] Temporal Cross-Validation ile ağırlık öğrenme
    log.info("--- 2. Weight Learning (5-fold temporal CV) ---")
    feat_cols = LOGISTIC_WEIGHT_FEATURE_COLUMNS
    if all(c in df.columns for c in feat_cols):
        sub_df = df.dropna(subset=feat_cols + ["label"]).copy()
        label_classes = sorted({int(v) for v in sub_df["label"].dropna().unique().tolist()})
        if len(label_classes) < 2:
            if weights_path.exists():
                log.warning(
                    "Only one label class for weight learning (%s). Preserving existing weights.",
                    label_classes,
                )
                status = "skipped"
                reason = "insufficient_class_balance_preserved_existing"
            else:
                log.warning(
                    "Only one label class for weight learning (%s). Refusing to write fake weights.",
                    label_classes,
                )
                status = "blocked"
                reason = "insufficient_class_balance"
            _write_weight_status(
                weights_status_path,
                {
                    "status": status,
                    "reason": reason,
                    "source": data_path.name,
                    "trained_on": len(sub_df),
                    "label_classes": label_classes,
                    "feature_columns": list(feat_cols),
                },
            )
        elif len(sub_df) >= 50:
            X = sub_df[feat_cols].astype(float).values
            y = sub_df["label"].values

            # 5-fold temporal (TimeSeriesSplit tarzı) cross-validation
            n_folds = 5
            fold_size = len(sub_df) // (n_folds + 1)
            cv_weights = []
            cv_scores = []

            for fold in range(n_folds):
                train_end = fold_size * (fold + 2)
                test_start = train_end
                test_end = min(test_start + fold_size, len(sub_df))
                if test_end <= test_start:
                    continue

                X_train, y_train = X[:train_end], y[:train_end]
                X_test, y_test = X[test_start:test_end], y[test_start:test_end]
                if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                    continue

                fold_model = LogisticRegression(class_weight="balanced", max_iter=1000)
                fold_model.fit(X_train, y_train)

                fold_score = float(fold_model.score(X_test, y_test))
                cv_scores.append(fold_score)
                cv_weights.append(fold_model.coef_[0].tolist())

            # Son model: tüm veri üzerinde fit
            final_model = LogisticRegression(class_weight="balanced", max_iter=1000)
            final_model.fit(X, y)

            w0 = clamp(float(final_model.intercept_[0]), -2.0, 2.0)
            w_ai = abs(clamp(float(final_model.coef_[0][0]), -1.0, 1.0))
            w_tech = abs(clamp(float(final_model.coef_[0][1]), -1.0, 1.0))
            w_sent = clamp(float(final_model.coef_[0][2]), -1.0, 1.0)

            avg_cv_score = float(np.mean(cv_scores)) if cv_scores else 0.0

            weights = _build_logistic_weights_payload(
                data_path=data_path,
                trained_on=len(sub_df),
                label_classes=label_classes,
                w0=w0,
                w_ai=w_ai,
                w_tech=w_tech,
                w_sent=w_sent,
                cv_score=avg_cv_score,
                cv_folds=len(cv_scores),
            )
            atomic_write_json(weights_path, weights)
            _write_weight_status(
                weights_status_path,
                {
                    "status": "trained",
                    "reason": "ready",
                    "source": data_path.name,
                    "trained_on": len(sub_df),
                    "label_classes": label_classes,
                    "feature_columns": list(feat_cols),
                    "cv_score": round(avg_cv_score, 4),
                    "cv_folds": len(cv_scores),
                },
            )
            log.info(
                "logistic_weights.json updated: w_ai=%.4f, w_tech=%.4f, w_sent=%.4f "
                "(CV=%.4f, folds=%d, n=%d)",
                w_ai, w_tech, w_sent, avg_cv_score, len(cv_scores), len(sub_df),
            )
        elif len(sub_df) >= 20:
            # Yeterli veri yok CV için, basit fit
            X = sub_df[feat_cols].astype(float).values
            y = sub_df["label"].values
            model = LogisticRegression(class_weight="balanced", max_iter=1000)
            model.fit(X, y)
            w0 = clamp(float(model.intercept_[0]), -2.0, 2.0)
            w_ai = abs(clamp(float(model.coef_[0][0]), -1.0, 1.0))
            w_tech = abs(clamp(float(model.coef_[0][1]), -1.0, 1.0))
            w_sent = clamp(float(model.coef_[0][2]), -1.0, 1.0)
            weights = _build_logistic_weights_payload(
                data_path=data_path,
                trained_on=len(sub_df),
                label_classes=label_classes,
                w0=w0,
                w_ai=w_ai,
                w_tech=w_tech,
                w_sent=w_sent,
                cv_score=None,
                cv_folds=0,
            )
            atomic_write_json(weights_path, weights)
            _write_weight_status(
                weights_status_path,
                {
                    "status": "trained",
                    "reason": "ready_no_cv",
                    "source": data_path.name,
                    "trained_on": len(sub_df),
                    "label_classes": label_classes,
                    "feature_columns": list(feat_cols),
                    "cv_folds": 0,
                },
            )
            log.info("logistic_weights.json updated (no CV, n=%d)", len(sub_df))
        else:
            if weights_path.exists():
                log.warning("Not enough data for weight learning (<20 rows). Preserving existing weights.")
                status = "skipped"
                reason = "insufficient_samples_preserved_existing"
            else:
                log.warning("Not enough data for weight learning (<20 rows). Refusing to write fake weights.")
                status = "blocked"
                reason = "insufficient_samples"
            _write_weight_status(
                weights_status_path,
                {
                    "status": status,
                    "reason": reason,
                    "source": data_path.name,
                    "trained_on": len(sub_df),
                    "label_classes": label_classes,
                    "feature_columns": list(feat_cols),
                },
            )
    else:
        log.warning("Missing feature columns for weight learning.")
        _write_weight_status(
            weights_status_path,
            {
                "status": "blocked",
                "reason": "missing_feature_columns",
                "source": data_path.name,
                "missing_feature_columns": [col for col in feat_cols if col not in df.columns],
                "feature_columns": list(feat_cols),
            },
        )

    # 3. Risk Schedule
    log.info("--- 3. Risk Schedule ---")
    req_cols = [conf_col, "r_multiple", "max_drawdown_pct"]
    monotonic_bins = [
        {"min": 0.55, "max": 0.60, "leverage": 1, "wallet_allocation_percent": 0.03},
        {"min": 0.60, "max": 0.65, "leverage": 1, "wallet_allocation_percent": 0.05},
        {"min": 0.65, "max": 0.70, "leverage": 1, "wallet_allocation_percent": 0.07},
        {"min": 0.70, "max": 0.75, "leverage": 1, "wallet_allocation_percent": 0.09},
        {"min": 0.75, "max": 0.80, "leverage": 1, "wallet_allocation_percent": 0.11},
        {"min": 0.80, "max": 0.85, "leverage": 1, "wallet_allocation_percent": 0.13},
        {"min": 0.85, "max": 0.90, "leverage": 1, "wallet_allocation_percent": 0.15},
        {"min": 0.90, "max": 0.95, "leverage": 1, "wallet_allocation_percent": 0.17},
        {"min": 0.95, "max": 1.01, "leverage": 1, "wallet_allocation_percent": 0.17},
    ]
    schedule_payload: dict[str, Any] = {
        "bins": monotonic_bins,
        "source": "monotonic_default_v2",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_schedule = True
    if all(c in df.columns for c in req_cols):
        sub_df = df.dropna(subset=req_cols).copy()
        if len(sub_df) > 0:
            support = []
            for entry in monotonic_bins:
                lo = float(entry["min"])
                hi = float(entry["max"])
                mask = (sub_df[conf_col].astype(float) >= lo) & (sub_df[conf_col].astype(float) < hi)
                support.append(
                    {
                        "min": lo,
                        "max": hi,
                        "samples": int(mask.sum()),
                    }
                )
            schedule_payload["bin_support"] = support
        else:
            if schedule_path.exists():
                write_schedule = False
                log.warning("No data for risk schedule support metrics. Preserving existing risk schedule.")
            else:
                log.warning("No data for risk schedule support metrics. Writing monotonic default schedule.")
    else:
        if schedule_path.exists():
            write_schedule = False
            log.warning("Missing columns for risk schedule support metrics. Preserving existing risk schedule.")
        else:
            log.warning("Missing columns for risk schedule support metrics. Writing monotonic default schedule.")

    if write_schedule:
        atomic_write_json(schedule_path, schedule_payload)
        log.info("risk_schedule.json updated.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    base_dir = Path(__file__).resolve().parent
    default_jsonl = base_dir / "metrics" / "calibration_trades.jsonl"
    default_csv = base_dir / "data" / "signal_dataset.csv"

    parser = argparse.ArgumentParser(description="Unified calibration runner")
    parser.add_argument("--data-path", type=str, default=None)
    default_output_dir = get_artifact_path("calibration.json").parent
    parser.add_argument("--output-dir", type=str, default=str(default_output_dir))
    parser.add_argument("--reports-only", action="store_true")
    args = parser.parse_args()

    if args.data_path:
        data_path = Path(args.data_path)
    else:
        data_path = default_jsonl if default_jsonl.exists() else default_csv

    calibrate_all(
        Path(data_path),
        Path(args.output_dir),
        reports_only=bool(args.reports_only),
    )
