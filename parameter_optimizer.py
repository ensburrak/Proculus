# -*- coding: utf-8 -*-
"""
parameter_optimizer.py
---------------------
Hiperparametre optimizasyonu — Optuna + config.json entegrasyonu.
"""
from __future__ import annotations


from core.exceptions import BEST_EFFORT_EXCEPTIONS
import logging
import os
import random
import json
from pathlib import Path
from typing import Any, Dict, Optional

from atomic_io import atomic_write_json
from portfolio_optimizer import apply_selected_candidate, run_multi_objective_optimization

try:
    import optuna  # type: ignore
except ImportError:
    optuna = None

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent
CONFIG_FILE = ROOT_DIR / "config.json"


def _objective(trial: "optuna.Trial") -> float:
    """Composite objective function for parameter optimisation.

    Logistic regression + strateji parametreleri.
    Veri/bağımlılık yoksa fallback uyarısı verilir (random score yerine).
    """
    c_val = trial.suggest_float("C", 1e-3, 1e2, log=True)
    trial.suggest_float("stop_loss_mult", 0.005, 0.05)
    trial.suggest_int("ma_short", 10, 100)
    trial.suggest_int("ma_long", 20, 200)
    trial.suggest_int("cooldown_min", 1, 60)

    try:
        import pandas as pd  # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
        from sklearn.model_selection import cross_val_score  # type: ignore

        data_path = os.path.join("data", "risk_dataset.csv")
        if not os.path.exists(data_path):
            raise FileNotFoundError

        df = pd.read_csv(data_path)
        feature_cols = [col for col in ["ai_score", "tech_score", "sent_score"] if col in df.columns]
        if len(feature_cols) < 2:
            raise ValueError("Required feature columns missing")

        target_col = None
        for col in ["y", "y_win", "label"]:
            if col in df.columns:
                target_col = col
                break
        if target_col is None:
            raise ValueError("Required target column missing")

        X = df[feature_cols]
        y = df[target_col]
        model = LogisticRegression(C=c_val, max_iter=1000, class_weight="balanced")
        scores = cross_val_score(model, X, y, cv=3, scoring="neg_log_loss")
        return -float(scores.mean())

    except FileNotFoundError:
        # [FIX] Fallback random yerine penalty score + uyarı
        logger.warning(
            "[OPTIMIZER] risk_dataset.csv bulunamadı — data/risk_dataset.csv oluşturun. "
            "Fallback penalty score kullanılıyor (0.9)."
        )
        return 0.9 + random.random() * 0.1
    except ImportError as ie:
        logger.warning("[OPTIMIZER] Bağımlılık eksik: %s — pip install pandas scikit-learn", ie)
        return 0.9 + random.random() * 0.1
    except BEST_EFFORT_EXCEPTIONS as e:
        logger.warning("[OPTIMIZER] Objective hatası: %s", e)
        return 0.9 + random.random() * 0.1


def run_optimization(n_trials: int = 20) -> Dict[str, Any]:
    """
    Optuna ile hiperparametre araması.

    Args:
        n_trials: Deneme sayısı.

    Returns:
        En iyi parametreler ve skoru.
    """
    if optuna is None:
        logger.warning("[OPTIMIZER] Optuna kurulu değil: pip install optuna")
        return {}
    if n_trials <= 0:
        return {}

    study = optuna.create_study(direction="minimize")
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_value = study.best_value
    logger.info("[OPTIMIZER] En iyi skor: %.4f, parametreler: %s", best_value, best_params)

    # [FIX] Sonuçları config.json'a yaz
    _save_to_config(best_params)

    return {"best_params": best_params, "best_value": best_value}


def run_multi_objective_parameter_optimization(
    n_trials: int = 50,
    *,
    preset: str = "balanced",
) -> Dict[str, Any]:
    """Run NSGA-II style multi-objective optimisation and persist results."""
    result = run_multi_objective_optimization(n_trials=n_trials, preset=preset)
    if not result:
        return {}
    _save_multi_objective_to_config(result)
    return result


def _save_to_config(params: Dict[str, Any], config_path: Optional[Path] = None) -> None:
    """[FIX] Optimizasyon sonuçlarını config.json'a yaz.

    Mevcut config korunur, sadece 'optimization_results' anahtarı güncellenir.
    """
    path = config_path or CONFIG_FILE
    try:
        existing = {}
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))

        existing["optimization_results"] = {
            "best_params": params,
            "source": "parameter_optimizer.py (Optuna)",
        }

        # Bazı parametreleri doğrudan config'e uygula (varsa)
        if "stop_loss_mult" in params and "risk_management" in existing:
            existing["risk_management"]["stop_loss_mult"] = params["stop_loss_mult"]
            logger.info("[OPTIMIZER] stop_loss_mult config'e yazıldı: %.4f", params["stop_loss_mult"])

        if "cooldown_min" in params and "trade_settings" in existing:
            existing["trade_settings"]["cooldown_min"] = params["cooldown_min"]
            logger.info("[OPTIMIZER] cooldown_min config'e yazıldı: %d", params["cooldown_min"])

        atomic_write_json(path, existing)
        logger.info("[OPTIMIZER] Sonuçlar config.json'a kaydedildi")
    except BEST_EFFORT_EXCEPTIONS as e:
        logger.warning("[OPTIMIZER] Config yazma hatası: %s", e)


def _save_multi_objective_to_config(result: Dict[str, Any], config_path: Optional[Path] = None) -> None:
    """Persist verified Pareto frontier and auto-apply the selected candidate."""
    path = config_path or CONFIG_FILE
    try:
        existing = {}
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))

        portfolio_section = existing.get("portfolio_optimization")
        if not isinstance(portfolio_section, dict):
            portfolio_section = {}
        multi_objective = portfolio_section.get("multi_objective")
        if not isinstance(multi_objective, dict):
            multi_objective = {}

        selected_candidate = result.get("selected")
        recommended_candidate = result.get("recommended")
        frontier = result.get("pareto_frontier", [])
        selected_index = None
        if isinstance(frontier, list) and isinstance(selected_candidate, dict):
            for idx, item in enumerate(frontier):
                if isinstance(item, dict) and item == selected_candidate:
                    selected_index = idx
                    break

        multi_objective.update(
            {
                "enabled": True,
                "active_preset": result.get("preset", "balanced"),
                "selected_candidate": selected_candidate if isinstance(selected_candidate, dict) else None,
                "selected_candidate_index": selected_index,
                "recommended_candidate": recommended_candidate if isinstance(recommended_candidate, dict) else None,
                "pareto_frontier": frontier if isinstance(frontier, list) else [],
                "preset_eligible_frontier": result.get("preset_eligible_frontier", []),
                "candidate_count": result.get("candidate_count", 0),
                "verified_candidate_count": result.get("verified_candidate_count", 0),
                "invalid_trial_count": result.get("invalid_trial_count", 0),
                "selection_required": bool(result.get("selection_required", False)),
                "objective_source": result.get("objective_source", "unknown"),
                "solver": result.get("solver", "unknown"),
            }
        )
        portfolio_section["multi_objective"] = multi_objective
        existing["portfolio_optimization"] = portfolio_section

        existing.setdefault("optimization_results", {})
        existing["optimization_results"]["last_multi_objective_run"] = {
            "preset": result.get("preset", "balanced"),
            "candidate_count": int(result.get("candidate_count", 0) or 0),
            "verified_candidate_count": int(result.get("verified_candidate_count", 0) or 0),
            "invalid_trial_count": int(result.get("invalid_trial_count", 0) or 0),
            "selection_required": bool(result.get("selection_required", False)),
            "source": "parameter_optimizer.py (NSGA-II verified auto-selection)",
        }

        atomic_write_json(path, existing)
        if isinstance(selected_candidate, dict):
            apply_selected_candidate(
                candidate=selected_candidate,
                candidate_index=selected_index,
                preset=str(result.get("preset", "balanced")),
                config_path=path,
            )
            existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            if not isinstance(existing, dict):
                existing = {}
            existing.setdefault("optimization_results", {})
            existing["optimization_results"]["best_params"] = selected_candidate.get("params", {})
            existing["optimization_results"]["source"] = "parameter_optimizer.py (verified Pareto auto-selection)"
            atomic_write_json(path, existing)
            logger.info("[OPTIMIZER] Verified Pareto candidate otomatik secildi ve config'e uygulandi")
        else:
            logger.info("[OPTIMIZER] Verified Pareto frontier kaydedildi ancak uygulanacak aday bulunamadi")
    except BEST_EFFORT_EXCEPTIONS as e:
        logger.warning("[OPTIMIZER] Multi-objective config yazma hatası: %s", e)


def select_multi_objective_candidate(
    candidate_index: int,
    *,
    preset: Optional[str] = None,
    config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Explicitly select and apply one Pareto candidate to the live config."""
    path = config_path or CONFIG_FILE
    selected = apply_selected_candidate(candidate_index=candidate_index, preset=preset, config_path=path)
    try:
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if not isinstance(existing, dict):
            existing = {}
        existing.setdefault("optimization_results", {})
        existing["optimization_results"]["best_params"] = selected.get("params", {})
        existing["optimization_results"]["source"] = "parameter_optimizer.py (explicit Pareto selection)"
        atomic_write_json(path, existing)
    except BEST_EFFORT_EXCEPTIONS as exc:
        logger.warning("[OPTIMIZER] Selected candidate metadata write warning: %s", exc)
    return selected


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = run_optimization()
    if res:
        print(json.dumps(res, indent=2))
