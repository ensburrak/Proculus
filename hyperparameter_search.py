# -*- coding: utf-8 -*-
"""
hyperparameter_search.py

Bu komut satiri araci, risk_dataset.csv filesi uzerinde lojistik regresyon
modeli icin basit bir hiperparametre taramasi yapar. Amac, farkli
cezalandirma parametreleri (C) ve solver kombinasyonlari icin modeli
valuelendirip en iyi parametreleri bulmaktir.

Calistirmak icin:

    (venv) python hyperparameter_search.py

Tarama resultlari ``data/hyperparameters.json`` filesina yazilir ve
console uzerine raporlanir. Veri filesi bulunamazsa, script hicbir
trade yapmadan cikar.

Not: Bu script, cevrimdisi usage icindir. Ana bot icerisinde
otomatik tetiklenmez.
"""
from __future__ import annotations


from core.exceptions import BEST_EFFORT_EXCEPTIONS
import json
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


def perform_grid_search(csv_path: Path, output_path: Path, verbose: bool = True) -> None:
    """Hiperparametre taramasi yap ve resultlari yaz.

    Args:
        csv_path: Veri kumesinin yolu (risk_dataset.csv).
        output_path: Sonuclarin yazilacagi json filesi.
        verbose: True ise ara resultlari ekrana yaz.
    """
    try:
        df = pd.read_csv(csv_path)
    except BEST_EFFORT_EXCEPTIONS as e:
        print(f"[hyperparameter_search] Veri seti could not be read: {e}")
        return
    # Hedef degisken
    y = df.get("y_win")
    if y is None:
        print("[hyperparameter_search] Veri kumesinde 'y_win' sutunu not found.")
        return
    # Ozellikler
    features = [col for col in df.columns if col not in ("y_win", "symbol", "timestamp")]
    X = df[features].values
    # Y14 FIX: C=50 aşırı uyum riski taşıyordu, düşürüldü
    param_grid = {
        'C': [0.01, 0.1, 1.0],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [200]
    }
    # Time-series validation: keep chronological order, no shuffled folds.
    n_splits = min(5, max(2, len(df) - 1))
    cv = TimeSeriesSplit(n_splits=n_splits)
    model = LogisticRegression()
    # Y14 FIX: accuracy → f1 (dengesiz sınıflar için daha anlamlı)
    search = GridSearchCV(model, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=0)
    try:
        search.fit(X, y)
    except BEST_EFFORT_EXCEPTIONS as e:
        print(f"[hyperparameter_search] Tarama failed: {e}")
        return
    best_params = search.best_params_
    best_score = search.best_score_
    if verbose:
        print(f"[hyperparameter_search] En iyi parametreler: {best_params}")
        print(f"[hyperparameter_search] Ortalama dogruluk: {best_score:.4f}")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as f:
            json.dump({'best_params': best_params, 'best_score': best_score}, f, ensure_ascii=False, indent=2)
        if verbose:
            print(f"[hyperparameter_search] Sonuclar {output_path} filesina yazildi.")
    except BEST_EFFORT_EXCEPTIONS as e:
        print(f"[hyperparameter_search] Sonuclar yazilamadi: {e}")


if __name__ == "__main__":
    csv_path = Path("data/risk_dataset.csv")
    output_path = Path("data/hyperparameters.json")
    perform_grid_search(csv_path, output_path)
