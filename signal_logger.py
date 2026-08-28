# -*- coding: utf-8 -*-
"""
signal_logger.py
-----------------

Bu modul, her potansiyel trade (enter sinyali) icin ozellikleri ve
sonradan etiketlenecek sonucu bir CSV filesina yazar. Amac, master
confidence skorunun ve diger sinyal bilesenlerinin gercek basari
oranlariyla istatistiksel olarak kalibre edilmesini saglayacak bir
veri seti toplamaktir. Kayit formati oldukca basittir ve time
damgasi, symbol, time dilimi (tf), ham master skoru (kalibrasyondan
once), AI/teknik/sentiment/RL skorlari, temel yon (base_decision),
beklenen entry fiyati, piyasa rejimi ve son olarak etiket (label)
icerir. Etiket 1=kazanc, 0=kayip olarak tanimlanir ve baslangicta
``None`` olarak yazilir; trade kapandiktan sonra ``update_label``
fonksiyonu ile update.

Dosya ``data/signal_dataset.csv`` konumunda tutulur. Ilk satirda
basliklar yer alir. Yeni kayitlar append modunda eklenir. Bir trade
ile sinyal kaydi arasinda baglanti kurmak icin benzersiz bir
``signal_id`` kullanilir; bu id, time damgasi ve symbolun birlesiminden
olusturulur ve trade kapandiginda bu id uzerinden etiket update.

Ornek satir:

```
signal_id,timestamp,symbol,tf,master_conf_raw,ai_score,tech_score,sent_score,rl_score,base_decision,price_entry_planned,regime,label
2025-11-30T04:03:10_BTC/USDT,2025-11-30T04:03:10,BTC/USDT,5m,0.6821,0.71,0.95,0.78,0.50,short,38350.2,BULL,
```

Bu modul, concurrency acisindan basit tutulmustur; file yazarken
kilitleme kullanilmaz. Yogun bir uretim ortaminda file tabanli
kilitleme veya veritabani entegrasyonu eklemek gerekebilir.
"""

from __future__ import annotations


from core.exceptions import BEST_EFFORT_EXCEPTIONS
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from atomic_io import file_lock
from runtime_paths import DATA_DIR

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Sinyal veri seti file yolu
SIGNAL_DATASET_FILE = DATA_DIR / "signal_dataset.csv"

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
RICH_LABEL_COLUMNS = [
    "label_class",
    "normalized_pnl",
    "sharpe_per_trade",
    "label_timestamp",
    "label_entry_time",
    "label_exit_time",
]
DATASET_COLUMNS = BASE_COLUMNS + RICH_LABEL_COLUMNS


def _normalize_headers(headers: list[str]) -> list[str]:
    merged = list(headers or BASE_COLUMNS)
    for col in RICH_LABEL_COLUMNS:
        if col not in merged:
            merged.append(col)
    return merged


def _rewrite_dataset(headers: list[str], rows: list[dict[str, Any]]) -> None:
    SIGNAL_DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = SIGNAL_DATASET_FILE.with_suffix(SIGNAL_DATASET_FILE.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in headers})
    tmp_path.replace(SIGNAL_DATASET_FILE)


def _ensure_header() -> list[str]:
    """Ensure the canonical dataset schema exists and return active headers."""
    SIGNAL_DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(SIGNAL_DATASET_FILE, timeout=10):
        if not SIGNAL_DATASET_FILE.exists():
            _rewrite_dataset(DATASET_COLUMNS, [])
            return list(DATASET_COLUMNS)
        with SIGNAL_DATASET_FILE.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = _normalize_headers(list(reader.fieldnames or []))
            rows = list(reader)
        if headers != list(reader.fieldnames or []):
            _rewrite_dataset(headers, rows)
        return headers


def log_signal(
    symbol: str,
    tf: str,
    master_conf_raw: float,
    ai_score: float,
    tech_score: float,
    sent_score: float,
    rl_score: float,
    base_decision: str,
    price_entry_planned: Optional[float],
    regime: Optional[str],
    timestamp: Optional[datetime] = None,
) -> str:
    """
    Yeni bir sinyal kaydi ekler ve benzersiz ``signal_id`` return.

    Args:
        symbol: Islem yapilan symbol (orn. "BTC/USDT").
        tf: Zaman dilimi (orn. "5m", "15m").
        master_conf_raw: Kalibrasyondan onceki master guven skoru (0..1).
        ai_score: AI modelinden gelen skor (0..1).
        tech_score: Teknik analiz skoru (0..1).
        sent_score: Sentiment skoru (0..1).
        rl_score: Takviye ogrenme skorunun 0..1 araligindaki valuei.
        base_decision: "long" veya "short" gibi temel karar.
        price_entry_planned: Islem icin planlanan entry fiyati.
        regime: Piyasa rejimi ("BULL", "BEAR", "SIDEWAYS" veya None).
        timestamp: Opsiyonel, sinyal timei. None ise UTC now kullanilir.

    Returns:
        signal_id: Kaydin benzersiz kimligi (``{timestamp_iso}_{symbol}``).
    """
    headers = _ensure_header()
    ts = timestamp or datetime.now(timezone.utc)
    ts_iso = ts.replace(microsecond=0).isoformat()
    signal_id = f"{ts_iso}_{symbol}"
    row = {
        "signal_id": signal_id,
        "timestamp": ts_iso,
        "symbol": symbol,
        "tf": tf,
        "master_conf_raw": round(float(master_conf_raw), 6) if master_conf_raw is not None else "",
        "ai_score": round(float(ai_score), 6) if ai_score is not None else "",
        "tech_score": round(float(tech_score), 6) if tech_score is not None else "",
        "sent_score": round(float(sent_score), 6) if sent_score is not None else "",
        "rl_score": round(float(rl_score), 6) if rl_score is not None else "",
        "base_decision": base_decision,
        "price_entry_planned": round(float(price_entry_planned), 6) if price_entry_planned is not None else "",
        "regime": regime or "",
        "label": "",
        "label_class": "",
        "normalized_pnl": "",
        "sharpe_per_trade": "",
        "label_timestamp": "",
        "label_entry_time": "",
        "label_exit_time": "",
    }
    with file_lock(SIGNAL_DATASET_FILE, timeout=10):
        with SIGNAL_DATASET_FILE.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writerow({key: row.get(key, "") for key in headers})
    return signal_id


def update_label(signal_id: str, label: int) -> None:
    """
    Varolan bir sinyal kaydinin etiketini (label) update.

    Bu fonksiyon, trade kapandiginda kazanc/kayip statusuna gore cagrilmali
    ve ilgili sinyalin ``label`` alanina 1 (kazanc) veya 0 (kayip) yazmalidir.
    ``signal_id`` eslesen ilk kayit update; birden fazla eslesme
    beklenmez.

    Args:
        signal_id: log_signal tarafindan return benzersiz kimlik.
        label: 1 (kazanc) veya 0 (kayip) valuei.
    """
    if label not in (0, 1):
        return
    headers = _ensure_header()
    if not SIGNAL_DATASET_FILE.exists():
        return
    updated = False
    with file_lock(SIGNAL_DATASET_FILE, timeout=10):
        with SIGNAL_DATASET_FILE.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        for row in rows:
            if str(row.get("signal_id") or "") == signal_id and not updated:
                row["label"] = str(int(label))
                row["label_timestamp"] = datetime.now(timezone.utc).isoformat()
                updated = True
        if updated:
            _rewrite_dataset(headers, rows)
    # update sessizce devam edilir

    # Otomatik kalibrasyon: yeterli etiketli sonuc birikince runtime artifact'larini gunceller.
    try:
        if updated:
            from decision.master_score_calibration import maybe_recalibrate_master_score

            maybe_recalibrate_master_score(source_hint=SIGNAL_DATASET_FILE)
    except BEST_EFFORT_EXCEPTIONS:
        pass
