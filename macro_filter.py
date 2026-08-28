# -*- coding: utf-8 -*-
"""
macro_filter.py

Basit bir ekonomik takvim filtresi. Bu modul, data/macro_events.json
filesinda tanimlanan onemli makro-ekonomik olaylari okur ve trade
timei bu olaylarin hemen oncesine veya sirasinda denk geliyorsa
guven skoruna bir risk indirimi uygular. Boylece yuksek volatilite
yapabilecek veri open sirasinda bot daha temkinli olur.

Ornek macro_events.json formati:
[
  {
    "name": "FOMC Meeting",
    "start": "2025-11-30T14:00:00Z",
    "end": "2025-11-30T15:00:00Z",
    "multiplier": 0.5,
    "pre_minutes": 60
  },
  ...
]

Alanlar:
  - name: Olayin adi (infolendirme amaclidir).
  - start: ISO8601 formatinda olayin baslangic timei (UTC).
  - end: ISO8601 formatinda olayin bitis timei (UTC). Opsiyoneldir; verilmezse
    start ile ayni kabul edilir.
  - multiplier: Olay sirasinda uygulanacak risk carpani. Ornegin 0.5,
    master confidence'i yariya indirir.
  - pre_minutes: Olaydan bu kadar dakika once de ayni carpan uygulanir.
    Opsiyoneldir; verilmezse default 30 dakika kullanilir.

Bu modul, olay listesinde sirayla tarama yapar ve eger simdiki time
(UTC) bir veya birden fazla olayin oncesine veya active durationsine denk
geliyorsa en kucuk carpani return. Aksi halde 1.0 return.
"""
from __future__ import annotations

from core.exceptions import BEST_EFFORT_EXCEPTIONS
from datetime import timezone

import json
import logging
from datetime import datetime, timedelta, timezone

log = logging.getLogger(__name__)
from pathlib import Path
from typing import Optional, List, Dict, Any

_events_cache: Optional[List[Dict[str, Any]]] = None

def _load_events() -> List[Dict[str, Any]]:
    """macro_events.json filesini okuyup olaylari return."""
    global _events_cache
    if _events_cache is not None:
        return _events_cache  # type: ignore
    try:
        path = Path(__file__).resolve().parent / "data" / "macro_events.json"
        if path.exists():
            txt = path.read_text(encoding="utf-8").strip()
            if txt:
                data = json.loads(txt)
                if isinstance(data, list):
                    _events_cache = data  # type: ignore
                    return _events_cache  # type: ignore
    except BEST_EFFORT_EXCEPTIONS as _exc:
        logging.getLogger(__name__).warning("_load_events error: %s", _exc)
    _events_cache = []
    return _events_cache  # type: ignore

def get_macro_risk_multiplier(
    now: Optional[datetime] = None,
    current_volatility: float = 0.0,
    avg_volatility: float = 0.0,
) -> float:
    """
    [FIX] Volatilite adaptif makro risk çarpanı.

    Eski: Saat geldi → sabit 0.5 multiplier.
    Yeni: Volatilite artışına göre kademeli risk azaltması.

    - Volatilite değişmedi → multiplier = 0.85 (hafif dikkat)
    - Volatilite 2x arttı → multiplier = 0.50 (dikkatli)
    - Volatilite 3x+ arttı → multiplier = 0.30 (tehlike)

    Args:
        now: Check edilecek time (UTC).
        current_volatility: Mevcut piyasa volatilitesi (ATR, spread vs.).
        avg_volatility: Ortalama volatilite (karşılaştırma için).

    Returns:
        0.2–1.0 arası risk çarpanı.
    """
    current = now or datetime.now(timezone.utc)
    events = _load_events()
    if not events:
        return 1.0
    multiplier = 1.0
    active_event_names: List[str] = []
    for ev in events:
        try:
            start_str = ev.get("start")
            if not start_str:
                continue
            start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
            end_str = ev.get("end") or start_str
            end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            pre_minutes = int(ev.get("pre_minutes", 30))
            pre_window_start = start_dt - timedelta(minutes=pre_minutes)
            if pre_window_start <= current <= end_dt:
                # [FIX] Volatilite adaptif multiplier hesapla
                base_m = float(ev.get("multiplier", 0.5))

                if avg_volatility > 0 and current_volatility > 0:
                    vol_ratio = current_volatility / avg_volatility
                    if vol_ratio >= 3.0:
                        # Volatilite 3x+ → tehlike
                        m = max(0.20, base_m * 0.6)
                    elif vol_ratio >= 2.0:
                        # Volatilite 2x → dikkatli
                        m = max(0.30, base_m * 0.8)
                    elif vol_ratio >= 1.5:
                        # Volatilite 1.5x → orta
                        m = max(0.50, base_m)
                    else:
                        # Volatilite artmadı → hafif dikkat
                        m = max(0.70, base_m * 1.4)
                else:
                    # Volatilite verisi yoksa orijinal multiplier'ı kullan
                    m = base_m

                name = ev.get("name")
                if isinstance(name, str) and name:
                    active_event_names.append(name)
                if m < multiplier:
                    multiplier = m
        except BEST_EFFORT_EXCEPTIONS:
            continue
    # Bound multiplier to [0.2, 1.0]
    if multiplier < 0.2:
        multiplier = 0.2
    if multiplier > 1.0:
        multiplier = 1.0
    if multiplier < 1.0 and active_event_names:
        names_preview = ", ".join(active_event_names[:3])
        log.info("Active macro events: %s → multiplier=%.2f", names_preview, multiplier)
    return multiplier
