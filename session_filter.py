"""
session_filter.py

Basit bir seans (session) filtresi. Bu modul, gunun saatine bagli olarak
farkli risk katsayilari ve trade izinleri tanimlar. Kullanici, `data/session_config.json`
filesinda seanslarin baslangic/bitis saatlerini ve risk carpanlarini
belirleyebilir. Bu sayede Asya, Londra ve New York gibi farkli seanslarda
bot davranisi (ornegin leverage veya trade sikligi) otomatik olarak
settinglanabilir.

Konfigurasyon formati (JSON listesi):
[
  {
    "name": "asia",
    "start": "00:00",
    "end": "07:59",
    "risk_multiplier": 0.6,
    "enabled": true
  },
  ...
]

Saatler HH:MM formatinda ve 24 saat diliminde girilmelidir. Modul,
yerel sistem saatini esas alir. Eger farkli bir time dilimi kullanmak
istiyorsaniz, seanslari UTC'ye gore tanimlamaniz onerilir.
"""
from __future__ import annotations


from core.exceptions import BEST_EFFORT_EXCEPTIONS
import json
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

DATA_DIR = Path(__file__).resolve().parent / "data"
_SESSION_FILE = DATA_DIR / "session_config.json"
_sessions_cache: Optional[List[Dict[str, Any]]] = None

def _load_sessions() -> List[Dict[str, Any]]:
    """Konfigurasyon filesindan seans tanimlarini yukler."""
    global _sessions_cache
    if _sessions_cache is None:
        sessions: List[Dict[str, Any]] = []
        try:
            if _SESSION_FILE.exists():
                txt = _SESSION_FILE.read_text(encoding="utf-8").strip()
                if txt:
                    data = json.loads(txt)
                    if isinstance(data, list):
                        for item in data:
                            if not isinstance(item, dict):
                                continue
                            start_str = str(item.get("start", "00:00")).strip()
                            end_str = str(item.get("end", "23:59")).strip()
                            try:
                                start_parts = start_str.split(":")
                                end_parts = end_str.split(":")
                                start_time = time(hour=int(start_parts[0]), minute=int(start_parts[1]))
                                end_time = time(hour=int(end_parts[0]), minute=int(end_parts[1]))
                            except BEST_EFFORT_EXCEPTIONS:
                                continue
                            sessions.append({
                                "name": str(item.get("name", "session")),
                                "start": start_time,
                                "end": end_time,
                                "risk_multiplier": float(item.get("risk_multiplier", 1.0)),
                                "enabled": bool(item.get("enabled", True))
                            })
        except BEST_EFFORT_EXCEPTIONS:
            sessions = []
        _sessions_cache = sessions
    return _sessions_cache or []

def _is_time_in_range(t: time, start: time, end: time) -> bool:
    """Verilen timein [start, end] araliginda olup olmadigini check eder."""
    if start <= end:
        return start <= t <= end
    # Aralik gece yarisini asiyorsa
    return t >= start or t <= end

def get_current_session(now: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
    """
    Su anki saate gore active seansi return. Eger hicbir seans eslesmiyorsa
    None return.

    Args:
        now: Opsiyonel olarak ozel bir datetime objesi. None ise sistem
             saatini kullanir.

    Returns:
        Aktif seans dict veya None.
    """
    sessions = _load_sessions()
    now_dt = now or datetime.now(timezone.utc)
    current_time = now_dt.time()
    for sess in sessions:
        if _is_time_in_range(current_time, sess["start"], sess["end"]):
            return sess
    return None

def get_risk_multiplier(now: Optional[datetime] = None) -> float:
    """Aktif seansin risk carpanini return. Yoksa 1.0."""
    sess = get_current_session(now)
    if sess and sess.get("enabled", True):
        try:
            return float(sess.get("risk_multiplier", 1.0))
        except BEST_EFFORT_EXCEPTIONS:
            return 1.0
    return 1.0

def is_trading_enabled(now: Optional[datetime] = None) -> bool:
    """Aktif seans trade yapmaya izin veriyor mu?"""
    sess = get_current_session(now)
    if sess is None:
        return True
    return bool(sess.get("enabled", True))
