# -*- coding: utf-8 -*-
"""
memory_guardian.py — Bellek Taşması Koruma Modülü
=================================================
Bot'un %100 bellek kullanımından çökmesini engeller.

Özellikler:
1. Periyodik gc.collect() ile döngüsel referansları temizler
2. Bellek eşiği kontrol eder (%85 aşılırsa acil temizlik yapar)
3. Global cache'leri (whale, arbitraj vb.) periyodik temizler
4. Deque maxlen güvenliği sağlar

Kullanım:
    from memory_guardian import start_memory_guardian, check_memory_now
    start_memory_guardian()  # Background thread olarak başlat
"""

import gc
import logging
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Optional

from core.exceptions import BEST_EFFORT_EXCEPTIONS

log = logging.getLogger("MemoryGuardian")

# === KONFIGÜRASYON ===
MEMORY_CHECK_INTERVAL = 60  # 1 dakikada bir kontrol
MAX_CACHE_ITEMS = 500  # Global cache'lerde maksimum öğe sayısı


@dataclass(frozen=True)
class CacheCleanupRegistration:
    name: str


_registered_cache_cleanups: dict[str, tuple[Callable[..., int], Any]] = {}
_registered_cache_cleanups_lock = threading.Lock()


def register_cache_cleanup(
    name: str,
    cleanup: Callable[..., int],
    *,
    lock: Any = None,
) -> CacheCleanupRegistration:
    """Register an explicit owner-provided cache cleanup hook."""
    key = str(name or "").strip()
    if not key:
        raise ValueError("cache cleanup name is required")
    if not callable(cleanup):
        raise TypeError("cleanup must be callable")
    with _registered_cache_cleanups_lock:
        _registered_cache_cleanups[key] = (cleanup, lock)
    return CacheCleanupRegistration(name=key)


def unregister_cache_cleanup(registration: CacheCleanupRegistration | str) -> None:
    key = registration.name if isinstance(registration, CacheCleanupRegistration) else str(registration)
    with _registered_cache_cleanups_lock:
        _registered_cache_cleanups.pop(key, None)


def _cleanup_registered_caches(aggressive: bool = False) -> int:
    cleaned = 0
    with _registered_cache_cleanups_lock:
        registrations = list(_registered_cache_cleanups.items())
    for name, (cleanup, lock) in registrations:
        try:
            if lock is not None:
                with lock:
                    cleaned += int(cleanup(aggressive=aggressive) or 0)
            else:
                cleaned += int(cleanup(aggressive=aggressive) or 0)
        except BEST_EFFORT_EXCEPTIONS as exc:
            log.warning("[MEM_GUARD] Registered cache cleanup failed: %s: %s", name, exc)
    return cleaned


# [FIX-6] Eşikler health_monitor.MONITORING_THRESHOLDS'dan okunur (tutarlılık)
try:
    from health_monitor import MONITORING_THRESHOLDS as _MT

    MEMORY_WARNING_THRESHOLD = _MT["memory_warning_pct"] / 100.0
    MEMORY_CRITICAL_THRESHOLD = _MT["memory_critical_pct"] / 100.0
    MEMORY_MAX_MB = float(_MT.get("memory_max_mb", 2048))
except ImportError:
    MEMORY_WARNING_THRESHOLD = 0.80
    MEMORY_CRITICAL_THRESHOLD = 0.90
    MEMORY_MAX_MB = 2048.0

# Süreç bazlı ek mutlak eşikler (MB). Yüzde ölçümü sistem RAM'ine göre düşük
# kalabileceği için, süreç RSS boyutunu da ayrı eşikle izliyoruz.
MEMORY_WARNING_MB = float(os.getenv("MEMORY_WARNING_MB", max(512.0, MEMORY_MAX_MB * 0.75)))
MEMORY_CRITICAL_MB = float(os.getenv("MEMORY_CRITICAL_MB", max(768.0, MEMORY_MAX_MB * 0.90)))


# === BELLEK ÖLÇÜMÜ ===
def get_memory_usage_mb() -> float:
    """Mevcut işlemin bellek kullanımını MB olarak döndürür."""
    try:
        import psutil

        process = psutil.Process(os.getpid())
        return float(process.memory_info().rss) / (1024 * 1024)
    except ImportError:
        # psutil yoksa /proc/self/status'tan oku (Linux)
        try:
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024  # KB → MB
        except BEST_EFFORT_EXCEPTIONS:
            pass
    except BEST_EFFORT_EXCEPTIONS:
        pass
    return 0.0


def get_memory_percent() -> float:
    """Mevcut sürecin toplam RAM içindeki oranını döndürür (0.0-1.0)."""
    try:
        import psutil

        process = psutil.Process(os.getpid())
        rss = process.memory_info().rss
        total = psutil.virtual_memory().total
        if total <= 0:
            return 0.0
        return float(rss) / float(total)
    except ImportError:
        return 0.0
    except BEST_EFFORT_EXCEPTIONS:
        return 0.0


# === CACHE TEMİZLEME ===
def _cleanup_global_caches(aggressive: bool = False) -> int:
    """
    Bilinen global cache'leri temizler.
    aggressive=True ise tümünü siler, False ise sadece eski olanları.
    """
    cleaned = 0

    # 1. whale_alert_provider cache
    try:
        import whale_alert_provider as wap

        if hasattr(wap, "_cache") and isinstance(wap._cache, dict):
            if aggressive:
                old_size = len(wap._cache)
                wap._cache.clear()
                cleaned += old_size
            elif len(wap._cache) > MAX_CACHE_ITEMS:
                # En eski yarısını sil
                items = sorted(wap._cache.items(), key=lambda x: x[1][1] if isinstance(x[1], tuple) else 0)
                for k, _ in items[: len(items) // 2]:
                    del wap._cache[k]
                    cleaned += 1
    except BEST_EFFORT_EXCEPTIONS:
        pass

    # 2. state_manager cache
    try:
        import state_manager as sm

        if hasattr(sm, "_state") and isinstance(sm._state, dict):
            if len(sm._state) > MAX_CACHE_ITEMS:
                old_keys = list(sm._state.keys())[:-MAX_CACHE_ITEMS]
                for k in old_keys:
                    del sm._state[k]
                    cleaned += 1
    except BEST_EFFORT_EXCEPTIONS:
        pass

    # 3. Arbitraj sensörü price_history
    try:
        import decision.arbitrage_sensor as das

        if hasattr(das, "_global_sensor") and das._global_sensor:
            sensor = das._global_sensor
            if hasattr(sensor, "price_history"):
                for sym, dq in sensor.price_history.items():
                    # maxlen yoksa 200'e kes
                    if hasattr(dq, "maxlen") and dq.maxlen is None:
                        while len(dq) > 200:
                            dq.popleft()
                            cleaned += 1
    except BEST_EFFORT_EXCEPTIONS:
        pass

    # 4. Proje modüllerindeki bilinen cache/history container'larını sınırlı tut.
    try:
        from ml.rl_predict import clear_prediction_caches

        rl_cleaned = clear_prediction_caches(force=aggressive, max_idle_sec=float(MEMORY_CHECK_INTERVAL) * 2.0)
        if rl_cleaned > 0:
            log.info("[MEM_GUARD] rl_predict temizlendi: %d", rl_cleaned)
        cleaned += rl_cleaned
    except BEST_EFFORT_EXCEPTIONS:
        pass

    # 5. LLM batch cache — en büyük sürekli büyüyen cache
    try:
        import ai_batch_manager as abm

        llm_cache = getattr(abm, "_LLM_CACHE", None)
        if llm_cache is not None:
            if aggressive:
                old_size = len(llm_cache.entries)
                llm_cache.clear()
                cleaned += old_size
            else:
                llm_cache._prune()
    except BEST_EFFORT_EXCEPTIONS:
        pass

    # 6. MTF analyzer instance cache
    try:
        from analysis.mtf_analyzer import MTFAnalyzer

        inst = getattr(MTFAnalyzer, "_instance", None)
        if inst is not None and hasattr(inst, "_cache") and isinstance(inst._cache, dict):
            old_size = len(inst._cache)
            if aggressive or old_size > MAX_CACHE_ITEMS:
                inst._cache.clear()
                cleaned += old_size
    except BEST_EFFORT_EXCEPTIONS:
        pass

    # 7. Feature cache
    try:
        import ml.feature_cache as fc

        store = getattr(fc, "_store", None) or getattr(fc, "_cache", None)
        if store is None:
            for attr in dir(fc):
                obj = getattr(fc, attr, None)
                if hasattr(obj, "_store") and isinstance(obj._store, dict):
                    store = obj._store
                    break
        if isinstance(store, dict) and (aggressive or len(store) > MAX_CACHE_ITEMS):
            old_size = len(store)
            store.clear()
            cleaned += old_size
    except BEST_EFFORT_EXCEPTIONS:
        pass

    registered_cleaned = _cleanup_registered_caches(aggressive=aggressive)
    cleaned += registered_cleaned

    pre_scan = cleaned
    scan_cleaned = _cleanup_project_module_caches(aggressive=aggressive)
    cleaned += scan_cleaned
    log.info(
        "[MEM_GUARD] Temizlik detayı: hardcoded_caches=%d, module_scan=%d, toplam=%d",
        pre_scan,
        scan_cleaned,
        cleaned,
    )
    return cleaned


_VENV_MARKERS = (
    os.sep + ".venv" + os.sep,
    os.sep + "venv" + os.sep,
    os.sep + "site-packages" + os.sep,
)


def _cleanup_project_module_caches(aggressive: bool = False) -> int:
    """
    Proje modüllerinde cache/history/buffer benzeri container alanlarını tarayıp
    kontrollü şekilde küçültür. Amaç, özelliği kapatmadan bellek şişmesini sınırlamak.
    """
    _ = aggressive
    root = os.path.dirname(os.path.abspath(__file__))
    keywords = ("cache", "history", "buffer", "queue", "pending")
    hard_cap = MAX_CACHE_ITEMS
    observed = 0
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        mod_file = getattr(mod, "__file__", None)
        if not mod_file:
            continue
        try:
            abs_mod_file = os.path.abspath(mod_file)
        except BEST_EFFORT_EXCEPTIONS:
            continue
        if not abs_mod_file.startswith(root):
            continue
        if any(marker in abs_mod_file for marker in _VENV_MARKERS):
            continue
        for attr_name in dir(mod):
            if attr_name.startswith("__"):
                continue
            low = attr_name.lower()
            if not any(k in low for k in keywords):
                continue
            try:
                obj = getattr(mod, attr_name)
                if isinstance(obj, (dict, list, set, deque)) and len(obj) > hard_cap:
                    observed += 1
                    log.warning(
                        "[MEM_GUARD] oversized unregistered cache observed: mod=%s attr=%s type=%s size=%d file=%s",
                        getattr(mod, "__name__", "?"),
                        attr_name,
                        type(obj).__name__,
                        len(obj),
                        abs_mod_file,
                    )
            except BEST_EFFORT_EXCEPTIONS:
                continue
    if observed:
        log.warning(
            "[MEM_GUARD] observed %d oversized unregistered caches; register cleanup hooks to enable mutation",
            observed,
        )
    return 0


def _force_gc() -> int:
    """Agresif garbage collection — döngüsel referansları temizle."""
    collected = 0
    for _ in range(3):  # 3 kez çağır (derin döngüsel referanslar için)
        collected += gc.collect()
    return collected


# === ANA İZLEME DÖNGÜSÜ ===
class MemoryGuardian:
    def __init__(self, check_interval: int = MEMORY_CHECK_INTERVAL) -> None:
        self.check_interval = check_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_mem_mb = 0.0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True, name="MemoryGuardian")
        self._thread.start()
        log.info(f"[MEM_GUARD] Bellek koruyucusu başlatıldı. Kontrol aralığı: {self.check_interval}s")

    def stop(self) -> None:
        self._running = False

    def _monitor_loop(self) -> None:
        while self._running:
            try:
                self._check_and_clean()
            except BEST_EFFORT_EXCEPTIONS as e:
                log.error(f"[MEM_GUARD] İzleme hatası: {e}")
            time.sleep(self.check_interval)

    def _check_and_clean(self) -> None:
        """Bellek durumunu kontrol et, gerekirse temizlik yap."""
        mem_mb = get_memory_usage_mb()
        mem_pct = get_memory_percent()
        self._last_mem_mb = mem_mb

        # Normal durum — periyodik gc
        gc_collected = _force_gc()

        is_critical = (mem_pct >= MEMORY_CRITICAL_THRESHOLD) or (mem_mb >= MEMORY_CRITICAL_MB)
        is_warning = (mem_pct >= MEMORY_WARNING_THRESHOLD) or (mem_mb >= MEMORY_WARNING_MB)

        if is_critical:
            # KRİTİK: %90+ → Acil temizlik
            log.warning(
                f"[MEM_GUARD] ⚠️ KRİTİK BELLEK! {mem_mb:.0f}MB "
                f"({mem_pct*100:.1f}% RAM). Eşikler: {MEMORY_CRITICAL_MB:.0f}MB / "
                f"{MEMORY_CRITICAL_THRESHOLD*100:.1f}%. Acil temizlik yapılıyor..."
            )
            cleaned = _cleanup_global_caches(aggressive=True)
            gc_extra = _force_gc()

            new_mem = get_memory_usage_mb()
            log.warning(
                f"[MEM_GUARD] Acil temizlik tamamlandı: {cleaned} cache öğesi silindi, "
                f"{gc_collected + gc_extra} nesne toplandı. "
                f"Bellek: {mem_mb:.0f}MB → {new_mem:.0f}MB"
            )

        elif is_warning:
            # UYARI: %80+ → Normal temizlik
            log.info(
                f"[MEM_GUARD] Bellek yüksek: {mem_mb:.0f}MB ({mem_pct*100:.1f}% RAM). "
                f"Eşikler: {MEMORY_WARNING_MB:.0f}MB / {MEMORY_WARNING_THRESHOLD*100:.1f}%. "
                f"Cache temizliği yapılıyor..."
            )
            cleaned = _cleanup_global_caches(aggressive=False)
            log.info(f"[MEM_GUARD] {cleaned} cache öğesi temizlendi.")

        else:
            # Normal: sadece gc yaptık
            if gc_collected > 0:
                log.debug(f"[MEM_GUARD] Periyodik GC: {gc_collected} nesne toplandı. Bellek: {mem_mb:.0f}MB")


# === GLOBAL INSTANCE ===
_guardian = MemoryGuardian()


def start_memory_guardian(check_interval: int = MEMORY_CHECK_INTERVAL) -> None:
    """Memory Guardian'ı background thread olarak başlat."""
    _guardian.check_interval = check_interval
    _guardian.start()


def stop_memory_guardian() -> None:
    """Memory Guardian'ı durdur."""
    _guardian.stop()


def check_memory_now() -> dict[str, float | int | str]:
    """Anlık bellek durumunu kontrol et ve döndür."""
    mem_mb = get_memory_usage_mb()
    mem_pct = get_memory_percent()
    gc_collected = _force_gc()

    return {
        "memory_mb": round(mem_mb, 1),
        "memory_percent": round(mem_pct * 100, 1),
        "gc_collected": gc_collected,
        "status": (
            "critical"
            if mem_pct >= MEMORY_CRITICAL_THRESHOLD
            else "warning" if mem_pct >= MEMORY_WARNING_THRESHOLD else "ok"
        ),
    }
