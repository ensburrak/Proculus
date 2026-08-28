# -*- coding: utf-8 -*-
"""
run_services.py
===============

[2026-01-15 PROFESSIONAL FIX #34]

Bu script, ana bot'tan BAĞIMSIZ çalışan kritik servisleri başlatır.
Bot kapansa bile bu servisler çalışmaya devam eder.

Servisler:
1. Stop Order Watchdog - Her 60 saniyede SL kontrolü
2. Sentiment Scheduler - Periyodik sentiment güncelleme

Kullanım:
    Terminal 1: py -3.12 main_bot_async.py
    Terminal 2: py -3.12 run_services.py

Windows Service olarak da kurulabilir.
"""

from core.exceptions import BEST_EFFORT_EXCEPTIONS
import asyncio
import logging
import os
import sys
import time
import threading
from pathlib import Path

# Proje kök dizinini path'e ekle
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("services")


def run_stop_watchdog():
    """Stop Order Watchdog'u başlat."""
    try:
        from stop_order_watchdog import run_watchdog
        log.info("🛡️ Stop Order Watchdog başlatılıyor...")
        run_watchdog()  # Bu sonsuz döngü
    except BEST_EFFORT_EXCEPTIONS as e:
        log.error(f"Stop Watchdog hatası: {e}")


def run_sentiment_scheduler():
    """Sentiment Scheduler'ı başlat."""
    try:
        from sentiment_scheduler import run_scheduler
        log.info("📊 Sentiment Scheduler başlatılıyor...")
        run_scheduler()  # Bu sonsuz döngü
    except ImportError:
        log.warning("Sentiment scheduler modülü bulunamadı")
    except BEST_EFFORT_EXCEPTIONS as e:
        log.error(f"Sentiment Scheduler hatası: {e}")



def run_health_monitor():
    """Health Monitor'i başlat."""
    try:
        from health_monitor import HealthMonitor
        log.info("🏥 Health Monitor başlatılıyor...")
        monitor = HealthMonitor()
        monitor.run()  # Bu sonsuz döngü
    except ImportError:
        log.warning("Health monitor modülü bulunamadı")
    except BEST_EFFORT_EXCEPTIONS as e:
        log.error(f"Health Monitor hatası: {e}")


def run_llm_consensus_worker():
    """LLM Consensus Worker'ı başlat."""
    try:
        from background_workers import LLMConsensusWorker
        log.info("🤖 LLM Consensus Worker başlatılıyor...")
        # Get interval from config.json if configured, default to 3600
        interval = 3600
        try:
            import json
            from pathlib import Path
            config_path = Path("config.json")
            if config_path.exists():
                cfg = json.loads(config_path.read_text(encoding="utf-8"))
                interval = int(cfg.get("llm_cost_optimizer", {}).get("llm_decoupled_interval_seconds", 3600))
        except Exception:
            pass
        worker = LLMConsensusWorker(interval=interval)
        worker.run()  # blocks in this service thread
    except BEST_EFFORT_EXCEPTIONS as e:
        log.error(f"LLM Consensus Worker hatası: {e}")


def main():
    print("=" * 60)
    print("🚀 BAĞIMSIZ SERVİSLER BAŞLATILIYOR")
    print("=" * 60)
    print()
    print("Bu script ana bot'tan BAĞIMSIZ çalışır.")
    print("Bot kapansa bile bu servisler korumaya devam eder.")
    print()
    print("Çalışan servisler:")
    print("  1. Stop Order Watchdog (SL koruma)")
    print("  2. Sentiment Scheduler (veri güncelleme)")
    print("  3. Health Monitor (sistem sağlığı)")
    print("  4. LLM Consensus Worker (arka plan LLM tahmini)")
    print()
    print("Durdurmak için: Ctrl+C")
    print("=" * 60)
    print()

    # SQLite veritabanını ilklendir
    try:
        from db_utils import initialise_db
        log.info("🗄️ SQLite veritabanı ilklendiriliyor...")
        initialise_db()
    except Exception as e:
        log.error(f"SQLite veritabanı ilklendirme hatası: {e}")
    
    # Thread'leri başlat
    threads = []
    
    # Stop Watchdog
    t1 = threading.Thread(target=run_stop_watchdog, daemon=False, name="StopWatchdog")
    t1.start()
    threads.append(t1)
    
    # Sentiment Scheduler
    t2 = threading.Thread(target=run_sentiment_scheduler, daemon=False, name="SentimentScheduler")
    t2.start()
    threads.append(t2)
    
    # Health Monitor
    t3 = threading.Thread(target=run_health_monitor, daemon=False, name="HealthMonitor")
    t3.start()
    threads.append(t3)

    # LLM Consensus Worker
    t4 = threading.Thread(target=run_llm_consensus_worker, daemon=False, name="LLMConsensusWorker")
    t4.start()
    threads.append(t4)
    
    log.info("✅ Tüm servisler başlatıldı")
    
    # Ana thread bekle
    try:
        while True:
            time.sleep(60)
            # Health check
            alive_count = sum(1 for t in threads if t.is_alive())
            log.info(f"💓 Servis durumu: {alive_count}/{len(threads)} aktif")
    except KeyboardInterrupt:
        log.info("🛑 Servisler durduruluyor...")
        sys.exit(0)


if __name__ == "__main__":
    main()
