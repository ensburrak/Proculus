"""
watchdog.py
-----------

This module implements a simple fail-safe watchdog that monitors system
resources (CPU and memory usage) and environment integrity.  When usage
exceeds configured thresholds or suspicious conditions (e.g. missing API
keys) are detected, the watchdog triggers a callback which can pause the
bot, notify an operator, or take other corrective action.

The watchdog is designed to run in a separate thread or process and
periodically sample the system.  It can be integrated into the bot by
starting it at initialisation and registering appropriate callbacks.

Example usage::

    from watchdog import Watchdog
    def on_alert(msg):
        print("Watchdog alert:", msg)
    wd = Watchdog(memory_threshold=0.9, cpu_threshold=0.9, interval=5, callback=on_alert)
    wd.start()
    # ... run bot ...
    wd.stop()

"""

from __future__ import annotations


from core.exceptions import BEST_EFFORT_EXCEPTIONS
import os
import threading
import time
import logging
from typing import Any, Callable, Optional

try:
    import psutil  # type: ignore
except BEST_EFFORT_EXCEPTIONS:
    psutil = None  # pragma: no cover

logger = logging.getLogger(__name__)


class Watchdog:
    """Monitors resource usage, environment integrity and connectivity.

    The watchdog periodically samples system statistics, checks for missing
    environment variables and optionally verifies network connectivity by
    pinging a URL.  When a threshold is exceeded or an anomaly is detected
    (such as missing API keys), the registered callback is invoked with a
    descriptive message.
    """

    def __init__(
        self,
        memory_threshold: float = None,
        memory_max_mb: float = None,
        cpu_threshold: float = None,
        interval: float = 10.0,
        callback: Optional[Callable[[str], None]] = None,
        env_keys: Optional[list[str]] = None,
        ping_url: Optional[str] = None,
        ping_failure_threshold: int = 3,
    ) -> None:
        """Initialise the watchdog.

        :param memory_threshold: fraction of total memory usage above which to alert
        :param cpu_threshold: fraction of total CPU usage above which to alert
        :param interval: interval in seconds between checks
        :param callback: function to call with an alert message when a threshold is breached
        :param env_keys: list of environment variable names that must be present; missing keys trigger alerts
        :param ping_url: optional URL to test network connectivity; failure triggers an alert
        """
        # [FIX-6.2] Eşikler health_monitor.MONITORING_THRESHOLDS'dan okunur (tutarlılık)
        if memory_threshold is None or cpu_threshold is None:
            try:
                from health_monitor import MONITORING_THRESHOLDS as _MT
                if memory_threshold is None:
                    memory_threshold = _MT["memory_critical_pct"] / 100.0
                if memory_max_mb is None:
                    memory_max_mb = float(_MT.get("memory_max_mb", 2048))
                if cpu_threshold is None:
                    cpu_threshold = _MT["cpu_max_pct"] / 100.0
            except ImportError:
                import logging as _log
                _log.info("[SYS_WATCHDOG] health_monitor import edilemedi — varsayılan eşikler kullanılıyor (memory=0.9, cpu=0.9)")
                memory_threshold = memory_threshold or 0.9
                memory_max_mb = memory_max_mb or 2048.0
                cpu_threshold = cpu_threshold or 0.9
        else:
            if memory_max_mb is None:
                memory_max_mb = 2048.0
        self.memory_threshold = memory_threshold
        self.memory_max_mb = float(memory_max_mb)
        self.cpu_threshold = cpu_threshold
        self.interval = interval
        self.callback = callback
        self.env_keys = env_keys or []
        self.ping_url = ping_url
        self.ping_failure_threshold = max(1, int(ping_failure_threshold or 1))
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._process: Optional[Any] = None
        self._ping_failures = 0
        self._ping_alert_active = False

    def _check_connectivity(self) -> list[str]:
        alerts: list[str] = []
        if not self.ping_url:
            return alerts

        failure_message: str | None = None
        try:
            import urllib.request

            request = urllib.request.Request(
                self.ping_url,
                headers={"User-Agent": "AutoTraderBot/1.0"},
            )
            with urllib.request.urlopen(request, timeout=5) as resp:
                if resp.status >= 400:
                    failure_message = f"Ping URL returned status {resp.status}"
        except BEST_EFFORT_EXCEPTIONS as exc:
            failure_message = f"Network connectivity check failed: {exc}"

        if failure_message is None:
            self._ping_failures = 0
            self._ping_alert_active = False
            return alerts

        self._ping_failures += 1
        if self._ping_failures >= self.ping_failure_threshold and not self._ping_alert_active:
            alerts.append(
                f"{failure_message} (consecutive_failures={self._ping_failures})"
            )
            self._ping_alert_active = True
        return alerts

    def _check(self) -> None:
        while not self._stop_event.is_set():
            alerts = []
            # Check environment variables
            for key in self.env_keys:
                if os.getenv(key) is None:
                    alerts.append(f"Missing required env var: {key}")
            alerts.extend(self._check_connectivity())
            if psutil is not None:
                try:
                    total_mb = psutil.virtual_memory().total / (1024 * 1024)
                    if self._process is None:
                        self._process = psutil.Process(os.getpid())
                    proc_mem_mb = self._process.memory_info().rss / (1024 * 1024)
                    proc_mem_ratio = (proc_mem_mb / total_mb) if total_mb > 0 else 0.0

                    if (proc_mem_ratio > self.memory_threshold) or (proc_mem_mb > self.memory_max_mb):
                        alerts.append(
                            "High memory usage: "
                            f"process={proc_mem_mb:.0f}MB ({proc_mem_ratio:.2%} of RAM), "
                            f"limit={self.memory_max_mb:.0f}MB"
                        )
                    # Measure only THIS PROCESS's CPU usage instead of system-wide
                    # Stateful process obj to avoid blocking sleep (GIL release) 
                    try:
                        if self._process is None:
                            self._process = psutil.Process(os.getpid())
                            self._process.cpu_percent()  # Initialize internally
                            proc_cpu = 0.0  # Skip first cycle
                        else:
                            proc_cpu = self._process.cpu_percent() / psutil.cpu_count()
                        cpu_ratio = proc_cpu / 100.0
                    except BEST_EFFORT_EXCEPTIONS:
                        # Fallback to system-wide if process measurement fails
                        cpu_ratio = psutil.cpu_percent(interval=1.0) / 100.0
                    if cpu_ratio > self.cpu_threshold:
                        alerts.append(f"High CPU usage: {cpu_ratio:.2%}")
                except BEST_EFFORT_EXCEPTIONS as exc:
                    logger.debug("psutil error: %s", exc)
            else:
                # psutil not available; use os.getloadavg as fallback on Unix
                try:
                    load1, _, _ = os.getloadavg()
                    cpu_ratio = load1 / os.cpu_count()
                    if cpu_ratio > self.cpu_threshold:
                        alerts.append(f"High load average: {cpu_ratio:.2%}")
                except BEST_EFFORT_EXCEPTIONS as _exc:
                    logging.getLogger(__name__).debug("Suppressed in _check: %s", _exc)
            if alerts and self.callback:
                for msg in alerts:
                    try:
                        self.callback(msg)
                    except BEST_EFFORT_EXCEPTIONS as exc:
                        logger.warning("Watchdog callback error: %s", exc)
            time.sleep(self.interval)

    def start(self) -> None:
        """Start monitoring in a background thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._check, name="WatchdogThread", daemon=True)
        self._thread.start()
        logger.info("Watchdog started")

    def stop(self) -> None:
        """Stop monitoring and join the thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            logger.info("Watchdog stopped")
