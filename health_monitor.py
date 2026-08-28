# -*- coding: utf-8 -*-
from __future__ import annotations

"""
health_monitor.py
=================

[2026-01-16 PROFESSIONAL FIX #75]

Profesyonel Bot Sağlık İzleme Sistemi.

Özellikler:
1. Bot çalışıyor mu kontrolü
2. Exchange bağlantısı kontrolü
3. Bellek/CPU kullanımı
4. Son trade zamanı kontrolü
5. Telegram uyarıları

Kullanım:
    py -3.12 health_monitor.py
    
    VEYA
    
    from health_monitor import HealthMonitor
    monitor = HealthMonitor()
    monitor.run()
"""



from core.exceptions import BEST_EFFORT_EXCEPTIONS
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import threading

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

log = logging.getLogger("health_monitor")

try:
    import requests
except ImportError:
    requests = None

from atomic_io import atomic_write_json, safe_append_jsonl
from runtime_paths import LOGS_DIR, METRICS_DIR, PROJECT_ROOT, STATE_DIR, get_trade_log_path

# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(__file__).resolve().parent
HEALTH_LOG = METRICS_DIR / "health_status.json"
HEALTH_HISTORY = METRICS_DIR / "health_history.jsonl"
RUNTIME_STATUS_FILE = METRICS_DIR / "runtime_status.json"


def _runtime_state_file(name: str) -> Path:
    root = Path(ROOT).resolve(strict=False)
    if root != PROJECT_ROOT.resolve(strict=False):
        return root / "state" / name
    return STATE_DIR / name

# Health check intervals (seconds)
CHECK_INTERVAL = 60  # Check every minute

# [FIX-6+6.1] Merkezi eşik değerleri — config.json'dan okunabilir
_DEFAULT_THRESHOLDS = {
    "memory_warning_pct": 80,
    "memory_critical_pct": 90,
    "memory_max_mb": 2048,
    "cpu_max_pct": 80,
    "inactive_minutes": 30,
    "heartbeat_minutes": 5,
}

def _load_monitoring_thresholds() -> dict:
    """[FIX-6.1+Z-38] config.json'dan monitoring ve alert eşiklerini oku."""
    try:
        cfg_path = Path(__file__).resolve().parent / "config.json"
        if cfg_path.exists():
            import json as _json
            cfg = _json.loads(cfg_path.read_text(encoding="utf-8"))
            merged = dict(_DEFAULT_THRESHOLDS)
            # Read from "monitoring" section
            mon = cfg.get("monitoring", {})
            merged.update({k: v for k, v in mon.items() if k in _DEFAULT_THRESHOLDS})
            # Z-38: Also read from "alerts" section and map to threshold keys
            alerts = cfg.get("alerts", {})
            alert_mapping = {
                "cpu_threshold": "cpu_max_pct",
                "memory_threshold": "memory_critical_pct",
            }
            for alert_key, threshold_key in alert_mapping.items():
                if alert_key in alerts:
                    merged[threshold_key] = alerts[alert_key]
            return merged
    except BEST_EFFORT_EXCEPTIONS:
        pass
    return dict(_DEFAULT_THRESHOLDS)

MONITORING_THRESHOLDS = _load_monitoring_thresholds()

# Backward compat aliases
MAX_MEMORY_MB = MONITORING_THRESHOLDS["memory_max_mb"]
MAX_CPU_PERCENT = MONITORING_THRESHOLDS["cpu_max_pct"]
MAX_INACTIVE_MINUTES = MONITORING_THRESHOLDS["inactive_minutes"]
MAX_NO_HEARTBEAT_MINUTES = MONITORING_THRESHOLDS["heartbeat_minutes"]


def reload_monitoring_thresholds() -> dict:
    """Reload monitoring thresholds from config.json and refresh module aliases."""
    global MONITORING_THRESHOLDS, MAX_MEMORY_MB, MAX_CPU_PERCENT, MAX_INACTIVE_MINUTES, MAX_NO_HEARTBEAT_MINUTES

    MONITORING_THRESHOLDS = _load_monitoring_thresholds()
    MAX_MEMORY_MB = MONITORING_THRESHOLDS["memory_max_mb"]
    MAX_CPU_PERCENT = MONITORING_THRESHOLDS["cpu_max_pct"]
    MAX_INACTIVE_MINUTES = MONITORING_THRESHOLDS["inactive_minutes"]
    MAX_NO_HEARTBEAT_MINUTES = MONITORING_THRESHOLDS["heartbeat_minutes"]
    return dict(MONITORING_THRESHOLDS)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class HealthStatus:
    """Bot health status."""
    timestamp: str
    overall_status: str  # "healthy", "warning", "critical"
    
    # Individual checks
    bot_running: bool = False
    exchange_connected: bool = False
    memory_ok: bool = True
    cpu_ok: bool = True
    recent_activity: bool = True
    
    # Metrics
    memory_mb: float = 0.0
    cpu_percent: float = 0.0
    cpu_percent_raw: float = 0.0
    uptime_seconds: int = 0
    active_positions: int = 0
    last_trade_minutes: int = 0
    
    # Errors
    errors: List[str] = None
    warnings: List[str] = None
    v2_operational: Dict[str, Any] | None = None
    open_trade_features: Dict[str, Any] | None = None
    runtime_phase: str | None = None
    last_runtime_error: str | None = None
    last_runtime_error_ts: str | None = None
    task_health: Dict[str, Any] | None = None
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "overall_status": self.overall_status,
            "bot_running": self.bot_running,
            "exchange_connected": self.exchange_connected,
            "memory_ok": self.memory_ok,
            "cpu_ok": self.cpu_ok,
            "recent_activity": self.recent_activity,
            "memory_mb": self.memory_mb,
            "cpu_percent": self.cpu_percent,
            "cpu_percent_raw": self.cpu_percent_raw,
            "uptime_seconds": self.uptime_seconds,
            "active_positions": self.active_positions,
            "last_trade_minutes": self.last_trade_minutes,
            "errors": self.errors or [],
            "warnings": self.warnings or [],
            "v2_operational": self.v2_operational or {},
            "open_trade_features": self.open_trade_features or {},
            "runtime_phase": self.runtime_phase,
            "last_runtime_error": self.last_runtime_error,
            "last_runtime_error_ts": self.last_runtime_error_ts,
            "task_health": self.task_health or {},
        }


# =============================================================================
# HEALTH MONITOR
# =============================================================================


class HealthMonitor:
    """
    Professional bot health monitoring system.
    
    Checks:
    1. Bot process running
    2. Exchange connectivity
    3. Memory usage
    4. CPU usage
    5. Recent trading activity
    6. Heartbeat file
    """
    
    def __init__(self):
        self.bot_process_name = "main_bot_async.py"
        self.bot_process_markers = (
            "main_bot_async.py",
            " -m runtime",
            "python -m runtime",
            "py -3.12 -m runtime",
            "runtime.entrypoint",
            "runtime\\entrypoint.py",
            "runtime/main_entry.py",
        )
        self.last_alert_time: Optional[datetime] = None
        self.consecutive_failures = 0
        self._last_live_safety_pause_reason: Optional[str] = None
        self._last_position_snapshot_stale = False
        self._last_position_snapshot_age_minutes: Optional[int] = None
        self._last_position_snapshot_count: Optional[int] = None
        self._last_exchange_positions_available = False
        self._last_private_exchange_auth_error: Optional[str] = None
        # [FIX-5.1B] İlk check hemen yapılsın (counter=4, ilk artışta 5 olur → check)
        self._api_key_check_counter = 4
        
        # Ensure directories
        METRICS_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _env_str(name: str, default: str = "") -> str:
        value = os.getenv(name)
        if value is not None:
            return str(value)
        try:
            from settings import env_str

            return str(env_str(name, default) or default)
        except BEST_EFFORT_EXCEPTIONS:
            return default

    @staticmethod
    def _env_bool(name: str, default: bool = False) -> bool:
        value = HealthMonitor._env_str(name, "")
        if value == "":
            return default
        return str(value).strip().lower() in ("1", "true", "yes", "y")

    @staticmethod
    def _shadow_mode_enabled() -> bool:
        return str(os.getenv("SHADOW_MODE", "")).strip().lower() in {"1", "true", "yes", "on"}

    def _private_api_healthcheck_enabled(self) -> bool:
        """Keep private exchange auth checks enabled outside paper/dry-run modes."""
        default_enabled = not (
            self._env_bool("PAPER_TRADING", False)
            or self._env_bool("WRAPPER_DRY_RUN", False)
            or self._shadow_mode_enabled()
        )
        return self._env_bool("OKX_PRIVATE_HEALTHCHECK_ENABLED", default_enabled)

    def _okx_rest_context(self) -> tuple[str, bool, dict]:
        """
        Return (base_url, use_testnet, extra_headers).
        OKX demo trading uses production domain + x-simulated-trading header.
        """
        base_url = self._env_str("OKX_REST_BASE_URL", "https://www.okx.com").strip().rstrip("/")
        use_testnet = self._env_bool("OKX_USE_TESTNET", True)
        extra_headers = {"x-simulated-trading": "1"} if use_testnet else {}
        return base_url, use_testnet, extra_headers

    def _okx_private_credentials(self) -> tuple[str, str, str]:
        if self._env_bool("OKX_USE_TESTNET", True):
            return (
                self._env_str("OKX_API_KEY", ""),
                self._env_str("OKX_API_SECRET", ""),
                self._env_str("OKX_API_PASSPHRASE", ""),
            )
        return (
            self._env_str("OKX_LIVE_API_KEY", ""),
            self._env_str("OKX_LIVE_API_SECRET", ""),
            self._env_str("OKX_LIVE_API_PASSPHRASE", ""),
        )

    def _build_okx_private_headers(self, method: str, request_path: str) -> Optional[dict]:
        import base64
        import hashlib
        import hmac

        api_key, api_secret, passphrase = self._okx_private_credentials()
        if not api_key or not api_secret:
            return None

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        msg = ts + method.upper() + request_path
        signature = base64.b64encode(
            hmac.HMAC(api_secret.encode(), msg.encode(), hashlib.sha256).digest()
        ).decode()
        headers = {
            "OK-ACCESS-KEY": api_key,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": ts,
            "OK-ACCESS-PASSPHRASE": passphrase,
            "Content-Type": "application/json",
        }
        _, _, extra_headers = self._okx_rest_context()
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def _fetch_okx_active_positions_count(self) -> Optional[int]:
        if not requests:
            return None
        try:
            base_url, _, _ = self._okx_rest_context()
            request_path = "/api/v5/account/positions"
            headers = self._build_okx_private_headers("GET", request_path)
            if not headers:
                return None
            response = requests.get(
                f"{base_url}{request_path}",
                headers=headers,
                timeout=5,
            )
            if response.status_code != 200:
                return None
            payload = response.json()
            rows = payload.get("data", []) if isinstance(payload, dict) else []
            if not isinstance(rows, list):
                return None
            count = 0
            for row in rows:
                if not isinstance(row, dict):
                    continue
                try:
                    if abs(float(row.get("pos") or 0.0)) > 0.0:
                        count += 1
                except (TypeError, ValueError):
                    continue
            return count
        except BEST_EFFORT_EXCEPTIONS:
            return None

    @staticmethod
    def _runtime_lock_path() -> Path:
        return ROOT / ".runtime.lock"

    @staticmethod
    def _parse_runtime_lock_pid(payload: Any) -> Optional[int]:
        if not isinstance(payload, dict):
            return None
        try:
            pid = int(payload.get("pid") or 0)
        except (TypeError, ValueError):
            return None
        return pid if pid > 0 else None

    @staticmethod
    def _is_process_alive(pid: Optional[int]) -> bool:
        if not pid or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except (OSError, ProcessLookupError):
            return False
        except PermissionError:
            return True
        return True

    def _runtime_lock_fallback(self) -> tuple[bool, Optional[int], float, float, int]:
        try:
            payload = json.loads(self._runtime_lock_path().read_text(encoding="utf-8"))
        except BEST_EFFORT_EXCEPTIONS:
            return False, None, 0.0, 0.0, 0

        pid = self._parse_runtime_lock_pid(payload)
        if not self._is_process_alive(pid):
            return False, None, 0.0, 0.0, 0

        heartbeat_ok, heartbeat_mins = self.check_heartbeat()
        if heartbeat_ok or (heartbeat_mins >= 0 and heartbeat_mins < MAX_NO_HEARTBEAT_MINUTES):
            return True, pid, 0.0, 0.0, 0
        return False, None, 0.0, 0.0, 0
    
    def check_bot_running(self) -> tuple[bool, Optional[int], float, float, int]:
        """Check if bot process is running."""
        if not PSUTIL_AVAILABLE:
            return True, None, 0.0, 0.0, 0  # Assume running if psutil not available
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'cpu_percent', 'create_time']):
                try:
                    cmdline = proc.info.get('cmdline') or []
                    cmdline_str = ' '.join(cmdline)
                    cmdline_lower = cmdline_str.lower()
                    
                    if any(marker in cmdline_lower for marker in self.bot_process_markers):
                        pid = proc.info.get('pid')
                        mem = proc.info.get('memory_info')
                        mem_mb = mem.rss / (1024 * 1024) if mem else 0
                        cpu = proc.cpu_percent(interval=0.1)
                        try:
                            created_at = float(proc.info.get("create_time") or 0.0)
                            uptime_seconds = max(0, int(time.time() - created_at)) if created_at > 0 else 0
                        except BEST_EFFORT_EXCEPTIONS:
                            uptime_seconds = 0
                        
                        return True, pid, mem_mb, cpu, uptime_seconds
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except BEST_EFFORT_EXCEPTIONS:
            fallback = self._runtime_lock_fallback()
            if fallback[0]:
                return fallback
        
        return self._runtime_lock_fallback()
    
    def check_heartbeat(self) -> tuple[bool, int]:
        """Check bot heartbeat file."""
        heartbeat_file = METRICS_DIR / "heartbeat.json"

        candidates: List[int] = []

        def _record_candidate(timestamp: Any) -> None:
            if not timestamp:
                return
            try:
                if isinstance(timestamp, (int, float)):
                    beat_time = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
                else:
                    beat_time = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
                now = datetime.now(beat_time.tzinfo)
                candidates.append(int((now - beat_time).total_seconds() / 60))
            except BEST_EFFORT_EXCEPTIONS:
                pass

        try:
            if heartbeat_file.exists():
                data = json.loads(heartbeat_file.read_text(encoding="utf-8"))
                _record_candidate(data.get("timestamp"))
        except BEST_EFFORT_EXCEPTIONS:
            pass

        try:
            if RUNTIME_STATUS_FILE.exists():
                data = json.loads(RUNTIME_STATUS_FILE.read_text(encoding="utf-8"))
                _record_candidate(data.get("last_heartbeat"))
        except BEST_EFFORT_EXCEPTIONS:
            pass

        if candidates:
            minutes_ago = min(candidates)
            return minutes_ago < MAX_NO_HEARTBEAT_MINUTES, minutes_ago

        return False, -1

    def check_exchange_connection(self) -> bool:
        """Check if exchange is reachable (public + private API)."""
        self._last_private_exchange_auth_error = None
        try:
            if not requests:
                return True
            base_url, _, extra_headers = self._okx_rest_context()
            
            # 1. Public API testi
            response = requests.get(
                f"{base_url}/api/v5/public/time",
                headers=extra_headers or None,
                timeout=5
            )
            if response.status_code != 200:
                return False
            
            # 2. [FIX-5.1] Private API key testi — 5 cycle'da bir (5 dk)
            self._api_key_check_counter += 1
            if self._api_key_check_counter >= 5:
                self._api_key_check_counter = 0
                if self._private_api_healthcheck_enabled():
                    if self._check_api_key_validity() is False:
                        return False
                else:
                    log.info("[HEALTH] OKX private API key testi paper/dry-run modda atlandı")
            return True
            
        except BEST_EFFORT_EXCEPTIONS:
            return False
    
    def _check_api_key_validity(self) -> bool:
        """[FIX-5] OKX API key geçerliliğini HMAC imzalı istek ile doğrula."""
        import os, base64, hmac, hashlib
        api_key, api_secret, passphrase = self._okx_private_credentials()
        base_url, use_testnet, extra_headers = self._okx_rest_context()
        if not api_key or not api_secret:
            # [FIX-5.2] API key yoksa warning log
            log.info("[HEALTH] OKX API Key yapılandırılmamış — private test atlandı")
            self._last_private_exchange_auth_error = "OKX private API healthcheck failed: credentials missing"
            return False
        try:
            from datetime import datetime, timezone
            ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z')
            msg = ts + "GET" + "/api/v5/account/balance"
            signature = base64.b64encode(
                hmac.HMAC(api_secret.encode(), msg.encode(), hashlib.sha256).digest()
            ).decode()
            headers = {
                "OK-ACCESS-KEY": api_key,
                "OK-ACCESS-SIGN": signature,
                "OK-ACCESS-TIMESTAMP": ts,
                "OK-ACCESS-PASSPHRASE": passphrase,
                "Content-Type": "application/json"
            }
            if extra_headers:
                headers.update(extra_headers)
            resp = requests.get(
                f"{base_url}/api/v5/account/balance",
                headers=headers, timeout=5
            )
            if resp.status_code != 200:
                mode = "TESTNET" if use_testnet else "LIVE"
                detail = ""
                try:
                    payload = resp.json()
                    detail = f" | code={payload.get('code')} msg={payload.get('msg')}"
                except BEST_EFFORT_EXCEPTIONS:
                    txt = (resp.text or "").strip().replace("\n", " ")
                    if txt:
                        detail = f" | body={txt[:180]}"
                log.warning(
                    "[HEALTH] OKX API key testi başarısız: mode=%s HTTP %s%s",
                    mode,
                    resp.status_code,
                    detail,
                )
                self._last_private_exchange_auth_error = (
                    f"OKX private API healthcheck failed: mode={mode} HTTP {resp.status_code}{detail}"
                )
                return False
            else:
                log.info("[HEALTH] OKX API key testi başarılı (mode=%s).", "TESTNET" if use_testnet else "LIVE")
        except BEST_EFFORT_EXCEPTIONS as exc:
            log.warning("[HEALTH] OKX private API healthcheck failed: %s", exc)
            self._last_private_exchange_auth_error = f"OKX private API healthcheck failed: {exc}"
            return False
            log.warning("[HEALTH] API key testi hatası: %s", exc)
    
    def check_last_trade(self) -> int:
        """Get minutes since last trade."""
        trade_log = get_trade_log_path()
        
        try:
            if trade_log.exists():
                data = json.loads(trade_log.read_text(encoding="utf-8"))
                trades = data if isinstance(data, list) else data.get("rows", [])
                
                if trades:
                    # Get most recent trade
                    latest = None
                    for trade in trades:
                        ts = trade.get("close_time") or trade.get("timestamp")
                        if ts:
                            trade_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                            if latest is None or trade_time > latest:
                                latest = trade_time
                    
                    if latest:
                        now = datetime.now(latest.tzinfo)
                        return int((now - latest).total_seconds() / 60)
        except BEST_EFFORT_EXCEPTIONS:
            pass
        
        return -1

    @staticmethod
    def _count_positions_from_payload(data: Any, *keys: str) -> tuple[bool, int]:
        if isinstance(data, list):
            return True, len(data)
        if not isinstance(data, dict):
            return False, 0

        for key in keys:
            if key not in data:
                continue
            positions = data.get(key)
            if isinstance(positions, list):
                return True, len(positions)
            if isinstance(positions, dict):
                return True, len(positions)
            return True, 0

        return False, 0

    @staticmethod
    def _local_paper_trading_enabled() -> bool:
        try:
            cfg_path = Path(__file__).resolve().parent / "config.json"
            if not cfg_path.exists():
                return False
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            paper_cfg = cfg.get("paper_trading", {}) if isinstance(cfg, dict) else {}
            return bool(paper_cfg.get("enabled", False)) if isinstance(paper_cfg, dict) else False
        except BEST_EFFORT_EXCEPTIONS:
            return False

    @staticmethod
    def _file_age_minutes(path: Path) -> Optional[int]:
        try:
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            return max(0, int((datetime.now(timezone.utc) - modified).total_seconds() / 60))
        except BEST_EFFORT_EXCEPTIONS:
            return None
    
    def check_active_positions(self) -> int:
        """Get number of active positions."""
        self._last_position_snapshot_stale = False
        self._last_position_snapshot_age_minutes = None
        self._last_position_snapshot_count = None
        self._last_exchange_positions_available = False

        candidate_files = []
        paper_enabled = self._local_paper_trading_enabled()
        live_metrics_snapshot = METRICS_DIR / "open_positions.json"

        if paper_enabled:
            candidate_files.append((METRICS_DIR / "paper_trading" / "paper_exchange_state.json", ("positions",)))
            paper_daily_files = sorted((METRICS_DIR / "paper_trading").glob("paper_daily_*.json"), reverse=True)
            if paper_daily_files:
                candidate_files.append((paper_daily_files[0], ("open_positions", "positions")))
        candidate_files.append((live_metrics_snapshot, ("positions", "open_positions")))

        zero_snapshot_found = False
        for path, keys in candidate_files:
            try:
                if not path.exists():
                    continue
                data = json.loads(path.read_text(encoding="utf-8"))
                found, count = self._count_positions_from_payload(data, *keys)
                if found:
                    if not paper_enabled and path == live_metrics_snapshot:
                        age_minutes = self._file_age_minutes(path)
                        self._last_position_snapshot_age_minutes = age_minutes
                        self._last_position_snapshot_count = count
                        if age_minutes is not None and age_minutes > MAX_INACTIVE_MINUTES:
                            self._last_position_snapshot_stale = True
                            if count == 0:
                                zero_snapshot_found = True
                            continue
                    if count > 0 or paper_enabled:
                        return count
                    zero_snapshot_found = True
            except BEST_EFFORT_EXCEPTIONS:
                continue

        if not paper_enabled:
            exchange_count = self._fetch_okx_active_positions_count()
            if exchange_count is not None:
                self._last_exchange_positions_available = True
                return exchange_count

        if self._last_position_snapshot_stale and self._last_position_snapshot_count is not None:
            return self._last_position_snapshot_count

        if zero_snapshot_found:
            return 0
        return 0

    def _open_trade_feature_diagnostics(self) -> Dict[str, Any]:
        path = _runtime_state_file("open_trades.json")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except BEST_EFFORT_EXCEPTIONS:
            return {
                "open_trade_count": 0,
                "missing_feature_window": [],
                "legacy_missing_feature_window": [],
                "online_learning_ineligible": [],
            }
        records = payload.get("open_trades") if isinstance(payload, dict) and isinstance(payload.get("open_trades"), dict) else {}
        missing: list[str] = []
        legacy_missing: list[str] = []
        ineligible: list[str] = []
        for symbol, record in records.items():
            if not isinstance(record, dict):
                continue
            symbol_text = str(symbol)
            has_feature_window = bool(record.get("online_feature_window") or record.get("feature_window"))
            feature_status = str(record.get("online_feature_window_status") or "").strip().lower()
            is_acknowledged_legacy_missing = (
                record.get("online_learning_eligible") is False
                and feature_status in {"missing_legacy_open", "legacy_missing_feature_window"}
            )
            if not has_feature_window:
                if is_acknowledged_legacy_missing:
                    legacy_missing.append(symbol_text)
                else:
                    missing.append(symbol_text)
            if record.get("online_learning_eligible") is False:
                ineligible.append(symbol_text)
        return {
            "open_trade_count": len(records),
            "missing_feature_window": missing,
            "legacy_missing_feature_window": legacy_missing,
            "online_learning_ineligible": ineligible,
        }

    @staticmethod
    def _normalize_process_cpu_percent(raw_cpu_pct: float) -> float:
        try:
            raw = max(0.0, float(raw_cpu_pct or 0.0))
        except (TypeError, ValueError):
            return 0.0
        cores = 1
        if PSUTIL_AVAILABLE:
            try:
                cores = int(psutil.cpu_count(logical=True) or 1)
            except BEST_EFFORT_EXCEPTIONS:
                cores = 1
        if cores <= 1:
            return round(raw, 3)
        return round(min(100.0, raw / float(cores)), 3)

    def _runtime_fail_safe_errors(self) -> List[str]:
        """Promote runtime danger flags to critical health errors."""
        try:
            if not RUNTIME_STATUS_FILE.exists():
                return []
            payload = json.loads(RUNTIME_STATUS_FILE.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return []
        except BEST_EFFORT_EXCEPTIONS:
            return []

        errors: List[str] = []
        flag_map = {
            "api_error": "Runtime API error flag",
            "stale_market_data": "Runtime stale market data flag",
            "market_data_stale": "Runtime stale market data flag",
            "balance_mismatch": "Runtime balance mismatch flag",
            "position_mismatch": "Runtime position mismatch flag",
            "unprotected_position": "Runtime unprotected position flag",
            "exchange_state_unavailable": "Runtime exchange state unavailable flag",
        }
        for key, message in flag_map.items():
            if bool(payload.get(key)):
                errors.append(message)

        try:
            api_error_rate = float(payload.get("api_error_rate") or 0.0)
            cfg_path = Path(__file__).resolve().parent / "config.json"
            threshold = 0.10
            if cfg_path.exists():
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                alerts = cfg.get("alerts", {}) if isinstance(cfg, dict) else {}
                if isinstance(alerts, dict):
                    threshold = float(alerts.get("api_error_rate_threshold", threshold) or threshold)
            if api_error_rate >= threshold > 0.0:
                errors.append(f"Runtime API error rate high: {api_error_rate:.4f} >= {threshold:.4f}")
        except BEST_EFFORT_EXCEPTIONS:
            pass

        reconciliation = payload.get("reconciliation")
        if isinstance(reconciliation, dict):
            if reconciliation.get("ok") is False:
                errors.append(f"Runtime reconciliation failed: {reconciliation.get('reason') or 'unknown'}")
            mismatched = int(reconciliation.get("mismatched") or 0)
            fixed = int(reconciliation.get("fixed") or 0)
            if mismatched > fixed:
                errors.append(f"Runtime position mismatch count: {mismatched - fixed}")

        watchdog = payload.get("trading_loop_watchdog")
        if isinstance(watchdog, dict):
            stale = bool(payload.get("trading_loop_stale")) or bool(watchdog.get("stale"))
            if not stale and bool(watchdog.get("active")):
                try:
                    idle_seconds = float(watchdog.get("idle_seconds") or 0.0)
                    max_idle_seconds = float(watchdog.get("max_idle_seconds") or 0.0)
                    if max_idle_seconds > 0.0 and idle_seconds > max_idle_seconds:
                        stale = True
                except (TypeError, ValueError):
                    stale = False
            if stale:
                reason = str(watchdog.get("reason") or "unknown")
                phase = str(watchdog.get("phase") or "unknown")
                try:
                    idle_seconds_int = int(float(watchdog.get("idle_seconds") or 0.0))
                except (TypeError, ValueError):
                    idle_seconds_int = 0
                try:
                    max_idle_seconds_int = int(float(watchdog.get("max_idle_seconds") or 0.0))
                except (TypeError, ValueError):
                    max_idle_seconds_int = 0
                errors.append(
                    "Runtime trading loop stale: "
                    f"reason={reason} phase={phase} idle={idle_seconds_int}s limit={max_idle_seconds_int}s"
                )

        return errors

    def _runtime_status_snapshot(self) -> Dict[str, Any]:
        try:
            if not RUNTIME_STATUS_FILE.exists():
                return {}
            payload = json.loads(RUNTIME_STATUS_FILE.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except BEST_EFFORT_EXCEPTIONS:
            return {}
    
    def perform_health_check(self) -> HealthStatus:
        """Perform complete health check."""
        errors = []
        warnings = []
        runtime_status = self._runtime_status_snapshot()
        runtime_phase = str(runtime_status.get("runtime_phase") or "").strip() or None
        last_runtime_error = str(runtime_status.get("last_runtime_error") or "").strip() or None
        last_runtime_error_ts = str(runtime_status.get("last_runtime_error_ts") or "").strip() or None
        
        # Check bot running
        bot_running, pid, mem_mb, cpu_pct, uptime_seconds = self.check_bot_running()
        if not bot_running:
            errors.append("Bot process not running!")
        if last_runtime_error:
            phase = runtime_phase or "unknown"
            errors.append(f"Runtime error ({phase}): {last_runtime_error}")
        
        # Check heartbeat
        heartbeat_ok, heartbeat_mins = self.check_heartbeat()
        if not heartbeat_ok and heartbeat_mins > 0:
            if bot_running and self._shadow_mode_enabled():
                warnings.append(
                    f"Heartbeat stale for {heartbeat_mins} minutes, but runtime process is active in shadow mode"
                )
            else:
                errors.append(f"No heartbeat for {heartbeat_mins} minutes")
        
        # Check exchange
        exchange_ok = self.check_exchange_connection()
        if self._last_private_exchange_auth_error:
            errors.append(self._last_private_exchange_auth_error)
        elif not exchange_ok:
            warnings.append("Exchange connection failed")
        
        errors.extend(self._runtime_fail_safe_errors())

        task_health: Dict[str, Any] = {}
        try:
            from runtime.task_health import summarize_required_task_health

            task_health, task_errors, task_warnings = summarize_required_task_health()
            errors.extend(task_errors)
            warnings.extend(task_warnings)
        except BEST_EFFORT_EXCEPTIONS as exc:
            warnings.append(f"background task health unavailable: {exc}")
        
        # Check memory
        memory_ok = mem_mb < MAX_MEMORY_MB
        if not memory_ok:
            warnings.append(f"High memory usage: {mem_mb:.0f}MB")
        
        # Check CPU
        cpu_health_pct = self._normalize_process_cpu_percent(cpu_pct)
        cpu_ok = cpu_health_pct < MAX_CPU_PERCENT
        if not cpu_ok:
            warnings.append(f"High CPU usage: {cpu_health_pct:.0f}% (raw process CPU: {cpu_pct:.0f}%)")
        
        # Check last trade
        last_trade_mins = self.check_last_trade()
        recent_activity = last_trade_mins < MAX_INACTIVE_MINUTES if last_trade_mins >= 0 else True
        if not recent_activity:
            warnings.append(f"No trade for {last_trade_mins} minutes")
        
        # [FAZA 9.3] Check secret rotation status
        try:
            from secret_env import check_all_secrets_rotation
            rotation_results = check_all_secrets_rotation()
            _expiring = [r for r in rotation_results if r.get("needs_rotation")]
            _warning = [r for r in rotation_results if r.get("warning") and not r.get("needs_rotation")]
            if _expiring:
                _keys = ", ".join(r["key"] for r in _expiring)
                errors.append(f"Secret rotation overdue: {_keys}")
                # Send Telegram alert for expired secrets
                try:
                    from telegram_notifier import send_message
                    _details = "\n".join(
                        f"  • <code>{r['key']}</code>: {r['age_days']} gün (max {r['max_age_days']})"
                        for r in _expiring
                    )
                    send_message(
                        f"🔑 <b>API Key Rotasyon GEREKLİ</b>\n\n"
                        f"Süresi dolmuş key'ler:\n{_details}\n\n"
                        f"⚠️ Lütfen key'leri yenileyin ve <code>mark_secret_rotated(key)</code> çağırın.",
                        parse_mode="HTML",
                    )
                except BEST_EFFORT_EXCEPTIONS:
                    pass
            elif _warning:
                _keys = ", ".join(r["key"] for r in _warning)
                warnings.append(f"Secret rotation approaching: {_keys}")
        except BEST_EFFORT_EXCEPTIONS:
            pass

        v2_operational: Dict[str, Any] = {}
        try:
            from decision.v2_operational_monitor import evaluate_current_v2_operational_health

            v2_operational = evaluate_current_v2_operational_health(ROOT)
            for alert in v2_operational.get("alerts", []):
                if not isinstance(alert, dict):
                    continue
                message = str(alert.get("message") or alert.get("code") or "v2 operational alert")
                if str(alert.get("severity") or "").lower() == "error":
                    errors.append(message)
                else:
                    warnings.append(message)
        except BEST_EFFORT_EXCEPTIONS as exc:
            warnings.append(f"v2 operational monitor unavailable: {exc}")

        # Check positions
        active_positions = self.check_active_positions()
        open_trade_features = self._open_trade_feature_diagnostics()
        if open_trade_features.get("missing_feature_window"):
            warnings.append(
                "Open trades missing online feature windows: "
                + ", ".join(open_trade_features.get("missing_feature_window", []))
            )
        if self._last_position_snapshot_stale and not self._last_exchange_positions_available:
            age = self._last_position_snapshot_age_minutes
            age_text = f" for {age} minutes" if age is not None else ""
            errors.append(f"Active position snapshot stale{age_text} and exchange positions unavailable")

        # Determine overall status
        if errors:
            overall_status = "critical"
        elif warnings:
            overall_status = "warning"
        else:
            overall_status = "healthy"
        
        return HealthStatus(
            timestamp=datetime.now(timezone.utc).isoformat(),
            overall_status=overall_status,
            bot_running=bot_running,
            exchange_connected=exchange_ok,
            memory_ok=memory_ok,
            cpu_ok=cpu_ok,
            recent_activity=recent_activity,
            memory_mb=mem_mb,
            cpu_percent=cpu_health_pct,
            cpu_percent_raw=cpu_pct,
            uptime_seconds=uptime_seconds,
            active_positions=active_positions,
            last_trade_minutes=last_trade_mins,
            errors=errors,
            warnings=warnings,
            v2_operational=v2_operational,
            open_trade_features=open_trade_features,
            runtime_phase=runtime_phase,
            last_runtime_error=last_runtime_error,
            last_runtime_error_ts=last_runtime_error_ts,
            task_health=task_health,
        )
    
    def save_status(self, status: HealthStatus):
        """Save health status to file."""
        try:
            atomic_write_json(HEALTH_LOG, status.to_dict())
            safe_append_jsonl(HEALTH_HISTORY, status.to_dict())
        except BEST_EFFORT_EXCEPTIONS:
            pass

    def _live_safety_watchdog_policy(self) -> tuple[bool, int]:
        """Return whether critical health failures must pause live-mainnet trading."""
        try:
            from core.config_loader import load_config
            from core.live_safety import evaluate_live_safety

            live_status = evaluate_live_safety(load_config())
            watchdog = live_status.get("independent_watchdog", {})
            if not isinstance(watchdog, dict):
                return False, 2
            enabled = bool(
                live_status.get("required")
                and watchdog.get("fail_safe_pause_on_critical", True)
            )
            threshold = int(watchdog.get("fail_safe_pause_after_consecutive_failures", 2) or 2)
            return enabled, max(1, threshold)
        except (ImportError, ModuleNotFoundError, TypeError, ValueError, RuntimeError):
            return False, 2

    def _apply_live_safety_fail_safe(self, status: HealthStatus) -> None:
        enabled, threshold = self._live_safety_watchdog_policy()
        if not enabled or self.consecutive_failures < threshold:
            return
        reason_detail = "; ".join((status.errors or status.warnings or [status.overall_status])[:3])
        reason = f"health_monitor live fail-safe pause: {reason_detail}"
        if reason == self._last_live_safety_pause_reason:
            return
        self._last_live_safety_pause_reason = reason
        try:
            import state_manager

            state_manager.request_pause(reason)
        except (ImportError, ModuleNotFoundError, AttributeError, OSError) as exc:
            log.error("[HEALTH] live fail-safe pause failed: %s", exc)
        try:
            from telegram_notifier import send_notification

            send_notification(reason, level="CRITICAL", _critical=True)
        except BEST_EFFORT_EXCEPTIONS:
            pass
    
    def send_alert(self, status: HealthStatus):
        """Send alert to Telegram."""
        # Rate limiting (1 alert per hour maximum)
        if hasattr(self, 'last_alert_time') and self.last_alert_time:
            if (datetime.now(timezone.utc) - self.last_alert_time).total_seconds() < 300:
                return
        
        try:
            emoji = "🚨" if status.overall_status == "critical" else "⚠️"
            
            message = f"""
{emoji} <b>BOT SAĞLIK UYARISI</b>

<b>Durum:</b> {status.overall_status.upper()}
<b>Bot Çalışıyor:</b> {'✅' if status.bot_running else '❌'}
<b>Exchange:</b> {'✅' if status.exchange_connected else '❌'}
<b>Memory:</b> {status.memory_mb:.0f}MB {'✅' if status.memory_ok else '⚠️'}
<b>CPU:</b> {status.cpu_percent:.0f}% {'✅' if status.cpu_ok else '⚠️'}
<b>Açık Pozisyon:</b> {status.active_positions}

<b>Hatalar:</b>
{chr(10).join(['❌ ' + e for e in (status.errors or [])] or ['Yok'])}

<b>Uyarılar:</b>
{chr(10).join(['⚠️ ' + w for w in (status.warnings or [])] or ['Yok'])}

⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S')}
"""
            from telegram_notifier import send_message

            if send_message(message, parse_mode="HTML", _critical=True):
                self.last_alert_time = datetime.now(timezone.utc)
                
        except BEST_EFFORT_EXCEPTIONS:
            pass
    
    def run_once(self) -> HealthStatus:
        """Run a single health check."""
        if not hasattr(self, 'consecutive_failures'):
            self.consecutive_failures = 0
            
        status = self.perform_health_check()
        self.save_status(status)
        
        # Send alert if not healthy
        if status.overall_status != "healthy":
            self.consecutive_failures += 1
            if self.consecutive_failures >= 2:  # Alert after 2 consecutive failures
                self.send_alert(status)
                self._apply_live_safety_fail_safe(status)
        else:
            self.consecutive_failures = 0
            self._last_live_safety_pause_reason = None
        
        return status
    
    def run(self, interval: int = CHECK_INTERVAL):
        """Run continuous health monitoring."""
        log.info("=" * 60)
        log.info("[HEALTH] HEALTH MONITOR STARTED")
        log.info("=" * 60)
        log.info("Check Interval: %ss", interval)
        log.info("Heartbeat Timeout: %sm", MAX_NO_HEARTBEAT_MINUTES)
        log.info("Inactivity Threshold: %sm", MAX_INACTIVE_MINUTES)
        log.info("=" * 60)
        
        try:
            while True:
                status = self.run_once()
                
                # Print status
                # Print status (ASCII only for Windows safety)
                status_symbol = "OK" if status.overall_status == "healthy" else "WARN" if status.overall_status == "warning" else "CRIT"
                log.info(
                    "[%s] %s | Bot: %s | Exch: %s | Mem: %.0fMB | Pos: %s",
                    status_symbol,
                    status.overall_status.upper(),
                    "ON" if status.bot_running else "OFF",
                    "CONN" if status.exchange_connected else "DISC",
                    status.memory_mb,
                    status.active_positions,
                )
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            log.info("Health monitor stopped")




# =============================================================================
# HEARTBEAT WRITER (for main bot to call)
# =============================================================================

def write_heartbeat():
    """Write heartbeat file. Call this from main bot periodically."""
    try:
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        heartbeat_file = METRICS_DIR / "heartbeat.json"
        timestamp = datetime.now(timezone.utc).isoformat()
        atomic_write_json(
            heartbeat_file,
            {
                "timestamp": timestamp,
                "pid": os.getpid(),
            },
        )
        try:
            from runtime_status import update_status

            update_status(last_heartbeat=timestamp)
        except BEST_EFFORT_EXCEPTIONS:
            pass
    except BEST_EFFORT_EXCEPTIONS:
        pass


def get_current_health() -> Dict:
    """Get current health status as dict."""
    monitor = HealthMonitor()
    status = monitor.perform_health_check()
    return status.to_dict()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║           🏥 BOT HEALTH MONITOR                              ║
║                                                              ║
║  Bot'un sağlık durumunu izler ve uyarı gönderir.             ║
║                                                              ║
║  İzlenen Metrikler:                                          ║
║  - Bot process çalışıyor mu?                                 ║
║  - Exchange bağlantısı                                       ║
║  - Memory/CPU kullanımı                                      ║
║  - Son trade zamanı                                          ║
║                                                              ║
║  Durdurmak için: Ctrl+C                                      ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    monitor = HealthMonitor()
    monitor.run()

