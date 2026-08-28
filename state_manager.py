# -*- coding: utf-8 -*-
"""
state_manager.py - Persistent State Management
==============================================

[2026-01-16 PROFESSIONAL FIX]

Bot'un durdurulma (pause), öldürülme (kill) ve diğer durumlarını
dosyaya kalıcı olarak kaydeder. Böylece bot yeniden başladığında
(auto-restart) kaldığı durumu hatırlar.

Kayıt Dosyası: data/bot_state.json
"""

from core.exceptions import BEST_EFFORT_EXCEPTIONS
import os
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Any
import logging
from atomic_io import atomic_write_json, file_lock, safe_append_jsonl, safe_read_json
from runtime_paths import BOT_STATE_FILE

try:
    from logger import get_logger
    log = get_logger(__name__)
except ImportError:
    log = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

STATE_FILE = BOT_STATE_FILE
PAUSE_FLAG_LEGACY = Path("runtime/PAUSE")  # For backward-compatibility
AUDIT_LOG_FILE = STATE_FILE.parent / "control_audit.jsonl"
CONTROL_COMMAND_LIMIT = 500


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

# =============================================================================
# PERSISTENT STATE MANAGER
# =============================================================================

import threading

class PersistentStateManager:
    """Manages persistent bot state (pause, kill, custom data)."""
    
    def __init__(self):
        self.state_file = STATE_FILE
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Any] = self._load()
        self._last_legacy_check: float = 0.0
        self._legacy_check_interval: float = 5.0  # saniyede 1 kereden fazla diske vurmasin
        self._lock = threading.RLock()

    @staticmethod
    def _fail_safe_state(reason: str) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return {
            "paused": True,
            "pause_reason": reason,
            "pause_time": now,
            "killed": True,
            "kill_reason": reason,
            "state_unavailable": True,
        }

    def _load(self) -> Dict[str, Any]:
        """Load state from JSON file."""
        try:
            missing = not self.state_file.exists()
            sentinel = object()
            payload = safe_read_json(self.state_file, default=sentinel)
            if payload is sentinel:
                if missing:
                    return {}
                reason = "state_file_read_failed"
                log.warning("State file load error: %s", reason)
                return self._fail_safe_state(reason)
            if isinstance(payload, dict):
                return payload
            reason = "state_file_invalid_payload"
            log.warning("State file load error: %s", reason)
            return self._fail_safe_state(reason)
        except BEST_EFFORT_EXCEPTIONS as e:
            log.warning("State file load error: {0}".format(e))
            return self._fail_safe_state(f"state_file_load_failed: {e}")

    def _save(self):
        """Save state synchronously via atomic writer."""
        with self._lock:
            if not atomic_write_json(self.state_file, dict(self._cache)):
                log.warning("State file save error: atomic write failed")
                self._cache.update(self._fail_safe_state("state_file_save_failed"))

    def append_audit_record(self, record: Dict[str, Any]) -> None:
        """Persist a control-plane audit event as JSONL."""
        payload = dict(record)
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
        audit_path = AUDIT_LOG_FILE
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if not safe_append_jsonl(audit_path, payload):
                log.warning("Audit log append failed: %s", audit_path)

    def _write_legacy_pause_flag(self, reason: str) -> None:
        PAUSE_FLAG_LEGACY.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(PAUSE_FLAG_LEGACY):
            with PAUSE_FLAG_LEGACY.open("w", encoding="utf-8") as handle:
                handle.write(reason or "paused")
                handle.write("\n")

    def _clear_legacy_pause_flag(self) -> None:
        with file_lock(PAUSE_FLAG_LEGACY):
            if PAUSE_FLAG_LEGACY.exists():
                PAUSE_FLAG_LEGACY.unlink()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._cache)

    def update_many(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            disk_state = self._load()
            if isinstance(disk_state, dict):
                self._cache.update(disk_state)
            self._cache.update(dict(patch))
            snapshot = dict(self._cache)
            if not atomic_write_json(self.state_file, snapshot):
                log.warning("State file save error: atomic write failed")
                self._cache.update(self._fail_safe_state("state_file_save_failed"))
                snapshot = dict(self._cache)
            return snapshot

    def set_paused(self, paused: bool, reason: str = ""):
        """Set pause state."""
        with self._lock:
            self._cache["paused"] = paused
            self._cache["pause_reason"] = reason
            self._cache["pause_time"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z") if paused else None
        
        # Legacy support
        if paused:
            try:
                self._write_legacy_pause_flag(reason)
            except BEST_EFFORT_EXCEPTIONS as exc:
                log.warning("Legacy pause flag write failed: %s", exc)
        else:
            try:
                self._clear_legacy_pause_flag()
            except BEST_EFFORT_EXCEPTIONS as exc:
                log.warning("Legacy pause flag clear failed: %s", exc)
            
        self._save()

    def is_paused(self) -> tuple[bool, str]:
        """Check if paused (with TTL cache for disk operations)."""
        now = time.time()
        # [FIX] Ölümcül Senkron Okuma (Disk) Blokajı 5 Saniyelik TTL zırhına alındı
        with self._lock:
            should_check_legacy = now - self._last_legacy_check > self._legacy_check_interval
            if should_check_legacy:
                self._last_legacy_check = now
        if should_check_legacy:
            disk_state = self._load()
            if isinstance(disk_state, dict):
                with self._lock:
                    self._cache.update(disk_state)
            # Check legacy file first (external tools might create it)
            if PAUSE_FLAG_LEGACY.exists():
                with self._lock:
                    paused_now = bool(self._cache.get("paused"))
                if not paused_now:
                    # Sync legacy to json
                    reason = "Legacy PAUSE file detected"
                    try:
                        with file_lock(PAUSE_FLAG_LEGACY):
                            reason = PAUSE_FLAG_LEGACY.read_text(encoding="utf-8")
                    except BEST_EFFORT_EXCEPTIONS as exc:
                        log.warning("Legacy pause flag read failed: %s", exc)
                    self.set_paused(True, reason)
        with self._lock:
            return self._cache.get("paused", False), self._cache.get("pause_reason", "")

    def set_key(self, key: str, value: Any):
        """Set custom key."""
        self.update_many({key: value})

    def get_key(self, key: str, default=None) -> Any:
        """Get custom key."""
        with self._lock:
            return self._cache.get(key, default)

# Global Instance
_STATE_MGR = PersistentStateManager()

# =============================================================================
# PUBLIC API (Backward Compatible)
# =============================================================================

def is_paused() -> bool:
    paused, _ = _STATE_MGR.is_paused()
    return paused

def get_pause_reason() -> str:
    _, reason = _STATE_MGR.is_paused()
    return reason

def request_pause(reason: str = "Manual pause"):
    _STATE_MGR.set_paused(True, reason)
    record_control_command("pause", reason=reason, source="state_manager")
    audit_control_action("pause", reason=reason, source="state_manager")
    log.info("⏸ Bot PAUSED: {0}".format(reason))

def clear_pause(reason: str = "Manual resume"):
    _STATE_MGR.set_paused(False)
    record_control_command("resume", reason=reason, source="state_manager")
    audit_control_action("resume", reason=reason, source="state_manager")
    log.info("▶️ Bot RESUMED")

def is_killed() -> bool:
    return _STATE_MGR.get_key("killed", False)

def request_kill(reason: str = "Kill switch"):
    _STATE_MGR.set_key("killed", True)
    _STATE_MGR.set_key("kill_reason", reason)
    record_control_command("kill", reason=reason, source="state_manager", fail_safe={"new_entries_blocked": True})
    audit_control_action("kill", reason=reason, source="state_manager")
    log.error("💀 Bot KILLED: {0}".format(reason))

def clear_kill(reason: str = "Kill cleared"):
    _STATE_MGR.set_key("killed", False)
    record_control_command("unkill", reason=reason, source="state_manager")
    audit_control_action("unkill", reason=reason, source="state_manager")
    log.info("Bot UNKILLED")


def pause_loop_sleep(sec: int = 10):
    """Wait while paused."""
    while is_paused():
        reason = get_pause_reason()
        log.info("⏳ Bot PAUSED ({0}). Waiting...".format(reason))
        time.sleep(sec)

def set_metric(key: str, value: Any):
    """Set custom metric (alias for set_key)."""
    _STATE_MGR.set_key(key, value)

def get_metric(key: str, default=None) -> Any:
    """Get custom metric (alias for get_key)."""
    return _STATE_MGR.get_key(key, default)


def get_state() -> Dict[str, Any]:
    """Return a shallow copy of the persisted state cache."""
    return _STATE_MGR.snapshot()


def update_state(patch: Dict[str, Any]) -> Dict[str, Any]:
    """Merge a patch into persisted state and return updated snapshot."""
    if not isinstance(patch, dict):
        raise TypeError("patch must be a dict")
    return _STATE_MGR.update_many(patch)


def _normalized_control_command_types(command_types: Any) -> set[str] | None:
    if command_types is None:
        return None
    if isinstance(command_types, str):
        values = [command_types]
    else:
        try:
            values = list(command_types)
        except TypeError:
            values = [command_types]
    normalized = {str(item or "").strip().lower().replace("-", "_") for item in values}
    return {item for item in normalized if item}


def _control_commands_from_state(state: Dict[str, Any]) -> list[Dict[str, Any]]:
    commands = state.get("control_commands", [])
    if not isinstance(commands, list):
        return []
    return [dict(item) for item in commands if isinstance(item, dict)]


def record_control_command(
    command_type: str,
    *,
    actor: str = "system",
    reason: str = "",
    source: str = "runtime",
    armed: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
    fail_safe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Append a persistent runtime control command for cross-process ack."""
    normalized_type = str(command_type or "").strip().lower().replace("-", "_") or "unknown"
    command = {
        "command_id": f"{normalized_type}-{uuid.uuid4().hex[:12]}",
        "command_type": normalized_type,
        "actor": str(actor or "system").strip() or "system",
        "reason": str(reason or "").strip(),
        "source": str(source or "runtime").strip() or "runtime",
        "armed": bool(armed),
        "requested_at": _utc_now_iso(),
        "runtime_ack_at": None,
        "status": "pending",
        "fail_safe": dict(fail_safe or {}),
        "metadata": dict(metadata or {}),
    }
    commands = _control_commands_from_state(get_state())
    commands.append(command)
    if len(commands) > CONTROL_COMMAND_LIMIT:
        commands = commands[-CONTROL_COMMAND_LIMIT:]
    update_state({"control_commands": commands})
    return dict(command)


def get_pending_control_commands(
    command_types: Any = None,
    *,
    limit: int = 100,
) -> list[Dict[str, Any]]:
    """Return pending persistent runtime control commands, oldest first."""
    allowed_types = _normalized_control_command_types(command_types)
    pending_statuses = {"pending", "queued", "requested"}
    commands = _control_commands_from_state(get_state())
    pending: list[Dict[str, Any]] = []
    for command in commands:
        command_type = str(command.get("command_type") or "").strip().lower().replace("-", "_")
        if allowed_types is not None and command_type not in allowed_types:
            continue
        status = str(command.get("status") or "pending").strip().lower()
        if status not in pending_statuses:
            continue
        pending.append(dict(command))
    if limit <= 0:
        return []
    return pending[:limit]


def ack_control_command(
    command_id: str,
    *,
    status: str = "runtime_acknowledged",
    handled_by: str = "runtime",
    fail_safe: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Mark a persistent control command as acknowledged by the runtime."""
    command_id = str(command_id or "").strip()
    if not command_id:
        return None
    commands = _control_commands_from_state(get_state())
    updated: Optional[Dict[str, Any]] = None
    for command in commands:
        if str(command.get("command_id") or "") != command_id:
            continue
        command["status"] = str(status or "runtime_acknowledged")
        command["runtime_ack_at"] = _utc_now_iso()
        command["handled_by"] = str(handled_by or "runtime")
        if fail_safe is not None:
            command["fail_safe"] = dict(fail_safe)
        updated = dict(command)
        break
    if updated is None:
        return None
    update_state({"control_commands": commands})
    return updated


def get_positions() -> Dict[str, Any]:
    """Return the canonical runtime positions snapshot."""
    positions = get_state().get("positions", {})
    return dict(positions) if isinstance(positions, dict) else {}


def set_positions(positions: Dict[str, Any]) -> Dict[str, Any]:
    """Persist the canonical runtime positions snapshot."""
    if not isinstance(positions, dict):
        raise TypeError("positions must be a dict")
    return update_state({"positions": dict(positions)})


def audit_control_action(
    action: str,
    *,
    reason: str = "",
    actor: str = "system",
    source: str = "runtime",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Persist a normalized audit record for control-plane actions."""
    record = {
        "action": str(action or "").strip() or "unknown",
        "reason": str(reason or "").strip(),
        "actor": str(actor or "system").strip() or "system",
        "source": str(source or "runtime").strip() or "runtime",
        "metadata": dict(metadata or {}),
    }
    _STATE_MGR.append_audit_record(record)
    return record


def get_audit_trail(limit: int = 200) -> list[Dict[str, Any]]:
    """Return the most recent audit events from disk."""
    if limit <= 0:
        return []
    if not AUDIT_LOG_FILE.exists():
        return []
    try:
        with file_lock(AUDIT_LOG_FILE):
            lines = AUDIT_LOG_FILE.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        log.warning("Audit log read failed: %s", exc)
        return []

    items: list[Dict[str, Any]] = []
    for line in lines[-limit:]:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            items.append(payload)
    return items

# =============================================================================
# LEGACY KILL SWITCH CLASS (Kep)
# =============================================================================
# Eski kodların kırılmaması için minimal DailyKillSwitch implementasyonu
# Yeni sistemde main_bot_async.py içindeki logic kullanılıyor
class DailyKillSwitch:
    def __init__(self, *args, **kwargs): pass
    def check_switch(self, balance): return False
