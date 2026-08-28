# -*- coding: utf-8 -*-
"""
auto_restart.py
===============

[2026-01-16 PROFESSIONAL FIX #74]

Profesyonel Bot Otomatik Yeniden Başlatma Sistemi.

Özellikler:
1. Bot çökerse otomatik yeniden başlat
2. Restart limiti YOK (sınırsız restart)
3. Telegram bildirimi
4. Detaylı hata loglaması
5. Graceful shutdown desteği

Kullanım:
    py -3.12 auto_restart.py

    VEYA

    from auto_restart import run_with_auto_restart
    run_with_auto_restart("module:runtime")
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.exceptions import BEST_EFFORT_EXCEPTIONS

log = logging.getLogger(__name__)

try:
    import msvcrt  # Windows single-instance file lock
except BEST_EFFORT_EXCEPTIONS:
    msvcrt = None

# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
RESTART_LOG = LOG_DIR / "restart_log.jsonl"
RUNNER_LOCK = LOG_DIR / "auto_restart.lock"

# Bot target to run
DEFAULT_BOT_SCRIPT = "main_bot_async.py"
DEFAULT_BOT_TARGET = "module:runtime"

# [FAZA 7.4] Wait time before restart (seconds) — increased from 10 to 30
RESTART_DELAY = 30

# [FAZA 7.4] Max crashes per hour before halting auto-restart — reduced from 10 to 5
MAX_RAPID_CRASHES = 5

# [FAZA 7.4] Per-hour crash rate limit
MAX_CRASHES_PER_HOUR = int(os.getenv("MAX_CRASHES_PER_HOUR", "5"))


def _build_child_env() -> dict[str, str]:
    """Force UTF-8 output so Windows supervisors do not stop draining stdout."""
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _resolve_python_exe() -> str:
    """Prefer local venv Python to avoid missing dependency crashes."""
    env_override = os.getenv("BOT_PYTHON_EXE", "").strip()
    if env_override:
        return env_override
    venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable or "python"


# Python executable
PYTHON_EXE = _resolve_python_exe()


# =============================================================================
# LOGGER
# =============================================================================


def log_message(message: str, level: str = "INFO"):
    """Log a message with timestamp."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] [{level}] {message}"
    log.info("%s", log_line)

    # Also write to log file
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / "auto_restart.log", "a", encoding="utf-8") as f:
            f.write(log_line + "\n")
    except BEST_EFFORT_EXCEPTIONS:
        pass


def log_restart_event(event: str, details: dict = None):
    """Log restart event to JSONL file."""
    import json

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, **(details or {})}
        with open(RESTART_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except BEST_EFFORT_EXCEPTIONS:
        pass


def acquire_runner_lock():
    """
    Ensure only one auto_restart runner is active.
    Returns an open file handle when lock is acquired, else None.
    """
    if msvcrt is None:
        return None
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        fh = open(RUNNER_LOCK, "a+", encoding="utf-8")
        fh.seek(0)
        try:
            # Lock first byte (non-blocking). Fails if another runner holds it.
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            fh.close()
            return None
        fh.seek(0)
        fh.truncate()
        fh.write(str(os.getpid()))
        fh.flush()
        return fh
    except BEST_EFFORT_EXCEPTIONS:
        return None


def release_runner_lock(lock_handle) -> None:
    if not lock_handle:
        return
    try:
        lock_handle.seek(0)
        msvcrt.locking(lock_handle.fileno(), msvcrt.LK_UNLCK, 1)
    except BEST_EFFORT_EXCEPTIONS:
        pass
    try:
        lock_handle.close()
    except BEST_EFFORT_EXCEPTIONS:
        pass


# =============================================================================
# TELEGRAM NOTIFICATION
# =============================================================================


def send_telegram_notification(message: str):
    """Send notification to Telegram."""
    try:
        from telegram_notifier import send_message

        return send_message(message, parse_mode="HTML", _critical=True)
    except BEST_EFFORT_EXCEPTIONS:
        return False


# =============================================================================
# BOT RUNNER WITH AUTO-RESTART
# =============================================================================


class BotRunner:
    """
    Runs the bot and automatically restarts on crash.

    Features:
    - [FAZA 7.4] Max 5 crashes per hour rate limiting
    - [FAZA 7.4] Min 30s delay between restarts
    - [FAZA 7.4] Crash traceback capture
    - Telegram notifications
    - Graceful shutdown support
    - Detailed logging
    """

    def __init__(self, target: str = DEFAULT_BOT_TARGET):
        self.target = str(target).strip() or DEFAULT_BOT_TARGET
        self.process: Optional[subprocess.Popen | asyncio.subprocess.Process] = None
        self.restart_count = 0
        self.start_time: Optional[datetime] = None
        self.should_stop = False
        self.last_exit_code = 0
        self._lock_handle = None
        # [FAZA 7.4] Per-hour crash timestamps for rate limiting
        self._crash_timestamps: list[float] = []
        # [FAZA 7.4] Last stderr/stdout for traceback capture
        self._last_output_lines: list[str] = []

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        log_message(f"Received signal {signum}, initiating graceful shutdown...")
        self.should_stop = True

        if self.process:
            try:
                self.process.terminate()
                if isinstance(self.process, subprocess.Popen):
                    self.process.wait(timeout=10)
            except BEST_EFFORT_EXCEPTIONS:
                try:
                    self.process.kill()
                except BEST_EFFORT_EXCEPTIONS:
                    pass

    def _build_command(self) -> list[str]:
        if self.target.startswith("module:"):
            module_name = self.target.split(":", 1)[1].strip()
            if not module_name:
                raise ValueError("module target requires a module name")
            return [PYTHON_EXE, "-m", module_name]
        script_path = Path(self.target)
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        return [PYTHON_EXE, str(script_path)]

    def start(self):
        """Start the bot process."""
        try:
            cmd = self._build_command()

            log_message(f"Starting bot: {' '.join(cmd)}")

            # Start process
            self.process = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=_build_child_env(),
            )

            self.start_time = datetime.now(timezone.utc)

            log_message(f"Bot started with PID: {self.process.pid}")
            log_restart_event(
                "BOT_STARTED",
                {
                    "pid": self.process.pid,
                    "restart_count": self.restart_count,
                },
            )

            return True

        except BEST_EFFORT_EXCEPTIONS as e:
            log_message(f"Failed to start bot: {e}", "ERROR")
            return False

    async def start_async(self) -> bool:
        """Start the bot process with non-blocking asyncio subprocess I/O."""
        try:
            cmd = self._build_command()

            log_message(f"Starting bot: {' '.join(cmd)}")

            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(ROOT),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=_build_child_env(),
            )

            self.start_time = datetime.now(timezone.utc)

            log_message(f"Bot started with PID: {self.process.pid}")
            log_restart_event(
                "BOT_STARTED",
                {
                    "pid": self.process.pid,
                    "restart_count": self.restart_count,
                },
            )
            return True
        except BEST_EFFORT_EXCEPTIONS as e:
            log_message(f"Failed to start bot: {e}", "ERROR")
            return False

    async def _drain_async_output(self) -> None:
        if not self.process or not getattr(self.process, "stdout", None):
            return
        stdout = self.process.stdout
        self._last_output_lines = []
        while True:
            line = await stdout.readline()
            if not line:
                break
            if isinstance(line, bytes):
                text = line.decode("utf-8", errors="replace")
            else:
                text = str(line)
            stripped = text.rstrip()
            log.info("%s", stripped)
            self._last_output_lines.append(stripped)
            if len(self._last_output_lines) > 50:
                self._last_output_lines.pop(0)

    async def wait_for_exit_async(self) -> int:
        """Wait for an asyncio subprocess exit without blocking stdout reads."""
        if not self.process:
            return -1
        drain_task = asyncio.create_task(self._drain_async_output())
        try:
            exit_code = await self.process.wait()
            await drain_task
            self.last_exit_code = int(exit_code if exit_code is not None else getattr(self.process, "returncode", -1))
            return self.last_exit_code
        except BEST_EFFORT_EXCEPTIONS as e:
            drain_task.cancel()
            log_message(f"Wait error: {e}", "ERROR")
            return -1

    async def run_once_async(self) -> int:
        """Start one child process and wait for exit through the async supervisor path."""
        if not await self.start_async():
            return -1
        return await self.wait_for_exit_async()

    def run_with_output(self):
        """Run and stream output, capturing last lines for traceback."""
        if not self.process or not self.process.stdout:
            return

        # [FAZA 7.4] Reset output buffer for traceback capture
        self._last_output_lines = []
        try:
            for line in iter(self.process.stdout.readline, ""):
                if line:
                    stripped = line.rstrip()
                    # Stream bot output
                    log.info("%s", stripped)
                    # [FAZA 7.4] Keep last 50 lines for traceback capture
                    self._last_output_lines.append(stripped)
                    if len(self._last_output_lines) > 50:
                        self._last_output_lines.pop(0)

                if self.process.poll() is not None:
                    break

        except OSError as e:
            log_message(f"Output stream OS error: {e}", "WARNING")
        except BEST_EFFORT_EXCEPTIONS as e:
            log_message(f"Output stream error: {e}", "WARNING")

    def wait_for_exit(self) -> int:
        """Wait for bot to exit and return exit code."""
        if not self.process:
            return -1

        try:
            if not isinstance(self.process, subprocess.Popen):
                return asyncio.run(self.wait_for_exit_async())
            self.run_with_output()
            self.process.wait()
            self.last_exit_code = self.process.returncode
            return self.last_exit_code
        except BEST_EFFORT_EXCEPTIONS as e:
            log_message(f"Wait error: {e}", "ERROR")
            return -1

    def run_forever(self):
        """Run bot with unlimited auto-restart."""
        if msvcrt is not None:
            self._lock_handle = acquire_runner_lock()
            if self._lock_handle is None:
                log_message(
                    "Another auto-restart runner is already active; exiting this instance.",
                    "ERROR",
                )
                return

        log_message("=" * 60)
        log_message("AUTO-RESTART BOT RUNNER STARTED")
        log_message("=" * 60)
        log_message(f"Target: {self.target}")
        log_message(f"Python: {PYTHON_EXE}")
        log_message(f"Restart Delay: {RESTART_DELAY}s")
        log_message("Press Ctrl+C for graceful shutdown")
        log_message("=" * 60)

        # Send startup notification
        send_telegram_notification(
            "🤖 <b>Auto-Restart Runner Başlatıldı</b>\n\n"
            f"Target: <code>{self.target}</code>\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S')}"
        )

        rapid_crash_count = 0
        while not self.should_stop:
            log_message(f"Starting bot process (Attempt {self.restart_count + 1})...")
            start_time_epoch = time.time()
            # Start the bot and wait through the async supervisor path.
            exit_code = asyncio.run(self.run_once_async())
            if exit_code == -1:
                log_message("Failed to start, retrying in 30 seconds...", "ERROR")
                time.sleep(30)  # A fixed delay for initial startup failure
                continue

            # Check if we should stop
            if self.should_stop:
                log_message("Shutdown requested, not restarting")
                break

            # Calculate uptime
            uptime = "Unknown"
            duration = 0
            if self.start_time:
                delta = datetime.now(timezone.utc) - self.start_time
                hours, remainder = divmod(int(delta.total_seconds()), 3600)
                minutes, seconds = divmod(remainder, 60)
                uptime = f"{hours}h {minutes}m {seconds}s"
                duration = time.time() - start_time_epoch  # Use epoch time for more accurate duration

            # [FAZA 7.4] Extract traceback from last output lines
            traceback_text = ""
            if self._last_output_lines:
                # Find traceback start
                tb_start = -1
                for i, line in enumerate(self._last_output_lines):
                    if "Traceback" in line or "Error" in line or "Exception" in line:
                        tb_start = i
                        break
                if tb_start >= 0:
                    traceback_text = "\n".join(self._last_output_lines[tb_start:])
                else:
                    traceback_text = "\n".join(self._last_output_lines[-10:])

            # If the bot crashed explicitly or ran for very short time
            delay = RESTART_DELAY
            if duration < 60:  # Define "rapid crash" as less than 60 seconds uptime
                rapid_crash_count += 1
                delay = RESTART_DELAY * (2 ** (rapid_crash_count - 1))
                log_message(
                    f"Rapid crash detected. Rapid crash count: {rapid_crash_count}. Next delay: {delay}s", "WARNING"
                )
            else:
                rapid_crash_count = 0

            if rapid_crash_count >= MAX_RAPID_CRASHES:
                err_msg = f"Bot crashed {MAX_RAPID_CRASHES} times in a row rapidly. Halting auto-restart."
                log_message(err_msg, "ERROR")
                send_telegram_notification(
                    f"🚨 <b>CRITICAL: Auto-Restart Durduruldu</b>\n\n"
                    f"{err_msg}\n\n"
                    f"<b>Son Traceback:</b>\n<pre>{traceback_text[:500]}</pre>"
                )
                break

            # [FAZA 7.4] Per-hour crash rate limiting
            now_ts = time.time()
            self._crash_timestamps.append(now_ts)
            # Remove timestamps older than 1 hour
            one_hour_ago = now_ts - 3600
            self._crash_timestamps = [t for t in self._crash_timestamps if t > one_hour_ago]
            if len(self._crash_timestamps) >= MAX_CRASHES_PER_HOUR:
                err_msg = f"Bot crashed {MAX_CRASHES_PER_HOUR} times in the last hour. Halting auto-restart."
                log_message(err_msg, "ERROR")
                send_telegram_notification(
                    f"🚨 <b>CRITICAL: Saatlik Crash Limiti Aşıldı</b>\n\n"
                    f"{err_msg}\n\n"
                    f"<b>Son Traceback:</b>\n<pre>{traceback_text[:500]}</pre>"
                )
                break

            self.restart_count += 1
            log_message(f"Bot exited with code {exit_code} after {uptime}", "WARNING")
            log_restart_event(
                "BOT_CRASHED",
                {
                    "exit_code": exit_code,
                    "uptime": uptime,
                    "restart_count": self.restart_count,
                    "traceback": traceback_text[:2000] if traceback_text else None,
                },
            )

            # Send Telegram notification with traceback
            emoji = "⚠️" if exit_code != 0 else "ℹ️"
            tb_snippet = f"\n\n<b>Traceback:</b>\n<pre>{traceback_text[:300]}</pre>" if traceback_text else ""
            send_telegram_notification(
                f"{emoji} <b>Bot Çöktü - Yeniden Başlatılıyor</b>\n\n"
                f"Exit Code: <code>{exit_code}</code>\n"
                f"Uptime: {uptime}\n"
                f"Restart #: {self.restart_count}\n"
                f"Saatlik Crash: {len(self._crash_timestamps)}/{MAX_CRASHES_PER_HOUR}\n"
                f"⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S')}"
                f"{tb_snippet}"
            )

            # Wait specific delay before restart, checking shutdown flag
            log_message(f"Waiting {delay}s before restart...")
            wait_time = 0
            while wait_time < delay and not self.should_stop:
                time.sleep(1)
                wait_time += 1

        # Final message
        log_message("Auto-restart runner stopped")
        log_restart_event(
            "RUNNER_STOPPED",
            {
                "total_restarts": self.restart_count,
            },
        )
        release_runner_lock(self._lock_handle)
        self._lock_handle = None

        send_telegram_notification(
            "🛑 <b>Auto-Restart Runner Durduruldu</b>\n\n"
            f"Toplam Restart: {self.restart_count}\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S')}"
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def run_with_auto_restart(target: str = DEFAULT_BOT_TARGET):
    """Run a script or module target with unlimited auto-restart."""
    runner = BotRunner(target)
    runner.run_forever()


def run_runtime_with_auto_restart() -> None:
    """Run the official runtime boot target with unlimited auto-restart."""
    run_with_auto_restart(DEFAULT_BOT_TARGET)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    target = DEFAULT_BOT_TARGET
    if len(sys.argv) > 1:
        target = sys.argv[1]

    print(
        "\n"
        "==============================================================\n"
        "                 AUTO-RESTART BOT RUNNER\n"
        "==============================================================\n"
        "Bot cokerse otomatik olarak yeniden baslatilir.\n"
        "Restart limiti YOK - Sinirsiz restart.\n"
        "Graceful shutdown icin: Ctrl+C\n"
        "==============================================================\n"
    )

    run_with_auto_restart(target)
