"""
atomic_io.py
-------------

This module provides simple atomic file write helpers with a cooperative
file‑based locking mechanism.  When multiple parts of the bot attempt to
write to the same JSON file concurrently, naive writes can lead to
corruption or lost updates.  To mitigate this, `safe_write_json`
performs writes through a temporary file and atomically replaces the
target on completion.  It also obtains a `.lock` file to serialize
writers across threads and processes.  If the lock is already held by
another writer, calls will block briefly and retry until the lock is
released.  The lock is implemented using ``os.O_EXCL`` so it is
portable across platforms and does not rely on external packages.

Typical usage::

    from atomic_io import safe_write_json, file_lock
    safe_write_json(Path("metrics/metrics.json"), data)
    
    # Or using context manager:
    with file_lock(Path("myfile.json")):
        # do atomic operations
        pass

The helper functions catch and suppress all exceptions internally;
callers should not rely on them raising exceptions on failure.  If
atomic writes fail for any reason, the original file is left
untouched.

CHANGELOG:
- v1.0: Initial version with safe_write_json
- v1.1: Added file_lock context manager
- v1.2: Added safe_append_jsonl function
- v1.3: Added safe_read_json function
"""

from __future__ import annotations


from core.exceptions import BEST_EFFORT_EXCEPTIONS
import atexit
import asyncio
import ctypes
import json
import logging
import os
import queue
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Tuple, Union

log = logging.getLogger(__name__)
_WARNED_PATH_ERRORS: set[str] = set()
_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()
_LOCK_STALE_AFTER_SECONDS = 30 * 60
_PROJECT_ROOT = Path(__file__).resolve().parent
_PROTECTED_RUNTIME_DIRS = (
    _PROJECT_ROOT / "state",
    _PROJECT_ROOT / "metrics",
    _PROJECT_ROOT / "logs",
)
_ASYNC_IO_QUEUE: queue.Queue[tuple[str, Path, Any] | object] = queue.Queue(
    maxsize=int(os.getenv("ATOMIC_IO_QUEUE_MAXSIZE", "10000"))
)
_ASYNC_IO_STOP = object()
_ASYNC_IO_WORKER: threading.Thread | None = None
_ASYNC_IO_WORKER_LOCK = threading.Lock()
_WIN_ERROR_ACCESS_DENIED = 5
_WIN_WAIT_TIMEOUT = 0x00000102


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except (OSError, ValueError):
        return False


def _assert_test_write_allowed(path: Path) -> None:
    if str(os.getenv("TESTING", "")).strip().lower() not in {"1", "true", "yes", "on"}:
        return
    target = path.resolve(strict=False)
    runtime_root_raw = os.getenv("ATB_RUNTIME_ROOT", "")
    runtime_root = Path(runtime_root_raw).resolve(strict=False) if runtime_root_raw else None
    if runtime_root is not None and _path_is_relative_to(target, runtime_root):
        return
    if any(_path_is_relative_to(target, protected) for protected in _PROTECTED_RUNTIME_DIRS):
        raise RuntimeError(f"Refusing test write to protected runtime path: {target}")


def _warn_once(path: Union[Path, str], message: str, exc: Exception | None = None) -> None:
    key = f"{Path(path)}::{message}"
    if key in _WARNED_PATH_ERRORS:
        return
    _WARNED_PATH_ERRORS.add(key)
    if exc is None:
        log.warning("%s: %s", message, path)
    else:
        log.warning("%s: %s (%s)", message, path, exc)


def _path_lock(path: Union[Path, str]) -> threading.RLock:
    try:
        normalized = _normalize_path(path)
    except BEST_EFFORT_EXCEPTIONS:
        normalized = Path(path).expanduser()
    key = os.path.normcase(os.fspath(normalized))
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _THREAD_LOCKS[key] = lock
        return lock


def _is_windows_process_alive(pid: int) -> bool:
    win_dll_factory = getattr(ctypes, "WinDLL", None)
    if win_dll_factory is None:
        return False
    try:
        kernel32 = win_dll_factory("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        open_process.restype = ctypes.c_void_p
        wait_for_single_object = kernel32.WaitForSingleObject
        wait_for_single_object.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        wait_for_single_object.restype = ctypes.c_ulong
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [ctypes.c_void_p]
        close_handle.restype = ctypes.c_int
        process_query_limited_information = 0x1000
        synchronize = 0x00100000
        handle = open_process(process_query_limited_information | synchronize, 0, int(pid))
        if not handle:
            return ctypes.get_last_error() == _WIN_ERROR_ACCESS_DENIED
        try:
            return wait_for_single_object(handle, 0) == _WIN_WAIT_TIMEOUT
        finally:
            close_handle(handle)
    except (AttributeError, OSError, TypeError, ValueError):
        return False


def _is_process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        return _is_windows_process_alive(pid)
    try:
        os.kill(pid, 0)
    except SystemError:
        return False
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        winerror = getattr(exc, "winerror", None)
        if winerror in {87, 1168}:
            return False
        return False
    return True


def _read_lock_pid(lock_path: Path) -> int | None:
    try:
        raw = lock_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw:
        return None
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            raw_pid = payload.get("pid")
        else:
            raw_pid = payload
    except json.JSONDecodeError:
        raw_pid = raw
    try:
        return int(raw_pid)
    except (TypeError, ValueError):
        return None


def _lock_file_is_stale(lock_path: Path) -> bool:
    pid = _read_lock_pid(lock_path)
    if pid is not None:
        return not _is_process_alive(pid)
    try:
        age_seconds = time.time() - lock_path.stat().st_mtime
    except OSError:
        return False
    return age_seconds >= _LOCK_STALE_AFTER_SECONDS


def _remove_stale_lock(lock_path: Path) -> bool:
    if not _lock_file_is_stale(lock_path):
        return False
    try:
        lock_path.unlink()
        return True
    except OSError as exc:
        _warn_once(lock_path, "atomic_io stale lock cleanup failed", exc)
        return False


def _acquire_lock(lock_path: Path, timeout: float = 5.0, poll_interval: float = 0.05) -> Tuple[int, Path] | None:
    """
    Acquire an exclusive file lock.  The lock is represented by creating
    a companion ``.lock`` file next to the target.  If the lock already
    exists, this function waits until it is released or the timeout is
    reached.  Returns the file descriptor and the lock file path on
    success, or ``None`` if the lock could not be acquired.

    Args:
        lock_path: Path to the lock file (should end with ``.lock``).
        timeout: Maximum seconds to wait for the lock.
        poll_interval: Seconds between lock acquisition attempts.
    """
    end_time = time.monotonic() + max(0.0, timeout)
    while True:
        try:
            # Use os.O_EXCL to ensure we fail if the file already exists
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            # Write PID for debugging (optional)
            try:
                os.write(fd, str(os.getpid()).encode("utf-8"))
            except OSError as exc:
                _warn_once(lock_path, "atomic_io lock pid write failed", exc)
            return fd, lock_path
        except FileExistsError:
            if _remove_stale_lock(lock_path):
                continue
            # lock is held by another process
            if time.monotonic() >= end_time:
                return None
            time.sleep(poll_interval)
        except OSError as exc:
            _warn_once(lock_path, "atomic_io lock acquisition failed", exc)
            return None


def _release_lock(fd: int, lock_path: Path) -> None:
    """
    Release a previously acquired file lock.  Closes the file descriptor
    and removes the lock file.  Exceptions are suppressed.

    On Windows, PermissionError can occur when another process still has a
    handle open.  We retry the unlink a few times with short delays.
    """
    try:
        os.close(fd)
    except OSError as exc:
        _warn_once(lock_path, "atomic_io fd close failed", exc)
    for attempt in range(3):
        try:
            os.unlink(str(lock_path))
            return
        except FileNotFoundError:
            return
        except PermissionError:
            # Windows: another process may still hold a handle briefly
            if attempt < 2:
                time.sleep(0.05 * (attempt + 1))
        except OSError as exc:
            _warn_once(lock_path, "atomic_io lock unlink failed", exc)
            return


class file_lock:
    """
    Context manager for file locking. Supports both synchronous 'with' and asynchronous 'async with'.
    """
    def __init__(self, path: Union[Path, str], timeout: float = 5.0):
        self.path = Path(path)
        self.timeout = timeout
        self.fd = None
        self.lp = None
        self.thread_lock = _path_lock(self.path)
        self.async_lock = None

    def __enter__(self) -> file_lock:
        try:
            import asyncio
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                f"Blocking file_lock synchronous context manager cannot run inside an event loop for {self.path}. "
                "Use 'async with file_lock(...)' or the async_safe_* helpers."
            )

        acquired = self.thread_lock.acquire(timeout=self.timeout)
        if not acquired:
            raise TimeoutError(f"Could not acquire thread lock for {self.path} within {self.timeout}s")
            
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        lock = _acquire_lock(lock_path, timeout=self.timeout)
        if lock is None:
            self.thread_lock.release()
            raise TimeoutError(f"Could not acquire lock for {self.path} within {self.timeout}s")
        self.fd, self.lp = lock
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.fd is not None and self.lp is not None:
                _release_lock(self.fd, self.lp)
        finally:
            self.thread_lock.release()

    async def __aenter__(self) -> file_lock:
        import asyncio
        self.async_lock = _async_path_lock(self.path)
        
        try:
            await asyncio.wait_for(self.async_lock.acquire(), timeout=self.timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Could not acquire async lock for {self.path} within {self.timeout}s")
        
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        end_time = time.monotonic() + max(0.0, self.timeout)
        poll_interval = 0.05
        
        while True:
            try:
                self.fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                try:
                    os.write(self.fd, str(os.getpid()).encode("utf-8"))
                except OSError as exc:
                    _warn_once(lock_path, "atomic_io async lock pid write failed", exc)
                self.lp = lock_path
                return self
            except FileExistsError:
                stale_cleaned = await asyncio.to_thread(_remove_stale_lock, lock_path)
                if stale_cleaned:
                    continue
                if time.monotonic() >= end_time:
                    self.async_lock.release()
                    raise TimeoutError(f"Could not acquire lock for {self.path} within {self.timeout}s")
                await asyncio.sleep(poll_interval)
            except OSError as exc:
                self.async_lock.release()
                _warn_once(lock_path, "atomic_io async lock acquisition failed", exc)
                raise exc

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        import asyncio
        try:
            if self.fd is not None and self.lp is not None:
                await asyncio.to_thread(_release_lock, self.fd, self.lp)
        finally:
            if self.async_lock is not None:
                self.async_lock.release()



def _normalize_path(path: Union[Path, str]) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve(strict=False)
    return candidate


def _append_jsonl_direct(path: Path, record: Any) -> None:
    payload = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    last_exc: OSError | None = None
    for attempt in range(3):
        try:
            with path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(payload)
            return
        except OSError as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(0.05 * (attempt + 1))
    if last_exc is not None:
        raise last_exc


def _async_io_worker_loop() -> None:
    while True:
        task = _ASYNC_IO_QUEUE.get()
        try:
            if task is _ASYNC_IO_STOP:
                return
            op, path, payload = task  # type: ignore[misc]
            if op == "append_jsonl":
                safe_append_jsonl(path, payload)
        except BEST_EFFORT_EXCEPTIONS as exc:
            log.warning("atomic_io async worker failed: %s", exc)
        finally:
            _ASYNC_IO_QUEUE.task_done()


def _ensure_async_io_worker() -> None:
    global _ASYNC_IO_WORKER
    with _ASYNC_IO_WORKER_LOCK:
        if _ASYNC_IO_WORKER is not None and _ASYNC_IO_WORKER.is_alive():
            return
        _ASYNC_IO_WORKER = threading.Thread(
            target=_async_io_worker_loop,
            name="atomic-io-worker",
            daemon=True,
        )
        _ASYNC_IO_WORKER.start()


def enqueue_append_jsonl(path: Union[Path, str], record: Any) -> bool:
    """Queue a JSONL append on a background worker without blocking the caller."""
    try:
        target = _normalize_path(path)
        _assert_test_write_allowed(target)
        _ensure_async_io_worker()
        _ASYNC_IO_QUEUE.put_nowait(("append_jsonl", target, record))
        return True
    except queue.Full as exc:
        _warn_once(path, "atomic_io async jsonl queue full", exc)
        return False
    except BEST_EFFORT_EXCEPTIONS as exc:
        _warn_once(path, "atomic_io async jsonl enqueue failed", exc)
        return False


def _inside_running_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def flush_async_io_queue(timeout: float = 5.0) -> bool:
    """Wait until queued async I/O writes have drained."""
    deadline = time.monotonic() + max(0.0, float(timeout))
    while _ASYNC_IO_QUEUE.unfinished_tasks:
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True


atexit.register(flush_async_io_queue)


def _write_json_direct(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    modes = ["r+", "w"] if path.exists() else ["w"]
    last_exc: OSError | None = None
    for mode in modes:
        for attempt in range(3):
            try:
                with path.open(mode, encoding="utf-8", newline="\n") as f:
                    if mode == "r+":
                        f.seek(0)
                    json.dump(data, f, ensure_ascii=False, indent=2, default=str)
                    f.truncate()
                return
            except OSError as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(0.05 * (attempt + 1))
    if last_exc is not None:
        raise last_exc


def safe_write_json(path: Union[Path, str], data: Any, *, warn_on_failure: bool = True) -> bool:
    """
    Atomically write JSON data to the given path.  This function
    acquires a `.lock` next to the target file to prevent concurrent
    writers from clobbering each other.  The data is first written to
    a temporary file and then replaced over the target using
    ``os.replace``, which is atomic on POSIX and Windows.  If locking
    fails, or writing fails, the function falls back silently.

    Args:
        path: Destination file path.
        data: A JSON‑serialisable object to write.
        warn_on_failure: Emit warning logs when atomic write degrades or fails.
        
    Returns:
        True if write succeeded, False otherwise.
    """
    path = _normalize_path(path)
    _assert_test_write_allowed(path)
    tmp_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        thread_lock = _path_lock(path)
        with thread_lock:
            lock_path = path.with_suffix(path.suffix + ".lock")
            lock = _acquire_lock(lock_path)
            if lock is None:
                return False
            fd, lp = lock
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp_name = f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
                tmp_path = path.with_name(tmp_name)
                _write_json_direct(tmp_path, data)
                for _attempt in range(10):
                    try:
                        os.replace(str(tmp_path), str(path))
                        break
                    except OSError:
                        if _attempt < 9:
                            time.sleep(0.05 * (_attempt + 1))
                        else:
                            _write_json_direct(path, data)
                            break
                return True
            finally:
                try:
                    if tmp_path is not None and tmp_path.exists():
                        tmp_path.unlink()
                except OSError as exc:
                    _warn_once(tmp_path, "atomic_io temp cleanup failed", exc)
                _release_lock(fd, lp)
    except BEST_EFFORT_EXCEPTIONS as exc:
        try:
            _write_json_direct(path, data)
            if warn_on_failure:
                _warn_once(path, "atomic_io json write degraded to direct write", exc)
            return True
        except BEST_EFFORT_EXCEPTIONS as fallback_exc:
            if warn_on_failure:
                _warn_once(path, "atomic_io json write failed", fallback_exc)
            return False


def atomic_write_json(path: Union[Path, str], data: Any, indent: int = 2, *, warn_on_failure: bool = True) -> bool:
    """Compatibility helper used across modules for atomic JSON writes."""
    # `safe_write_json` already writes with indent=2 and ensure_ascii=False.
    _ = indent
    return safe_write_json(path, data, warn_on_failure=warn_on_failure)


def atomic_write_json_under_existing_lock(path: Union[Path, str], data: Any, *, warn_on_failure: bool = True) -> bool:
    """Atomically replace JSON when the caller already owns ``file_lock(path)``."""
    path = _normalize_path(path)
    _assert_test_write_allowed(path)
    tmp_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_name = f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        tmp_path = path.with_name(tmp_name)
        _write_json_direct(tmp_path, data)
        os.replace(str(tmp_path), str(path))
        return True
    except BEST_EFFORT_EXCEPTIONS as exc:
        if warn_on_failure:
            _warn_once(path, "atomic_io locked json write failed", exc)
        return False
    finally:
        try:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()
        except OSError as exc:
            _warn_once(tmp_path, "atomic_io locked temp cleanup failed", exc)


def safe_write_text(
    path: Union[Path, str],
    text: str,
    *,
    encoding: str = "utf-8",
    newline: str | None = None,
    warn_on_failure: bool = True,
) -> bool:
    """Atomically write text with the same lock discipline used for JSON."""
    path = _normalize_path(path)
    _assert_test_write_allowed(path)
    tmp_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        thread_lock = _path_lock(path)
        with thread_lock:
            lock_path = path.with_suffix(path.suffix + ".lock")
            lock = _acquire_lock(lock_path)
            if lock is None:
                return False
            fd, lp = lock
            try:
                tmp_name = f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
                tmp_path = path.with_name(tmp_name)
                tmp_path.write_text(text, encoding=encoding, newline=newline)
                os.replace(str(tmp_path), str(path))
                return True
            finally:
                try:
                    if tmp_path is not None and tmp_path.exists():
                        tmp_path.unlink()
                except OSError as exc:
                    _warn_once(tmp_path, "atomic_io temp text cleanup failed", exc)
                _release_lock(fd, lp)
    except BEST_EFFORT_EXCEPTIONS as exc:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding=encoding, newline=newline)
            if warn_on_failure:
                _warn_once(path, "atomic_io text write degraded to direct write", exc)
            return True
        except BEST_EFFORT_EXCEPTIONS as fallback_exc:
            if warn_on_failure:
                _warn_once(path, "atomic_io text write failed", fallback_exc)
            return False


def safe_read_json(path: Union[Path, str], default: Any = None) -> Any:
    """
    Safely read JSON data from the given path with locking.
    
    Args:
        path: Source file path.
        default: Value to return if file doesn't exist or read fails.
        
    Returns:
        Parsed JSON data or default value.
    """
    path = Path(path)
    thread_lock = _path_lock(path)
    try:
        with thread_lock:
            if not path.exists():
                return default
            lock_path = path.with_suffix(path.suffix + ".lock")
            lock = _acquire_lock(lock_path, timeout=2.0)
            if lock is None:
                return json.loads(path.read_text(encoding="utf-8"))
            fd, lp = lock
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            finally:
                _release_lock(fd, lp)
    except BEST_EFFORT_EXCEPTIONS:
        return default


def safe_append_jsonl(path: Union[Path, str], record: Any) -> bool:
    """
    Append a JSON record to a JSONL file with the same lock discipline.

    - Her kayit tek satir JSON olarak yazilir.
    - Lock ile yaris kosullari engellenir.
    
    Args:
        path: Path to the JSONL file.
        record: A JSON-serializable object to append.
        
    Returns:
        True if append succeeded, False otherwise.
    """
    path = _normalize_path(path)
    _assert_test_write_allowed(path)
    if _inside_running_event_loop():
        if enqueue_append_jsonl(path, record):
            return True
        try:
            _append_jsonl_direct(path, record)
            _warn_once(path, "atomic_io jsonl append degraded to direct append inside event loop")
            return True
        except BEST_EFFORT_EXCEPTIONS as fallback_exc:
            _warn_once(path, "atomic_io jsonl append failed inside event loop", fallback_exc)
            return False
    try:
        with file_lock(path, timeout=10):
            _append_jsonl_direct(path, record)
        return True
    except BEST_EFFORT_EXCEPTIONS as exc:
        try:
            _append_jsonl_direct(path, record)
            _warn_once(path, "atomic_io jsonl append degraded to direct append", exc)
            return True
        except BEST_EFFORT_EXCEPTIONS as fallback_exc:
            _warn_once(path, "atomic_io jsonl append failed", fallback_exc)
            return False


def safe_update_json(path: Union[Path, str], updates: dict[str, Any]) -> bool:
    """
    Atomically update specific keys in a JSON file.
    
    Reads the existing JSON, merges updates, and writes back atomically.
    
    Args:
        path: Path to the JSON file.
        updates: Dictionary of key-value pairs to update.
        
    Returns:
        True if update succeeded, False otherwise.
    """
    try:
        path = _normalize_path(path)
        _assert_test_write_allowed(path)
        with file_lock(path, timeout=10):
            # Read existing data
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(data, dict):
                        data = {}
                except BEST_EFFORT_EXCEPTIONS:
                    data = {}
            else:
                data = {}
            
            # Merge updates
            data.update(updates)
            
            # Write back
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            os.replace(str(tmp_path), str(path))
            return True
    except BEST_EFFORT_EXCEPTIONS as exc:
        _warn_once(path, "atomic_io json update failed", exc)
        return False


# ==============================================================================
# ASYNC-NATIVE LOCKING AND I/O HELPERS
# ==============================================================================

_ASYNC_THREAD_LOCKS: dict[str, asyncio.Lock] = {}
_ASYNC_THREAD_LOCKS_GUARD = threading.Lock()

def _async_path_lock(path: Union[Path, str]) -> asyncio.Lock:
    import asyncio
    try:
        normalized = _normalize_path(path)
    except BEST_EFFORT_EXCEPTIONS:
        normalized = Path(path).expanduser()
    key = os.path.normcase(os.fspath(normalized))
    with _ASYNC_THREAD_LOCKS_GUARD:
        lock = _ASYNC_THREAD_LOCKS.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _ASYNC_THREAD_LOCKS[key] = lock
        return lock


async_file_lock = file_lock


async def async_safe_write_json(path: Union[Path, str], data: Any, timeout: float = 5.0, *, warn_on_failure: bool = True) -> bool:
    """Async version of safe_write_json that doesn't block the event loop."""
    import asyncio
    path = _normalize_path(path)
    try:
        async with async_file_lock(path, timeout=timeout):
            def perform_write():
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp_name = f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
                tmp_path = path.with_name(tmp_name)
                _write_json_direct(tmp_path, data)
                try:
                    os.replace(str(tmp_path), str(path))
                except OSError as replace_exc:
                    try:
                        if tmp_path.exists():
                            tmp_path.unlink()
                    except OSError:
                        pass
                    raise replace_exc
                return True
            
            return await asyncio.to_thread(perform_write)
    except BEST_EFFORT_EXCEPTIONS as exc:
        try:
            def perform_fallback_write():
                _write_json_direct(path, data)
            await asyncio.to_thread(perform_fallback_write)
            if warn_on_failure:
                _warn_once(path, "atomic_io async json write degraded to direct write", exc)
            return True
        except BEST_EFFORT_EXCEPTIONS as fallback_exc:
            if warn_on_failure:
                _warn_once(path, "atomic_io async json write failed", fallback_exc)
            return False


async def async_safe_read_json(path: Union[Path, str], default: Any = None, timeout: float = 5.0) -> Any:
    """Async version of safe_read_json that doesn't block the event loop."""
    import asyncio
    path = Path(path)
    try:
        async with async_file_lock(path, timeout=timeout):
            def perform_read():
                if not path.exists():
                    return default
                return json.loads(path.read_text(encoding="utf-8"))
            return await asyncio.to_thread(perform_read)
    except BEST_EFFORT_EXCEPTIONS:
        try:
            def perform_fallback_read():
                if not path.exists():
                    return default
                return json.loads(path.read_text(encoding="utf-8"))
            return await asyncio.to_thread(perform_fallback_read)
        except BEST_EFFORT_EXCEPTIONS:
            return default


async def async_safe_append_jsonl(path: Union[Path, str], record: Any, timeout: float = 5.0) -> bool:
    """Async version of safe_append_jsonl that doesn't block the event loop."""
    import asyncio
    path = _normalize_path(path)
    try:
        async with async_file_lock(path, timeout=timeout):
            await asyncio.to_thread(_append_jsonl_direct, path, record)
        return True
    except BEST_EFFORT_EXCEPTIONS as exc:
        try:
            await asyncio.to_thread(_append_jsonl_direct, path, record)
            _warn_once(path, "atomic_io async jsonl append degraded to direct append", exc)
            return True
        except BEST_EFFORT_EXCEPTIONS as fallback_exc:
            _warn_once(path, "atomic_io async jsonl append failed", fallback_exc)
            return False


async def async_safe_update_json(
    path: Union[Path, str],
    updates_or_fn: Union[dict[str, Any], Any],
    timeout: float = 5.0
) -> bool:
    """
    Async version of safe_update_json that doesn't block the event loop.
    Supports passing either a dictionary of updates or a callable update_fn(data).
    """
    import asyncio
    path = _normalize_path(path)
    try:
        async with async_file_lock(path, timeout=timeout):
            def perform_update():
                if path.exists():
                    try:
                        data = json.loads(path.read_text(encoding="utf-8"))
                        if not isinstance(data, dict):
                            data = {}
                    except BEST_EFFORT_EXCEPTIONS:
                        data = {}
                else:
                    data = {}
                
                if callable(updates_or_fn):
                    result = updates_or_fn(data)
                    if isinstance(result, dict):
                        data = result
                elif isinstance(updates_or_fn, dict):
                    data.update(updates_or_fn)
                
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = path.with_suffix(path.suffix + ".tmp")
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2, default=str)
                os.replace(str(tmp_path), str(path))
                return True
                
            return await asyncio.to_thread(perform_update)
    except BEST_EFFORT_EXCEPTIONS as exc:
        _warn_once(path, "atomic_io async json update failed", exc)
        return False
