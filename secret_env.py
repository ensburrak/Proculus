# -*- coding: utf-8 -*-
"""Helpers for resolving env vars from plain values or Docker-style *_FILE secrets."""
from __future__ import annotations


from core.exceptions import BEST_EFFORT_EXCEPTIONS
import os
import time
import logging
import base64
import json
from dataclasses import dataclass
from pathlib import Path
from atomic_io import atomic_write_json

from dotenv import dotenv_values
from env_mode import resolve_dotenv_fallback

_PLACEHOLDER_PREFIXES = ("REPLACE_ME", "CHANGE_ME", "SET_ME", "__MOVED_TO_SECRETS_ENC__")
_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_EXTERNAL_SECRET_ROOT = Path.home() / ".autotraderbot" / "AutoTraderBot" / "secrets"
def default_secret_root() -> Path:
    return Path(
        os.getenv("ATB_SECRET_ROOT")
        or os.getenv("APP_SECRET_ROOT")
        or os.getenv("APP_BASE_DIR")
        or _DEFAULT_EXTERNAL_SECRET_ROOT
    ).expanduser()
PASSWORD_SECRET_BUNDLE_VERSION = 1
PASSWORD_SECRET_BUNDLE_KDF = "pbkdf2-sha256"
PASSWORD_SECRET_BUNDLE_ITERATIONS = 390_000
WINDOWS_DPAPI_SECRET_BUNDLE_KDF = "windows-dpapi"


@dataclass(frozen=True)
class SecretResolution:
    key: str
    value: str | None
    source: str
    path: Path | None = None


@dataclass(frozen=True)
class EnvironmentBootstrapPlan:
    process_env: dict[str, str]
    secrets_key_file: str
    secrets_file: str
    secrets_path: Path
    explicit_dotenv_policy: str | None
    explicit_env_file: str | None
    env_path: Path
    allow_dotenv_fallback: bool


def resolve_secret_file_path(
    path_value: str | os.PathLike[str],
    *,
    base_path: str | os.PathLike[str] | None = None,
) -> Path:
    candidate = Path(str(path_value).strip()).expanduser()
    if candidate.is_absolute():
        return candidate
    if base_path is not None and str(os.getenv("ALLOW_PROJECT_SECRETS", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return (Path(base_path).expanduser() / candidate).resolve(strict=False)
    return (default_secret_root() / candidate).resolve(strict=False)


def _load_fernet_class():
    try:
        from cryptography.fernet import Fernet

        return Fernet
    except ImportError as exc:
        raise RuntimeError(
            "Password encrypted secrets require the 'cryptography' package."
        ) from exc


def _derive_password_secret_key(password: str, salt: bytes, iterations: int) -> bytes:
    if not password:
        raise RuntimeError("SECRETS_PASSWORD is required for password encrypted secrets.")
    try:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    except ImportError as exc:
        raise RuntimeError(
            "Password encrypted secrets require the 'cryptography' package."
        ) from exc

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=int(iterations),
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode("utf-8")))


def _resolve_bundle_password(password: str | None = None) -> str:
    resolved = password if password is not None else os.getenv("SECRETS_PASSWORD")
    if not resolved:
        raise RuntimeError("SECRETS_PASSWORD is required for password encrypted secrets.")
    return str(resolved)


def encrypt_password_secret_bundle(
    payload: dict[str, object],
    out_path: str | os.PathLike[str],
    *,
    password: str | None = None,
    iterations: int = PASSWORD_SECRET_BUNDLE_ITERATIONS,
) -> Path:
    """Write a password-encrypted JSON secrets bundle.

    The encrypted file stores only KDF metadata plus Fernet ciphertext. The
    plaintext payload is never written by this helper.
    """
    if not isinstance(payload, dict) or not payload:
        raise ValueError("secret bundle payload must be a non-empty dict")
    secret_password = _resolve_bundle_password(password)
    salt = os.urandom(16)
    key = _derive_password_secret_key(secret_password, salt, iterations)
    Fernet = _load_fernet_class()
    ciphertext = Fernet(key).encrypt(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    envelope = {
        "version": PASSWORD_SECRET_BUNDLE_VERSION,
        "kdf": PASSWORD_SECRET_BUNDLE_KDF,
        "iterations": int(iterations),
        "salt": base64.b64encode(salt).decode("ascii"),
        "payload": ciphertext.decode("ascii"),
    }
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target, envelope)
    return target


def decrypt_password_secret_bundle(
    path: str | os.PathLike[str],
    *,
    password: str | None = None,
) -> dict[str, str]:
    """Decrypt a password-encrypted JSON secrets bundle."""
    envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(envelope, dict):
        raise RuntimeError("Password secrets bundle must be a JSON object.")
    if int(envelope.get("version", -1)) != PASSWORD_SECRET_BUNDLE_VERSION:
        raise RuntimeError("Unsupported password secrets bundle version.")
    if str(envelope.get("kdf", "")) != PASSWORD_SECRET_BUNDLE_KDF:
        raise RuntimeError("Unsupported password secrets bundle KDF.")
    iterations = int(envelope.get("iterations", 0))
    if iterations < 100_000:
        raise RuntimeError("Password secrets bundle KDF iterations are too low.")
    try:
        salt = base64.b64decode(str(envelope["salt"]).encode("ascii"))
        ciphertext = str(envelope["payload"]).encode("ascii")
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Invalid password secrets bundle envelope.") from exc
    key = _derive_password_secret_key(_resolve_bundle_password(password), salt, iterations)
    Fernet = _load_fernet_class()
    decoded = json.loads(Fernet(key).decrypt(ciphertext).decode("utf-8"))
    if not isinstance(decoded, dict):
        raise RuntimeError("Password secrets payload must decode to a JSON object.")
    return {str(k): "" if v is None else str(v) for k, v in decoded.items()}


def windows_dpapi_available() -> bool:
    return os.name == "nt"


def _windows_dpapi_transform(data: bytes, *, protect: bool) -> bytes:
    if not windows_dpapi_available():
        raise RuntimeError("Windows DPAPI is only available on Windows.")
    import ctypes
    from ctypes import wintypes

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [
            ("cbData", wintypes.DWORD),
            ("pbData", ctypes.POINTER(ctypes.c_ubyte)),
        ]

    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    in_buffer = ctypes.create_string_buffer(data)
    in_blob = DATA_BLOB(
        len(data),
        ctypes.cast(in_buffer, ctypes.POINTER(ctypes.c_ubyte)),
    )
    out_blob = DATA_BLOB()
    if protect:
        ok = crypt32.CryptProtectData(
            ctypes.byref(in_blob),
            None,
            None,
            None,
            None,
            0,
            ctypes.byref(out_blob),
        )
    else:
        ok = crypt32.CryptUnprotectData(
            ctypes.byref(in_blob),
            None,
            None,
            None,
            None,
            0,
            ctypes.byref(out_blob),
        )
    if not ok:
        raise ctypes.WinError()
    try:
        return ctypes.string_at(out_blob.pbData, out_blob.cbData)
    finally:
        kernel32.LocalFree(ctypes.cast(out_blob.pbData, wintypes.HLOCAL))


def encrypt_windows_dpapi_secret_bundle(
    payload: dict[str, object],
    out_path: str | os.PathLike[str],
) -> Path:
    """Write a Windows-user-bound encrypted JSON secrets bundle."""
    if not isinstance(payload, dict) or not payload:
        raise ValueError("secret bundle payload must be a non-empty dict")
    plaintext = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ciphertext = _windows_dpapi_transform(plaintext, protect=True)
    envelope = {
        "version": PASSWORD_SECRET_BUNDLE_VERSION,
        "kdf": WINDOWS_DPAPI_SECRET_BUNDLE_KDF,
        "payload": base64.b64encode(ciphertext).decode("ascii"),
    }
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target, envelope)
    return target


def decrypt_windows_dpapi_secret_bundle(path: str | os.PathLike[str]) -> dict[str, str]:
    envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(envelope, dict):
        raise RuntimeError("Windows DPAPI secrets bundle must be a JSON object.")
    if int(envelope.get("version", -1)) != PASSWORD_SECRET_BUNDLE_VERSION:
        raise RuntimeError("Unsupported Windows DPAPI secrets bundle version.")
    if str(envelope.get("kdf", "")) != WINDOWS_DPAPI_SECRET_BUNDLE_KDF:
        raise RuntimeError("Unsupported Windows DPAPI secrets bundle KDF.")
    try:
        ciphertext = base64.b64decode(str(envelope["payload"]).encode("ascii"))
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Invalid Windows DPAPI secrets bundle envelope.") from exc
    decoded = json.loads(_windows_dpapi_transform(ciphertext, protect=False).decode("utf-8"))
    if not isinstance(decoded, dict):
        raise RuntimeError("Windows DPAPI secrets payload must decode to a JSON object.")
    return {str(k): "" if v is None else str(v) for k, v in decoded.items()}


def resolve_secret_env(key: str, default: str | None = None) -> SecretResolution:
    """Resolve an env var, falling back to KEY_FILE / KEY_PATH if present."""
    value = os.getenv(key)
    if value is not None and str(value).strip() != "":
        return SecretResolution(key=key, value=str(value), source="env")

    for file_key in (f"{key}_FILE", f"{key}_PATH"):
        file_path = os.getenv(file_key)
        if file_path is None or str(file_path).strip() == "":
            continue
        try:
            resolved_path = resolve_secret_file_path(file_path)
            value = resolved_path.read_text(encoding="utf-8").strip()
            return SecretResolution(key=key, value=value, source=file_key, path=resolved_path)
        except OSError:
            return SecretResolution(key=key, value=default, source=file_key, path=resolve_secret_file_path(file_path))

    return SecretResolution(key=key, value=default, source="default")


def resolve_dotenv_path(
    base_path: str | os.PathLike[str],
    explicit_path: str | None = None,
) -> Path:
    if explicit_path:
        candidate = Path(explicit_path).expanduser()
        return candidate if candidate.is_absolute() else Path(base_path) / candidate
    return (default_secret_root() / ".env").resolve(strict=False)


def _is_project_dotenv(path: Path, base_path: str | os.PathLike[str]) -> bool:
    try:
        resolved = path.resolve(strict=False)
        base = Path(base_path).expanduser().resolve(strict=False)
        return resolved == base / ".env"
    except OSError:
        return False


def _project_dotenv_allowed(env_snapshot: dict[str, str]) -> bool:
    return str(env_snapshot.get("ALLOW_PROJECT_DOTENV", "") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def build_environment_bootstrap_plan(
    *,
    base_path: str | os.PathLike[str],
    process_env: dict[str, str] | None = None,
) -> EnvironmentBootstrapPlan:
    env_snapshot = dict(process_env or os.environ)
    secrets_key_file = env_snapshot.get("SECRETS_KEY_FILE", "master.key")
    secrets_file = env_snapshot.get("SECRETS_FILE", "secrets.enc")
    secrets_path = resolve_secret_file_path(secrets_file or "secrets.enc", base_path=base_path)
    explicit_dotenv_policy = env_snapshot.get("ALLOW_DOTENV_FALLBACK")
    explicit_env_file = env_snapshot.get("APP_ENV_FILE") or env_snapshot.get("DOTENV_PATH")
    env_path = resolve_dotenv_path(base_path, explicit_env_file)
    project_dotenv_blocked = (
        explicit_env_file is not None
        and _is_project_dotenv(env_path, base_path)
        and not _project_dotenv_allowed(env_snapshot)
    )
    env_exists_for_policy = env_path.exists() and not project_dotenv_blocked
    if explicit_dotenv_policy is None and env_exists_for_policy:
        try:
            file_policy = dotenv_values(env_path).get("ALLOW_DOTENV_FALLBACK")
            if file_policy is not None:
                explicit_dotenv_policy = str(file_policy)
        except OSError:
            explicit_dotenv_policy = None
    allow_dotenv_fallback = resolve_dotenv_fallback(
        explicit_dotenv_policy,
        secrets_exists=secrets_path.exists(),
        project_env_exists=env_exists_for_policy,
    )
    return EnvironmentBootstrapPlan(
        process_env=env_snapshot,
        secrets_key_file=secrets_key_file,
        secrets_file=secrets_file,
        secrets_path=secrets_path,
        explicit_dotenv_policy=explicit_dotenv_policy,
        explicit_env_file=explicit_env_file,
        env_path=env_path,
        allow_dotenv_fallback=allow_dotenv_fallback,
    )


def get_secret_env(key: str, default: str | None = None) -> str | None:
    result = resolve_secret_env(key, default)
    if result.value is not None and result.source != "default":
        record_secret_usage(key)
    return result.value


def has_secret_env(key: str) -> bool:
    value = get_secret_env(key)
    if value is None:
        return False
    stripped = value.strip()
    if not stripped:
        return False
    return not any(stripped.startswith(prefix) for prefix in _PLACEHOLDER_PREFIXES)


_rotation_logger = logging.getLogger("secret_rotation")

# Secret rotation tracking
_SECRET_MAX_AGE_DAYS = int(os.environ.get("SECRET_MAX_AGE_DAYS", "90"))
_ROTATION_STATE_FILE = default_secret_root() / "secrets" / ".rotation_state.json"


def _load_rotation_state() -> dict:
    """Load secret rotation state."""
    try:
        if _ROTATION_STATE_FILE.exists():
            import json
            return json.loads(_ROTATION_STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    return {}


def _save_rotation_state(state: dict) -> None:
    """Save secret rotation state."""
    try:
        import json
        _ROTATION_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(_ROTATION_STATE_FILE, state)
    except OSError:
        pass


def record_secret_usage(key: str) -> None:
    """Record when a secret was first used (for age tracking)."""
    state = _load_rotation_state()
    if key not in state:
        state[key] = {
            "first_seen": time.time(),
            "last_rotated": time.time(),
        }
        _save_rotation_state(state)


def _send_rotation_telegram_alert(key: str, age_days: float, max_days: int, *, critical: bool = False) -> None:
    """[FAZA 9.3] Send Telegram notification for secret rotation warnings."""
    try:
        from telegram_notifier import send_message
        emoji = "🚨" if critical else "⚠️"
        level = "GEREKLİ" if critical else "YAKLAŞIYOR"
        send_message(
            f"{emoji} <b>API Key Rotasyon {level}</b>\n\n"
            f"Key: <code>{key}</code>\n"
            f"Yaş: {age_days:.0f} gün / {max_days} gün\n"
            f"{'❌ Süre doldu — hemen yenileyin!' if critical else '⏰ Yakında süresi dolacak.'}",
            parse_mode="HTML",
        )
    except BEST_EFFORT_EXCEPTIONS:
        pass


def check_secret_rotation(key: str) -> dict:
    """Check if a secret needs rotation.

    Returns:
        dict with keys: needs_rotation (bool), age_days (float), max_age_days (int), warning (str|None)
    """
    state = _load_rotation_state()
    entry = state.get(key, {})
    last_rotated = entry.get("last_rotated", entry.get("first_seen", time.time()))
    age_days = (time.time() - last_rotated) / 86400.0
    needs_rotation = age_days >= _SECRET_MAX_AGE_DAYS
    warning = None

    if needs_rotation:
        warning = f"Secret '{key}' is {age_days:.0f} days old (max {_SECRET_MAX_AGE_DAYS}). Rotation recommended."
        _rotation_logger.warning("[SECRET_ROTATION] %s", warning)
        # [FAZA 9.3] Send Telegram alert for overdue rotation
        _send_rotation_telegram_alert(key, age_days, _SECRET_MAX_AGE_DAYS, critical=True)
    elif age_days >= _SECRET_MAX_AGE_DAYS * 0.8:  # Warn at 80% of max age
        warning = f"Secret '{key}' is {age_days:.0f} days old. Approaching rotation deadline ({_SECRET_MAX_AGE_DAYS} days)."
        _rotation_logger.info("[SECRET_ROTATION] %s", warning)
        # [FAZA 9.3] Send Telegram warning for approaching deadline
        _send_rotation_telegram_alert(key, age_days, _SECRET_MAX_AGE_DAYS, critical=False)

    return {
        "needs_rotation": needs_rotation,
        "age_days": round(age_days, 1),
        "max_age_days": _SECRET_MAX_AGE_DAYS,
        "warning": warning,
    }


def mark_secret_rotated(key: str) -> None:
    """Mark a secret as freshly rotated."""
    state = _load_rotation_state()
    if key not in state:
        state[key] = {"first_seen": time.time()}
    state[key]["last_rotated"] = time.time()
    _save_rotation_state(state)
    _rotation_logger.info("[SECRET_ROTATION] Secret '%s' marked as rotated.", key)


def check_all_secrets_rotation() -> list[dict]:
    """Check rotation status for all tracked secrets."""
    state = _load_rotation_state()
    results = []
    for key in state:
        results.append({"key": key, **check_secret_rotation(key)})
    return results


__all__ = [
    "EnvironmentBootstrapPlan",
    "PASSWORD_SECRET_BUNDLE_KDF",
    "PASSWORD_SECRET_BUNDLE_ITERATIONS",
    "PASSWORD_SECRET_BUNDLE_VERSION",
    "WINDOWS_DPAPI_SECRET_BUNDLE_KDF",
    "build_environment_bootstrap_plan",
    "decrypt_password_secret_bundle",
    "decrypt_windows_dpapi_secret_bundle",
    "encrypt_password_secret_bundle",
    "encrypt_windows_dpapi_secret_bundle",
    "default_secret_root",
    "SecretResolution",
    "check_all_secrets_rotation",
    "check_secret_rotation",
    "get_secret_env",
    "has_secret_env",
    "mark_secret_rotated",
    "record_secret_usage",
    "resolve_dotenv_path",
    "resolve_secret_env",
    "resolve_secret_file_path",
    "windows_dpapi_available",
]
