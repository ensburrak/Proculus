# -*- coding: utf-8 -*-
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from core.config_loader import resolve_config_path
from core.service_container import registry
from dotenv import dotenv_values
from env_mode import strict_env_enabled
from secret_env import (
    EnvironmentBootstrapPlan,
    PASSWORD_SECRET_BUNDLE_KDF,
    WINDOWS_DPAPI_SECRET_BUNDLE_KDF,
    build_environment_bootstrap_plan,
    decrypt_password_secret_bundle,
    decrypt_windows_dpapi_secret_bundle,
    get_secret_env,
    has_secret_env,
    resolve_dotenv_path,
    resolve_secret_file_path,
)
from runtime_paths import LOGS_DIR as RUNTIME_LOGS_DIR

Fernet = registry.import_or("cryptography.fernet.Fernet", fallback=None)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = Path(BASE_DIR)
log = logging.getLogger("settings")
_ENV_BOOTSTRAPPED = False


def _apply_env_mapping(values: dict[str, str], *, overwrite: bool = False) -> None:
    for key, value in values.items():
        if value is None:
            continue
        if overwrite or key not in os.environ:
            os.environ[key] = str(value)


def load_encrypted_secrets(
    key_file: str = "master.key",
    secrets_file: str = "secrets.enc",
) -> dict[str, str]:
    """Load encrypted secrets from a Fernet-encrypted JSON file."""
    secrets_path = resolve_secret_file_path(secrets_file or "secrets.enc", base_path=BASE_PATH)
    envelope = None
    try:
        envelope = json.loads(secrets_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        envelope = None

    if isinstance(envelope, dict) and envelope.get("kdf") == PASSWORD_SECRET_BUNDLE_KDF:
        return decrypt_password_secret_bundle(secrets_path)
    if isinstance(envelope, dict) and envelope.get("kdf") == WINDOWS_DPAPI_SECRET_BUNDLE_KDF:
        return decrypt_windows_dpapi_secret_bundle(secrets_path)

    if Fernet is None:
        raise RuntimeError(
            "Encrypted secrets requested but cryptography is not installed. "
            "Install the 'cryptography' package."
        )

    key_path = resolve_secret_file_path(key_file or "master.key", base_path=BASE_PATH)

    with open(key_path, "rb") as f:
        key = f.read().strip()
    cipher = Fernet(key)
    with open(secrets_path, "rb") as f:
        payload = f.read()
    data = cipher.decrypt(payload)
    decoded = json.loads(data.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise RuntimeError("Encrypted secrets payload must decode to a JSON object.")
    return {str(k): "" if v is None else str(v) for k, v in decoded.items()}


def _build_environment_bootstrap_plan() -> EnvironmentBootstrapPlan:
    return build_environment_bootstrap_plan(
        base_path=BASE_PATH,
        process_env=dict(os.environ),
    )


def _resolve_dotenv_path(base_path: str | os.PathLike[str], explicit_path: str | None = None) -> Path:
    """Backward-compatible local alias used by bootstrap tests and legacy callers."""
    return resolve_dotenv_path(base_path, explicit_path)


def _bootstrap_environment() -> Path:
    plan = _build_environment_bootstrap_plan()
    if plan.secrets_path.exists():
        decrypted = load_encrypted_secrets(
            key_file=plan.secrets_key_file,
            secrets_file=plan.secrets_file,
        )
        _apply_env_mapping(decrypted, overwrite=False)

    if plan.allow_dotenv_fallback:
        if plan.env_path.exists():
            dotenv_map = {
                str(k): "" if v is None else str(v)
                for k, v in dotenv_values(plan.env_path).items()
                if k
            }
            _apply_env_mapping(dotenv_map, overwrite=False)
        else:
            log.warning(".env fallback requested but file not found: %s", plan.env_path)
    elif plan.env_path.exists() and not plan.secrets_path.exists():
        if plan.explicit_dotenv_policy is not None:
            log.warning(
                "Plaintext .env exists but ALLOW_DOTENV_FALLBACK is disabled; "
                "set ALLOW_DOTENV_FALLBACK=1 for local development or migrate to secrets.enc."
            )

    # Re-apply the original process environment so it always wins.
    _apply_env_mapping(plan.process_env, overwrite=True)
    return plan.env_path


def _environment_bootstrap_enabled() -> bool:
    skip_flag = str(os.getenv("AUTOTRADER_SKIP_ENV_BOOTSTRAP", "")).strip().lower()
    if skip_flag in {"1", "true", "yes", "on"}:
        return False
    return str(os.getenv("TESTING", "")).strip() != "1"


def ensure_environment_bootstrap(*, force: bool = False) -> Path:
    global ENV_PATH, _ENV_BOOTSTRAPPED
    if _ENV_BOOTSTRAPPED and not force:
        return ENV_PATH
    if _environment_bootstrap_enabled():
        ENV_PATH = _bootstrap_environment()
    else:
        ENV_PATH = resolve_dotenv_path(BASE_PATH, os.getenv("APP_ENV_FILE") or os.getenv("DOTENV_PATH"))
    _ENV_BOOTSTRAPPED = True
    return ENV_PATH


ENV_PATH = resolve_dotenv_path(BASE_PATH, os.getenv("APP_ENV_FILE") or os.getenv("DOTENV_PATH"))
ensure_environment_bootstrap()
CONFIG_PATH = str(resolve_config_path())

def env_str(key: str, default: str = "") -> str:
    v = get_secret_env(key)
    return v if v is not None else default


def _normalize_deepseek_decision_model(model_name: str) -> str:
    value = str(model_name or "").strip()
    if not value or value.lower() in {"deepseek-chat", "deepseek-reasoner"}:
        return "deepseek-v4-flash"
    return value


def env_bool(key: str, default: bool = False) -> bool:
    v = get_secret_env(key)
    if v is None:
        return default
    return str(v).lower() in ("1", "true", "yes", "y")

def env_float(key: str, default: float = 0.0) -> float:
    v = get_secret_env(key)
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def env_int(key: str, default: int = 0) -> int:
    v = get_secret_env(key)
    try:
        return int(float(v)) if v is not None else int(default)
    except (TypeError, ValueError):
        return int(default)

# ---------------------------------------------------------------------------
# Environment validation
#
# The bot historically started even when critical API keys were missing,
# failing later during the first live API call (often with confusing errors).
# We validate required variables early and provide actionable messages.

def missing_env(keys: list[str]) -> list[str]:
    """Return a list of missing/empty environment variables."""
    out: list[str] = []
    for k in keys:
        if not has_secret_env(k):
            out.append(k)
    return out


def require_env(keys: list[str], *, context: str = "", strict: bool = True) -> None:
    """Validate that required env vars exist.

    If strict is True and any variable is missing, raise RuntimeError with a
    readable message. If strict is False, only log via stdout (settings.py has
    no logger dependency).
    """
    missing = missing_env(keys)
    if not missing:
        return

    ctx = f" ({context})" if context else ""
    msg = (
        "Missing required environment variables" + ctx + ": "
        + ", ".join(missing)
        + "\n\nFix:\n"
        + "1) Provide the variables via secrets.enc/_FILE env vars or an explicit APP_ENV_FILE\n"
        + "2) Add the variables above\n"
        + "3) Re-run the bot\n"
    )
    if strict:
        raise RuntimeError(msg)
    log.warning("%s", msg)


# OKX
OKX_API_KEY        = env_str("OKX_API_KEY")
OKX_API_SECRET     = env_str("OKX_API_SECRET")
OKX_API_PASSPHRASE = env_str("OKX_API_PASSPHRASE")
OKX_USE_TESTNET    = env_bool("OKX_USE_TESTNET", True)
ENABLE_OKX_EXECUTION = env_bool("ENABLE_OKX_EXECUTION", True)

# OKX (Live - veri çekme ve canlı işlem için)
OKX_LIVE_API_KEY        = env_str("OKX_LIVE_API_KEY", "")
OKX_LIVE_API_SECRET     = env_str("OKX_LIVE_API_SECRET", "")
OKX_LIVE_API_PASSPHRASE = env_str("OKX_LIVE_API_PASSPHRASE", "")

# Binance Futures
BINANCE_API_KEY     = env_str("BINANCE_API_KEY", "")
BINANCE_API_SECRET  = env_str("BINANCE_API_SECRET", "")
BINANCE_USE_TESTNET = env_bool("BINANCE_USE_TESTNET", True)
ENABLE_BINANCE_EXECUTION = env_bool("ENABLE_BINANCE_EXECUTION", False)
PRIMARY_EXCHANGE = env_str("PRIMARY_EXCHANGE", "okx").strip().lower() or "okx"
ENABLE_PARALLEL_EXCHANGES = env_bool("ENABLE_PARALLEL_EXCHANGES", False)
PARALLEL_EXCHANGES = env_str("PARALLEL_EXCHANGES", "okx,binance")

# OpenAI (şu an boş olabilir)
OPENAI_API_KEY        = env_str("OPENAI_API_KEY")
OPENAI_MODEL          = env_str("OPENAI_MODEL", "")
OPENAI_FILTER_MODEL   = env_str("OPENAI_FILTER_MODEL", OPENAI_MODEL or "gpt-4o-mini")
OPENAI_DECISION_MODEL = env_str("OPENAI_DECISION_MODEL", OPENAI_MODEL or "gpt-4o-mini")
OPENAI_FALLBACK_MODEL = env_str("OPENAI_FALLBACK_MODEL", OPENAI_MODEL or "gpt-4o-mini")

# DeepSeek configuration
# Use environment variables if set; fallback to defaults for development.
DEEPSEEK_API_KEY        = env_str("DEEPSEEK_API_KEY", "")
DEEPSEEK_DECISION_MODEL = _normalize_deepseek_decision_model(
    env_str("DEEPSEEK_DECISION_MODEL", "deepseek-v4-flash")
)

# Genel
LOG_DIR = str(RUNTIME_LOGS_DIR)
LOGS_DIR = LOG_DIR
LIVE_MODE = env_bool("LIVE_MODE", False)
WRAPPER_DRY_RUN = env_bool("WRAPPER_DRY_RUN", not LIVE_MODE)
SHADOW_MODE = env_bool("SHADOW_MODE", False)

# Two-key safety: going live requires BOTH WRAPPER_DRY_RUN=False AND LIVE_MODE=1.
# Shadow mode is a third axis with higher precedence at dispatch time:
# SHADOW_MODE > WRAPPER_DRY_RUN > LIVE_MODE.
# When shadow mode is enabled, orders are logged but never sent to the exchange,
# so the live-execution startup guard is intentionally skipped.
if not SHADOW_MODE and not WRAPPER_DRY_RUN and not LIVE_MODE:
    raise RuntimeError(
        "SAFETY: WRAPPER_DRY_RUN=False ama LIVE_MODE=1 degil. "
        "Canli ticaret icin her iki switch de explicit set edilmelidir "
        "(WRAPPER_DRY_RUN=False ve LIVE_MODE=1). "
        "Kaza korumasi: bot baslatilmadi."
    )
if SHADOW_MODE:
    log.info("[STARTUP] Shadow mode active - no real execution")

MAX_TICKER_CONCURRENCY = max(1, env_int("MAX_TICKER_CONCURRENCY", 4))
MAX_ANALYZE_CONCURRENCY = max(1, env_int("MAX_ANALYZE_CONCURRENCY", 4))
MAX_BINANCE_FALLBACK_CONCURRENCY = max(1, env_int("MAX_BINANCE_FALLBACK_CONCURRENCY", 2))
MAX_SYMBOLS_PER_LOOP_CAP = max(1, env_int("MAX_SYMBOLS_PER_LOOP_CAP", 60))
ENABLE_BINANCE_FALLBACK = env_bool("ENABLE_BINANCE_FALLBACK", True)

# Durum dosyaları
RUNTIME_DIR = os.path.join(BASE_DIR, "runtime")
PAUSE_FLAG = os.path.join(RUNTIME_DIR, "PAUSE")

# Kill bayrağı: var ise bot güvenli bir şekilde kapanır. Arayüzde KILL
# düğmesine basıldığında bu dosya oluşturulur ve bot bir sonraki döngüde
# kendisini durdurur.
KILL_FLAG = os.path.join(RUNTIME_DIR, "KILL")

# ---------------------------------------------------------------------------
# Runtime env validation intentionally does not run at import time.
# Startup entrypoints enforce strict secrets via validate_startup_config()
# and explicit require_env(...) checks.
