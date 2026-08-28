# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import settings
from runtime_paths import LOGS_DIR as RUNTIME_LOGS_DIR


LOGGING_EXCEPTIONS = (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError)
_CONFIGURED = False


class _SafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """RotatingFileHandler that gracefully handles Windows file-lock errors.

    On Windows, ``os.rename()`` inside ``doRollover()`` raises
    ``PermissionError (WinError 32)`` when another process (or a stale
    handle from the same process) still holds the log file open.

    This subclass catches the error and silently skips the rotation so
    that the *current* log message is still written to the existing file
    instead of crashing the entire application.
    """

    def doRollover(self) -> None:  # noqa: N802
        try:
            super().doRollover()
        except PermissionError:
            # File is locked by another process – skip rotation this time.
            # The next rollover attempt (on the following log write that
            # exceeds maxBytes) will succeed once the lock is released.
            pass

_SENSITIVE_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    re.compile(r"[A-Fa-f0-9]{32,64}"),
    re.compile(r"(?:api[_-]?key|secret|password|passphrase|token)\s*[=:]\s*\S+", re.IGNORECASE),
]

_STANDARD_LOG_RECORD_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}

_TRANSLATIONS_EN_TO_TR = [
    (r"\bruntime bootstrap started\b", "çalışma zamanı başlatma süreci başladı"),
    (r"\bselected runtime mode\b", "çalışma modu seçildi"),
    (r"\bconfig reload failed\b", "konfigürasyon yeniden yükleme başarısız"),
    (r"\bevent published\b", "olay yayımlandı"),
    (r"\bprocessing command\b", "komut işleniyor"),
    (r"\brate limit reached\b", "oran sınırına ulaşıldı"),
    (r"\bcircuit breaker tripped\b", "devre kesici tetiklendi"),
    (r"\ball retries failed\b", "tüm yeniden denemeler başarısız oldu"),
    (r"\bjson parse error\b", "json ayrıştırma hatası"),
    (r"\bcache hit\b", "önbellek isabeti"),
    (r"\bstats updated\b", "istatistikler güncellendi"),
    (r"\binitialized\b", "başlatıldı"),
    (r"\benabled\b", "etkin"),
    (r"\bdisabled\b", "devre dışı"),
    (r"\bstarting\b", "başlatılıyor"),
    (r"\bstarted\b", "başlatıldı"),
    (r"\bstopping\b", "durduruluyor"),
    (r"\bstopped\b", "durduruldu"),
    (r"\bcontinue\b", "devam"),
    (r"\bcancelled\b", "iptal edildi"),
    (r"\bfailed\b", "başarısız"),
    (r"\bsuccessful\b", "başarılı"),
    (r"\bstatus\b", "durum"),
    (r"\bactive\b", "aktif"),
    (r"\binactive\b", "pasif"),
    (r"\bnot found\b", "bulunamadı"),
    (r"\bread failed\b", "okuma başarısız"),
    (r"\bwrite failed\b", "yazma başarısız"),
    (r"\bload failed\b", "yükleme başarısız"),
    (r"\berror\b", "hata"),
    (r"\bwarning\b", "uyarı"),
    (r"\bwarnings\b", "uyarılar"),
    (r"\bexternal\b", "harici"),
    (r"\bdaily\b", "günlük"),
    (r"\bweekly\b", "haftalık"),
    (r"\bmonthly\b", "aylık"),
    (r"\brisk\b", "risk"),
    (r"\bposition\b", "pozisyon"),
    (r"\border\b", "emir"),
    (r"\bopen\b", "açık"),
    (r"\bclosed\b", "kapalı"),
    (r"\bbalance\b", "bakiye"),
    (r"\bdefault\b", "varsayılan"),
    (r"\binvalid\b", "geçersiz"),
    (r"\bsymbol\b", "sembol"),
    (r"\bfile\b", "dosya"),
    (r"\bmodule\b", "modül"),
    (r"\bloaded\b", "yüklendi"),
    (r"\bwill be used\b", "kullanılacak"),
    (r"\bcleaned\b", "temizlendi"),
    (r"\bcalculated\b", "hesaplandı"),
    (r"\bcommand\b", "komut"),
    (r"\bconfig\b", "konfigürasyon"),
    (r"\breload(ed|ing)?\b", "yeniden yüklendi"),
    (r"\brefresh(ed|ing)?\b", "yenilendi"),
    (r"\bskipped\b", "atlanıyor"),
]

_TRANSLATIONS_TR_TO_EN = [
    (r"\bçalışma zamanı başlatma süreci başladı\b", "runtime bootstrap started"),
    (r"\bçalışma modu seçildi\b", "selected runtime mode"),
    (r"\bkonfigürasyon yeniden yükleme başarısız\b", "config reload failed"),
    (r"\bolay yayımlandı\b", "event published"),
    (r"\bkomut işleniyor\b", "processing command"),
    (r"\boran sınırına ulaşıldı\b", "rate limit reached"),
    (r"\bdevre kesici tetiklendi\b", "circuit breaker tripped"),
    (r"\btüm yeniden denemeler başarısız oldu\b", "all retries failed"),
    (r"\bjson ayrıştırma hatası\b", "json parse error"),
    (r"\bönbellek isabeti\b", "cache hit"),
    (r"\bistatistikler güncellendi\b", "stats updated"),
    (r"\bbaşlatılıyor\b", "starting"),
    (r"\bbaşlatıldı\b", "started"),
    (r"\bdurduruluyor\b", "stopping"),
    (r"\bdurduruldu\b", "stopped"),
    (r"\bdevam\b", "continue"),
    (r"\biptal edildi\b", "cancelled"),
    (r"\bbaşarısız\b", "failed"),
    (r"\bbaşarılı\b", "successful"),
    (r"\bdurum\b", "status"),
    (r"\baktif\b", "active"),
    (r"\bpasif\b", "inactive"),
    (r"\bbulunamadı\b", "not found"),
    (r"\bokuma başarısız\b", "read failed"),
    (r"\byazma başarısız\b", "write failed"),
    (r"\byükleme başarısız\b", "load failed"),
    (r"\bhata\b", "error"),
    (r"\buyarılar\b", "warnings"),
    (r"\buyarı\b", "warning"),
    (r"\bharici\b", "external"),
    (r"\bgünlük\b", "daily"),
    (r"\bhaftalık\b", "weekly"),
    (r"\baylık\b", "monthly"),
    (r"\bris[kK]\b", "risk"),
    (r"\bpozisyon\b", "position"),
    (r"\bemir\b", "order"),
    (r"\baçık\b", "open"),
    (r"\bkapalı\b", "closed"),
    (r"\bbakiye\b", "balance"),
    (r"\bvarsayılan\b", "default"),
    (r"\bgeçersiz\b", "invalid"),
    (r"\bsembol\b", "symbol"),
    (r"\bdosya\b", "file"),
    (r"\bmodül\b", "module"),
    (r"\byüklendi\b", "loaded"),
    (r"\bkullanılacak\b", "will be used"),
    (r"\btemizlendi\b", "cleaned"),
    (r"\bhesaplandı\b", "calculated"),
    (r"\bkomut\b", "command"),
    (r"\bkonfigürasyon\b", "config"),
    (r"\byeniden yüklendi\b", "reloaded"),
    (r"\byenilendi\b", "refreshed"),
    (r"\batlanıyor\b", "skipped"),
]

_LEVEL_LABELS = {
    "tr": {
        "DEBUG": "AYRINTI",
        "INFO": "BILGI",
        "WARNING": "UYARI",
        "ERROR": "HATA",
        "CRITICAL": "KRITIK",
    },
    "en": {
        "DEBUG": "DEBUG",
        "INFO": "INFO",
        "WARNING": "WARNING",
        "ERROR": "ERROR",
        "CRITICAL": "CRITICAL",
    },
}

_FIELD_LABELS = {
    "tr": {
        "event": "olay",
        "context": "bağlam",
    },
    "en": {
        "event": "event",
        "context": "context",
    },
}


@dataclass(frozen=True)
class LoggingPreferences:
    language: str
    console_json: bool
    file_json: bool


def _bool_from_value(value: Any, default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_language(value: Any) -> str:
    normalized = str(value or "tr").strip().lower()
    if normalized.startswith("en"):
        return "en"
    return "tr"


def _load_logging_config() -> dict[str, Any]:
    candidates = [
        Path(getattr(settings, "CONFIG_PATH", "")),
        Path(getattr(settings, "BASE_DIR", ".")) / "config.json",
    ]
    for path in candidates:
        try:
            if not str(path):
                continue
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    logging_cfg = raw.get("logging", {})
                    if isinstance(logging_cfg, dict):
                        return logging_cfg
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return {}


def resolve_logging_preferences(
    *,
    config: dict[str, Any] | None = None,
    environ: dict[str, str] | None = None,
) -> LoggingPreferences:
    cfg = config if isinstance(config, dict) else _load_logging_config()
    env = environ if environ is not None else os.environ

    legacy_json = env.get("LOG_JSON")
    language = _normalize_language(env.get("LOG_LANGUAGE") or cfg.get("language") or "tr")
    console_json = _bool_from_value(
        env.get("LOG_CONSOLE_JSON", legacy_json if legacy_json is not None else cfg.get("console_json")),
        default=False,
    )
    file_json = _bool_from_value(
        env.get("LOG_FILE_JSON", legacy_json if legacy_json is not None else cfg.get("file_json")),
        default=True,
    )
    return LoggingPreferences(language=language, console_json=console_json, file_json=file_json)


def _base_clean_message(msg: str) -> str:
    try:
        text = str(msg)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except LOGGING_EXCEPTIONS:
        return str(msg)


def _apply_translation(text: str, language: str) -> str:
    patterns = _TRANSLATIONS_EN_TO_TR if language == "tr" else _TRANSLATIONS_TR_TO_EN
    translated = text
    for pattern, repl in patterns:
        translated = re.sub(pattern, repl, translated, flags=re.IGNORECASE)
    return translated


def _redact_sensitive(msg: str) -> str:
    for pattern in _SENSITIVE_PATTERNS:
        msg = pattern.sub("***REDACTED***", msg)
    return msg


def _localize_message(msg: str, language: str) -> str:
    cleaned = _base_clean_message(msg)
    translated = _apply_translation(cleaned, _normalize_language(language))
    return _redact_sensitive(translated)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return _redact_sensitive(_base_clean_message(repr(value)))


def _extract_record_context(record: logging.LogRecord) -> Dict[str, Any]:
    context: Dict[str, Any] = {}
    for key, value in record.__dict__.items():
        if key in _STANDARD_LOG_RECORD_ATTRS or key.startswith("_"):
            continue
        context[str(key)] = _json_safe(value)
    return context


class JsonFormatter(logging.Formatter):
    def __init__(self, *, language: str = "tr") -> None:
        super().__init__()
        self.language = _normalize_language(language)

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": _localize_message(record.getMessage(), self.language),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "thread": record.threadName,
            "language": self.language,
        }
        extra_context = _extract_record_context(record)
        if extra_context:
            event_name = extra_context.pop("event", None)
            if event_name:
                payload["event"] = event_name
            payload["context"] = extra_context
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class HumanTextFormatter(logging.Formatter):
    def __init__(self, *, language: str = "tr") -> None:
        super().__init__(datefmt="%Y-%m-%d %H:%M:%S")
        self.language = _normalize_language(language)

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, self.datefmt)
        level = _LEVEL_LABELS[self.language].get(record.levelname, record.levelname)
        message = _localize_message(record.getMessage(), self.language)
        parts = [timestamp, level, record.name, message]

        context = _extract_record_context(record)
        event_name = context.pop("event", None)
        field_labels = _FIELD_LABELS[self.language]
        if event_name:
            parts.append(f"{field_labels['event']}={event_name}")
        if context:
            ctx = ", ".join(f"{key}={value}" for key, value in context.items())
            parts.append(f"{field_labels['context']}: {ctx}")
        if record.exc_info:
            parts.append(self.formatException(record.exc_info))
        return " | ".join(parts)


class SanitizeFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = _redact_sensitive(_base_clean_message(record.getMessage()))
            record.args = ()
        except LOGGING_EXCEPTIONS as exc:
            logging.getLogger(__name__).debug("Suppressed in sanitize filter: %s", exc)
        return True


def _build_formatter(*, json_enabled: bool, language: str) -> logging.Formatter:
    if json_enabled:
        return JsonFormatter(language=language)
    return HumanTextFormatter(language=language)


def _configure_handler(handler: logging.Handler, formatter: logging.Formatter, sanitize_filter: logging.Filter) -> None:
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    has_sanitize = any(isinstance(existing, SanitizeFilter) for existing in handler.filters)
    if not has_sanitize:
        handler.addFilter(sanitize_filter)


def _configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    preferences = resolve_logging_preferences()
    sanitize_filter = SanitizeFilter()
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    logs_dir = RUNTIME_LOGS_DIR
    os.makedirs(logs_dir, exist_ok=True)

    console_formatter = _build_formatter(json_enabled=preferences.console_json, language=preferences.language)
    file_formatter = _build_formatter(json_enabled=preferences.file_json, language=preferences.language)

    if not root.handlers:
        console = logging.StreamHandler()
        _configure_handler(console, console_formatter, sanitize_filter)
        root.addHandler(console)

        log_path = os.path.join(logs_dir, f"app_{datetime.now(timezone.utc):%Y%m}.log")
        file_handler = _SafeRotatingFileHandler(
            log_path,
            maxBytes=10_000_000,
            backupCount=5,
            encoding="utf-8",
        )
        _configure_handler(file_handler, file_formatter, sanitize_filter)
        root.addHandler(file_handler)
    else:
        for handler in root.handlers:
            formatter = file_formatter if isinstance(handler, logging.handlers.RotatingFileHandler) else console_formatter
            _configure_handler(handler, formatter, sanitize_filter)

    has_root_sanitize = any(isinstance(existing, SanitizeFilter) for existing in root.filters)
    if not has_root_sanitize:
        root.addFilter(sanitize_filter)

    _CONFIGURED = True


def get_logger(name: str = "autotrader") -> logging.Logger:
    _configure_logging()
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = True
    return logger


def log_event(logger: logging.Logger, level: int, event: str, message: str, **context: Any) -> None:
    payload = {"event": str(event)}
    payload.update(context)
    logger.log(level, message, extra=payload)
