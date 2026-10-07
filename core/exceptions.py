from __future__ import annotations

import asyncio
from typing import Any, Dict

OPTIONAL_IMPORT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ImportError,
    ModuleNotFoundError,
    AttributeError,
)

DATA_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    KeyError,
    TypeError,
    ValueError,
)

NUMERIC_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ArithmeticError,
    OverflowError,
    ZeroDivisionError,
)

IO_EXCEPTIONS: tuple[type[BaseException], ...] = (
    OSError,
    EOFError,
)

_THIRD_PARTY_EXCEPTIONS: tuple[type[BaseException], ...] = ()
try:
    import requests

    _THIRD_PARTY_EXCEPTIONS += (requests.RequestException,)
except OPTIONAL_IMPORT_EXCEPTIONS:
    pass

# Keep aiohttp out of this import path. On Windows, importing aiohttp can block
# in platform/WMI during API startup; aiohttp-specific call sites catch it locally.
_THIRD_PARTY_EXCEPTIONS += (asyncio.TimeoutError,)

try:
    import ccxt

    _THIRD_PARTY_EXCEPTIONS += (ccxt.BaseError,)
except OPTIONAL_IMPORT_EXCEPTIONS:
    pass

try:
    import openai

    _THIRD_PARTY_EXCEPTIONS += (openai.OpenAIError,)
except OPTIONAL_IMPORT_EXCEPTIONS:
    pass

BEST_EFFORT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    RuntimeError,
) + OPTIONAL_IMPORT_EXCEPTIONS + DATA_EXCEPTIONS + NUMERIC_EXCEPTIONS + IO_EXCEPTIONS + _THIRD_PARTY_EXCEPTIONS


class CoreError(Exception):
    """Base typed exception for core/runtime trading flow."""

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        context: Dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.context = context or {}
        self.cause = cause

    def __str__(self) -> str:
        base = super().__str__()
        if self.context:
            ctx = " ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{base} [{ctx}]"
        return base


class ExchangeCallError(CoreError):
    """Raised when exchange API methods cannot be called successfully."""


class MarketDataError(CoreError):
    """Raised for ticker/price parsing and market data issues."""


class BalanceFetchError(CoreError):
    """Raised when account balance cannot be read safely."""


class DecisionEngineError(CoreError):
    """Raised when decision pipeline integration fails."""


class DecisionHandlingError(CoreError):
    """Raised when a decision cannot be safely translated to execution."""


class OrderValidationError(CoreError):
    """Raised when order request is invalid."""


class OrderExecutionError(CoreError):
    """Raised for order placement/cancel failures."""
