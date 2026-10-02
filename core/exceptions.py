from __future__ import annotations

import asyncio
from typing import Any

OPTIONAL_IMPORT_EXCEPTIONS = (ImportError, ModuleNotFoundError, AttributeError)
DATA_EXCEPTIONS = (AttributeError, KeyError, TypeError, ValueError)
NUMERIC_EXCEPTIONS = (ArithmeticError, OverflowError, ZeroDivisionError)
IO_EXCEPTIONS = (OSError, EOFError)
BEST_EFFORT_EXCEPTIONS = (
    RuntimeError,
    asyncio.TimeoutError,
) + OPTIONAL_IMPORT_EXCEPTIONS + DATA_EXCEPTIONS + NUMERIC_EXCEPTIONS + IO_EXCEPTIONS


class CoreError(Exception):
    def __init__(self, message: str, *, stage: str = "core", context: dict[str, Any] | None = None, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.stage = stage
        self.context = context or {}
        self.cause = cause


class DecisionEngineError(CoreError):
    pass


class MarketDataError(CoreError):
    pass


class OrderExecutionError(CoreError):
    pass
