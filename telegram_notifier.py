"""Backward-compatible shim for the notifier package."""

from __future__ import annotations

import sys as _sys
import types as _types
from typing import Any

from notifier import client as _client

for _name in dir(_client):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_client, _name)


class _DelegatingModule(_types.ModuleType):
    def __getattr__(self, name: str) -> Any:
        return getattr(_client, name)

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if not name.startswith("__") and name not in {"_client", "_DelegatingModule"}:
            setattr(_client, name, value)


_sys.modules[__name__].__class__ = _DelegatingModule
__all__ = [name for name in globals() if not name.startswith("__")]
