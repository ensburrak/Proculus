# -*- coding: utf-8 -*-
"""Compatibility wrapper for the background worker runtime.

The implementation is intentionally named ``background_workers`` because this
process uses in-process threads and queues, not external microservice IPC.
"""
from __future__ import annotations

import sys as _sys
import types as _types

import background_workers as _impl

for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)


class _DelegatingModule(_types.ModuleType):
    def __getattr__(self, name):
        return getattr(_impl, name)

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if not name.startswith("__") and name not in {"_impl", "_DelegatingModule"}:
            setattr(_impl, name, value)


_sys.modules[__name__].__class__ = _DelegatingModule
__all__ = [name for name in globals() if not name.startswith("__")]
