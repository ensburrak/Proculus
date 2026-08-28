# -*- coding: utf-8 -*-
"""Backward-compatible CLI wrapper for the official runtime entrypoint."""

from __future__ import annotations

from runtime.entrypoint import (
    OFFICIAL_RUNTIME_BOOT_COMMAND,
    OFFICIAL_RUNTIME_ENTRYPOINT,
    main as _official_runtime_main,
)


def main() -> None:
    """Compatibility shim; the official boot path is `python -m runtime`."""
    _official_runtime_main()


if __name__ == "__main__":
    main()
