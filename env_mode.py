# -*- coding: utf-8 -*-
"""Environment mode helpers shared by runtime, tests, and tooling."""
from __future__ import annotations

import os
import sys


_TRUTHY = {"1", "true", "yes", "y", "on"}
_FALSY = {"0", "false", "no", "n", "off"}


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in _TRUTHY


def is_testing_mode() -> bool:
    """Return True when running under pytest or explicit TESTING=1."""
    if env_bool("TESTING", False):
        return True
    if "PYTEST_CURRENT_TEST" in os.environ:
        return True
    return "pytest" in sys.modules


def strict_env_enabled(default: bool = True) -> bool:
    """Resolve STRICT_ENV with a safe testing-aware default."""
    if "STRICT_ENV" in os.environ:
        return env_bool("STRICT_ENV", default)
    return False if is_testing_mode() else bool(default)


def resolve_dotenv_fallback(
    raw_value: str | None,
    *,
    secrets_exists: bool,
    project_env_exists: bool,
) -> bool:
    """Resolve whether plaintext .env loading should be allowed.

    Policy:
    - Explicit env var always wins.
    - Encrypted secrets remain the highest-precedence project source.
    - Plaintext `.env` fallback is opt-in only. Runtime should prefer process env,
      encrypted secrets, or file-based secret mounts over implicit local files.
    """
    if raw_value is not None:
        normalized = str(raw_value).strip().lower()
        if normalized in _TRUTHY:
            return True
        if normalized in _FALSY:
            return False
    _ = (secrets_exists, project_env_exists)
    return False
