"""Runtime feature-flag inventory.

This module is intentionally kept because the runtime currently has more than
five independently owned operational flags spanning execution, ML, scheduling,
and monitoring. Small one-off toggles should live in config files instead.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any


_TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class FeatureFlagDefinition:
    name: str
    default: bool
    owner: str
    description: str
    category: str


FEATURE_FLAGS: tuple[FeatureFlagDefinition, ...] = (
    FeatureFlagDefinition("USE_CORE_ENGINE", True, "runtime", "Use the core engine runtime instead of the legacy runtime.", "runtime"),
    FeatureFlagDefinition("ALLOW_LEGACY_RUNTIME_FALLBACK", False, "runtime", "Allow the core runtime to fall back to the legacy runtime after a recoverable boot failure.", "runtime"),
    FeatureFlagDefinition("ENABLE_OKX_EXECUTION", True, "execution", "Enable OKX execution adapter paths.", "exchange"),
    FeatureFlagDefinition("ENABLE_BINANCE_EXECUTION", False, "execution", "Enable Binance execution adapter paths.", "exchange"),
    FeatureFlagDefinition("ENABLE_PARALLEL_EXCHANGES", False, "execution", "Execute on multiple exchanges in parallel.", "exchange"),
    FeatureFlagDefinition("ENABLE_BINANCE_FALLBACK", True, "execution", "Allow Binance market-data fallback when the primary exchange is unavailable.", "exchange"),
    FeatureFlagDefinition("ENABLE_DERIBIT_WS", False, "runtime", "Enable the Deribit websocket market monitor.", "runtime"),
    FeatureFlagDefinition("ENABLE_LIQUIDATION_WS", False, "runtime", "Enable liquidation websocket ingestion.", "runtime"),
    FeatureFlagDefinition("ENABLE_WHALE_MONITOR", False, "runtime", "Enable whale-monitor background services.", "runtime"),
    FeatureFlagDefinition("ENABLE_SENTIMENT_SCHEDULER", False, "runtime", "Enable the background sentiment scheduler.", "runtime"),
    FeatureFlagDefinition("ENABLE_STOP_ORDER_WATCHDOG", True, "runtime", "Enable the stop-order watchdog service.", "runtime"),
    FeatureFlagDefinition("ENABLE_HEALTH_MONITOR", True, "runtime", "Enable process health monitoring.", "runtime"),
    FeatureFlagDefinition("ENABLE_SPOT_HEDGE_SCANNER", False, "runtime", "Enable the spot hedge funding scanner.", "runtime"),
    FeatureFlagDefinition("ENABLE_PROMETHEUS_EXPORTER", False, "runtime", "Enable the Prometheus exporter background server.", "runtime"),
    FeatureFlagDefinition("ENABLE_DASHBOARD_LISTENER", True, "runtime", "Enable the dashboard listener background service.", "runtime"),
    FeatureFlagDefinition("ENABLE_MACRO_SENSOR", False, "runtime", "Enable the macro sensor background worker.", "runtime"),
    FeatureFlagDefinition("ENABLE_SYS_WATCHDOG", True, "runtime", "Enable the system watchdog safety monitor.", "runtime"),
    FeatureFlagDefinition("ENABLE_SCHEDULED_ML_TRAINING", True, "ml", "Enable scheduled ML training orchestration.", "ml"),
    FeatureFlagDefinition("ENABLE_TRANSFORMER_TRAIN", True, "ml", "Enable scheduled transformer training jobs.", "ml"),
    FeatureFlagDefinition("ENABLE_RL_TRAIN", True, "ml", "Enable scheduled RL training jobs.", "ml"),
    FeatureFlagDefinition("ENABLE_HYPERPARAM_SEARCH", False, "ml", "Enable scheduled hyperparameter search runs.", "ml"),
    FeatureFlagDefinition("ENABLE_LEGACY_RISK_SCRIPTS", False, "risk", "Run legacy risk calibration/dataset compatibility scripts.", "risk"),
    FeatureFlagDefinition("ENABLE_WEEKLY_WALK_FORWARD", True, "ml", "Run weekly walk-forward backtest sanity after scheduled training.", "ml"),
    FeatureFlagDefinition("ENABLE_V5_BREADTH_SHADOW", False, "research", "Evaluate breadth_donchian_10_v5 in shadow-only mode and persist prospective evidence; never grants execution authority.", "research"),
)

_FLAG_MAP = {definition.name: definition for definition in FEATURE_FLAGS}


def flag_enabled(name: str, default: bool | None = None) -> bool:
    definition = _FLAG_MAP.get(name)
    resolved_default = definition.default if definition is not None else bool(default)
    if default is not None:
        resolved_default = bool(default)
    raw = os.getenv(name)
    if raw is None:
        return resolved_default
    return str(raw).strip().lower() in _TRUE_VALUES


def feature_flag_inventory() -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for definition in FEATURE_FLAGS:
        payload = asdict(definition)
        payload["enabled"] = flag_enabled(definition.name, definition.default)
        inventory.append(payload)
    return inventory


__all__ = ["FEATURE_FLAGS", "FeatureFlagDefinition", "feature_flag_inventory", "flag_enabled"]
