from __future__ import annotations

from core.feature_flags import feature_flag_inventory, flag_enabled


def test_flag_enabled_uses_defaults_when_env_missing(monkeypatch):
    monkeypatch.delenv("USE_CORE_ENGINE", raising=False)
    assert flag_enabled("USE_CORE_ENGINE", True) is True


def test_flag_enabled_reads_env_override(monkeypatch):
    monkeypatch.setenv("ENABLE_BINANCE_EXECUTION", "1")
    assert flag_enabled("ENABLE_BINANCE_EXECUTION", False) is True


def test_feature_flag_inventory_contains_metadata(monkeypatch):
    monkeypatch.setenv("ENABLE_WHALE_MONITOR", "1")
    inventory = feature_flag_inventory()
    whale_flag = next(item for item in inventory if item["name"] == "ENABLE_WHALE_MONITOR")
    assert whale_flag["owner"] == "runtime"
    assert whale_flag["enabled"] is True


def test_v5_breadth_shadow_flag_is_registered_and_default_off(monkeypatch):
    monkeypatch.delenv("ENABLE_V5_BREADTH_SHADOW", raising=False)
    inventory = feature_flag_inventory()
    flag = next(item for item in inventory if item["name"] == "ENABLE_V5_BREADTH_SHADOW")
    assert flag["owner"] == "research"
    assert flag["category"] == "research"
    assert flag["default"] is False
    assert flag["enabled"] is False
