from __future__ import annotations

from pathlib import Path

from runtime.paper_probe_ledger import (
    authorize_learning_probe_entry,
    progress_snapshot,
    register_learning_probe_entry,
    settle_open_positions,
)


SETUP = "bull_trend.pullback.long.15m.v2"


def _config(**probe_overrides):
    probe = {
        "enabled": True,
        "eligible_runtime_modes": ["paper", "sim"],
        "tracked_setups": [SETUP],
        "target_trades_per_setup": 100,
        "early_freeze_min_trades": 20,
        "early_freeze_recent_pf": 0.75,
        "early_freeze_expectancy_r": -0.25,
        "max_concurrent_positions": 1,
        "setup_cooldown_minutes": 0,
        "daily_budget_pct": 0.005,
        "risk_per_trade_pct": 0.0025,
    }
    probe.update(probe_overrides)
    return {
        "pipeline_v2": {"learning_probe_mode": probe},
        "paper_trading": {
            "initial_capital": 10_000,
            "slippage_bps": 0,
            "taker_fee": 0.0005,
            "funding_rate_8h": 0.0,
            "partial_fill_rate": 1.0,
        },
        "risk": {
            "daily_loss_limit_pct": 0.005,
            "weekly_loss_limit_pct": 0.02,
            "max_wallet_pct": 0.20,
            "daily_loss_timezone": "Europe/Istanbul",
            "daily_limits": {
                "reduce_risk_threshold": 0.50,
                "hedge_only_threshold": 0.75,
            },
        },
    }


def _item(ts: str, *, high: float, low: float, close: float = 100.0):
    return {
        "symbol": "BTC/USDT:USDT",
        "runtime_mode": "paper",
        "tf": "15m",
        "price": close,
        "ta_pack": {"atr": 1.0, "atr_ratio": 0.01},
        "market_bar": {
            "timestamp": ts,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
        },
    }


def _decision():
    return {
        "action": "enter",
        "direction": "long",
        "setup_id": SETUP,
        "strategy": "trend_pullback_resumption",
        "regime": "bull",
        "master_confidence": 0.70,
        "risk_scale": 0.25,
        "lev": 1,
        "stop_atr_mult": 1.0,
        "tp_r_target": 2.0,
        "max_hold_hours": 24.0,
        "learning_probe": {"active": True, "tracked": True},
    }


def test_probe_registers_and_closes_target_with_net_r(tmp_path: Path) -> None:
    cfg = _config()
    state = tmp_path / "probe.json"
    entry = _item("2026-10-01T10:00:00+00:00", high=100.2, low=99.8)

    registered = register_learning_probe_entry(entry, _decision(), cfg, state_path=state)
    assert registered["allowed"] is True
    assert registered["registered"] is True
    assert registered["stop_loss"] == 99.0
    assert registered["take_profit"] == 102.0

    closed = settle_open_positions(
        [_item("2026-10-01T10:15:00+00:00", high=102.5, low=99.5, close=102.2)],
        cfg,
        state_path=state,
        persist_trade_log=False,
    )
    assert len(closed) == 1
    assert closed[0]["exit_reason"] == "take_profit"
    assert closed[0]["r_multiple"] > 1.8

    snapshot = progress_snapshot(cfg, state_path=state)
    setup = snapshot["setups"][SETUP]
    assert setup["closed_trades"] == 1
    assert setup["wins"] == 1
    assert setup["profit_factor"] > 1.0
    assert snapshot["open_positions"] == {}


def test_probe_same_bar_exit_is_stop_first(tmp_path: Path) -> None:
    cfg = _config()
    state = tmp_path / "probe.json"
    register_learning_probe_entry(
        _item("2026-10-01T11:00:00+00:00", high=100.1, low=99.9),
        _decision(),
        cfg,
        state_path=state,
    )
    closed = settle_open_positions(
        [_item("2026-10-01T11:15:00+00:00", high=103.0, low=98.0, close=101.0)],
        cfg,
        state_path=state,
        persist_trade_log=False,
    )
    assert len(closed) == 1
    assert closed[0]["exit_reason"] == "stop_loss"
    assert closed[0]["r_multiple"] < -1.0


def test_probe_daily_risk_budget_fails_closed(tmp_path: Path) -> None:
    cfg = _config(daily_budget_pct=0.001)
    state = tmp_path / "probe.json"
    decision = _decision()

    register_learning_probe_entry(
        _item("2026-10-01T12:00:00+00:00", high=100.1, low=99.9),
        decision,
        cfg,
        state_path=state,
    )
    settle_open_positions(
        [_item("2026-10-01T12:15:00+00:00", high=102.5, low=99.5, close=102.0)],
        cfg,
        state_path=state,
        persist_trade_log=False,
    )
    gate = authorize_learning_probe_entry(
        _item("2026-10-01T12:30:00+00:00", high=100.1, low=99.9),
        decision,
        cfg,
        state_path=state,
    )
    assert gate["allowed"] is False
    assert gate["reason"] == "probe_daily_budget"


def test_probe_early_freezes_repeated_negative_edge(tmp_path: Path) -> None:
    cfg = _config(
        early_freeze_min_trades=2,
        early_freeze_recent_pf=0.75,
        early_freeze_expectancy_r=-0.25,
    )
    state = tmp_path / "probe.json"
    decision = _decision()

    for base_hour in (13, 14):
        registered = register_learning_probe_entry(
            _item(f"2026-10-01T{base_hour:02d}:00:00+00:00", high=100.1, low=99.9),
            decision,
            cfg,
            state_path=state,
        )
        assert registered["allowed"] is True
        closed = settle_open_positions(
            [_item(f"2026-10-01T{base_hour:02d}:15:00+00:00", high=100.2, low=98.0, close=99.0)],
            cfg,
            state_path=state,
            persist_trade_log=False,
        )
        assert closed[0]["exit_reason"] == "stop_loss"

    snapshot = progress_snapshot(cfg, state_path=state)
    setup = snapshot["setups"][SETUP]
    assert setup["closed_trades"] == 2
    assert setup["frozen"] is True
    assert setup["freeze_reason"] == "early_negative_edge"

    gate = authorize_learning_probe_entry(
        _item("2026-10-01T15:00:00+00:00", high=100.1, low=99.9),
        decision,
        cfg,
        state_path=state,
    )
    assert gate["allowed"] is False
    assert gate["reason"] == "probe_setup_frozen"
