from __future__ import annotations

import asyncio
from types import SimpleNamespace

import controller_async
import main_bot_async
import risk_manager
from core.decision_pipeline import DecisionPipeline
from runtime.execution_bridge import execute_decision


def _bull_item(*, edge_validated: bool = True, regime: str = "bull") -> dict:
    return {
        "symbol": "BTC/USDT",
        "runtime_mode": "paper",
        "regime": regime,
        "edge_validated": edge_validated,
        "expected_edge_bps": 50.0,
        "estimated_slippage_bps": 2.0,
        "spread_bps": 1.0,
        "expected_hold_hours": 4.0,
        "setup_expectancy": {
            "samples": 150,
            "profit_factor": 1.30,
            "expectancy_r": 0.12,
            "expectancy_r_ci95_low": 0.03,
        },
        "ta_pack": {
            "base_decision": "long",
            "rsi": 56.0,
            "adx": 28.0,
            "atr_ratio": 0.012,
            "ema": {"fast": 109.2, "slow": 108.8},
        },
        "mtf_features": {
            "4h": {
                "ema_fast": 108.0,
                "ema_slow": 105.0,
                "macd_hist": 0.30,
                "recent_closes": [106.0, 107.0, 108.0, 109.0, 110.0],
                "recent_lows": [105.5, 106.5, 107.5, 108.5, 109.2],
                "recent_highs": [106.5, 107.5, 108.5, 109.5, 110.5],
            },
            "1h": {
                "ema_fast": 109.0,
                "ema_slow": 107.0,
                "macd_hist": 0.10,
                "recent_closes": [107.2, 108.0, 108.5, 109.0, 109.5],
                "recent_lows": [106.8, 107.5, 108.0, 108.4, 108.9],
                "recent_highs": [107.5, 108.4, 108.9, 109.4, 109.8],
            },
            "15m": {
                "ema_fast": 109.2,
                "ema_slow": 108.8,
                "macd_hist": 0.04,
                "recent_closes": [109.0, 109.1, 109.0, 109.15, 109.35],
                "recent_lows": [108.9, 109.0, 108.85, 109.0, 109.1],
                "recent_highs": [109.1, 109.2, 109.15, 109.3, 109.45],
            },
        },
    }


def test_controller_async_runs_real_v2_decision_path() -> None:
    result = asyncio.run(controller_async.decide_batch([_bull_item()]))
    decision = result["BTC/USDT"]

    assert decision["action"] == "enter"
    assert decision["direction"] == "long"
    assert decision["pipeline"] == "v2"
    assert decision["setup_id"].startswith("bull_trend.pullback.long")
    assert decision["ai_authority"]["directional"] is False
    assert 0.0 < decision["risk_scale"] <= 1.0


def test_strict_edge_evidence_fails_closed() -> None:
    result = asyncio.run(DecisionPipeline().decide_batch([_bull_item(edge_validated=False)]))
    decision = result["BTC/USDT"]

    assert decision["action"] == "hold"
    assert decision["risk_scale"] == 0.0
    assert "edge evidence" in decision["reason"]


def test_costed_edge_is_required_for_capital_authority() -> None:
    item = _bull_item()
    item.pop("expected_edge_bps")
    result = asyncio.run(controller_async.decide_batch([item]))
    decision = result["BTC/USDT"]

    assert decision["action"] == "hold"
    assert decision["risk_scale"] == 0.0
    assert "costed_edge_missing" in decision["reason"]


def test_shock_regime_is_hard_no_trade() -> None:
    item = _bull_item()
    item["regime"] = "shock"
    result = asyncio.run(controller_async.decide_batch([item]))
    decision = result["BTC/USDT"]

    assert decision["action"] == "hold"
    assert decision["lev"] == 0
    assert decision["risk_scale"] == 0.0


def test_paper_execution_bridge_never_sends_real_order() -> None:
    class DummyExchange:
        async def create_order(self, *args, **kwargs):
            raise AssertionError("paper mode must never call create_order")

    item = _bull_item()
    decision = asyncio.run(controller_async.decide_batch([item]))["BTC/USDT"]
    result = asyncio.run(execute_decision(DummyExchange(), item, decision, {}))

    assert result["status"] == "simulated"
    assert result["order_sent"] is False


def test_legacy_facades_import_against_new_packages() -> None:
    assert callable(main_bot_async.main)
    assert callable(risk_manager.compute_stop_loss)
    assert controller_async.OFFICIAL_DECISION_PIPELINE == "proculus_pipeline_v2"
