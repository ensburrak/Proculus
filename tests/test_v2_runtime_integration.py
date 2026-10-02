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


def _stoch_ta(**overrides) -> dict:
    base = {
        "stoch_rsi_warmup_ok": True,
        "stoch_rsi_90_prev_k": 6.0,
        "stoch_rsi_90_prev_d": 8.0,
        "stoch_rsi_90_k": 8.0,
        "stoch_rsi_90_d": 6.0,
        "ema_fast": 105.0,
        "ema_slow": 100.0,
        "adx": 24.0,
        "plus_di": 28.0,
        "minus_di": 14.0,
        "atr_pct": 0.012,
        "volume_spike_ratio": 1.0,
    }
    base.update(overrides)
    return base


def test_stochrsi90_is_an_independent_direction_authority() -> None:
    from decision.stochrsi_parallel import evaluate_stochrsi90

    result = evaluate_stochrsi90(
        item={"symbol": "BTC/USDT", "runtime_mode": "paper", "regime": "bull"},
        ta=_stoch_ta(),
    )

    assert result["action"] == "enter"
    assert result["direction"] == "long"
    assert result["pipeline"] == "stochrsi_parallel"
    assert result["authority"] == "independent"
    assert result["model_confirmation_required"] is False
    assert result["max_leverage"] == 1.0


def test_stochrsi90_fails_closed_in_never_trade_regimes() -> None:
    from decision.stochrsi_parallel import evaluate_stochrsi90

    for regime in ("shock", "unknown", "conflict"):
        result = evaluate_stochrsi90(
            item={"symbol": "BTC/USDT", "runtime_mode": "paper", "regime": regime},
            ta=_stoch_ta(),
        )
        assert result["action"] == "hold"
        assert result["direction"] == "neutral"
        assert result["risk_scale"] == 0.0



def test_stochrsi90_snapshot_is_computed_from_closed_price_history() -> None:
    import math

    from decision.stochrsi_parallel import compute_stochrsi90_snapshot

    closes = [
        100.0 + 0.025 * idx + 2.0 * math.sin(idx / 7.0)
        for idx in range(320)
    ]
    snapshot = compute_stochrsi90_snapshot(closes)

    assert snapshot["stoch_rsi_warmup_ok"] is True
    for key in (
        "stoch_rsi_90_k",
        "stoch_rsi_90_d",
        "stoch_rsi_90_prev_k",
        "stoch_rsi_90_prev_d",
    ):
        assert 0.0 <= float(snapshot[key]) <= 100.0


def test_runtime_ta_pack_includes_native_stochrsi90_snapshot() -> None:
    import math

    import pandas as pd

    from runtime.runtime_ta_pack import build_ta_pack_from_multidata

    close = [
        100.0 + 0.025 * idx + 2.0 * math.sin(idx / 7.0)
        for idx in range(320)
    ]
    frame = pd.DataFrame(
        {
            "close": close,
            "open": close,
            "high": [value + 0.4 for value in close],
            "low": [value - 0.4 for value in close],
            "volume": [1000.0 + (idx % 11) * 20.0 for idx in range(320)],
            "ema_fast": pd.Series(close).ewm(span=20, adjust=False).mean(),
            "ema_slow": pd.Series(close).ewm(span=50, adjust=False).mean(),
            "adx": [24.0] * 320,
            "atr": [1.2] * 320,
            "atr_ratio": [0.012] * 320,
        }
    )

    ta = build_ta_pack_from_multidata({"15m": frame})

    assert ta["stoch_rsi_warmup_ok"] is True
    assert 0.0 <= float(ta["stoch_rsi_90_k"]) <= 100.0
    assert 0.0 <= float(ta["stoch_rsi_90_d"]) <= 100.0


def test_runtime_exposes_stochrsi_as_parallel_authority_without_overwriting_v2(
    monkeypatch,
) -> None:
    import runtime.runtime_loop_services as loop_services

    item = _bull_item()
    item["ta_pack"].update(_stoch_ta())

    async def fake_analyze(_exchange, _symbol, runtime_mode="paper"):
        result = dict(item)
        result["runtime_mode"] = runtime_mode
        return result

    monkeypatch.setattr(loop_services, "_analyze_one", fake_analyze)

    class DummyExchange:
        async def create_order(self, *args, **kwargs):
            raise AssertionError("paper runtime must not submit an exchange order")

    latest = asyncio.run(
        loop_services.trading_loop_async_service(
            DummyExchange(),
            ["BTC/USDT"],
            runtime_mode="paper",
            once=True,
        )
    )
    decision = latest["BTC/USDT"]

    assert decision["pipeline"] == "v2"
    assert decision["parallel_decisions"]["stochrsi"]["pipeline"] == "stochrsi_parallel"
    assert decision["parallel_decisions"]["stochrsi"]["authority"] == "independent"
    assert decision["parallel_decisions"]["stochrsi"]["execution"]["order_sent"] is False



def test_combined_backtest_only_arbitrates_at_final_same_symbol_boundary() -> None:
    import pandas as pd

    from tools.fresh_two_year_v2_backtest import (
        Candidate,
        collapse_independent_candidates,
    )

    when = pd.Timestamp("2026-01-01T00:15:00Z")
    common = {
        "symbol": "BTC/USDT:USDT",
        "decision_idx": 10,
        "entry_idx": 11,
        "entry_time": when,
        "side": "long",
        "regime": "bull",
        "risk_scale": 0.25,
        "leverage": 1.0,
        "atr": 1.0,
        "decision_price": 100.0,
    }
    v2 = Candidate(
        **common,
        setup_id="bull_trend.pullback.long.15m.v2",
        strategy="trend_pullback",
        confidence=0.74,
    )
    stoch = Candidate(
        **common,
        setup_id="stochrsi90.bull.long.15m.v1",
        strategy="stochrsi90_independent",
        confidence=0.81,
    )

    selected = collapse_independent_candidates([v2, stoch])

    assert selected == [stoch]



def test_cross_sectional_ranker_keeps_only_top_confidence_entries() -> None:
    from decision.cross_sectional_ranker import rank_enter_decisions

    decisions = {
        "BTC/USDT": {
            "action": "enter",
            "direction": "long",
            "master_confidence": 0.91,
            "risk_scale": 0.5,
            "lev": 1,
        },
        "ETH/USDT": {
            "action": "enter",
            "direction": "long",
            "master_confidence": 0.72,
            "risk_scale": 0.5,
            "lev": 1,
        },
        "SOL/USDT": {"action": "hold", "master_confidence": 0.99},
    }

    ranked = rank_enter_decisions(
        decisions,
        top_fraction=0.15,
        min_candidates=2,
    )

    assert ranked["BTC/USDT"]["action"] == "enter"
    assert ranked["ETH/USDT"]["action"] == "hold"
    assert ranked["ETH/USDT"]["risk_scale"] == 0.0
    assert "cross-sectional" in ranked["ETH/USDT"]["reason"]
    assert ranked["SOL/USDT"]["action"] == "hold"
