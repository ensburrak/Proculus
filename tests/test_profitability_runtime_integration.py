import asyncio
import math

import pandas as pd

from decision.cross_sectional_ranker import rank_enter_decisions
from decision.stochrsi_parallel import evaluate_stochrsi90


def _stoch_ta() -> dict:
    return {
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


def test_stochrsi90_is_independent_and_model_free() -> None:
    result = evaluate_stochrsi90(
        item={"symbol": "BTC/USDT", "runtime_mode": "paper", "regime": "bull"},
        ta=_stoch_ta(),
        config={
            "pipeline_v2": {
                "learning_probe_mode": {
                    "never_trade_regimes": ["shock", "conflict", "unknown"]
                }
            },
            "stochrsi_parallel": {
                "enabled": True,
                "paper_orders_enabled": True,
                "demo_orders_enabled": False,
                "live_orders_enabled": False,
            },
        },
    )

    assert result["action"] == "enter"
    assert result["direction"] == "long"
    assert result["authority"] == "independent"
    assert result["model_confirmation_required"] is False
    assert result["execution_allowed"] is True
    assert result["max_leverage"] == 1.0


def test_stochrsi90_never_trade_regimes_fail_closed() -> None:
    for regime in ("shock", "conflict", "unknown"):
        result = evaluate_stochrsi90(
            item={"symbol": "BTC/USDT", "runtime_mode": "paper", "regime": regime},
            ta=_stoch_ta(),
            config={
                "pipeline_v2": {
                    "learning_probe_mode": {
                        "never_trade_regimes": ["shock", "conflict", "unknown"]
                    }
                },
                "stochrsi_parallel": {"enabled": True},
            },
        )
        assert result["action"] == "hold"
        assert result["risk_scale"] == 0.0


def test_cross_sectional_ranker_keeps_only_top_entry() -> None:
    decisions = {
        "BTC/USDT": {
            "action": "enter",
            "direction": "long",
            "master_confidence": 0.91,
            "risk_scale": 0.25,
            "lev": 1,
        },
        "ETH/USDT": {
            "action": "enter",
            "direction": "long",
            "master_confidence": 0.72,
            "risk_scale": 0.25,
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
    assert ranked["ETH/USDT"]["cross_sectional_rank"]["selected"] is False


def test_runtime_ta_pack_contains_native_stochrsi90() -> None:
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


def _runtime_bull_item() -> dict:
    ta = {
        "base_decision": "long",
        "rsi": 56.0,
        "adx": 28.0,
        "atr_ratio": 0.012,
        "ema": {"fast": 109.2, "slow": 108.8},
    }
    ta.update(_stoch_ta())
    return {
        "symbol": "BTC/USDT",
        "runtime_mode": "paper",
        "regime": "bull",
        "edge_validated": True,
        "ta_pack": ta,
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


def test_runtime_exposes_parallel_stochrsi_without_double_live_order(monkeypatch) -> None:
    import runtime.runtime_loop_services as loop_services

    item = _runtime_bull_item()

    async def fake_analyze(_exchange, _symbol, runtime_mode="paper"):
        result = dict(item)
        result["runtime_mode"] = runtime_mode
        return result

    monkeypatch.setattr(loop_services, "_analyze_one", fake_analyze)

    class DummyExchange:
        async def create_order(self, *args, **kwargs):
            raise AssertionError("paper runtime must never submit a real order")

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
