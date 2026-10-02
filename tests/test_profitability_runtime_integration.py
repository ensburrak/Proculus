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
