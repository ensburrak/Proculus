from __future__ import annotations

import meta_strategy_selector as selector


def test_shock_is_hard_no_trade(monkeypatch) -> None:
    monkeypatch.setattr(selector, "_pipeline_v2_enabled", lambda: False)

    result = selector.select_strategy(regime="SHOCK", regime_confidence=0.95)

    assert result["action"] == "no_trade"
    assert result["direction_bias"] == "neutral"
    assert result["risk_size_cap"] == 0.0


def test_aligned_ai_cannot_boost_directional_confidence(monkeypatch) -> None:
    monkeypatch.setattr(selector, "_pipeline_v2_enabled", lambda: False)

    baseline = selector.select_strategy(regime="BULL", regime_confidence=0.75)
    with_ai = selector.select_strategy(
        regime="BULL",
        regime_confidence=0.75,
        ai_scores={"chatgpt": 0.90, "deepseek": 0.90},
    )

    assert with_ai["direction_bias"] == "long"
    assert with_ai["confidence"] == baseline["confidence"]


def test_breakout_strategy_never_amplifies_position_size() -> None:
    multipliers = selector.get_strategy_multipliers("breakout")

    assert multipliers["position_size"] <= 1.0


def test_transition_requires_confirmation_in_fallback(monkeypatch) -> None:
    monkeypatch.setattr(selector, "_pipeline_v2_enabled", lambda: False)

    result = selector.select_strategy(regime="TRANSITION", regime_confidence=0.8)

    assert result["action"] == "no_trade"
    assert result["strategy"] == "confirmation"
    assert result["risk_size_cap"] == 0.0
