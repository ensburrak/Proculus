from __future__ import annotations

from decision.profitability_control import apply_profitability_control, evaluate_profitability_control
from research.profitability_evidence import (
    build_tca,
    counterfactual_report,
    evaluate_champion_challenger,
    evaluate_release_gate,
    summarize_edge,
)


def _cfg() -> dict:
    return {
        "execution_fees": {
            "taker_fee": 0.0005,
            "funding_estimate": 0.0002,
            "slippage_tolerance": 0.0005,
        },
        "pipeline_v2": {
            "strict_edge_evidence": True,
            "profitability_control": {
                "require_costed_edge": True,
                "min_edge_buffer_bps": 5,
                "min_setup_samples": 100,
                "min_profit_factor": 1.15,
                "min_expectancy_r": 0.05,
                "sample_limited_size_scale": 0.25,
                "transition_min_confidence": 0.74,
            },
        },
    }


def _entry(**updates):
    payload = {
        "action": "enter",
        "direction": "long",
        "base_decision": "long",
        "regime": "bull",
        "setup_id": "bull_trend.pullback.long.15m.v1",
        "master_confidence": 0.80,
        "risk_scale": 0.80,
        "lev": 2,
        "edge_contract": {"validated": True},
    }
    payload.update(updates)
    return payload


def _item(**updates):
    payload = {
        "edge_validated": True,
        "expected_edge_bps": 45.0,
        "estimated_slippage_bps": 2.0,
        "spread_bps": 1.0,
        "expected_hold_hours": 4.0,
        "setup_expectancy": {
            "samples": 150,
            "profit_factor": 1.30,
            "expectancy_r": 0.12,
            "expectancy_r_ci95_low": 0.03,
        },
    }
    payload.update(updates)
    return payload


def test_profitability_control_blocks_hard_no_trade_regime() -> None:
    result = evaluate_profitability_control(
        decision=_entry(regime="shock"),
        item=_item(),
        config=_cfg(),
    )
    assert result.allowed is False
    assert result.reason == "hard_no_trade_regime:shock"


def test_profitability_control_requires_edge_to_clear_costs() -> None:
    result = evaluate_profitability_control(
        decision=_entry(),
        item=_item(expected_edge_bps=5.0),
        config=_cfg(),
    )
    assert result.allowed is False
    assert result.reason == "edge_does_not_clear_costs"


def test_profitability_control_never_amplifies_size_or_leverage() -> None:
    decision = _entry(risk_scale=0.40, lev=2)
    guarded = apply_profitability_control(
        decision=decision,
        item=_item(setup_expectancy={"samples": 20}),
        config=_cfg(),
    )
    assert guarded["action"] == "enter"
    assert guarded["risk_scale"] <= 0.25
    assert guarded["risk_scale"] <= decision["risk_scale"]
    assert guarded["lev"] == 1


def test_transition_is_confirmation_only_and_small() -> None:
    blocked = evaluate_profitability_control(
        decision=_entry(regime="transition", setup_id="generic.transition.long", master_confidence=0.90),
        item=_item(),
        config=_cfg(),
    )
    assert blocked.allowed is False
    assert blocked.reason == "transition_confirmation_missing"

    passed = evaluate_profitability_control(
        decision=_entry(
            regime="transition",
            setup_id="transition_confirm.retest.long.15m.v1",
            master_confidence=0.80,
        ),
        item=_item(),
        config=_cfg(),
    )
    assert passed.allowed is True
    assert passed.risk_scale <= 0.35
    assert passed.max_leverage <= 1.0


def test_negative_setup_expectancy_is_fail_closed() -> None:
    result = evaluate_profitability_control(
        decision=_entry(),
        item=_item(
            setup_expectancy={
                "samples": 150,
                "profit_factor": 0.95,
                "expectancy_r": -0.03,
                "expectancy_r_ci95_low": -0.08,
            }
        ),
        config=_cfg(),
    )
    assert result.allowed is False
    assert result.reason in {"setup_profit_factor_below_gate", "setup_expectancy_below_gate"}


def test_edge_observatory_and_tca_are_cost_aware() -> None:
    summary = summarize_edge(
        [
            {"r_multiple_net": 1.0, "gross_pnl_usd": 12, "net_pnl_usd": 10, "slippage_bps": 2},
            {"r_multiple_net": -0.5, "gross_pnl_usd": -4, "net_pnl_usd": -5, "slippage_bps": 4},
            {"r_multiple_net": 0.8, "gross_pnl_usd": 10, "net_pnl_usd": 8, "slippage_bps": 3},
        ]
    )
    assert summary.trades == 3
    assert summary.expectancy_r > 0
    assert summary.cost_drag_usd == 5
    assert summary.avg_slippage_bps == 3

    tca = build_tca(
        [
            {
                "filled": True,
                "order_type": "market",
                "implementation_shortfall_bps": 6,
                "slippage_bps": 3,
                "latency_ms": 120,
                "adverse_selection_bps": 2,
                "fee_usd": 1,
                "funding_usd": 0.2,
                "cost_drag_usd": 2,
            },
            {"filled": False, "status": "missed"},
        ]
    )
    assert tca.fills == 1
    assert tca.missed_fills == 1
    assert tca.avg_latency_ms == 120


def test_counterfactual_and_champion_challenger_contracts() -> None:
    cf = counterfactual_report(
        [
            {"accepted": False, "counterfactual_r": -1.0},
            {"accepted": False, "counterfactual_r": 0.4},
            {"accepted": True, "counterfactual_r": 0.5},
        ]
    )
    assert cf["avoided_losses"] == 1
    assert cf["missed_winners"] == 1

    promotion = evaluate_champion_challenger(
        champion={"expectancy_r": 0.08, "profit_factor": 1.2, "max_drawdown_r": 8},
        challenger={
            "expectancy_r": 0.12,
            "profit_factor": 1.35,
            "expectancy_r_ci95_low": 0.03,
            "max_drawdown_r": 8.5,
        },
    )
    assert promotion["eligible_for_shadow_promotion"] is True


def test_release_gate_requires_oos_tca_integrity_and_soak() -> None:
    cfg = {
        "min_oos_trades": 250,
        "min_oos_profit_factor": 1.15,
        "min_oos_expectancy_r": 0.05,
        "min_positive_fold_share": 0.67,
        "min_shadow_hours": 72,
        "min_paper_days": 14,
        "min_canary_days": 14,
        "require_zero_liquidations": True,
        "require_data_integrity": True,
        "require_tca": True,
    }
    evidence = {
        "oos": {
            "trades": 300,
            "profit_factor": 1.3,
            "expectancy_r": 0.1,
            "expectancy_r_ci95_low": 0.02,
            "positive_fold_share": 0.75,
        },
        "shadow": {"hours": 72},
        "paper": {"days": 14},
        "canary": {"days": 14, "liquidations": 0},
        "data_integrity": {"passed": True},
        "tca": {"fills": 100},
    }
    assert evaluate_release_gate(evidence, cfg)["live_allowed"] is True
    evidence["oos"]["expectancy_r_ci95_low"] = -0.01
    blocked = evaluate_release_gate(evidence, cfg)
    assert blocked["live_allowed"] is False
    assert "oos_expectancy_ci" in blocked["blockers"]
