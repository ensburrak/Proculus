from __future__ import annotations

import json

from decision.oos_setup_gate import evaluate_oos_setup_eligibility


def _write_registry(tmp_path, families):
    path = tmp_path / "oos.json"
    path.write_text(
        json.dumps({"schema": "test", "families": families}),
        encoding="utf-8",
    )
    return path


def test_scoped_family_is_fail_closed_when_not_promoted(tmp_path) -> None:
    path = _write_registry(tmp_path, {"bull_trend": {"promoted": False}})
    decision = evaluate_oos_setup_eligibility(
        "bull_trend.pullback.long.15m.v1",
        registry_path=path,
    )
    assert decision.allowed is False
    assert decision.reason == "oos_edge_not_promoted"


def test_scoped_family_is_allowed_only_when_promoted(tmp_path) -> None:
    path = _write_registry(tmp_path, {"bull_trend": {"promoted": True}})
    decision = evaluate_oos_setup_eligibility(
        "bull_trend.pullback.long.15m.v1",
        registry_path=path,
    )
    assert decision.allowed is True
    assert decision.reason == "oos_edge_promoted"


def test_unscoped_family_is_unchanged(tmp_path) -> None:
    path = _write_registry(tmp_path, {})
    decision = evaluate_oos_setup_eligibility(
        "transition_confirm.retest.long.15m.v1",
        registry_path=path,
    )
    assert decision.allowed is True
    assert decision.reason == "oos_gate_not_scoped"


def test_missing_registry_blocks_scoped_family(tmp_path) -> None:
    decision = evaluate_oos_setup_eligibility(
        "stochrsi_opportunity.bull.long.15m.v1",
        registry_path=tmp_path / "missing.json",
    )
    assert decision.allowed is False


def test_v5_cannot_be_unlocked_with_promoted_flag_alone(tmp_path) -> None:
    path = _write_registry(tmp_path, {"breadth_donchian_10_v5": {"promoted": True}})
    decision = evaluate_oos_setup_eligibility(
        "breadth_donchian_10_v5.entry.long.4h.v1",
        registry_path=path,
    )
    assert decision.allowed is False
    assert decision.reason == "v5_promotion_prerequisites_missing"
    assert set(decision.blockers) == {
        "prospective_shadow_passed",
        "exact_repo_selfhosted_passed",
        "live_cost_guard_integrated",
        "execution_authority",
    }


def test_v5_gate_requires_all_execution_prerequisites(tmp_path) -> None:
    record = {
        "promoted": True,
        "prospective_shadow_passed": True,
        "exact_repo_selfhosted_status": "passed",
        "live_cost_guard_integrated": False,
        "execution_authority": True,
    }
    path = _write_registry(tmp_path, {"breadth_donchian_10_v5": record})
    decision = evaluate_oos_setup_eligibility(
        "breadth_donchian_10_v5.entry.short.4h.v1",
        registry_path=path,
    )
    assert decision.allowed is False
    assert decision.blockers == ("live_cost_guard_integrated",)


def test_v5_gate_allows_only_completely_attested_promotion(tmp_path) -> None:
    record = {
        "promoted": True,
        "prospective_shadow_passed": True,
        "exact_repo_selfhosted_status": "passed",
        "live_cost_guard_integrated": True,
        "execution_authority": True,
    }
    path = _write_registry(tmp_path, {"breadth_donchian_10_v5": record})
    decision = evaluate_oos_setup_eligibility(
        "breadth_donchian_10_v5.entry.long.4h.v1",
        registry_path=path,
    )
    assert decision.allowed is True
    assert decision.reason == "oos_edge_promoted"
