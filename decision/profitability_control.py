from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math
from typing import Any

HARD_NO_TRADE = frozenset({"shock", "conflict", "unknown"})
DEFAULT_REGIME_CAPS = {
    "bull": 1.0,
    "bear": 1.0,
    "range": 0.75,
    "compression": 0.60,
    "transition": 0.35,
    "shock": 0.0,
    "conflict": 0.0,
    "unknown": 0.0,
}


@dataclass(frozen=True)
class ProfitabilityControlResult:
    allowed: bool
    reason: str
    risk_scale: float
    max_leverage: float
    checks: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _f(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _load_config(config: dict[str, Any] | None = None, config_file: str | Path | None = None) -> dict[str, Any]:
    if isinstance(config, dict):
        return config
    if config_file is None:
        return {}
    try:
        payload = json.loads(Path(config_file).read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _edge_validated(item: dict[str, Any], decision: dict[str, Any]) -> bool:
    if item.get("edge_validated") is True or decision.get("edge_validated") is True:
        return True
    for source in (item.get("edge"), decision.get("edge_contract"), decision.get("edge")):
        if isinstance(source, dict):
            if source.get("validated") is True:
                return True
            if str(source.get("status") or "").lower() in {"approved", "validated", "confirmed"}:
                return True
    return False


def _round_trip_cost_bps(item: dict[str, Any], cfg: dict[str, Any]) -> float:
    fees = cfg.get("execution_fees") if isinstance(cfg.get("execution_fees"), dict) else {}
    paper = cfg.get("paper_trading") if isinstance(cfg.get("paper_trading"), dict) else {}
    taker = _f(item.get("taker_fee")) or _f(fees.get("taker_fee")) or _f(paper.get("taker_fee")) or 0.0005
    slippage_bps = _f(item.get("estimated_slippage_bps"))
    if slippage_bps is None:
        slippage_bps = _f(paper.get("slippage_bps"))
    if slippage_bps is None:
        slip_frac = _f(item.get("estimated_slippage_pct"))
        if slip_frac is None:
            slip_frac = _f(fees.get("slippage_tolerance"))
        slippage_bps = max(0.0, float(slip_frac or 0.0) * 10000.0)
    spread_bps = max(0.0, float(_f(item.get("spread_bps")) or 0.0))
    funding = abs(float(_f(item.get("funding_rate_8h")) or _f(fees.get("funding_estimate")) or 0.0))
    expected_hold_hours = max(0.0, float(_f(item.get("expected_hold_hours")) or 4.0))
    fee_bps = max(0.0, taker) * 10000.0 * 2.0
    funding_bps = funding * 10000.0 * expected_hold_hours / 8.0
    return fee_bps + 2.0 * max(0.0, slippage_bps) + 2.0 * spread_bps + funding_bps


def _expected_edge_bps(item: dict[str, Any], decision: dict[str, Any]) -> float | None:
    for source in (item, decision, item.get("edge"), decision.get("edge_contract"), decision.get("edge")):
        if not isinstance(source, dict):
            continue
        for key in ("expected_edge_bps", "net_edge_bps", "edge_bps"):
            value = _f(source.get(key))
            if value is not None:
                return value
        for key in ("expected_edge", "net_edge", "edge"):
            value = _f(source.get(key))
            if value is not None:
                return value * 10000.0 if abs(value) <= 1.0 else value
    return None


def _setup_stats(item: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    for source in (item.get("setup_expectancy"), decision.get("setup_expectancy"), item.get("edge_stats")):
        if isinstance(source, dict):
            return source
    return {}


def evaluate_profitability_control(
    *,
    decision: dict[str, Any],
    item: dict[str, Any],
    config: dict[str, Any] | None = None,
    config_file: str | Path | None = None,
) -> ProfitabilityControlResult:
    cfg = _load_config(config, config_file)
    pipeline = cfg.get("pipeline_v2") if isinstance(cfg.get("pipeline_v2"), dict) else {}
    pcfg = pipeline.get("profitability_control") if isinstance(pipeline.get("profitability_control"), dict) else {}
    action = str(decision.get("action") or "").strip().lower()
    if pcfg.get("enabled") is False:
        checks = {"action": action, "bypass": "profitability_control_disabled"}
        return ProfitabilityControlResult(
            True,
            "disabled",
            max(0.0, min(1.0, float(_f(decision.get("risk_scale")) or 1.0))),
            max(0.0, float(_f(decision.get("max_leverage")) or _f(decision.get("lev")) or 1.0)),
            checks,
        )
    regime = str(decision.get("regime") or item.get("regime") or "unknown").strip().lower()
    setup_id = str(decision.get("setup_id") or "")
    checks: dict[str, Any] = {"action": action, "regime": regime, "setup_id": setup_id}

    if action not in {"enter", "buy", "sell"}:
        return ProfitabilityControlResult(True, "non_entry", 0.0, 0.0, checks)

    if regime in HARD_NO_TRADE:
        checks["regime_gate"] = "blocked"
        return ProfitabilityControlResult(False, f"hard_no_trade_regime:{regime}", 0.0, 0.0, checks)

    existing_scale = max(0.0, min(1.0, float(_f(decision.get("risk_scale")) or 1.0)))
    existing_lev = max(0.0, float(_f(decision.get("max_leverage")) or _f(decision.get("lev")) or 1.0))
    caps = dict(DEFAULT_REGIME_CAPS)
    if isinstance(pcfg.get("regime_size_caps"), dict):
        for key, value in pcfg["regime_size_caps"].items():
            parsed = _f(value)
            if parsed is not None:
                caps[str(key).lower()] = max(0.0, min(1.0, parsed))
    scale = min(existing_scale, caps.get(regime, 0.0))
    max_lev = min(existing_lev, 1.0 if regime in {"range", "compression", "transition"} else 2.0)

    if regime == "transition":
        confirmed = (
            setup_id.startswith("transition_confirm.")
            and float(_f(decision.get("master_confidence")) or _f(decision.get("confidence")) or 0.0)
            >= float(_f(pcfg.get("transition_min_confidence")) or 0.74)
        )
        checks["transition_confirmed"] = confirmed
        if not confirmed:
            return ProfitabilityControlResult(False, "transition_confirmation_missing", 0.0, 0.0, checks)
        scale = min(scale, 0.35)
        max_lev = min(max_lev, 1.0)

    strict_edge = bool(pipeline.get("strict_edge_evidence", True))
    edge_ok = _edge_validated(item, decision)
    checks["edge_validated"] = edge_ok
    if strict_edge and not edge_ok:
        return ProfitabilityControlResult(False, "strict_edge_evidence_missing", 0.0, 0.0, checks)

    expected_edge_bps = _expected_edge_bps(item, decision)
    round_trip_cost_bps = _round_trip_cost_bps(item, cfg)
    min_buffer = float(_f(pcfg.get("min_edge_buffer_bps")) or 5.0)
    checks.update({
        "expected_edge_bps": expected_edge_bps,
        "round_trip_cost_bps": round(round_trip_cost_bps, 6),
        "min_edge_buffer_bps": min_buffer,
    })
    if expected_edge_bps is None:
        if bool(pcfg.get("require_costed_edge", True)):
            return ProfitabilityControlResult(False, "costed_edge_missing", 0.0, 0.0, checks)
    elif expected_edge_bps <= round_trip_cost_bps + min_buffer:
        return ProfitabilityControlResult(False, "edge_does_not_clear_costs", 0.0, 0.0, checks)

    meta = decision.get("meta_quality") if isinstance(decision.get("meta_quality"), dict) else {}
    meta_p = _f(item.get("meta_quality_probability"))
    if meta_p is None:
        meta_p = _f(meta.get("probability"))
    if meta_p is not None:
        block_below = float(_f(pcfg.get("meta_block_below")) or 0.45)
        reduce_below = float(_f(pcfg.get("meta_reduce_below")) or 0.60)
        checks["meta_quality_probability"] = meta_p
        if meta_p < block_below:
            return ProfitabilityControlResult(False, "meta_quality_veto", 0.0, 0.0, checks)
        if meta_p < reduce_below:
            scale = min(scale, float(_f(pcfg.get("meta_reduced_size_scale")) or 0.50))

    stats = _setup_stats(item, decision)
    samples = int(_f(stats.get("samples") or stats.get("trade_count") or stats.get("n")) or 0)
    pf = _f(stats.get("profit_factor"))
    exp_r = _f(stats.get("expectancy_r"))
    lower_ci = _f(stats.get("expectancy_r_ci95_low") or stats.get("lower_95_expectancy_r"))
    checks["setup_expectancy"] = {"samples": samples, "profit_factor": pf, "expectancy_r": exp_r, "ci95_low": lower_ci}
    min_samples = int(_f(pcfg.get("min_setup_samples")) or 100)
    if samples < min_samples:
        scale = min(scale, float(_f(pcfg.get("sample_limited_size_scale")) or 0.25))
        max_lev = min(max_lev, 1.0)
    else:
        if pf is not None and pf < float(_f(pcfg.get("min_profit_factor")) or 1.15):
            return ProfitabilityControlResult(False, "setup_profit_factor_below_gate", 0.0, 0.0, checks)
        if exp_r is not None and exp_r < float(_f(pcfg.get("min_expectancy_r")) or 0.05):
            return ProfitabilityControlResult(False, "setup_expectancy_below_gate", 0.0, 0.0, checks)
        if lower_ci is not None and lower_ci <= 0.0:
            return ProfitabilityControlResult(False, "setup_expectancy_ci_not_positive", 0.0, 0.0, checks)

    scale = max(0.0, min(existing_scale, scale, 1.0))
    if scale < 1.0:
        max_lev = min(max_lev, 1.0)
    checks["final_scale"] = scale
    checks["final_max_leverage"] = max_lev
    if scale <= 0.0:
        return ProfitabilityControlResult(False, "size_resolved_to_zero", 0.0, 0.0, checks)
    return ProfitabilityControlResult(True, "profitability_gate_passed", scale, max_lev, checks)


def apply_profitability_control(
    *,
    decision: dict[str, Any],
    item: dict[str, Any],
    config: dict[str, Any] | None = None,
    config_file: str | Path | None = None,
) -> dict[str, Any]:
    payload = dict(decision)
    result = evaluate_profitability_control(decision=payload, item=item, config=config, config_file=config_file)
    payload["profitability_xray"] = result.to_dict()
    if str(payload.get("action") or "").lower() not in {"enter", "buy", "sell"}:
        return payload
    if not result.allowed:
        payload["action"] = "hold"
        payload["direction"] = "neutral"
        payload["base_decision"] = "neutral"
        payload["risk_scale"] = 0.0
        payload["lev"] = 0
        payload["max_leverage"] = 0.0
        previous = str(payload.get("reason") or "").strip()
        payload["reason"] = f"{previous} | profitability:{result.reason}".strip(" |")
        return payload
    payload["risk_scale"] = round(min(float(_f(payload.get("risk_scale")) or 1.0), result.risk_scale), 6)
    current_lev = float(_f(payload.get("lev")) or _f(payload.get("max_leverage")) or result.max_leverage)
    payload["lev"] = int(max(1.0, min(current_lev, result.max_leverage)))
    payload["max_leverage"] = float(payload["lev"])
    return payload


__all__ = ["ProfitabilityControlResult", "apply_profitability_control", "evaluate_profitability_control"]
