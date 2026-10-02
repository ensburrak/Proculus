from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, asdict
from math import sqrt
from statistics import mean, pstdev
from typing import Any, Iterable


@dataclass(frozen=True)
class EdgeSummary:
    trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float | None
    payoff_ratio: float | None
    expectancy_r: float
    expectancy_r_ci95_low: float
    expectancy_r_ci95_high: float
    avg_r: float
    max_drawdown_r: float
    sharpe_like: float
    sortino_like: float
    gross_pnl_usd: float
    net_pnl_usd: float
    cost_drag_usd: float
    avg_slippage_bps: float | None
    avg_mae_r: float | None
    avg_mfe_r: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TcaSummary:
    fills: int
    missed_fills: int
    maker_fills: int
    taker_fills: int
    avg_implementation_shortfall_bps: float | None
    avg_slippage_bps: float | None
    avg_latency_ms: float | None
    avg_adverse_selection_bps: float | None
    total_fee_usd: float
    total_funding_usd: float
    total_cost_drag_usd: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _f(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed == parsed else default


def _optional_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def summarize_edge(trades: Iterable[dict[str, Any]]) -> EdgeSummary:
    rows = [dict(row) for row in trades if isinstance(row, dict)]
    r_values: list[float] = []
    net_pnls: list[float] = []
    gross_pnls: list[float] = []
    slippage_bps: list[float] = []
    mae_r: list[float] = []
    mfe_r: list[float] = []

    for row in rows:
        r = _f(row.get("r_multiple_net", row.get("r_multiple", row.get("r", 0.0))))
        r_values.append(r)
        net_pnls.append(_f(row.get("net_pnl_usd", row.get("pnl_usd", row.get("net_pnl", 0.0)))))
        gross_pnls.append(_f(row.get("gross_pnl_usd", row.get("gross_pnl", row.get("pnl_usd", 0.0)))))
        if row.get("slippage_bps") is not None:
            slippage_bps.append(_f(row.get("slippage_bps")))
        if row.get("mae_r") is not None:
            mae_r.append(_f(row.get("mae_r")))
        if row.get("mfe_r") is not None:
            mfe_r.append(_f(row.get("mfe_r")))

    wins = [r for r in r_values if r > 0]
    losses = [r for r in r_values if r <= 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    pf = (gross_win / gross_loss) if gross_loss > 0 else (None if not wins else float("inf"))
    avg_win = mean(wins) if wins else 0.0
    avg_loss = abs(mean(losses)) if losses else 0.0
    payoff = (avg_win / avg_loss) if avg_loss > 0 else (None if not wins else float("inf"))

    n = len(r_values)
    exp_r = mean(r_values) if r_values else 0.0
    sd = pstdev(r_values) if len(r_values) > 1 else 0.0
    se = sd / sqrt(n) if n > 1 else 0.0
    ci_low = exp_r - 1.96 * se
    ci_high = exp_r + 1.96 * se

    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in r_values:
        equity += r
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    downside = [min(0.0, r) for r in r_values]
    downside_sd = sqrt(sum(x * x for x in downside) / len(downside)) if downside else 0.0
    sharpe_like = (exp_r / sd * sqrt(n)) if sd > 0 and n > 1 else 0.0
    sortino_like = (exp_r / downside_sd * sqrt(n)) if downside_sd > 0 and n > 1 else 0.0

    gross = sum(gross_pnls)
    net = sum(net_pnls)
    return EdgeSummary(
        trades=n,
        wins=len(wins),
        losses=len(losses),
        win_rate=(len(wins) / n) if n else 0.0,
        profit_factor=pf,
        payoff_ratio=payoff,
        expectancy_r=exp_r,
        expectancy_r_ci95_low=ci_low,
        expectancy_r_ci95_high=ci_high,
        avg_r=exp_r,
        max_drawdown_r=max_dd,
        sharpe_like=sharpe_like,
        sortino_like=sortino_like,
        gross_pnl_usd=gross,
        net_pnl_usd=net,
        cost_drag_usd=gross - net,
        avg_slippage_bps=_optional_mean(slippage_bps),
        avg_mae_r=_optional_mean(mae_r),
        avg_mfe_r=_optional_mean(mfe_r),
    )


def summarize_by_key(trades: Iterable[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trades:
        if isinstance(row, dict):
            groups[str(row.get(key) or "unknown")].append(dict(row))
    return {group: summarize_edge(rows).to_dict() for group, rows in sorted(groups.items())}


def build_tca(events: Iterable[dict[str, Any]]) -> TcaSummary:
    rows = [dict(row) for row in events if isinstance(row, dict)]
    shortfalls: list[float] = []
    slips: list[float] = []
    latencies: list[float] = []
    adverse: list[float] = []
    fees = 0.0
    funding = 0.0
    cost_drag = 0.0
    missed = 0
    maker = 0
    taker = 0
    fills = 0

    for row in rows:
        status = str(row.get("status") or row.get("fill_status") or "").lower()
        filled = bool(row.get("filled")) or status in {"filled", "partial", "complete"}
        if not filled:
            missed += 1
            continue
        fills += 1
        order_type = str(row.get("order_type") or "market").lower()
        if order_type == "maker" or order_type == "limit":
            maker += 1
        else:
            taker += 1
        if row.get("implementation_shortfall_bps") is not None:
            shortfalls.append(_f(row.get("implementation_shortfall_bps")))
        if row.get("slippage_bps") is not None:
            slips.append(_f(row.get("slippage_bps")))
        if row.get("latency_ms") is not None:
            latencies.append(_f(row.get("latency_ms")))
        if row.get("adverse_selection_bps") is not None:
            adverse.append(_f(row.get("adverse_selection_bps")))
        fees += _f(row.get("fee_usd"))
        funding += _f(row.get("funding_usd"))
        cost_drag += _f(row.get("cost_drag_usd"))

    return TcaSummary(
        fills=fills,
        missed_fills=missed,
        maker_fills=maker,
        taker_fills=taker,
        avg_implementation_shortfall_bps=_optional_mean(shortfalls),
        avg_slippage_bps=_optional_mean(slips),
        avg_latency_ms=_optional_mean(latencies),
        avg_adverse_selection_bps=_optional_mean(adverse),
        total_fee_usd=fees,
        total_funding_usd=funding,
        total_cost_drag_usd=cost_drag,
    )


def counterfactual_report(candidates: Iterable[dict[str, Any]]) -> dict[str, Any]:
    accepted: list[float] = []
    rejected: list[float] = []
    avoided_losses = 0
    missed_winners = 0
    for row in candidates:
        if not isinstance(row, dict):
            continue
        realized_r = _f(row.get("counterfactual_r", row.get("realized_r", 0.0)))
        accepted_flag = bool(row.get("accepted")) or str(row.get("decision") or "").lower() in {"enter", "accepted"}
        if accepted_flag:
            accepted.append(realized_r)
        else:
            rejected.append(realized_r)
            if realized_r < 0:
                avoided_losses += 1
            elif realized_r > 0:
                missed_winners += 1
    return {
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted_expectancy_r": mean(accepted) if accepted else 0.0,
        "rejected_counterfactual_expectancy_r": mean(rejected) if rejected else 0.0,
        "avoided_losses": avoided_losses,
        "missed_winners": missed_winners,
        "net_filter_value_r": -sum(rejected),
    }


def evaluate_champion_challenger(
    *,
    champion: dict[str, Any],
    challenger: dict[str, Any],
    max_oos_degradation_pct: float = 25.0,
) -> dict[str, Any]:
    c_exp = _f(champion.get("expectancy_r"))
    h_exp = _f(challenger.get("expectancy_r"))
    c_pf = _f(champion.get("profit_factor"))
    h_pf = _f(challenger.get("profit_factor"))
    h_ci = _f(challenger.get("expectancy_r_ci95_low"))
    c_dd = abs(_f(champion.get("max_drawdown_r")))
    h_dd = abs(_f(challenger.get("max_drawdown_r")))
    degradation = ((c_exp - h_exp) / max(abs(c_exp), 1e-9) * 100.0) if c_exp > 0 else 0.0
    improved = h_exp > c_exp and h_pf >= max(1.0, c_pf) and h_ci > 0.0
    dd_ok = h_dd <= max(c_dd * 1.25, c_dd + 1.0)
    eligible = improved and dd_ok and degradation <= float(max_oos_degradation_pct)
    return {
        "eligible_for_shadow_promotion": eligible,
        "champion_expectancy_r": c_exp,
        "challenger_expectancy_r": h_exp,
        "champion_profit_factor": c_pf,
        "challenger_profit_factor": h_pf,
        "challenger_ci95_low": h_ci,
        "drawdown_ok": dd_ok,
        "oos_degradation_pct": degradation,
    }


def evaluate_release_gate(evidence: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(config or {})
    blockers: list[str] = []
    oos = evidence.get("oos") if isinstance(evidence.get("oos"), dict) else {}
    shadow = evidence.get("shadow") if isinstance(evidence.get("shadow"), dict) else {}
    paper = evidence.get("paper") if isinstance(evidence.get("paper"), dict) else {}
    canary = evidence.get("canary") if isinstance(evidence.get("canary"), dict) else {}
    integrity = evidence.get("data_integrity") if isinstance(evidence.get("data_integrity"), dict) else {}
    tca = evidence.get("tca") if isinstance(evidence.get("tca"), dict) else {}

    min_trades = int(_f(cfg.get("min_oos_trades"), 250))
    min_pf = _f(cfg.get("min_oos_profit_factor"), 1.15)
    min_exp = _f(cfg.get("min_oos_expectancy_r"), 0.05)
    if int(_f(oos.get("trades"))) < min_trades:
        blockers.append("oos_trade_count")
    if _f(oos.get("profit_factor")) < min_pf:
        blockers.append("oos_profit_factor")
    if _f(oos.get("expectancy_r")) < min_exp:
        blockers.append("oos_expectancy")
    if bool(cfg.get("require_positive_expectancy_ci95_low", True)) and _f(oos.get("expectancy_r_ci95_low")) <= 0:
        blockers.append("oos_expectancy_ci")
    if _f(oos.get("positive_fold_share")) < _f(cfg.get("min_positive_fold_share"), 0.67):
        blockers.append("oos_fold_stability")
    if _f(shadow.get("hours")) < _f(cfg.get("min_shadow_hours"), 72):
        blockers.append("shadow_duration")
    if _f(paper.get("days")) < _f(cfg.get("min_paper_days"), 14):
        blockers.append("paper_duration")
    if _f(canary.get("days")) < _f(cfg.get("min_canary_days"), 14):
        blockers.append("canary_duration")
    if bool(cfg.get("require_zero_liquidations", True)) and int(_f(canary.get("liquidations"))) > 0:
        blockers.append("canary_liquidation")
    if bool(cfg.get("require_data_integrity", True)) and integrity.get("passed") is not True:
        blockers.append("data_integrity")
    if bool(cfg.get("require_tca", True)) and int(_f(tca.get("fills"))) <= 0:
        blockers.append("tca_missing")

    stage = "live" if not blockers else (
        "canary" if not any(x.startswith("oos_") or x in {"data_integrity", "tca_missing"} for x in blockers)
        else "research"
    )
    return {
        "allowed_stage": stage,
        "live_allowed": not blockers,
        "blockers": blockers,
        "fail_closed": True,
    }


__all__ = [
    "EdgeSummary",
    "TcaSummary",
    "build_tca",
    "counterfactual_report",
    "evaluate_champion_challenger",
    "evaluate_release_gate",
    "summarize_by_key",
    "summarize_edge",
]
