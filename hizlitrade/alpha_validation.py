from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .oos import (
    SettledTrade,
    aggregate_markets,
    load_research_trades,
    purged_market_walk_forward_folds,
)
from .validation import trade_return_summary


@dataclass(frozen=True, slots=True)
class AdaptivePolicyConfig:
    min_symbol_train_markets: int = 20
    min_setup_train_markets: int = 20
    min_ml_train_trades: int = 80


@dataclass(frozen=True, slots=True)
class LogisticModel:
    means: tuple[float, ...]
    scales: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float
    threshold: float

    def probability(self, trade: SettledTrade) -> float:
        raw = _feature_vector(trade)
        standardized = [
            (value - mean) / scale
            for value, mean, scale in zip(raw, self.means, self.scales, strict=True)
        ]
        z = self.bias + sum(
            weight * value
            for weight, value in zip(self.weights, standardized, strict=True)
        )
        if z >= 0:
            exp_neg = math.exp(-min(z, 60.0))
            return 1.0 / (1.0 + exp_neg)
        exp_pos = math.exp(max(z, -60.0))
        return exp_pos / (1.0 + exp_pos)


@dataclass(frozen=True, slots=True)
class PortfolioReplayConfig:
    initial_balance: float = 100.0
    max_open_exposure_pct: float = 0.20
    max_market_exposure_pct: float = 0.05
    max_symbol_exposure_pct: float = 0.10
    max_concurrent_positions: int = 8
    max_daily_loss_pct: float = 0.05
    max_drawdown_pct: float = 0.10


@dataclass(frozen=True, slots=True)
class PortfolioReplayResult:
    accepted_trades: tuple[SettledTrade, ...]
    rejected: tuple[tuple[str, int], ...]
    final_cash: float
    realized_pnl_usd: float
    max_drawdown_pct: float

    @property
    def rejection_counts(self) -> dict[str, int]:
        return dict(self.rejected)


@dataclass(frozen=True, slots=True)
class FittedPolicy:
    selected_symbols: tuple[str, ...]
    symbol_gate_available: bool
    selected_setup_families: tuple[str, ...]
    setup_gate_available: bool
    regime_low_max: float | None
    regime_medium_max: float | None
    selected_regimes: tuple[str, ...]
    regime_gate_available: bool
    min_consensus: float
    consensus_gate_available: bool
    consensus_no_trade: bool
    min_net_edge: float | None
    max_prediction_spread: float | None
    execution_gate_available: bool
    execution_no_trade: bool
    ml_model: LogisticModel | None
    ml_gate_available: bool
    ml_no_trade: bool


def _direction(trade: SettledTrade) -> float:
    return 1.0 if getattr(trade, "outcome", "YES").upper() == "YES" else -1.0


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def replay_portfolio(
    trades: list[SettledTrade],
    config: PortfolioReplayConfig | None = None,
) -> PortfolioReplayResult:
    """Replay selected research trades under the live paper-account risk envelope.

    Admission uses only information known at signal time. Outcomes affect cash
    only when the recorded settlement timestamp is reached. Open binary
    positions are carried at cost for equity accounting because historical
    counterfactual rows do not contain a continuous executable mark stream;
    the hard worst-case budget remains cash-based, matching RiskEngine.
    """
    config = config or PortfolioReplayConfig()
    if config.initial_balance <= 0:
        raise ValueError("initial_balance must be positive")
    if config.max_concurrent_positions <= 0:
        raise ValueError("max_concurrent_positions must be positive")

    def trade_priority(trade: SettledTrade) -> tuple[int, float, str, str]:
        edge = _finite(trade.net_edge)
        edge_rank = edge if edge is not None else -1e9
        return (
            trade.signal_ts_ns,
            -edge_rank,
            trade.market_id,
            trade.instrument,
        )

    ordered = sorted(trades, key=trade_priority)
    cash = config.initial_balance
    start_of_day_equity = config.initial_balance
    peak_equity = config.initial_balance
    max_drawdown = 0.0
    risk_day: int | None = None
    open_positions: list[SettledTrade] = []
    accepted: list[SettledTrade] = []
    rejections: Counter[str] = Counter()

    def equity() -> float:
        return cash + sum(max(trade.cost_basis_usd, 0.0) for trade in open_positions)

    def refresh_drawdown() -> None:
        nonlocal peak_equity, max_drawdown
        current = equity()
        peak_equity = max(peak_equity, current)
        if peak_equity > 0:
            max_drawdown = max(max_drawdown, (peak_equity - current) / peak_equity)

    def settle_due(timestamp_ns: int) -> None:
        nonlocal cash
        remaining: list[SettledTrade] = []
        for trade in open_positions:
            if trade.settled_ts_ns <= timestamp_ns:
                cash += trade.cost_basis_usd + trade.realized_pnl_usd
            else:
                remaining.append(trade)
        open_positions[:] = remaining
        refresh_drawdown()

    for trade in ordered:
        settle_due(trade.signal_ts_ns)
        day = trade.signal_ts_ns // 86_400_000_000_000
        if risk_day != day:
            risk_day = day
            start_of_day_equity = equity()

        current_equity = equity()
        if current_equity <= 0 or cash <= 0:
            rejections["account_has_no_deployable_capital"] += 1
            continue
        if start_of_day_equity <= 0:
            rejections["invalid_start_of_day_equity"] += 1
            continue

        daily_loss = max(start_of_day_equity - current_equity, 0.0)
        if daily_loss / start_of_day_equity >= config.max_daily_loss_pct:
            rejections["daily_loss_kill_switch"] += 1
            continue
        drawdown = max(peak_equity - current_equity, 0.0)
        if peak_equity > 0 and drawdown / peak_equity >= config.max_drawdown_pct:
            rejections["max_drawdown_kill_switch"] += 1
            continue

        notional = max(trade.cost_basis_usd, 0.0)
        if notional <= 0:
            rejections["invalid_notional"] += 1
            continue
        worst_case_cash = cash - notional
        worst_daily_loss = max(start_of_day_equity - worst_case_cash, 0.0)
        if worst_daily_loss / start_of_day_equity > config.max_daily_loss_pct:
            rejections["daily_loss_budget_at_risk"] += 1
            continue
        worst_drawdown = max(peak_equity - worst_case_cash, 0.0)
        if peak_equity > 0 and worst_drawdown / peak_equity > config.max_drawdown_pct:
            rejections["drawdown_budget_at_risk"] += 1
            continue
        if len(open_positions) >= config.max_concurrent_positions:
            rejections["max_concurrent_positions"] += 1
            continue

        open_exposure = sum(position.cost_basis_usd for position in open_positions)
        if open_exposure + notional > current_equity * config.max_open_exposure_pct:
            rejections["open_exposure_limit"] += 1
            continue
        market_exposure = sum(
            position.cost_basis_usd
            for position in open_positions
            if position.market_id == trade.market_id
        )
        if market_exposure + notional > current_equity * config.max_market_exposure_pct:
            rejections["per_market_exposure_limit"] += 1
            continue
        symbol_exposure = sum(
            position.cost_basis_usd
            for position in open_positions
            if position.symbol == trade.symbol
        )
        if symbol_exposure + notional > current_equity * config.max_symbol_exposure_pct:
            rejections["correlated_symbol_exposure_limit"] += 1
            continue
        if notional > cash:
            rejections["insufficient_cash"] += 1
            continue

        cash -= notional
        open_positions.append(trade)
        accepted.append(trade)
        refresh_drawdown()

    settle_due(10**30)
    return PortfolioReplayResult(
        accepted_trades=tuple(accepted),
        rejected=tuple(sorted(rejections.items())),
        final_cash=cash,
        realized_pnl_usd=cash - config.initial_balance,
        max_drawdown_pct=max_drawdown,
    )


def _portfolio_summary(result: PortfolioReplayResult) -> dict[str, Any]:
    accepted = list(result.accepted_trades)
    summary = _summary(accepted)
    return {
        **summary,
        "final_cash": result.final_cash,
        "portfolio_realized_pnl_usd": result.realized_pnl_usd,
        "max_drawdown_pct": result.max_drawdown_pct,
        "rejections": result.rejection_counts,
    }


def technical_consensus_score(trade: SettledTrade) -> float:
    direction = _direction(trade)
    values = (
        getattr(trade, "oracle_basis_bps", None),
        getattr(trade, "momentum_250ms_bps", None),
        getattr(trade, "momentum_1s_bps", None),
        getattr(trade, "momentum_3s_bps", None),
        getattr(trade, "book_imbalance", None),
    )
    observed = 0
    aligned = 0
    for raw in values:
        value = _finite(raw)
        if value is None or abs(value) <= 1e-12:
            continue
        observed += 1
        if value * direction > 0:
            aligned += 1
    return aligned / observed if observed else 0.0


def _setup_family(trade: SettledTrade) -> str:
    duration_ms = _finite(getattr(trade, "market_duration_ms", None))
    if duration_ms is None or duration_ms <= 0:
        return "unknown"
    minutes = max(1, round(duration_ms / 60_000))
    return f"{minutes}m"


def _fit_setup_families(
    trades: list[SettledTrade],
    min_markets: int,
) -> tuple[tuple[str, ...], bool]:
    grouped: dict[tuple[str, str], list[SettledTrade]] = {}
    for trade in trades:
        grouped.setdefault((trade.market_id, _setup_family(trade)), []).append(trade)

    per_family: dict[str, list[float]] = {}
    all_returns: list[float] = []
    for (_market_id, family), rows in grouped.items():
        cost = sum(row.cost_basis_usd for row in rows)
        pnl = sum(row.realized_pnl_usd for row in rows)
        value = pnl / cost if cost > 0 else 0.0
        per_family.setdefault(family, []).append(value)
        all_returns.append(value)

    all_families = tuple(sorted(per_family))
    evidence_available = any(
        len(returns) >= min_markets for returns in per_family.values()
    )
    if not evidence_available:
        return all_families, False

    selected = tuple(
        sorted(
            family
            for family, returns in per_family.items()
            if len(returns) >= min_markets and statistics.fmean(returns) > 0
        )
    )
    selected_returns = [
        value
        for family in selected
        for value in per_family[family]
    ]
    selected_score = _selection_score(selected_returns)
    baseline_score = _selection_score(all_returns)
    if selected and selected_score > max(baseline_score, 0.0):
        return selected, True
    if baseline_score > 0.0:
        return all_families, True
    return (), True


def _selection_score(values: list[float]) -> float:
    if len(values) < 5:
        return float("-inf")
    summary = trade_return_summary(values)
    lower = summary.get("mean_ci95_low")
    return float(lower) if lower is not None else float("-inf")


def _filter_improves(
    baseline: list[SettledTrade],
    selected: list[SettledTrade],
) -> bool:
    baseline_markets = _market_returns(baseline)
    selected_markets = _market_returns(selected)
    if len(selected_markets) < 5 or len(selected_markets) >= len(baseline_markets):
        return False
    baseline_score = _selection_score(list(baseline_markets.values()))
    selected_score = _selection_score(list(selected_markets.values()))
    return selected_score > max(baseline_score, 0.0)


def _market_returns(trades: list[SettledTrade]) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[SettledTrade]] = {}
    for trade in trades:
        grouped.setdefault((trade.market_id, trade.symbol), []).append(trade)
    result: dict[tuple[str, str], float] = {}
    for key, rows in grouped.items():
        cost = sum(row.cost_basis_usd for row in rows)
        pnl = sum(row.realized_pnl_usd for row in rows)
        result[key] = pnl / cost if cost > 0 else 0.0
    return result


def _fit_symbols(
    trades: list[SettledTrade],
    min_markets: int,
) -> tuple[tuple[str, ...], bool]:
    market_returns = _market_returns(trades)
    per_symbol: dict[str, list[float]] = {}
    for (_, symbol), value in market_returns.items():
        per_symbol.setdefault(symbol, []).append(value)
    all_symbols = tuple(sorted(per_symbol))
    evidence_available = any(
        len(returns) >= min_markets for returns in per_symbol.values()
    )
    if not evidence_available:
        return all_symbols, False

    selected = tuple(
        sorted(
            symbol
            for symbol, returns in per_symbol.items()
            if len(returns) >= min_markets and statistics.fmean(returns) > 0
        )
    )
    baseline = list(market_returns.values())
    filtered = [
        value
        for (_, symbol), value in market_returns.items()
        if symbol in selected
    ]
    selected_score = _selection_score(filtered)
    baseline_score = _selection_score(baseline)
    if selected and selected_score > max(baseline_score, 0.0):
        return selected, True
    if baseline_score > 0.0:
        return all_symbols, True
    return (), True


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def _fit_regimes(
    trades: list[SettledTrade],
) -> tuple[float | None, float | None, tuple[str, ...], bool]:
    values = [
        abs(value)
        for trade in trades
        if (value := _finite(getattr(trade, "momentum_1s_bps", None))) is not None
    ]
    if len(values) < 9:
        return None, None, ("unknown",), False
    low = _quantile(values, 1 / 3)
    high = _quantile(values, 2 / 3)
    if low is None or high is None:
        return None, None, ("unknown",), False
    grouped: dict[str, list[SettledTrade]] = {
        "low": [],
        "medium": [],
        "high": [],
        "unknown": [],
    }
    for trade in trades:
        label = _regime_label(trade, low, high)
        grouped[label].append(trade)
    all_regimes = ("low", "medium", "high", "unknown")
    selected = tuple(
        label
        for label in ("low", "medium", "high")
        if grouped[label]
        and statistics.fmean(_market_returns(grouped[label]).values()) > 0
    )
    selected_rows = [
        trade for label in selected for trade in grouped[label]
    ]
    baseline_rows = [
        trade for label in all_regimes for trade in grouped[label]
    ]
    selected_score = _selection_score(list(_market_returns(selected_rows).values()))
    baseline_score = _selection_score(list(_market_returns(baseline_rows).values()))
    if selected and selected_score > max(baseline_score, 0.0):
        return low, high, selected, True
    if baseline_score > 0.0:
        return low, high, all_regimes, True
    return low, high, (), True


def _regime_label(
    trade: SettledTrade,
    low: float | None,
    high: float | None,
) -> str:
    value = _finite(getattr(trade, "momentum_1s_bps", None))
    if value is None or low is None or high is None:
        return "unknown"
    absolute = abs(value)
    if absolute <= low:
        return "low"
    if absolute <= high:
        return "medium"
    return "high"


def _fit_consensus_threshold(
    trades: list[SettledTrade],
) -> tuple[float, bool, bool]:
    observed = [
        trade
        for trade in trades
        if any(
            _finite(getattr(trade, field, None)) is not None
            for field in (
                "oracle_basis_bps",
                "momentum_1s_bps",
                "book_imbalance",
            )
        )
    ]
    if len(_market_returns(observed)) < 5:
        return 0.0, False, False

    baseline_score = _selection_score(list(_market_returns(trades).values()))
    best_score = baseline_score
    best_threshold = 0.0
    for threshold in (0.50, 0.60, 0.75, 1.00):
        selected = [
            trade
            for trade in trades
            if technical_consensus_score(trade) >= threshold
        ]
        score = _selection_score(list(_market_returns(selected).values()))
        if score > max(best_score, 0.0):
            best_score = score
            best_threshold = threshold
    if best_threshold > 0.0:
        return best_threshold, True, False
    if baseline_score > 0.0:
        return 0.0, True, False
    return 0.0, True, True


def _fit_execution_quality(
    trades: list[SettledTrade],
) -> tuple[float | None, float | None, bool, bool]:
    edge_values = sorted(
        value
        for trade in trades
        if (value := _finite(getattr(trade, "net_edge", None))) is not None
    )
    spread_values = sorted(
        value
        for trade in trades
        if (value := _finite(getattr(trade, "prediction_spread", None))) is not None
    )
    if len(edge_values) < 5 or len(_market_returns(trades)) < 5:
        return None, None, False, False

    edge_candidates = tuple(
        dict.fromkeys(
            value
            for value in (
                edge_values[0],
                _quantile(edge_values, 0.50),
                _quantile(edge_values, 0.75),
            )
            if value is not None
        )
    )
    spread_candidates: tuple[float | None, ...]
    if len(spread_values) >= 5:
        spread_candidates = tuple(
            dict.fromkeys(
                value
                for value in (
                    spread_values[-1],
                    _quantile(spread_values, 0.75),
                    _quantile(spread_values, 0.50),
                )
                if value is not None
            )
        )
    else:
        spread_candidates = (None,)

    baseline_score = _selection_score(list(_market_returns(trades).values()))
    best_score = baseline_score
    best: tuple[float | None, float | None] = (None, None)
    for min_edge in edge_candidates:
        for max_spread in spread_candidates:
            selected: list[SettledTrade] = []
            for trade in trades:
                edge = _finite(getattr(trade, "net_edge", None))
                spread = _finite(getattr(trade, "prediction_spread", None))
                if edge is None or edge < min_edge:
                    continue
                if max_spread is not None and (spread is None or spread > max_spread):
                    continue
                selected.append(trade)
            score = _selection_score(list(_market_returns(selected).values()))
            if score > max(best_score, 0.0):
                best_score = score
                best = (min_edge, max_spread)
    if best != (None, None):
        return best[0], best[1], True, False
    if baseline_score > 0.0:
        return 0.0, None, True, False
    return None, None, True, True


def _feature_vector(trade: SettledTrade) -> tuple[float, ...]:
    direction = _direction(trade)
    directional = (
        getattr(trade, "oracle_basis_bps", None),
        getattr(trade, "momentum_250ms_bps", None),
        getattr(trade, "momentum_1s_bps", None),
        getattr(trade, "momentum_3s_bps", None),
        getattr(trade, "book_imbalance", None),
    )
    result = [
        (_finite(value) or 0.0) * direction
        for value in directional
    ]
    result.extend(
        [
            _finite(getattr(trade, "net_edge", None)) or 0.0,
            -(_finite(getattr(trade, "prediction_spread", None)) or 0.0),
            abs((_finite(getattr(trade, "fair_probability", None)) or 0.5) - 0.5),
            math.log1p(max(_finite(getattr(trade, "time_to_expiry_ms", None)) or 0.0, 0.0)),
        ]
    )
    return tuple(result)


def _fit_logistic(
    trades: list[SettledTrade],
    min_trades: int,
) -> tuple[LogisticModel | None, bool, bool]:
    if len(trades) < min_trades or len(_market_returns(trades)) < 5:
        return None, False, False
    xs = [_feature_vector(trade) for trade in trades]
    ys = [1.0 if trade.realized_return > 0 else 0.0 for trade in trades]
    if len(set(ys)) < 2:
        return None, False, False
    width = len(xs[0])
    means = tuple(statistics.fmean(row[index] for row in xs) for index in range(width))
    scales: list[float] = []
    for index, mean in enumerate(means):
        variance = statistics.fmean((row[index] - mean) ** 2 for row in xs)
        scales.append(max(math.sqrt(variance), 1e-9))
    normalized = [
        tuple(
            (value - mean) / scale
            for value, mean, scale in zip(row, means, scales, strict=True)
        )
        for row in xs
    ]
    market_counts = Counter(trade.market_id for trade in trades)
    sample_weights = [1.0 / market_counts[trade.market_id] for trade in trades]
    total_weight = sum(sample_weights)
    weights = [0.0] * width
    bias = 0.0
    learning_rate = 0.08
    l2 = 0.02
    for _ in range(250):
        grad_w = [0.0] * width
        grad_b = 0.0
        for row, label, sample_weight in zip(
            normalized,
            ys,
            sample_weights,
            strict=True,
        ):
            z = bias + sum(weight * value for weight, value in zip(weights, row, strict=True))
            probability = 1.0 / (1.0 + math.exp(-max(min(z, 60.0), -60.0)))
            error = (probability - label) * sample_weight
            grad_b += error
            for index, value in enumerate(row):
                grad_w[index] += error * value
        bias -= learning_rate * grad_b / total_weight
        for index in range(width):
            weights[index] -= learning_rate * (
                grad_w[index] / total_weight + l2 * weights[index]
            )

    provisional = LogisticModel(
        means=means,
        scales=tuple(scales),
        weights=tuple(weights),
        bias=bias,
        threshold=0.55,
    )
    baseline_score = _selection_score(list(_market_returns(trades).values()))
    best_score = baseline_score
    best_threshold: float | None = None
    min_selected_markets = max(5, len(_market_returns(trades)) // 4)
    for threshold in (0.50, 0.55, 0.60, 0.65, 0.70):
        selected = [
            trade
            for trade in trades
            if provisional.probability(trade) >= threshold
        ]
        if len(_market_returns(selected)) < min_selected_markets:
            continue
        score = _selection_score(list(_market_returns(selected).values()))
        if score > max(best_score, 0.0):
            best_score = score
            best_threshold = threshold
    if best_threshold is None:
        if baseline_score > 0.0:
            return None, True, False
        return None, True, True
    return (
        LogisticModel(
            means=means,
            scales=tuple(scales),
            weights=tuple(weights),
            bias=bias,
            threshold=best_threshold,
        ),
        True,
        False,
    )


def fit_policy(
    trades: list[SettledTrade],
    config: AdaptivePolicyConfig,
) -> FittedPolicy:
    symbols, symbol_gate_available = _fit_symbols(
        trades,
        config.min_symbol_train_markets,
    )
    current = (
        [trade for trade in trades if trade.symbol in symbols]
        if symbol_gate_available
        else list(trades)
    )

    setup_families, setup_gate_available = _fit_setup_families(
        current,
        config.min_setup_train_markets,
    )
    current = (
        [
            trade
            for trade in current
            if _setup_family(trade) in setup_families
        ]
        if setup_gate_available
        else current
    )

    low, high, regimes, regime_gate_available = _fit_regimes(current)
    current = (
        [
            trade
            for trade in current
            if _regime_label(trade, low, high) in regimes
        ]
        if regime_gate_available
        else current
    )

    consensus = _fit_consensus_threshold(current)
    current = [
        trade
        for trade in current
        if technical_consensus_score(trade) >= consensus
    ]

    min_edge, max_spread = _fit_execution_quality(current)
    if min_edge is not None:
        filtered: list[SettledTrade] = []
        for trade in current:
            edge = _finite(getattr(trade, "net_edge", None))
            spread = _finite(getattr(trade, "prediction_spread", None))
            if edge is None or edge < min_edge:
                continue
            if max_spread is not None and (spread is None or spread > max_spread):
                continue
            filtered.append(trade)
        if _filter_improves(current, filtered):
            current = filtered
        else:
            min_edge, max_spread = None, None

    ml_model = _fit_logistic(current, config.min_ml_train_trades)
    return FittedPolicy(
        selected_symbols=symbols,
        symbol_gate_available=symbol_gate_available,
        selected_setup_families=setup_families,
        setup_gate_available=setup_gate_available,
        regime_low_max=low,
        regime_medium_max=high,
        selected_regimes=regimes,
        regime_gate_available=regime_gate_available,
        min_consensus=consensus,
        consensus_gate_available=consensus_gate_available,
        consensus_no_trade=consensus_no_trade,
        min_net_edge=min_edge,
        max_prediction_spread=max_spread,
        execution_gate_available=execution_gate_available,
        execution_no_trade=execution_no_trade,
        ml_model=ml_model,
        ml_gate_available=ml_gate_available,
        ml_no_trade=ml_no_trade,
    )


def _summary(trades: list[SettledTrade]) -> dict[str, Any]:
    trade_stats = trade_return_summary([trade.realized_return for trade in trades])
    market_stats = trade_return_summary(list(_market_returns(trades).values()))
    return {
        "trades": len(trades),
        "unique_markets": len({trade.market_id for trade in trades}),
        "return_stats": trade_stats,
        "market_return_stats": market_stats,
        "realized_pnl_usd": sum(trade.realized_pnl_usd for trade in trades),
        "cost_basis_usd": sum(trade.cost_basis_usd for trade in trades),
    }


def _apply_policy_stages(
    trades: list[SettledTrade],
    policy: FittedPolicy,
    config: AdaptivePolicyConfig,
) -> dict[str, list[SettledTrade]]:
    stages: dict[str, list[SettledTrade]] = {"baseline": list(trades)}
    current = (
        [
            trade
            for trade in stages["baseline"]
            if trade.symbol in policy.selected_symbols
        ]
        if policy.symbol_gate_available
        else list(stages["baseline"])
    )
    stages["dynamic_symbol"] = current
    current = (
        [
            trade
            for trade in current
            if _setup_family(trade) in policy.selected_setup_families
        ]
        if policy.setup_gate_available
        else current
    )
    stages["setup_alignment"] = current
    current = (
        [
            trade
            for trade in current
            if _regime_label(
                trade,
                policy.regime_low_max,
                policy.regime_medium_max,
            )
            in policy.selected_regimes
        ]
        if policy.regime_gate_available
        else current
    )
    stages["regime_veto"] = current
    if policy.consensus_gate_available:
        current = (
            []
            if policy.consensus_no_trade
            else [
                trade
                for trade in current
                if technical_consensus_score(trade) >= policy.min_consensus
            ]
        )
    stages["technical_consensus"] = current
    if policy.execution_gate_available:
        if policy.execution_no_trade:
            current = []
        elif policy.min_net_edge is not None:
            filtered: list[SettledTrade] = []
            for trade in current:
                edge = _finite(getattr(trade, "net_edge", None))
                spread = _finite(getattr(trade, "prediction_spread", None))
                if edge is None or edge < policy.min_net_edge:
                    continue
                if (
                    policy.max_prediction_spread is not None
                    and (spread is None or spread > policy.max_prediction_spread)
                ):
                    continue
                filtered.append(trade)
            current = filtered
    stages["execution_quality"] = current
    if policy.ml_gate_available:
        if policy.ml_no_trade:
            current = []
        elif policy.ml_model is not None:
            current = [
                trade
                for trade in current
                if policy.ml_model.probability(trade) >= policy.ml_model.threshold
            ]
    stages["point_in_time_ml"] = current
    return stages


def evaluate_policy_stages(
    trades: list[SettledTrade],
    policy: FittedPolicy,
    config: AdaptivePolicyConfig,
) -> dict[str, list[SettledTrade]]:
    """Apply a frozen train-fitted policy to OOS trades for ablation."""
    return _apply_policy_stages(trades, policy, config)


def adaptive_walk_forward_report(
    root: Path,
    *,
    min_train_markets: int = 100,
    test_markets: int = 50,
    step_markets: int | None = None,
    min_symbol_train_markets: int = 20,
    min_setup_train_markets: int = 20,
    min_ml_train_trades: int = 80,
) -> dict[str, Any]:
    trades = list(load_research_trades(root))
    markets = aggregate_markets(tuple(trades))
    folds = purged_market_walk_forward_folds(
        markets,
        min_train_size=min_train_markets,
        test_size=test_markets,
        step=step_markets,
    )
    market_by_id = {row.market_id: row for row in markets}
    config = AdaptivePolicyConfig(
        min_symbol_train_markets=min_symbol_train_markets,
        min_setup_train_markets=min_setup_train_markets,
        min_ml_train_trades=min_ml_train_trades,
    )
    fold_reports: list[dict[str, Any]] = []
    final_oos: dict[tuple[str, str, int], SettledTrade] = {}
    positive_folds = 0

    for index, fold in enumerate(folds):
        train_markets = tuple(market_by_id[market_id] for market_id in fold.train_market_ids)
        train_ids = set(fold.train_market_ids)
        test_ids = set(fold.test_market_ids)
        if train_ids & test_ids:
            raise RuntimeError("market leakage detected across adaptive walk-forward fold")
        if train_markets and max(row.settled_ts_ns for row in train_markets) >= fold.test_start_ts_ns:
            raise RuntimeError("future settlement label leaked into adaptive train window")
        train_trades = [trade for trade in trades if trade.market_id in train_ids]
        test_trades = [trade for trade in trades if trade.market_id in test_ids]
        policy = fit_policy(train_trades, config)
        stages = _apply_policy_stages(test_trades, policy, config)
        final_stage = stages["point_in_time_ml"]
        final_summary = _summary(final_stage)
        final_mean = final_summary["market_return_stats"].get("mean")
        if final_mean is not None and float(final_mean) > 0:
            positive_folds += 1
        for trade in final_stage:
            final_oos[(trade.market_id, trade.instrument, trade.signal_ts_ns)] = trade

        policy_payload = {
            "selected_symbols": list(policy.selected_symbols),
            "symbol_gate_available": policy.symbol_gate_available,
            "selected_setup_families": list(policy.selected_setup_families),
            "setup_gate_available": policy.setup_gate_available,
            "regime_thresholds_abs_momentum_1s_bps": {
                "low_max": policy.regime_low_max,
                "medium_max": policy.regime_medium_max,
            },
            "selected_regimes": list(policy.selected_regimes),
            "regime_gate_available": policy.regime_gate_available,
            "min_consensus": policy.min_consensus,
            "consensus_gate_available": policy.consensus_gate_available,
            "consensus_no_trade": policy.consensus_no_trade,
            "min_net_edge": policy.min_net_edge,
            "max_prediction_spread": policy.max_prediction_spread,
            "execution_gate_available": policy.execution_gate_available,
            "execution_no_trade": policy.execution_no_trade,
            "ml_gate_available": policy.ml_gate_available,
            "ml_no_trade": policy.ml_no_trade,
            "ml": (
                {
                    "threshold": policy.ml_model.threshold,
                    "weights": list(policy.ml_model.weights),
                    "bias": policy.ml_model.bias,
                    "means": list(policy.ml_model.means),
                    "scales": list(policy.ml_model.scales),
                }
                if policy.ml_model is not None
                else None
            ),
        }
        fold_reports.append(
            {
                "fold": index,
                "train_market_ids": sorted(train_ids),
                "test_market_ids": sorted(test_ids),
                "train_markets": len(train_ids),
                "test_markets": len(test_ids),
                "purged_train_markets": fold.purged_train_markets,
                "train_start_ts_ns": fold.train_start_ts_ns,
                "train_end_ts_ns": fold.train_end_ts_ns,
                "train_last_settled_ts_ns": fold.train_last_settled_ts_ns,
                "test_start_ts_ns": fold.test_start_ts_ns,
                "test_end_ts_ns": fold.test_end_ts_ns,
                "policy": policy_payload,
                "ablation": {
                    name: _summary(rows)
                    for name, rows in stages.items()
                },
                "portfolio_ablation": {
                    name: _portfolio_summary(replay_portfolio(rows))
                    for name, rows in stages.items()
                },
            }
        )

    final_trades = list(final_oos.values())
    oos_summary = _summary(final_trades)
    portfolio_result = replay_portfolio(final_trades)
    portfolio_oos = _portfolio_summary(portfolio_result)
    blockers: list[str] = []
    if len(markets) < min_train_markets + test_markets:
        blockers.append("insufficient_markets_for_first_walk_forward_fold")
    if len(folds) < 3:
        blockers.append("fewer_than_3_walk_forward_folds")
    if oos_summary["unique_markets"] < 250:
        blockers.append("fewer_than_250_unique_oos_markets")
    ci_low = oos_summary["market_return_stats"].get("mean_ci95_low")
    if ci_low is None or float(ci_low) <= 0:
        blockers.append("adaptive_oos_return_ci_not_positive")
    if folds and positive_folds / len(folds) < 2 / 3:
        blockers.append("fewer_than_two_thirds_positive_adaptive_oos_folds")
    if not final_trades:
        blockers.append("adaptive_policy_selected_no_oos_trades")
    portfolio_ci_low = portfolio_oos["market_return_stats"].get("mean_ci95_low")
    if portfolio_oos["unique_markets"] < 100:
        blockers.append("fewer_than_100_portfolio_oos_markets")
    if portfolio_ci_low is None or float(portfolio_ci_low) <= 0:
        blockers.append("portfolio_oos_return_ci_not_positive")
    portfolio_pf = portfolio_oos["market_return_stats"].get("profit_factor")
    if portfolio_pf is None or (
        not math.isinf(float(portfolio_pf))
        and float(portfolio_pf) < 1.15
    ):
        blockers.append("portfolio_oos_profit_factor_below_policy")

    return {
        "method": "adaptive_purged_expanding_market_level_walk_forward",
        "unique_markets": len(markets),
        "settled_trades": len(trades),
        "settlement_event_type": "ResearchSettlement",
        "parameters": {
            "min_train_markets": min_train_markets,
            "test_markets": test_markets,
            "step_markets": step_markets or test_markets,
            "min_symbol_train_markets": min_symbol_train_markets,
            "min_setup_train_markets": min_setup_train_markets,
            "min_ml_train_trades": min_ml_train_trades,
        },
        "folds": fold_reports,
        "positive_folds": positive_folds,
        "final_stage": "point_in_time_ml",
        "oos": oos_summary,
        "portfolio_oos": portfolio_oos,
        "qualified": not blockers,
        "blockers": blockers,
        "notes": [
            "Every selector/model is fitted only on the expanding train window.",
            "Training labels must settle strictly before each OOS test block begins.",
            "Markets sharing one signal timestamp are kept together in one test block.",
            "Market IDs are disjoint between train and test inside every fold.",
            "Dynamic symbol, setup, regime, consensus, execution-quality and ML layers are reported sequentially for ablation.",
            "Every OOS stage is additionally replayed through the current paper risk envelope before qualification.",
            "Qualification confidence intervals and profit factor are computed at market level, not by treating same-market trades as independent.",
            "Train-time gate/threshold selection also uses market-level evidence, and logistic gradients weight each market equally regardless of signal count.",
            "This report is research evidence only and does not enable live trading.",
        ],
    }


def write_adaptive_walk_forward_report(
    root: Path,
    destination: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    report = adaptive_walk_forward_report(root, **kwargs)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
