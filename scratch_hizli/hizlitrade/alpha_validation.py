from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .calibration import (
    ProbabilityCalibration,
    calibrated_net_edge,
    fit_probability_calibration,
)
from .oos import (
    SettledTrade,
    aggregate_markets,
    load_research_trades,
    purged_market_walk_forward_folds,
)
from .validation import trade_return_summary


def adaptive_policy_fingerprint() -> str:
    """Bind promotion attestation to every policy-critical implementation file."""
    package_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for filename in ("alpha_validation.py", "calibration.py", "oos.py"):
        path = package_root / filename
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class AdaptivePolicyConfig:
    min_symbol_train_markets: int = 20
    min_setup_train_markets: int = 20
    min_session_train_markets: int = 20
    min_ml_train_trades: int = 80
    min_calibration_bin_samples: int = 30
    min_calibrated_net_edge: float = 0.025
    min_final_train_markets: int = 20
    min_alpha_mean_return: float = 0.10
    stability_windows: int = 3
    min_positive_stability_windows: int = 2
    challenger_validation_fraction: float = 0.33
    min_challenger_validation_markets: int = 10

    def __post_init__(self) -> None:
        if self.min_symbol_train_markets <= 0:
            raise ValueError("min_symbol_train_markets must be positive")
        if self.min_setup_train_markets <= 0:
            raise ValueError("min_setup_train_markets must be positive")
        if self.min_session_train_markets <= 0:
            raise ValueError("min_session_train_markets must be positive")
        if self.min_ml_train_trades <= 1:
            raise ValueError("min_ml_train_trades must be greater than one")
        if self.min_calibration_bin_samples <= 0:
            raise ValueError("min_calibration_bin_samples must be positive")
        if self.min_calibrated_net_edge < 0:
            raise ValueError("min_calibrated_net_edge must be non-negative")
        if self.min_final_train_markets <= 0:
            raise ValueError("min_final_train_markets must be positive")
        if not 0 <= self.min_alpha_mean_return <= 1:
            raise ValueError("min_alpha_mean_return must be in [0, 1]")
        if self.stability_windows <= 0:
            raise ValueError("stability_windows must be positive")
        if not 1 <= self.min_positive_stability_windows <= self.stability_windows:
            raise ValueError(
                "min_positive_stability_windows must be between one and stability_windows"
            )
        if not 0.20 <= self.challenger_validation_fraction <= 0.50:
            raise ValueError("challenger_validation_fraction must be in [0.20, 0.50]")
        if self.min_challenger_validation_markets < 5:
            raise ValueError("min_challenger_validation_markets must be at least five")


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
class FittedPolicy:
    selected_symbols: tuple[str, ...]
    selected_setup_families: tuple[str, ...]
    selected_sessions: tuple[str, ...]
    regime_low_max: float | None
    regime_medium_max: float | None
    selected_regimes: tuple[str, ...]
    min_consensus: float
    min_net_edge: float | None
    max_prediction_spread: float | None
    probability_calibration: ProbabilityCalibration | None
    ml_model: LogisticModel | None
    selection_mode: str
    challenger_validation_reason: str
    challenger_validation_markets: int
    challenger_validation_mean_return: float | None
    challenger_validation_ci95_low: float | None
    alpha_evidence_qualified: bool
    alpha_evidence_reason: str
    alpha_train_markets: int
    alpha_train_mean_return: float | None
    alpha_train_ci95_low: float | None
    alpha_stability_positive_windows: int
    alpha_stability_total_windows: int


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


def technical_consensus_score(trade: SettledTrade) -> float:
    """Score only information observable before execution with correct sign semantics.

    Spot/oracle features are expressed in underlying-asset direction, so they
    are sign-adjusted for YES vs NO. Prediction-book imbalance is already for
    the outcome token being bought; positive bid imbalance supports buying that
    token for both YES and NO and must not be sign-flipped.
    """
    direction = _direction(trade)
    observed = 0
    aligned = 0
    for raw in (
        getattr(trade, "oracle_basis_bps", None),
        getattr(trade, "momentum_250ms_bps", None),
        getattr(trade, "momentum_1s_bps", None),
        getattr(trade, "momentum_3s_bps", None),
    ):
        value = _finite(raw)
        if value is None or abs(value) <= 1e-12:
            continue
        observed += 1
        if value * direction > 0:
            aligned += 1

    imbalance = _finite(getattr(trade, "book_imbalance", None))
    if imbalance is not None and abs(imbalance) > 1e-12:
        observed += 1
        if imbalance > 0:
            aligned += 1
    return aligned / observed if observed else 0.0


def _setup_family(trade: SettledTrade) -> str:
    duration_ms = _finite(getattr(trade, "market_duration_ms", None))
    if duration_ms is None or duration_ms <= 0:
        return "unknown"
    minutes = max(1, round(duration_ms / 60_000))
    return f"{minutes}m"


def _session_label(trade: SettledTrade) -> str:
    seconds = trade.signal_ts_ns // 1_000_000_000
    hour_utc = (seconds // 3_600) % 24
    start = (hour_utc // 3) * 3
    end = start + 3
    return f"utc_{start:02d}_{end:02d}"


def _fit_sessions(
    trades: list[SettledTrade],
    min_markets: int,
) -> tuple[str, ...]:
    grouped: dict[tuple[str, str], list[SettledTrade]] = {}
    for trade in trades:
        grouped.setdefault((trade.market_id, _session_label(trade)), []).append(trade)

    per_session: dict[str, list[float]] = {}
    all_returns: list[float] = []
    for (_market_id, session), rows in grouped.items():
        cost = sum(row.cost_basis_usd for row in rows)
        pnl = sum(row.realized_pnl_usd for row in rows)
        value = pnl / cost if cost > 0 else 0.0
        per_session.setdefault(session, []).append(value)
        all_returns.append(value)

    all_sessions = tuple(sorted(per_session))
    selected = tuple(
        sorted(
            session
            for session, returns in per_session.items()
            if len(returns) >= min_markets and statistics.fmean(returns) > 0
        )
    )
    if not selected or selected == all_sessions:
        return all_sessions
    selected_returns = [
        value
        for session in selected
        for value in per_session[session]
    ]
    selected_score = _selection_score(selected_returns)
    baseline_score = _selection_score(all_returns)
    return selected if selected_score > max(baseline_score, 0.0) else all_sessions


def _fit_setup_families(
    trades: list[SettledTrade],
    min_markets: int,
) -> tuple[str, ...]:
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
    selected = tuple(
        sorted(
            family
            for family, returns in per_family.items()
            if len(returns) >= min_markets and statistics.fmean(returns) > 0
        )
    )
    if not selected or selected == all_families:
        return all_families
    selected_returns = [
        value
        for family in selected
        for value in per_family[family]
    ]
    selected_score = _selection_score(selected_returns)
    baseline_score = _selection_score(all_returns)
    return selected if selected_score > max(baseline_score, 0.0) else all_families


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
    if len({trade.market_id for trade in selected}) < 5 or len(selected) >= len(baseline):
        return False
    baseline_score = _market_selection_score(baseline)
    selected_score = _market_selection_score(selected)
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


def _market_selection_score(trades: list[SettledTrade]) -> float:
    """Score statistically independent market-level returns, not correlated trade rows."""
    return _selection_score(list(_market_returns(trades).values()))


def _ordered_market_returns(trades: list[SettledTrade]) -> list[tuple[int, float]]:
    grouped: dict[tuple[str, str], list[SettledTrade]] = {}
    for trade in trades:
        grouped.setdefault((trade.market_id, trade.symbol), []).append(trade)

    rows: list[tuple[int, float]] = []
    for grouped_trades in grouped.values():
        cost = sum(trade.cost_basis_usd for trade in grouped_trades)
        if cost <= 0:
            continue
        pnl = sum(trade.realized_pnl_usd for trade in grouped_trades)
        first_signal = min(trade.signal_ts_ns for trade in grouped_trades)
        rows.append((first_signal, pnl / cost))
    rows.sort(key=lambda item: item[0])
    return rows


def _alpha_evidence_diagnostics(
    trades: list[SettledTrade],
    config: AdaptivePolicyConfig,
) -> tuple[bool, str, int, float | None, float | None, int, int]:
    ordered = _ordered_market_returns(trades)
    market_count = len(ordered)
    returns = [value for _timestamp, value in ordered]
    summary = trade_return_summary(returns)
    mean_raw = summary.get("mean")
    ci_low_raw = summary.get("mean_ci95_low")
    mean_return = float(mean_raw) if mean_raw is not None else None
    ci_low = float(ci_low_raw) if ci_low_raw is not None else None

    if market_count < config.min_final_train_markets:
        return (
            False,
            "insufficient_final_train_markets",
            market_count,
            mean_return,
            ci_low,
            0,
            0,
        )
    if mean_return is None or mean_return <= 0:
        return False, "non_positive_train_edge", market_count, mean_return, ci_low, 0, 0
    if mean_return < config.min_alpha_mean_return:
        return (
            False,
            "train_expectancy_below_minimum",
            market_count,
            mean_return,
            ci_low,
            0,
            0,
        )
    if ci_low is None or ci_low <= 0:
        return False, "train_edge_ci_not_positive", market_count, mean_return, ci_low, 0, 0

    # Require enough independent markets in every chronological stability bucket.
    minimum_for_windows = config.stability_windows * 5
    if market_count < minimum_for_windows:
        return (
            False,
            "insufficient_markets_for_stability_windows",
            market_count,
            mean_return,
            ci_low,
            0,
            0,
        )

    positive_windows = 0
    total_windows = config.stability_windows
    for window in range(total_windows):
        start = window * market_count // total_windows
        end = (window + 1) * market_count // total_windows
        values = returns[start:end]
        if values and statistics.fmean(values) > 0:
            positive_windows += 1

    if positive_windows < config.min_positive_stability_windows:
        return (
            False,
            "train_edge_not_time_stable",
            market_count,
            mean_return,
            ci_low,
            positive_windows,
            total_windows,
        )
    return (
        True,
        "qualified",
        market_count,
        mean_return,
        ci_low,
        positive_windows,
        total_windows,
    )


def _chronological_market_partition(
    trades: list[SettledTrade],
    *,
    validation_fraction: float,
    min_validation_markets: int,
    min_fit_markets: int,
) -> tuple[list[SettledTrade], list[SettledTrade]]:
    ordered = _ordered_market_returns(trades)
    market_ids_in_order: list[str] = []
    seen: set[str] = set()
    for timestamp, _return in ordered:
        matching = min(
            (
                trade
                for trade in trades
                if trade.signal_ts_ns == timestamp
                and trade.market_id not in seen
            ),
            key=lambda trade: trade.market_id,
            default=None,
        )
        if matching is None:
            continue
        seen.add(matching.market_id)
        market_ids_in_order.append(matching.market_id)

    count = len(market_ids_in_order)
    validation_count = max(
        min_validation_markets,
        int(round(count * validation_fraction)),
    )
    fit_count = count - validation_count
    if fit_count < min_fit_markets or validation_count < min_validation_markets:
        return [], []

    fit_ids = set(market_ids_in_order[:fit_count])
    validation_ids = set(market_ids_in_order[fit_count:])
    return (
        [trade for trade in trades if trade.market_id in fit_ids],
        [trade for trade in trades if trade.market_id in validation_ids],
    )


def _validation_diagnostics(
    trades: list[SettledTrade],
    *,
    min_markets: int,
) -> tuple[bool, str, int, float | None, float | None]:
    returns = list(_market_returns(trades).values())
    summary = trade_return_summary(returns)
    count = len(returns)
    mean_raw = summary.get("mean")
    ci_raw = summary.get("mean_ci95_low")
    mean_return = float(mean_raw) if mean_raw is not None else None
    ci_low = float(ci_raw) if ci_raw is not None else None
    if count < min_markets:
        return False, "candidate_validation_insufficient_markets", count, mean_return, ci_low
    if mean_return is None or mean_return <= 0:
        return False, "candidate_validation_mean_not_positive", count, mean_return, ci_low
    if ci_low is None or ci_low <= 0:
        return False, "candidate_validation_ci_not_positive", count, mean_return, ci_low
    return True, "qualified", count, mean_return, ci_low


def _fit_joint_consensus_edge_gate(
    trades: list[SettledTrade],
    config: AdaptivePolicyConfig,
) -> tuple[
    float | None,
    float | None,
    str,
    int,
    float | None,
    float | None,
]:
    """Fit a compact interaction catalog and confirm it on newer train data.

    Candidate thresholds are selected only on the older inner-fit partition.
    The single winning candidate is then frozen and evaluated once on a newer
    inner-validation partition. Validation never participates in candidate
    ranking, reducing the multiple-testing optimism of same-sample selection.
    """
    fit_trades, validation_trades = _chronological_market_partition(
        trades,
        validation_fraction=config.challenger_validation_fraction,
        min_validation_markets=config.min_challenger_validation_markets,
        min_fit_markets=config.min_final_train_markets,
    )
    if not fit_trades or not validation_trades:
        return (
            None,
            None,
            "insufficient_inner_train_partition",
            0,
            None,
            None,
        )

    edge_values = sorted(
        value
        for trade in fit_trades
        if (value := _finite(getattr(trade, "net_edge", None))) is not None
    )
    if len(edge_values) < config.min_final_train_markets:
        return None, None, "insufficient_inner_fit_edge_values", 0, None, None

    baseline_score = _market_selection_score(fit_trades)
    best_rank = (max(baseline_score, 0.0), float("-inf"), 0)
    best: tuple[float | None, float | None] = (None, None)

    edge_candidates = tuple(
        dict.fromkeys(
            value
            for value in (
                _quantile(edge_values, 0.50),
                _quantile(edge_values, 0.75),
            )
            if value is not None
        )
    )
    for consensus in (0.50, 0.60, 0.75, 1.00):
        for min_edge in edge_candidates:
            selected = [
                trade
                for trade in fit_trades
                if technical_consensus_score(trade) >= consensus
                and (edge := _finite(getattr(trade, "net_edge", None))) is not None
                and edge >= min_edge
            ]
            if len(selected) >= len(fit_trades):
                continue
            diagnostics = _alpha_evidence_diagnostics(selected, config)
            qualified = diagnostics[0]
            mean_return = diagnostics[3]
            ci_low = diagnostics[4]
            market_count = diagnostics[2]
            if not qualified or mean_return is None or ci_low is None:
                continue
            rank = (ci_low, mean_return, market_count)
            if rank > best_rank:
                best_rank = rank
                best = (consensus, min_edge)

    consensus, min_edge = best
    if consensus is None or min_edge is None:
        return None, None, "no_inner_fit_challenger", 0, None, None

    validation_selected = [
        trade
        for trade in validation_trades
        if technical_consensus_score(trade) >= consensus
        and (edge := _finite(getattr(trade, "net_edge", None))) is not None
        and edge >= min_edge
    ]
    (
        validation_qualified,
        validation_reason,
        validation_markets,
        validation_mean,
        validation_ci,
    ) = _validation_diagnostics(
        validation_selected,
        min_markets=config.min_challenger_validation_markets,
    )
    if not validation_qualified:
        return (
            None,
            None,
            validation_reason,
            validation_markets,
            validation_mean,
            validation_ci,
        )
    return (
        consensus,
        min_edge,
        "qualified",
        validation_markets,
        validation_mean,
        validation_ci,
    )


def _fit_symbols(
    trades: list[SettledTrade],
    min_markets: int,
) -> tuple[str, ...]:
    market_returns = _market_returns(trades)
    per_symbol: dict[str, list[float]] = {}
    for (_, symbol), value in market_returns.items():
        per_symbol.setdefault(symbol, []).append(value)
    all_symbols = tuple(sorted(per_symbol))
    selected = tuple(
        sorted(
            symbol
            for symbol, returns in per_symbol.items()
            if len(returns) >= min_markets and statistics.fmean(returns) > 0
        )
    )
    if not selected or selected == all_symbols:
        return all_symbols
    baseline = list(market_returns.values())
    filtered = [
        value
        for (_, symbol), value in market_returns.items()
        if symbol in selected
    ]
    selected_score = _selection_score(filtered)
    baseline_score = _selection_score(baseline)
    return selected if selected_score > max(baseline_score, 0.0) else all_symbols


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def _fit_regimes(
    trades: list[SettledTrade],
) -> tuple[float | None, float | None, tuple[str, ...]]:
    values = [
        abs(value)
        for trade in trades
        if (value := _finite(getattr(trade, "momentum_1s_bps", None))) is not None
    ]
    if len(values) < 9:
        return None, None, ("unknown",)
    low = _quantile(values, 1 / 3)
    high = _quantile(values, 2 / 3)
    if low is None or high is None:
        return None, None, ("unknown",)
    grouped: dict[str, list[SettledTrade]] = {"low": [], "medium": [], "high": [], "unknown": []}
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
    selected_score = _market_selection_score(selected_rows)
    baseline_score = _market_selection_score(trades)
    if not selected or selected_score <= max(baseline_score, 0.0):
        selected = all_regimes
    return low, high, selected


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


def _fit_consensus_threshold(trades: list[SettledTrade]) -> float:
    baseline_score = _market_selection_score(trades)
    best_score = baseline_score
    best_threshold = 0.0
    for threshold in (0.50, 0.60, 0.75, 1.00):
        selected = [
            trade
            for trade in trades
            if technical_consensus_score(trade) >= threshold
        ]
        score = _market_selection_score(selected)
        if score > max(best_score, 0.0):
            best_score = score
            best_threshold = threshold
    return best_threshold


def _fit_execution_quality(
    trades: list[SettledTrade],
) -> tuple[float | None, float | None]:
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
    if len(edge_values) < 5:
        return None, None

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

    best_mean = float("-inf")
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
            if len({trade.market_id for trade in selected}) < 5:
                continue
            score = _market_selection_score(selected)
            if score > best_mean:
                best_mean = score
                best = (min_edge, max_spread)
    return best if best_mean > 0 else (None, None)


def _feature_vector(trade: SettledTrade) -> tuple[float, ...]:
    direction = _direction(trade)
    directional = (
        getattr(trade, "oracle_basis_bps", None),
        getattr(trade, "momentum_250ms_bps", None),
        getattr(trade, "momentum_1s_bps", None),
        getattr(trade, "momentum_3s_bps", None),
    )
    result = [
        (_finite(value) or 0.0) * direction
        for value in directional
    ]
    # Prediction-book imbalance is already expressed for the outcome token
    # being bought; preserve its sign for both YES and NO instruments.
    result.append(_finite(getattr(trade, "book_imbalance", None)) or 0.0)
    result.extend(
        [
            _finite(getattr(trade, "net_edge", None)) or 0.0,
            -(_finite(getattr(trade, "prediction_spread", None)) or 0.0),
            abs((_finite(getattr(trade, "fair_probability", None)) or 0.5) - 0.5),
            math.log1p(max(_finite(getattr(trade, "time_to_expiry_ms", None)) or 0.0, 0.0)),
        ]
    )
    return tuple(result)


def _fit_probability_calibration(
    trades: list[SettledTrade],
    config: AdaptivePolicyConfig,
) -> ProbabilityCalibration | None:
    return fit_probability_calibration(
        [
            (_finite(getattr(trade, "fair_probability", None)), bool(trade.won))
            for trade in trades
        ],
        min_bin_samples=config.min_calibration_bin_samples,
    )


def _passes_calibration(
    trade: SettledTrade,
    model: ProbabilityCalibration | None,
    *,
    min_net_edge: float,
) -> bool:
    if model is None:
        return True
    raw_probability = _finite(getattr(trade, "fair_probability", None))
    raw_net_edge = _finite(getattr(trade, "net_edge", None))
    if raw_probability is None or raw_net_edge is None:
        return False
    adjusted_edge, _calibrated_probability, _samples, _applied = calibrated_net_edge(
        raw_probability=raw_probability,
        raw_net_edge=raw_net_edge,
        model=model,
    )
    return adjusted_edge is not None and adjusted_edge >= min_net_edge


def _fit_logistic(trades: list[SettledTrade], min_trades: int) -> LogisticModel | None:
    if len(trades) < min_trades:
        return None
    xs = [_feature_vector(trade) for trade in trades]
    ys = [1.0 if trade.realized_return > 0 else 0.0 for trade in trades]
    if len(set(ys)) < 2:
        return None
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
    weights = [0.0] * width
    bias = 0.0
    learning_rate = 0.08
    l2 = 0.02
    for _ in range(250):
        grad_w = [0.0] * width
        grad_b = 0.0
        for row, label in zip(normalized, ys, strict=True):
            z = bias + sum(weight * value for weight, value in zip(weights, row, strict=True))
            probability = 1.0 / (1.0 + math.exp(-max(min(z, 60.0), -60.0)))
            error = probability - label
            grad_b += error
            for index, value in enumerate(row):
                grad_w[index] += error * value
        count = float(len(normalized))
        bias -= learning_rate * grad_b / count
        for index in range(width):
            weights[index] -= learning_rate * (grad_w[index] / count + l2 * weights[index])

    provisional = LogisticModel(
        means=means,
        scales=tuple(scales),
        weights=tuple(weights),
        bias=bias,
        threshold=0.55,
    )
    baseline_score = _market_selection_score(trades)
    best_score = baseline_score
    best_threshold: float | None = None
    for threshold in (0.50, 0.55, 0.60, 0.65, 0.70):
        selected = [
            trade
            for trade in trades
            if provisional.probability(trade) >= threshold
        ]
        if len({trade.market_id for trade in selected}) < max(5, min_trades // 8):
            continue
        score = _market_selection_score(selected)
        if score > max(best_score, 0.0):
            best_score = score
            best_threshold = threshold
    if best_threshold is None:
        return None
    return LogisticModel(
        means=means,
        scales=tuple(scales),
        weights=tuple(weights),
        bias=bias,
        threshold=best_threshold,
    )


def fit_policy(
    trades: list[SettledTrade],
    config: AdaptivePolicyConfig,
) -> FittedPolicy:
    symbols = _fit_symbols(trades, config.min_symbol_train_markets)
    current = [trade for trade in trades if trade.symbol in symbols]

    setup_families = _fit_setup_families(current, config.min_setup_train_markets)
    current = [
        trade for trade in current
        if _setup_family(trade) in setup_families
    ]

    sessions = _fit_sessions(current, config.min_session_train_markets)
    current = [
        trade for trade in current
        if _session_label(trade) in sessions
    ]

    low, high, regimes = _fit_regimes(current)
    current = [
        trade
        for trade in current
        if _regime_label(trade, low, high) in regimes
    ]

    (
        joint_consensus,
        joint_edge,
        challenger_validation_reason,
        challenger_validation_markets,
        challenger_validation_mean_return,
        challenger_validation_ci95_low,
    ) = _fit_joint_consensus_edge_gate(current, config)
    if joint_consensus is not None and joint_edge is not None:
        selection_mode = "joint_consensus_edge"
        consensus = joint_consensus
        min_edge = joint_edge
        max_spread = None
        current = [
            trade
            for trade in current
            if technical_consensus_score(trade) >= consensus
            and (edge := _finite(getattr(trade, "net_edge", None))) is not None
            and edge >= min_edge
        ]
    else:
        selection_mode = "sequential"
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

    probability_calibration = _fit_probability_calibration(current, config)
    if probability_calibration is not None:
        current = [
            trade
            for trade in current
            if _passes_calibration(
                trade,
                probability_calibration,
                min_net_edge=config.min_calibrated_net_edge,
            )
        ]

    ml_model = _fit_logistic(current, config.min_ml_train_trades)
    if ml_model is not None:
        current = [
            trade
            for trade in current
            if ml_model.probability(trade) >= ml_model.threshold
        ]

    (
        alpha_evidence_qualified,
        alpha_evidence_reason,
        alpha_train_markets,
        alpha_train_mean_return,
        alpha_train_ci95_low,
        alpha_stability_positive_windows,
        alpha_stability_total_windows,
    ) = _alpha_evidence_diagnostics(current, config)

    return FittedPolicy(
        selected_symbols=symbols,
        selected_setup_families=setup_families,
        selected_sessions=sessions,
        regime_low_max=low,
        regime_medium_max=high,
        selected_regimes=regimes,
        min_consensus=consensus,
        min_net_edge=min_edge,
        max_prediction_spread=max_spread,
        probability_calibration=probability_calibration,
        ml_model=ml_model,
        selection_mode=selection_mode,
        challenger_validation_reason=challenger_validation_reason,
        challenger_validation_markets=challenger_validation_markets,
        challenger_validation_mean_return=challenger_validation_mean_return,
        challenger_validation_ci95_low=challenger_validation_ci95_low,
        alpha_evidence_qualified=alpha_evidence_qualified,
        alpha_evidence_reason=alpha_evidence_reason,
        alpha_train_markets=alpha_train_markets,
        alpha_train_mean_return=alpha_train_mean_return,
        alpha_train_ci95_low=alpha_train_ci95_low,
        alpha_stability_positive_windows=alpha_stability_positive_windows,
        alpha_stability_total_windows=alpha_stability_total_windows,
    )


def _summary(trades: list[SettledTrade]) -> dict[str, Any]:
    market_returns = list(_market_returns(trades).values())
    return {
        "trades": len(trades),
        "unique_markets": len(market_returns),
        "return_stats": trade_return_summary(market_returns),
        "trade_return_stats": trade_return_summary(
            [trade.realized_return for trade in trades]
        ),
        "realized_pnl_usd": sum(trade.realized_pnl_usd for trade in trades),
        "cost_basis_usd": sum(trade.cost_basis_usd for trade in trades),
    }


def _apply_policy_stages(
    trades: list[SettledTrade],
    policy: FittedPolicy,
    config: AdaptivePolicyConfig,
) -> dict[str, list[SettledTrade]]:
    stages: dict[str, list[SettledTrade]] = {"baseline": list(trades)}
    current = [
        trade
        for trade in stages["baseline"]
        if trade.symbol in policy.selected_symbols
    ]
    stages["dynamic_symbol"] = current
    current = [
        trade
        for trade in current
        if _setup_family(trade) in policy.selected_setup_families
    ]
    stages["setup_alignment"] = current
    current = [
        trade for trade in current
        if _session_label(trade) in policy.selected_sessions
    ]
    stages["session_filter"] = current
    current = [
        trade for trade in current
        if _regime_label(trade, policy.regime_low_max, policy.regime_medium_max)
        in policy.selected_regimes
    ]
    stages["regime_veto"] = current
    current = [
        trade for trade in current
        if technical_consensus_score(trade) >= policy.min_consensus
    ]
    stages["technical_consensus"] = current
    if policy.min_net_edge is not None:
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
    current = [
        trade
        for trade in current
        if _passes_calibration(
            trade,
            policy.probability_calibration,
            min_net_edge=config.min_calibrated_net_edge,
        )
    ]
    stages["realized_calibration"] = current
    if policy.ml_model is not None:
        current = [
            trade
            for trade in current
            if policy.ml_model.probability(trade) >= policy.ml_model.threshold
        ]
    stages["point_in_time_ml"] = current
    stages["alpha_evidence_gate"] = (
        list(current) if policy.alpha_evidence_qualified else []
    )
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
    min_session_train_markets: int = 20,
    min_ml_train_trades: int = 80,
    min_calibration_bin_samples: int = 30,
    min_calibrated_net_edge: float = 0.025,
    min_final_train_markets: int = 20,
    min_alpha_mean_return: float = 0.10,
    stability_windows: int = 3,
    min_positive_stability_windows: int = 2,
    challenger_validation_fraction: float = 0.33,
    min_challenger_validation_markets: int = 10,
    required_evidence_config_sha256: str | None = None,
) -> dict[str, Any]:
    all_trades = list(load_research_trades(root))
    if required_evidence_config_sha256 is None:
        trades = all_trades
    else:
        trades = [
            trade
            for trade in all_trades
            if trade.evidence_config_sha256 == required_evidence_config_sha256
        ]
    excluded_config_trades = len(all_trades) - len(trades)
    markets = aggregate_markets(tuple(trades))
    folds = purged_market_walk_forward_folds(
        markets,
        min_train_size=min_train_markets,
        test_size=test_markets,
        step=step_markets,
    )
    config = AdaptivePolicyConfig(
        min_symbol_train_markets=min_symbol_train_markets,
        min_setup_train_markets=min_setup_train_markets,
        min_session_train_markets=min_session_train_markets,
        min_ml_train_trades=min_ml_train_trades,
        min_calibration_bin_samples=min_calibration_bin_samples,
        min_calibrated_net_edge=min_calibrated_net_edge,
        min_final_train_markets=min_final_train_markets,
        min_alpha_mean_return=min_alpha_mean_return,
        stability_windows=stability_windows,
        min_positive_stability_windows=min_positive_stability_windows,
        challenger_validation_fraction=challenger_validation_fraction,
        min_challenger_validation_markets=min_challenger_validation_markets,
    )
    fold_reports: list[dict[str, Any]] = []
    final_oos: dict[tuple[str, str, int], SettledTrade] = {}
    positive_folds = 0

    by_id = {row.market_id: row for row in markets}
    for index, fold in enumerate(folds):
        train_markets = tuple(by_id[market_id] for market_id in fold.train_market_ids)
        test_market_rows = tuple(by_id[market_id] for market_id in fold.test_market_ids)
        train_ids = set(fold.train_market_ids)
        test_ids = set(fold.test_market_ids)
        if train_ids & test_ids:
            raise RuntimeError("market leakage detected across adaptive walk-forward fold")
        if (
            train_markets
            and max(row.settled_ts_ns for row in train_markets)
            >= fold.test_start_ts_ns
        ):
            raise RuntimeError(
                "future settlement label leaked into adaptive walk-forward train window"
            )
        train_trades = [trade for trade in trades if trade.market_id in train_ids]
        test_trades = [trade for trade in trades if trade.market_id in test_ids]
        policy = fit_policy(train_trades, config)
        stages = _apply_policy_stages(test_trades, policy, config)
        final_stage = stages["alpha_evidence_gate"]
        final_summary = _summary(final_stage)
        final_mean = final_summary["return_stats"].get("mean")
        if final_mean is not None and float(final_mean) > 0:
            positive_folds += 1
        for trade in final_stage:
            final_oos[(trade.market_id, trade.instrument, trade.signal_ts_ns)] = trade

        policy_payload = {
            "selected_symbols": list(policy.selected_symbols),
            "selected_setup_families": list(policy.selected_setup_families),
            "selected_sessions": list(policy.selected_sessions),
            "regime_thresholds_abs_momentum_1s_bps": {
                "low_max": policy.regime_low_max,
                "medium_max": policy.regime_medium_max,
            },
            "selected_regimes": list(policy.selected_regimes),
            "selection_mode": policy.selection_mode,
            "challenger_validation": {
                "reason": policy.challenger_validation_reason,
                "markets": policy.challenger_validation_markets,
                "mean_return": policy.challenger_validation_mean_return,
                "ci95_low": policy.challenger_validation_ci95_low,
            },
            "min_consensus": policy.min_consensus,
            "min_net_edge": policy.min_net_edge,
            "max_prediction_spread": policy.max_prediction_spread,
            "probability_calibration": (
                policy.probability_calibration.as_dict()
                if policy.probability_calibration is not None
                else None
            ),
            "min_calibrated_net_edge": config.min_calibrated_net_edge,
            "alpha_evidence": {
                "qualified": policy.alpha_evidence_qualified,
                "reason": policy.alpha_evidence_reason,
                "train_markets": policy.alpha_train_markets,
                "train_mean_return": policy.alpha_train_mean_return,
                "train_ci95_low": policy.alpha_train_ci95_low,
                "stability_positive_windows": policy.alpha_stability_positive_windows,
                "stability_total_windows": policy.alpha_stability_total_windows,
            },
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
            }
        )

    final_trades = list(final_oos.values())
    oos_summary = _summary(final_trades)
    blockers: list[str] = []
    if len(markets) < min_train_markets + test_markets:
        blockers.append("insufficient_markets_for_first_walk_forward_fold")
    if len(folds) < 3:
        blockers.append("fewer_than_3_walk_forward_folds")
    if oos_summary["unique_markets"] < 250:
        blockers.append("fewer_than_250_unique_oos_markets")
    oos_mean = oos_summary["return_stats"].get("mean")
    if oos_mean is None or float(oos_mean) < min_alpha_mean_return:
        blockers.append("adaptive_oos_expectancy_below_minimum")
    ci_low = oos_summary["return_stats"].get("mean_ci95_low")
    if ci_low is None or float(ci_low) <= 0:
        blockers.append("adaptive_oos_return_ci_not_positive")
    if folds and positive_folds / len(folds) < 2 / 3:
        blockers.append("fewer_than_two_thirds_positive_adaptive_oos_folds")
    if not final_trades:
        blockers.append("adaptive_policy_selected_no_oos_trades")

    return {
        "method": "adaptive_purged_expanding_market_level_walk_forward",
        "unique_markets": len(markets),
        "settled_trades": len(trades),
        "settled_trades_all_configs": len(all_trades),
        "excluded_config_mismatch_or_legacy_trades": excluded_config_trades,
        "required_evidence_config_sha256": required_evidence_config_sha256,
        "settlement_event_type": "ResearchSettlement",
        "parameters": {
            "min_train_markets": min_train_markets,
            "test_markets": test_markets,
            "step_markets": step_markets or test_markets,
            "min_symbol_train_markets": min_symbol_train_markets,
            "min_setup_train_markets": min_setup_train_markets,
            "min_session_train_markets": min_session_train_markets,
            "min_ml_train_trades": min_ml_train_trades,
            "min_calibration_bin_samples": min_calibration_bin_samples,
            "min_calibrated_net_edge": min_calibrated_net_edge,
            "min_final_train_markets": min_final_train_markets,
            "min_alpha_mean_return": min_alpha_mean_return,
            "stability_windows": stability_windows,
            "min_positive_stability_windows": min_positive_stability_windows,
            "challenger_validation_fraction": challenger_validation_fraction,
            "min_challenger_validation_markets": min_challenger_validation_markets,
        },
        "folds": fold_reports,
        "positive_folds": positive_folds,
        "final_stage": "alpha_evidence_gate",
        "oos": oos_summary,
        "qualified": not blockers,
        "blockers": blockers,
        "notes": [
            "Every selector/model is fitted only on labels settled before the next OOS test signal.",
            "When an evidence-config fingerprint is supplied, other config epochs are excluded before fitting.",
            "Market IDs are disjoint between train and test inside every fold.",
            "Markets sharing the same signal timestamp remain in one test block.",
            "Dynamic symbol, setup, UTC session, regime, consensus, execution-quality, realized-calibration and ML layers are reported sequentially for ablation.",
            "A compact train-only joint consensus+edge challenger is selected on an older inner-fit partition and must independently retain a positive market-level confidence bound on a newer inner-validation partition before it can replace sequential thresholds.",
            "Realized probability calibration is fitted only on purged train settlements and can only reduce modeled confidence/edge.",
            "The final alpha-evidence gate fails closed to NO_TRADE unless the train-selected portfolio reaches the configured minimum expectancy, has a positive lower 95% confidence bound, and remains positive across the configured chronological stability windows.",
            "A no-trade fold never satisfies promotion volume requirements and cannot be used to manufacture positive OOS evidence.",
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
