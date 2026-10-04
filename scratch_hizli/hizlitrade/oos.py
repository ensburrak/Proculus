from __future__ import annotations

import bisect
import json
import math
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from .validation import trade_return_summary


@dataclass(frozen=True, slots=True)
class SettledTrade:
    market_id: str
    instrument: str
    symbol: str
    signal_ts_ns: int
    settled_ts_ns: int
    cost_basis_usd: float
    realized_pnl_usd: float
    shares: float
    fair_probability: float | None
    won: bool
    momentum_1s_bps: float | None
    oracle_basis_bps: float | None
    prediction_spread: float | None
    quality_regime: str | None = None
    quality_score: float | None = None
    quality_size_multiplier: float | None = None
    evidence_config_sha256: str | None = None
    run_id: str | None = None
    outcome: str = ""
    momentum_250ms_bps: float | None = None
    momentum_3s_bps: float | None = None
    book_imbalance: float | None = None
    time_to_expiry_ms: float | None = None
    net_edge: float | None = None
    signal_executable_price: float | None = None
    market_duration_ms: float | None = None

    @property
    def realized_return(self) -> float:
        if self.cost_basis_usd <= 0:
            return 0.0
        return self.realized_pnl_usd / self.cost_basis_usd


@dataclass(frozen=True, slots=True)
class MarketOutcome:
    market_id: str
    symbol: str
    signal_ts_ns: int
    settled_ts_ns: int
    trade_count: int
    cost_basis_usd: float
    realized_pnl_usd: float
    realized_return: float
    mean_abs_momentum_1s_bps: float | None
    mean_abs_oracle_basis_bps: float | None


@dataclass(frozen=True, slots=True)
class PurgedMarketFold:
    train_market_ids: tuple[str, ...]
    test_market_ids: tuple[str, ...]
    train_start_ts_ns: int | None
    train_end_ts_ns: int | None
    train_last_settled_ts_ns: int | None
    test_start_ts_ns: int
    test_end_ts_ns: int
    purged_train_markets: int


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, str | int | float | Decimal):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _event_rows(root: Path, event_type: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in sorted(root.glob("*/*.ndjson")):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"malformed JSON at {path}:{line_number}") from exc
                if isinstance(event, dict) and event.get("event_type") == event_type:
                    rows.append(event)
    return rows


def load_settled_trades(
    root: Path,
    *,
    feature_match_max_age_ms: int = 2_000,
    settlement_event_type: str = "PaperSettlement",
) -> tuple[SettledTrade, ...]:
    """Join settled trades to the latest same-market point-in-time feature at signal time."""
    if feature_match_max_age_ms <= 0:
        raise ValueError("feature_match_max_age_ms must be positive")
    feature_index: dict[
        tuple[str, str, str | None, str | None],
        list[tuple[int, dict[str, Any]]],
    ] = defaultdict(list)
    for row in _event_rows(root, "LeadLagFeature"):
        try:
            key = (
                str(row["market_id"]),
                str(row["instrument"]),
                str(row.get("evidence_config_sha256") or "") or None,
                str(row.get("run_id") or "") or None,
            )
            ts = int(row["created_ts_ns"])
        except (KeyError, TypeError, ValueError):
            continue
        feature_index[key].append((ts, row))
    for rows in feature_index.values():
        rows.sort(key=lambda item: item[0])

    trades: list[SettledTrade] = []
    for row in _event_rows(root, settlement_event_type):
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            continue
        market_id = str(row.get("market_id") or "")
        instrument = str(row.get("instrument") or "")
        symbol = str(metadata.get("symbol") or "").upper()
        try:
            signal_ts_ns = int(metadata["signal_created_ts_ns"])
            settled_ts_ns = int(row["settled_ts_ns"])
        except (KeyError, TypeError, ValueError):
            continue
        cost = _number(row.get("cost_basis_usd"))
        pnl = _number(row.get("realized_pnl_usd"))
        shares = _number(row.get("shares"))
        if not market_id or not instrument or cost is None or cost <= 0 or pnl is None or shares is None:
            continue

        evidence_config_sha256 = str(row.get("evidence_config_sha256") or "") or None
        run_id = str(row.get("run_id") or "") or None
        feature: dict[str, Any] | None = None
        candidates = feature_index.get(
            (market_id, instrument, evidence_config_sha256, run_id),
            [],
        )
        if candidates:
            timestamps = [item[0] for item in candidates]
            index = bisect.bisect_right(timestamps, signal_ts_ns) - 1
            if index >= 0:
                candidate_ts, candidate = candidates[index]
                if 0 <= signal_ts_ns - candidate_ts <= feature_match_max_age_ms * 1_000_000:
                    feature = candidate

        outcome = str(row.get("outcome") or "").upper()
        winning_outcome = str(row.get("winning_outcome") or "").upper()
        fair_probability = _number(metadata.get("fair_probability"))
        trades.append(
            SettledTrade(
                market_id=market_id,
                instrument=instrument,
                symbol=symbol or "UNKNOWN",
                signal_ts_ns=signal_ts_ns,
                settled_ts_ns=settled_ts_ns,
                cost_basis_usd=cost,
                realized_pnl_usd=pnl,
                shares=shares,
                fair_probability=fair_probability,
                won=bool(outcome and winning_outcome and outcome == winning_outcome),
                momentum_1s_bps=(
                    _number(feature.get("spot_momentum_1s_bps")) if feature is not None else None
                ),
                oracle_basis_bps=(
                    _number(feature.get("oracle_basis_bps")) if feature is not None else None
                ),
                prediction_spread=(
                    _number(feature.get("prediction_spread")) if feature is not None else None
                ),
                quality_regime=(
                    str(metadata.get("quality_regime") or "") or None
                ),
                quality_score=_number(metadata.get("quality_score")),
                quality_size_multiplier=_number(metadata.get("quality_size_multiplier")),
                evidence_config_sha256=evidence_config_sha256,
                run_id=run_id,
                outcome=outcome,
                momentum_250ms_bps=(
                    _number(feature.get("spot_momentum_250ms_bps"))
                    if feature is not None
                    else None
                ),
                momentum_3s_bps=(
                    _number(feature.get("spot_momentum_3s_bps")) if feature is not None else None
                ),
                book_imbalance=(
                    _number(feature.get("book_imbalance")) if feature is not None else None
                ),
                time_to_expiry_ms=(
                    _number(feature.get("time_to_expiry_ms")) if feature is not None else None
                ),
                net_edge=_number(metadata.get("net_edge")),
                signal_executable_price=_number(metadata.get("signal_executable_price")),
                market_duration_ms=_number(metadata.get("market_duration_ms")),
            )
        )
    return tuple(sorted(trades, key=lambda trade: (trade.signal_ts_ns, trade.market_id, trade.instrument)))


def load_research_trades(
    root: Path,
    *,
    feature_match_max_age_ms: int = 2_000,
) -> tuple[SettledTrade, ...]:
    """Load isolated counterfactual settlements without mixing them into paper PnL."""
    return load_settled_trades(
        root,
        feature_match_max_age_ms=feature_match_max_age_ms,
        settlement_event_type="ResearchSettlement",
    )


def aggregate_markets(trades: tuple[SettledTrade, ...]) -> tuple[MarketOutcome, ...]:
    grouped: dict[str, list[SettledTrade]] = defaultdict(list)
    for trade in trades:
        grouped[trade.market_id].append(trade)

    outcomes: list[MarketOutcome] = []
    for market_id, rows in grouped.items():
        cost = sum(row.cost_basis_usd for row in rows)
        pnl = sum(row.realized_pnl_usd for row in rows)
        symbols = {row.symbol for row in rows}
        momenta = [abs(row.momentum_1s_bps) for row in rows if row.momentum_1s_bps is not None]
        bases = [abs(row.oracle_basis_bps) for row in rows if row.oracle_basis_bps is not None]
        outcomes.append(
            MarketOutcome(
                market_id=market_id,
                symbol=next(iter(symbols)) if len(symbols) == 1 else "MIXED",
                signal_ts_ns=min(row.signal_ts_ns for row in rows),
                settled_ts_ns=max(row.settled_ts_ns for row in rows),
                trade_count=len(rows),
                cost_basis_usd=cost,
                realized_pnl_usd=pnl,
                realized_return=pnl / cost if cost > 0 else 0.0,
                mean_abs_momentum_1s_bps=statistics.fmean(momenta) if momenta else None,
                mean_abs_oracle_basis_bps=statistics.fmean(bases) if bases else None,
            )
        )
    return tuple(sorted(outcomes, key=lambda item: (item.signal_ts_ns, item.market_id)))


def purged_market_walk_forward_folds(
    markets: tuple[MarketOutcome, ...],
    *,
    min_train_size: int,
    test_size: int,
    step: int | None = None,
) -> tuple[PurgedMarketFold, ...]:
    """Build causal expanding OOS folds from only labels known before test time.

    Markets sharing one signal timestamp stay in the same test block. A prior
    market is eligible for training only when its final settlement timestamp is
    strictly earlier than the first signal timestamp of the test block.
    """
    if min_train_size <= 0 or test_size <= 0:
        raise ValueError("market window sizes must be positive")
    stride = step or test_size
    if stride <= 0:
        raise ValueError("step must be positive")
    if not markets:
        return ()

    ordered = tuple(sorted(markets, key=lambda item: (item.signal_ts_ns, item.market_id)))
    groups: list[tuple[MarketOutcome, ...]] = []
    current: list[MarketOutcome] = []
    current_ts: int | None = None
    for market in ordered:
        if current_ts is None or market.signal_ts_ns == current_ts:
            current.append(market)
            current_ts = market.signal_ts_ns
            continue
        groups.append(tuple(current))
        current = [market]
        current_ts = market.signal_ts_ns
    if current:
        groups.append(tuple(current))

    folds: list[PurgedMarketFold] = []
    group_index = 1
    while group_index < len(groups):
        test_start_ts = groups[group_index][0].signal_ts_ns
        prior = [market for group in groups[:group_index] for market in group]
        train = [market for market in prior if market.settled_ts_ns < test_start_ts]
        if len(train) < min_train_size:
            group_index += 1
            continue

        test: list[MarketOutcome] = []
        test_end_group = group_index
        while test_end_group < len(groups) and len(test) < test_size:
            test.extend(groups[test_end_group])
            test_end_group += 1
        if len(test) < test_size:
            break

        folds.append(
            PurgedMarketFold(
                train_market_ids=tuple(market.market_id for market in train),
                test_market_ids=tuple(market.market_id for market in test),
                train_start_ts_ns=train[0].signal_ts_ns if train else None,
                train_end_ts_ns=train[-1].signal_ts_ns if train else None,
                train_last_settled_ts_ns=max(
                    (market.settled_ts_ns for market in train),
                    default=None,
                ),
                test_start_ts_ns=test[0].signal_ts_ns,
                test_end_ts_ns=test[-1].signal_ts_ns,
                purged_train_markets=len(prior) - len(train),
            )
        )

        if step is None or stride == test_size:
            group_index = test_end_group
            continue

        advanced = 0
        next_group = group_index
        while next_group < len(groups) and advanced < stride:
            advanced += len(groups[next_group])
            next_group += 1
        group_index = max(next_group, group_index + 1)

    return tuple(folds)


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def _regime_thresholds(rows: tuple[MarketOutcome, ...]) -> tuple[float, float] | None:
    values = [
        row.mean_abs_momentum_1s_bps
        for row in rows
        if row.mean_abs_momentum_1s_bps is not None
    ]
    if len(values) < 9:
        return None
    low = _quantile(values, 1 / 3)
    high = _quantile(values, 2 / 3)
    if low is None or high is None:
        return None
    return low, high


def _regime_summary(
    train: tuple[MarketOutcome, ...],
    test: tuple[MarketOutcome, ...],
) -> dict[str, Any]:
    thresholds = _regime_thresholds(train)
    if thresholds is None:
        return {"available": False, "reason": "insufficient_train_momentum_features"}
    low, high = thresholds
    grouped: dict[str, list[float]] = {"low": [], "medium": [], "high": [], "unknown": []}
    for row in test:
        value = row.mean_abs_momentum_1s_bps
        if value is None:
            grouped["unknown"].append(row.realized_return)
        elif value <= low:
            grouped["low"].append(row.realized_return)
        elif value <= high:
            grouped["medium"].append(row.realized_return)
        else:
            grouped["high"].append(row.realized_return)
    return {
        "available": True,
        "train_thresholds_abs_momentum_1s_bps": {"low_max": low, "medium_max": high},
        "test": {name: trade_return_summary(values) for name, values in grouped.items()},
    }


def _quality_attribution(trades: tuple[SettledTrade, ...]) -> dict[str, Any]:
    grouped: dict[str, list[SettledTrade]] = defaultdict(list)
    for trade in trades:
        grouped[trade.quality_regime or "unknown"].append(trade)

    result: dict[str, Any] = {}
    for regime, rows in sorted(grouped.items()):
        returns = [row.realized_return for row in rows]
        result[regime] = {
            "trades": len(rows),
            "cost_basis_usd": sum(row.cost_basis_usd for row in rows),
            "realized_pnl_usd": sum(row.realized_pnl_usd for row in rows),
            "return_stats": trade_return_summary(returns),
            "mean_quality_score": (
                statistics.fmean(
                    row.quality_score for row in rows if row.quality_score is not None
                )
                if any(row.quality_score is not None for row in rows)
                else None
            ),
            "mean_size_multiplier": (
                statistics.fmean(
                    row.quality_size_multiplier
                    for row in rows
                    if row.quality_size_multiplier is not None
                )
                if any(row.quality_size_multiplier is not None for row in rows)
                else None
            ),
        }
    return result


def market_walk_forward_report(
    root: Path,
    *,
    min_train_markets: int = 100,
    test_markets: int = 50,
    step_markets: int | None = None,
    required_evidence_config_sha256: str | None = None,
) -> dict[str, Any]:
    """Evaluate settled performance on non-overlapping market-level OOS folds.

    All fills/settlements belonging to one market are collapsed before splitting,
    preventing same-event leakage across train and test. Regime boundaries are
    derived from each fold's training window only and then frozen for its test window.
    """
    all_trades = load_settled_trades(root)
    if required_evidence_config_sha256 is None:
        trades = all_trades
    else:
        trades = tuple(
            trade for trade in all_trades
            if trade.evidence_config_sha256 == required_evidence_config_sha256
        )
    excluded_config_trades = len(all_trades) - len(trades)
    markets = aggregate_markets(trades)
    folds = purged_market_walk_forward_folds(
        markets,
        min_train_size=min_train_markets,
        test_size=test_markets,
        step=step_markets,
    )
    fold_reports: list[dict[str, Any]] = []
    oos_market_ids: set[str] = set()
    oos_returns: list[float] = []
    positive_folds = 0
    oos_trades: list[SettledTrade] = []

    by_id = {row.market_id: row for row in markets}
    for index, fold in enumerate(folds):
        train = tuple(by_id[market_id] for market_id in fold.train_market_ids)
        test = tuple(by_id[market_id] for market_id in fold.test_market_ids)
        overlap = {row.market_id for row in train} & {row.market_id for row in test}
        if overlap:
            raise RuntimeError("market leakage detected across walk-forward fold")
        if train and max(row.settled_ts_ns for row in train) >= fold.test_start_ts_ns:
            raise RuntimeError("future settlement label leaked into walk-forward train window")
        test_returns = [row.realized_return for row in test]
        summary = trade_return_summary(test_returns)
        if summary["mean"] is not None and float(summary["mean"]) > 0:
            positive_folds += 1
        test_market_ids = {row.market_id for row in test}
        fold_trades = tuple(trade for trade in trades if trade.market_id in test_market_ids)
        new_market_ids = test_market_ids - oos_market_ids
        for row in test:
            if row.market_id in new_market_ids:
                oos_market_ids.add(row.market_id)
                oos_returns.append(row.realized_return)
        oos_trades.extend(
            trade for trade in fold_trades if trade.market_id in new_market_ids
        )
        fold_reports.append(
            {
                "fold": index,
                "train_markets": len(train),
                "test_markets": len(test),
                "purged_train_markets": fold.purged_train_markets,
                "train_start_ts_ns": fold.train_start_ts_ns,
                "train_end_ts_ns": fold.train_end_ts_ns,
                "train_last_settled_ts_ns": fold.train_last_settled_ts_ns,
                "test_start_ts_ns": fold.test_start_ts_ns,
                "test_end_ts_ns": fold.test_end_ts_ns,
                "test_return_stats": summary,
                "regimes": _regime_summary(train, test),
                "quality_attribution": _quality_attribution(fold_trades),
            }
        )

    oos_summary = trade_return_summary(oos_returns)
    blockers: list[str] = []
    if len(markets) < min_train_markets + test_markets:
        blockers.append("insufficient_markets_for_first_walk_forward_fold")
    if len(folds) < 3:
        blockers.append("fewer_than_3_walk_forward_folds")
    if len(oos_market_ids) < 250:
        blockers.append("fewer_than_250_unique_oos_markets")
    mean_ci_low = oos_summary.get("mean_ci95_low")
    if mean_ci_low is None or float(mean_ci_low) <= 0:
        blockers.append("oos_market_return_ci_not_positive")
    if folds and positive_folds / len(folds) < 2 / 3:
        blockers.append("fewer_than_two_thirds_positive_oos_folds")

    return {
        "method": "purged_expanding_market_level_walk_forward",
        "settled_trades": len(trades),
        "settled_trades_all_configs": len(all_trades),
        "excluded_config_mismatch_or_legacy_trades": excluded_config_trades,
        "required_evidence_config_sha256": required_evidence_config_sha256,
        "unique_markets": len(markets),
        "feature_matched_trades": sum(trade.momentum_1s_bps is not None for trade in trades),
        "parameters": {
            "min_train_markets": min_train_markets,
            "test_markets": test_markets,
            "step_markets": step_markets or test_markets,
        },
        "folds": fold_reports,
        "positive_folds": positive_folds,
        "unique_oos_markets": len(oos_market_ids),
        "oos_market_return_stats": oos_summary,
        "quality_attribution": _quality_attribution(tuple(oos_trades)),
        "qualified": not blockers,
        "blockers": blockers,
        "notes": [
            "Splitting occurs only after all same-market settlements are aggregated.",
            "Training labels must settle strictly before the next OOS test block starts.",
            "Markets sharing one signal timestamp remain in the same test block.",
            "No parameter fitting occurs inside the OOS test windows.",
            "Volatility-regime thresholds are estimated from each fold's train window only.",
        ],
    }


def write_market_walk_forward_report(root: Path, destination: Path, **kwargs: Any) -> dict[str, Any]:
    report = market_walk_forward_report(root, **kwargs)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def market_records_as_dicts(root: Path) -> list[dict[str, Any]]:
    return [asdict(row) for row in aggregate_markets(load_settled_trades(root))]
