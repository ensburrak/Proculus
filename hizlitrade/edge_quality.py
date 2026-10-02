from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from .models import PaperSettlement, ResearchSettlement
from .oos import SettledTrade


@dataclass(slots=True)
class _CalibrationBin:
    count: int = 0
    wins: int = 0


@dataclass(slots=True)
class _ReturnStats:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0

    def observe(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.total_sq += value * value

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    def upper_mean_bound(self, z: float = 1.645) -> float | None:
        """One-sided 95% normal bound; used only after a minimum sample gate."""
        if self.count == 0:
            return None
        if self.count == 1:
            return self.mean
        mean = self.mean
        numerator = self.total_sq - self.count * mean * mean
        variance = max(numerator / (self.count - 1), 0.0)
        standard_error = math.sqrt(variance / self.count)
        return mean + z * standard_error


@dataclass(frozen=True, slots=True)
class EdgeQualityAssessment:
    fair_probability: Decimal
    calibration_applied: bool
    calibration_samples: int
    symbol_samples: int = 0
    setup_samples: int = 0
    reject_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SequentialEdgeQualityResult:
    accepted_trades: tuple[SettledTrade, ...]
    rejected: tuple[tuple[str, int], ...]
    calibration_applied: int
    observed_settlements: int

    @property
    def rejection_counts(self) -> dict[str, int]:
        return dict(self.rejected)


def _directional_bucket(value: float | None, *, outcome: str) -> str:
    if value is None or not math.isfinite(value):
        return "unknown"
    direction = 1.0 if outcome.upper() == "YES" else -1.0
    aligned = value * direction
    if abs(aligned) <= 1e-12:
        return "flat"
    return "aligned" if aligned > 0 else "opposed"


def setup_key_from_values(
    *,
    outcome: str,
    oracle_basis_bps: float | None = None,
    momentum_1s_bps: float | None = None,
    book_imbalance: float | None = None,
    time_to_expiry_ms: float | None = None,
    market_duration_ms: float | None = None,
) -> str:
    """Build one stable point-in-time setup family without outcome leakage."""
    normalized_outcome = outcome.upper()
    tte = time_to_expiry_ms
    if tte is None or not math.isfinite(tte):
        tte_bucket = "unknown"
    elif tte <= 300_000:
        tte_bucket = "short"
    elif tte <= 900_000:
        tte_bucket = "medium"
    else:
        tte_bucket = "long"

    duration = market_duration_ms
    if duration is None or not math.isfinite(duration) or duration <= 0:
        duration_bucket = "unknown"
    else:
        duration_bucket = f"{max(1, round(duration / 60_000))}m"

    return "|".join(
        (
            normalized_outcome,
            f"basis_{_directional_bucket(oracle_basis_bps, outcome=normalized_outcome)}",
            f"mom_{_directional_bucket(momentum_1s_bps, outcome=normalized_outcome)}",
            f"book_{_directional_bucket(book_imbalance, outcome=normalized_outcome)}",
            f"tte_{tte_bucket}",
            f"dur_{duration_bucket}",
        )
    )


def setup_key_from_trade(trade: SettledTrade) -> str:
    """Use the exact runtime family when persisted, otherwise reconstruct it."""
    persisted = (trade.edge_quality_setup or "").strip()
    if persisted:
        return persisted
    return setup_key_from_values(
        outcome=trade.outcome,
        oracle_basis_bps=trade.oracle_basis_bps,
        momentum_1s_bps=trade.momentum_1s_bps,
        book_imbalance=trade.book_imbalance,
        time_to_expiry_ms=trade.time_to_expiry_ms,
        market_duration_ms=trade.market_duration_ms,
    )


class EdgeQualityModel:
    """Point-in-time quality guard learned only from final realized settlements.

    Empirical calibration is deliberately one-sided: it can reduce an
    overconfident fair probability but can never increase it. Symbol/setup
    vetoes require enough realized samples and a negative one-sided 95% upper
    confidence bound. No mark, open position, or future settlement is accepted
    as training truth.
    """

    def __init__(
        self,
        *,
        calibration_bins: int = 10,
        min_calibration_bin_samples: int = 30,
        min_symbol_samples: int = 30,
        min_setup_samples: int = 30,
        calibration_prior_strength: Decimal = Decimal("20"),
    ) -> None:
        if calibration_bins <= 0:
            raise ValueError("calibration_bins must be positive")
        if min_calibration_bin_samples <= 0:
            raise ValueError("min_calibration_bin_samples must be positive")
        if min_symbol_samples <= 0:
            raise ValueError("min_symbol_samples must be positive")
        if min_setup_samples <= 0:
            raise ValueError("min_setup_samples must be positive")
        if calibration_prior_strength < 0:
            raise ValueError("calibration_prior_strength must be non-negative")

        self.calibration_bins = calibration_bins
        self.min_calibration_bin_samples = min_calibration_bin_samples
        self.min_symbol_samples = min_symbol_samples
        self.min_setup_samples = min_setup_samples
        self.calibration_prior_strength = calibration_prior_strength
        self._calibration = [_CalibrationBin() for _ in range(calibration_bins)]
        self._symbol_returns: dict[str, _ReturnStats] = {}
        self._setup_returns: dict[str, _ReturnStats] = {}

    @staticmethod
    def _probability(value: object) -> Decimal | None:
        try:
            probability = Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError):
            return None
        if not probability.is_finite() or probability < 0 or probability > 1:
            return None
        return probability

    def _bin_index(self, probability: Decimal) -> int:
        index = int(probability * self.calibration_bins)
        return min(index, self.calibration_bins - 1)

    def _observe(
        self,
        *,
        fair_probability: object,
        won: bool,
        symbol: str,
        setup_key: str | None,
        realized_return: float | None,
    ) -> bool:
        learned = False
        probability = self._probability(fair_probability)
        if probability is not None:
            bucket = self._calibration[self._bin_index(probability)]
            bucket.count += 1
            bucket.wins += int(won)
            learned = True

        if realized_return is not None and math.isfinite(realized_return):
            symbol_key = symbol.strip().upper()
            if symbol_key:
                self._symbol_returns.setdefault(symbol_key, _ReturnStats()).observe(
                    realized_return
                )
                learned = True
            setup = (setup_key or "").strip()
            if setup:
                self._setup_returns.setdefault(setup, _ReturnStats()).observe(
                    realized_return
                )
                learned = True
        return learned

    def observe_settlement(
        self,
        settlement: PaperSettlement | ResearchSettlement,
    ) -> bool:
        """Learn from one final settlement event."""
        outcome = settlement.outcome.upper()
        winning = settlement.winning_outcome.upper()
        fair_probability = settlement.metadata.get("fair_probability")
        symbol = str(settlement.metadata.get("symbol") or "")
        setup_key = str(settlement.metadata.get("edge_quality_setup") or "") or None
        realized_return: float | None = None
        if settlement.cost_basis_usd > 0:
            realized_return = float(
                settlement.realized_pnl_usd / settlement.cost_basis_usd
            )
        return self._observe(
            fair_probability=fair_probability,
            won=outcome in {"YES", "NO"} and outcome == winning,
            symbol=symbol,
            setup_key=setup_key,
            realized_return=realized_return,
        )

    def observe_trade(self, trade: SettledTrade) -> bool:
        """Learn from one historical settled trade after its label is observable."""
        return self._observe(
            fair_probability=trade.fair_probability,
            won=trade.won,
            symbol=trade.symbol,
            setup_key=setup_key_from_trade(trade),
            realized_return=trade.realized_return,
        )

    @staticmethod
    def _negative_with_confidence(
        stats: _ReturnStats | None,
        minimum: int,
    ) -> bool:
        if stats is None or stats.count < minimum:
            return False
        upper = stats.upper_mean_bound()
        return upper is not None and upper < 0.0

    def veto_reason(self, *, symbol: str, setup_key: str | None) -> str | None:
        symbol_stats = self._symbol_returns.get(symbol.strip().upper())
        setup_stats = self._setup_returns.get(setup_key or "")
        if self._negative_with_confidence(symbol_stats, self.min_symbol_samples):
            return "symbol_expectancy_negative"
        if self._negative_with_confidence(setup_stats, self.min_setup_samples):
            return "setup_expectancy_negative"
        return None

    def assess(
        self,
        *,
        raw_probability: Decimal,
        symbol: str,
        setup_key: str | None,
    ) -> EdgeQualityAssessment:
        """Return conservative calibration plus statistically supported vetoes."""
        probability = self._probability(raw_probability)
        if probability is None:
            raise ValueError("raw_probability must be finite and inside [0, 1]")

        bucket = self._calibration[self._bin_index(probability)]
        calibration_applied = bucket.count >= self.min_calibration_bin_samples
        calibrated = probability
        if calibration_applied:
            prior = self.calibration_prior_strength
            posterior = (
                Decimal(bucket.wins) + prior * probability
            ) / (Decimal(bucket.count) + prior)
            calibrated = min(
                probability,
                max(Decimal(0), min(Decimal(1), posterior)),
            )

        symbol_stats = self._symbol_returns.get(symbol.strip().upper())
        setup_stats = self._setup_returns.get(setup_key or "")
        return EdgeQualityAssessment(
            fair_probability=calibrated,
            calibration_applied=calibration_applied,
            calibration_samples=bucket.count,
            symbol_samples=symbol_stats.count if symbol_stats is not None else 0,
            setup_samples=setup_stats.count if setup_stats is not None else 0,
            reject_reason=self.veto_reason(symbol=symbol, setup_key=setup_key),
        )


def warm_edge_quality_model(
    trades: Iterable[SettledTrade],
    *,
    model: EdgeQualityModel,
    before_ns: int,
) -> int:
    """Warm a shadow model from unique labels observable before startup.

    Historical rows may be duplicated by copied evidence directories or repeated
    exports. Count a settlement identity once, and never learn from a label whose
    settlement timestamp is at or beyond the startup cutoff.
    """
    if before_ns <= 0:
        raise ValueError("before_ns must be positive")
    seen: set[tuple[str, str, int, int]] = set()
    learned = 0
    eligible = sorted(
        (trade for trade in trades if 0 < trade.settled_ts_ns < before_ns),
        key=lambda trade: (
            trade.settled_ts_ns,
            trade.market_id,
            trade.instrument,
            trade.signal_ts_ns,
        ),
    )
    for trade in eligible:
        identity = (
            trade.market_id,
            trade.instrument,
            trade.signal_ts_ns,
            trade.settled_ts_ns,
        )
        if identity in seen:
            continue
        seen.add(identity)
        if model.observe_trade(trade):
            learned += 1
    return learned


def sequential_edge_quality_filter(
    trades: list[SettledTrade],
    *,
    model: EdgeQualityModel | None = None,
    min_net_edge: float = 0.025,
) -> SequentialEdgeQualityResult:
    """Apply edge quality sequentially without future-label leakage.

    Before each signal, the model is updated only with trades whose settlement
    timestamp is strictly earlier than that signal. Counterfactual settlements
    remain eligible as research truth even when a prior policy would have
    rejected their paper deployment; this prevents the quality model from
    creating a self-confirming blind spot.
    """
    if min_net_edge < 0:
        raise ValueError("min_net_edge must be non-negative")
    quality = model or EdgeQualityModel()
    ordered_signals = sorted(
        trades,
        key=lambda trade: (
            trade.signal_ts_ns,
            trade.market_id,
            trade.instrument,
        ),
    )
    ordered_settlements = sorted(
        trades,
        key=lambda trade: (
            trade.settled_ts_ns,
            trade.market_id,
            trade.instrument,
        ),
    )
    settlement_index = 0
    observed = 0
    calibration_applied = 0
    rejected: Counter[str] = Counter()
    accepted: list[SettledTrade] = []

    for trade in ordered_signals:
        while (
            settlement_index < len(ordered_settlements)
            and ordered_settlements[settlement_index].settled_ts_ns
            < trade.signal_ts_ns
        ):
            if quality.observe_trade(ordered_settlements[settlement_index]):
                observed += 1
            settlement_index += 1

        setup_key = setup_key_from_trade(trade)
        reason = quality.veto_reason(symbol=trade.symbol, setup_key=setup_key)
        if reason is not None:
            rejected[reason] += 1
            continue

        if trade.fair_probability is not None:
            raw_probability = Decimal(str(trade.fair_probability))
            assessment = quality.assess(
                raw_probability=raw_probability,
                symbol=trade.symbol,
                setup_key=setup_key,
            )
            if assessment.calibration_applied:
                calibration_applied += 1
            if assessment.reject_reason is not None:
                rejected[assessment.reject_reason] += 1
                continue
            if trade.signal_net_edge is not None:
                calibrated_net_edge = (
                    trade.signal_net_edge
                    - float(raw_probability - assessment.fair_probability)
                )
                if calibrated_net_edge < min_net_edge:
                    rejected["calibrated_edge_below_minimum"] += 1
                    continue

        accepted.append(trade)

    return SequentialEdgeQualityResult(
        accepted_trades=tuple(accepted),
        rejected=tuple(sorted(rejected.items())),
        calibration_applied=calibration_applied,
        observed_settlements=observed,
    )
