from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_probability: float | None
    observed_rate: float | None


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    count: int
    brier_score: float | None
    log_loss: float | None
    expected_calibration_error: float | None
    bins: tuple[CalibrationBin, ...]


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def _validate_probability(value: float) -> float:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("probabilities must be finite values in [0, 1]")
    return value


def binary_calibration(
    probabilities: Sequence[float],
    outcomes: Sequence[int | bool],
    *,
    bins: int = 10,
) -> CalibrationReport:
    """Compute proper scoring rules and equal-width reliability bins.

    Outcomes must already be final resolutions. No marked/unrealized labels are
    accepted by this function, which keeps calibration tied to realized truth.
    """
    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must have equal length")
    if bins <= 0:
        raise ValueError("bins must be positive")
    if not probabilities:
        empty_bins = tuple(
            CalibrationBin(i / bins, (i + 1) / bins, 0, None, None)
            for i in range(bins)
        )
        return CalibrationReport(0, None, None, None, empty_bins)

    probs = [_validate_probability(float(value)) for value in probabilities]
    labels: list[int] = []
    for value in outcomes:
        label = int(value)
        if label not in {0, 1}:
            raise ValueError("binary outcomes must be 0/1")
        labels.append(label)

    brier = sum((p - y) ** 2 for p, y in zip(probs, labels, strict=True)) / len(probs)
    epsilon = 1e-15
    log_loss = -sum(
        y * math.log(min(max(p, epsilon), 1 - epsilon))
        + (1 - y) * math.log(min(max(1 - p, epsilon), 1 - epsilon))
        for p, y in zip(probs, labels, strict=True)
    ) / len(probs)

    bucket_probabilities: list[list[float]] = [[] for _ in range(bins)]
    bucket_outcomes: list[list[int]] = [[] for _ in range(bins)]
    for p, y in zip(probs, labels, strict=True):
        index = min(int(p * bins), bins - 1)
        bucket_probabilities[index].append(p)
        bucket_outcomes[index].append(y)

    reliability: list[CalibrationBin] = []
    ece = 0.0
    for index, (bucket_p, bucket_y) in enumerate(
        zip(bucket_probabilities, bucket_outcomes, strict=True)
    ):
        lower = index / bins
        upper = (index + 1) / bins
        if not bucket_p:
            reliability.append(CalibrationBin(lower, upper, 0, None, None))
            continue
        mean_probability = statistics.fmean(bucket_p)
        observed_rate = statistics.fmean(bucket_y)
        count = len(bucket_p)
        ece += count / len(probs) * abs(mean_probability - observed_rate)
        reliability.append(
            CalibrationBin(lower, upper, count, mean_probability, observed_rate)
        )

    return CalibrationReport(len(probs), brier, log_loss, ece, tuple(reliability))


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 2_000,
    seed: int = 17,
) -> tuple[float, float] | None:
    """Deterministic non-parametric bootstrap CI for a sample mean."""
    if not values:
        return None
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if resamples < 100:
        raise ValueError("resamples must be at least 100")
    clean = [float(value) for value in values]
    if not all(math.isfinite(value) for value in clean):
        raise ValueError("bootstrap values must be finite")
    if len(clean) == 1:
        return clean[0], clean[0]

    rng = random.Random(seed)
    sample_size = len(clean)
    means = sorted(
        statistics.fmean(clean[rng.randrange(sample_size)] for _ in range(sample_size))
        for _ in range(resamples)
    )
    alpha = (1.0 - confidence) / 2.0
    lower_index = max(0, min(resamples - 1, int(alpha * resamples)))
    upper_index = max(0, min(resamples - 1, int((1.0 - alpha) * resamples) - 1))
    return means[lower_index], means[upper_index]


def walk_forward_splits(
    length: int,
    *,
    min_train_size: int,
    test_size: int,
    step: int | None = None,
) -> tuple[WalkForwardFold, ...]:
    """Generate expanding-window, strictly time-ordered train/test folds."""
    if length < 0 or min_train_size <= 0 or test_size <= 0:
        raise ValueError("length must be non-negative and window sizes positive")
    stride = step or test_size
    if stride <= 0:
        raise ValueError("step must be positive")

    folds: list[WalkForwardFold] = []
    train_end = min_train_size
    while train_end + test_size <= length:
        folds.append(WalkForwardFold(0, train_end, train_end, train_end + test_size))
        train_end += stride
    return tuple(folds)


def trade_return_summary(returns: Sequence[float]) -> dict[str, float | int | None]:
    """Summarize realized per-trade returns with a bootstrap expectancy interval."""
    if not returns:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "win_rate": None,
            "profit_factor": None,
            "mean_ci95_low": None,
            "mean_ci95_high": None,
        }
    clean = [float(value) for value in returns]
    gains = sum(value for value in clean if value > 0)
    losses = -sum(value for value in clean if value < 0)
    ci = bootstrap_mean_ci(clean)
    return {
        "count": len(clean),
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "win_rate": sum(value > 0 for value in clean) / len(clean),
        "profit_factor": gains / losses if losses > 0 else (math.inf if gains > 0 else None),
        "mean_ci95_low": ci[0] if ci is not None else None,
        "mean_ci95_high": ci[1] if ci is not None else None,
    }


def stress_edge(
    net_edges: Sequence[float],
    *,
    extra_costs: Sequence[float] = (0.0, 0.0025, 0.005, 0.01, 0.02),
) -> tuple[dict[str, float | int], ...]:
    """Apply explicit extra execution-cost shocks without refitting the strategy."""
    clean_edges = [float(value) for value in net_edges]
    rows: list[dict[str, float | int]] = []
    for cost in extra_costs:
        if cost < 0:
            raise ValueError("extra costs must be non-negative")
        stressed = [edge - cost for edge in clean_edges]
        rows.append(
            {
                "extra_cost": cost,
                "count": len(stressed),
                "positive_count": sum(value > 0 for value in stressed),
                "mean_stressed_edge": statistics.fmean(stressed) if stressed else 0.0,
            }
        )
    return tuple(rows)
