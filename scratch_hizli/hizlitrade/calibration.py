from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    count: int
    wins: int


@dataclass(frozen=True, slots=True)
class ProbabilityCalibration:
    bins: tuple[CalibrationBin, ...]
    prior_strength: Decimal
    min_bin_samples: int

    def calibrate(self, probability: float) -> tuple[float, int, bool]:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be inside [0, 1]")
        index = min(int(probability * len(self.bins)), len(self.bins) - 1)
        bucket = self.bins[index]
        if bucket.count < self.min_bin_samples:
            return probability, bucket.count, False

        raw = Decimal(str(probability))
        posterior = (
            Decimal(bucket.wins) + self.prior_strength * raw
        ) / (Decimal(bucket.count) + self.prior_strength)
        # Calibration is deliberately one-sided for deployment research:
        # historical evidence may reduce overconfidence but never manufacture
        # a larger edge than the transparent fair-value model produced.
        calibrated = min(raw, max(Decimal(0), min(Decimal(1), posterior)))
        return float(calibrated), bucket.count, True

    def as_dict(self) -> dict[str, object]:
        return {
            "bins": [
                {"count": bucket.count, "wins": bucket.wins}
                for bucket in self.bins
            ],
            "prior_strength": str(self.prior_strength),
            "min_bin_samples": self.min_bin_samples,
        }


def fit_probability_calibration(
    rows: list[tuple[float | None, bool]],
    *,
    bins: int = 10,
    min_bin_samples: int = 30,
    prior_strength: Decimal = Decimal("20"),
) -> ProbabilityCalibration | None:
    if bins <= 0:
        raise ValueError("bins must be positive")
    if min_bin_samples <= 0:
        raise ValueError("min_bin_samples must be positive")
    if prior_strength < 0:
        raise ValueError("prior_strength must be non-negative")

    counts = [0] * bins
    wins = [0] * bins
    observed = 0
    for probability, won in rows:
        if probability is None or not 0.0 <= probability <= 1.0:
            continue
        index = min(int(probability * bins), bins - 1)
        counts[index] += 1
        wins[index] += int(won)
        observed += 1
    if observed < min_bin_samples:
        return None

    model = ProbabilityCalibration(
        bins=tuple(
            CalibrationBin(count=count, wins=win)
            for count, win in zip(counts, wins, strict=True)
        ),
        prior_strength=prior_strength,
        min_bin_samples=min_bin_samples,
    )
    # A model with no individually mature bin cannot affect any future signal;
    # keep the policy explicit by treating it as unavailable.
    if not any(bucket.count >= min_bin_samples for bucket in model.bins):
        return None
    return model


def calibrated_net_edge(
    *,
    raw_probability: float,
    raw_net_edge: float | None,
    model: ProbabilityCalibration | None,
) -> tuple[float | None, float, int, bool]:
    if model is None:
        return raw_net_edge, raw_probability, 0, False
    calibrated_probability, samples, applied = model.calibrate(raw_probability)
    if raw_net_edge is None:
        return None, calibrated_probability, samples, applied
    shrink = max(raw_probability - calibrated_probability, 0.0)
    return raw_net_edge - shrink, calibrated_probability, samples, applied
