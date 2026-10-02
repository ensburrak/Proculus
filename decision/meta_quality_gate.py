from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MetaQualityGateResult:
    allowed: bool
    size_scale: float
    probability: float | None
    enforced: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "size_scale": self.size_scale,
            "probability": self.probability,
            "enforced": self.enforced,
            "reason": self.reason,
            "direction_authority": False,
        }


def _clamp(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return default


def evaluate_meta_quality_gate(*, probability: float | None, calibrated: bool, config: dict[str, Any] | None, runtime_mode: str) -> MetaQualityGateResult:
    cfg = dict(config or {})
    if not bool(cfg.get("enabled", False)):
        return MetaQualityGateResult(True, 1.0, None, False, "disabled")
    if not calibrated or probability is None:
        return MetaQualityGateResult(True, 1.0, None, False, "not_calibrated")

    p = _clamp(probability, 0.0, 1.0, 0.5)
    live = str(runtime_mode).lower() in {"live", "production", "real"}
    if live and not bool(cfg.get("enforce_live", False)):
        return MetaQualityGateResult(True, 1.0, p, False, "live_enforcement_disabled")

    block_below = _clamp(cfg.get("block_below", 0.45), 0.0, 1.0, 0.45)
    reduce_below = _clamp(cfg.get("reduce_below", 0.60), block_below, 1.0, 0.60)
    reduced = _clamp(cfg.get("reduced_size_scale", 0.50), 0.05, 1.0, 0.50)
    if p < block_below:
        return MetaQualityGateResult(False, 0.0, p, True, "below_block_threshold")
    if p < reduce_below:
        return MetaQualityGateResult(True, reduced, p, True, "reduced_size")
    return MetaQualityGateResult(True, 1.0, p, True, "pass")
