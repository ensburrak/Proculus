from __future__ import annotations

from typing import Any

from .regime_policy import get_regime_policy
from .strategy_experts import ExpertSignal
from .strategy_experts.compression_breakout import evaluate_compression
from .strategy_experts.range_revert import evaluate_range
from .strategy_experts.trend import evaluate_trend


def route_to_expert(*, regime: str, item: dict[str, Any], ta: dict[str, Any]) -> ExpertSignal | None:
    """Route one closed-candle market snapshot to exactly one regime expert.

    Direction belongs to deterministic experts. ML/LLM layers are not allowed
    to manufacture or flip direction here.
    """
    policy=get_regime_policy(regime)
    normalized=str(policy["regime"])
    if policy["action"]=="no_trade":
        return None

    max_leverage=max(1,int(policy.get("max_leverage") or 1))
    signal: ExpertSignal | None
    if normalized=="bull":
        signal=evaluate_trend(side="long",item=item,ta=ta,max_leverage=max_leverage)
    elif normalized=="bear":
        signal=evaluate_trend(side="short",item=item,ta=ta,max_leverage=max_leverage)
    elif normalized=="range":
        signal=evaluate_range(item=item,ta=ta,max_leverage=max_leverage)
    elif normalized=="compression":
        signal=evaluate_compression(item=item,ta=ta,max_leverage=max_leverage)
    else:
        signal=None

    if signal is None:
        return None
    if signal.direction not in set(policy.get("directions") or set()):
        return None
    if float(signal.confidence) < float(policy.get("min_confidence") or 1.0):
        return None
    return signal


__all__=["ExpertSignal","route_to_expert"]
