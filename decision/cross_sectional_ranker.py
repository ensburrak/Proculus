from __future__ import annotations

import math
from typing import Any


def _confidence(decision: dict[str, Any]) -> float:
    for key in ("master_confidence", "confidence", "score"):
        try:
            value = float(decision.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return max(0.0, min(1.0, value))
    return 0.0


def rank_enter_decisions(
    decisions: dict[str, dict[str, Any]],
    *,
    top_fraction: float = 0.15,
    min_candidates: int = 2,
) -> dict[str, dict[str, Any]]:
    """Apply a cross-sectional quality gate without blending decision engines.

    Existing non-entry decisions are preserved. When enough entries compete in
    the same batch, only the top fraction by confidence retains execution
    authority. The remainder fail closed to hold.
    """
    ranked = {str(symbol): dict(payload) for symbol, payload in decisions.items()}
    entrants = [
        (symbol, payload)
        for symbol, payload in ranked.items()
        if str(payload.get("action") or "").lower() == "enter"
    ]
    if len(entrants) < max(1, int(min_candidates)):
        return ranked

    fraction = max(0.0, min(1.0, float(top_fraction)))
    keep_count = max(1, int(math.ceil(len(entrants) * fraction)))
    ordered = sorted(
        entrants,
        key=lambda item: (-_confidence(item[1]), str(item[0])),
    )
    keep = {symbol for symbol, _ in ordered[:keep_count]}

    for rank, (symbol, payload) in enumerate(ordered, start=1):
        audit = (
            dict(payload.get("cross_sectional_rank") or {})
            if isinstance(payload.get("cross_sectional_rank"), dict)
            else {}
        )
        audit.update(
            {
                "rank": rank,
                "candidate_count": len(ordered),
                "keep_count": keep_count,
                "top_fraction": fraction,
                "confidence": _confidence(payload),
                "selected": symbol in keep,
            }
        )
        payload["cross_sectional_rank"] = audit
        if symbol in keep:
            continue

        previous_reason = str(payload.get("reason") or "").strip()
        veto = "cross-sectional rank veto"
        payload["action"] = "hold"
        payload["direction"] = "neutral"
        payload["base_decision"] = "neutral"
        payload["risk_scale"] = 0.0
        payload["lev"] = 0
        payload["max_leverage"] = 0.0
        payload["reason"] = f"{previous_reason} | {veto}" if previous_reason else veto

    return ranked


__all__ = ["rank_enter_decisions"]
