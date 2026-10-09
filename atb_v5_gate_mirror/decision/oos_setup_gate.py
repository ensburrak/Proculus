from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY_PATH = _ROOT / "config" / "oos_setup_eligibility.json"
SCOPED_FAMILIES = frozenset(
    {
        "bull_trend",
        "bear_trend",
        "range_revert",
        "compression_breakout",
        "stochrsi_opportunity",
        "trend_breakout_v3",
        "pullback_reclaim_v3",
        "range_failed_break_v3",
        "squeeze_expansion_v3",
        "vwap_reclaim_v3",
        "stoch_trend_reentry_v3",
        "donchian_4h_v3_20",
        "donchian_4h_v3_40",
        "donchian_4h_v3_10",
        "persistent_breakout_4h_v4",
        "dual_momentum_breakout_4h_v4",
        "breakout_retest_4h_v4",
        "volatility_expansion_breakout_4h_v4",
        "breadth_donchian_40_v5",
        "breadth_donchian_10_v5",
        "breadth_momentum_breakout_v5",
        "cross_sectional_momentum_v5",
    }
)


@dataclass(frozen=True)
class OosSetupEligibilityDecision:
    allowed: bool
    reason: str
    setup_id: str
    family: str
    blockers: tuple[str, ...] = ()
    audit: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": bool(self.allowed),
            "reason": self.reason,
            "setup_id": self.setup_id,
            "family": self.family,
            "blockers": list(self.blockers),
            "audit": dict(self.audit),
        }


def _family(setup_id: Any) -> str:
    return str(setup_id or "").strip().lower().split(".", 1)[0]


def _load_registry(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def evaluate_oos_setup_eligibility(
    setup_id: Any,
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> OosSetupEligibilityDecision:
    sid = str(setup_id or "").strip()
    family = _family(sid)
    if family not in SCOPED_FAMILIES:
        return OosSetupEligibilityDecision(
            True,
            "oos_gate_not_scoped",
            sid,
            family,
            audit={"registry_path": str(registry_path)},
        )

    registry = _load_registry(registry_path)
    families = registry.get("families") if isinstance(registry.get("families"), dict) else {}
    record = families.get(family) if isinstance(families.get(family), dict) else {}

    promoted = record.get("promoted") is True
    audit = {
        "registry_path": str(registry_path),
        "schema": registry.get("schema"),
        "evidence_date": registry.get("evidence_date"),
        "family_record": dict(record),
        "fail_closed": True,
    }
    if promoted and family == "breadth_donchian_10_v5":
        # A profitable historical replay is only a research gate. Require an
        # independently completed prospective ledger, exact-repo verification
        # and an explicitly reviewed execution-cost safeguard before authority.
        prerequisites = {
            "prospective_shadow_passed": record.get("prospective_shadow_passed") is True,
            "exact_repo_selfhosted_passed": record.get("exact_repo_selfhosted_status") == "passed",
            "live_cost_guard_integrated": record.get("live_cost_guard_integrated") is True,
            "execution_authority": record.get("execution_authority") is True,
        }
        missing = tuple(key for key, passed in prerequisites.items() if not passed)
        audit["v5_execution_prerequisites"] = prerequisites
        if missing:
            return OosSetupEligibilityDecision(
                False,
                "v5_promotion_prerequisites_missing",
                sid,
                family,
                blockers=missing,
                audit=audit,
            )

    if promoted:
        return OosSetupEligibilityDecision(
            True,
            "oos_edge_promoted",
            sid,
            family,
            audit=audit,
        )

    blocker = str(record.get("blocker") or "oos_edge_not_promoted")
    return OosSetupEligibilityDecision(
        False,
        blocker,
        sid,
        family,
        blockers=(blocker,),
        audit=audit,
    )


__all__ = [
    "DEFAULT_REGISTRY_PATH",
    "OosSetupEligibilityDecision",
    "SCOPED_FAMILIES",
    "evaluate_oos_setup_eligibility",
]
