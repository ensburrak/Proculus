from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research.profitability_evidence import evaluate_release_gate

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = ROOT / "reports" / "profitability_release_latest.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def evaluate_strategy_release(
    *,
    item: dict[str, Any],
    decision: dict[str, Any],
    config: dict[str, Any],
    evidence_path: Path | None = None,
) -> dict[str, Any]:
    mode = str(item.get("runtime_mode") or item.get("mode") or "paper").lower()
    setup_id = str(decision.get("setup_id") or "")
    audit: dict[str, Any] = {
        "runtime_mode": mode,
        "setup_id": setup_id,
        "fail_closed": True,
    }

    # Research modes never send real exchange orders and don't need a live
    # release attestation to collect learning evidence.
    if mode in {"paper", "sim", "demo", "dry", "dry-run", "shadow", "testnet", "sandbox"}:
        return {**audit, "allowed": True, "reason": "non_live_research_mode"}

    if mode not in {"live", "production", "real"}:
        return {**audit, "allowed": False, "reason": "runtime_mode_not_released"}

    pcfg = config.get("profitability_evidence") if isinstance(config.get("profitability_evidence"), dict) else {}
    path = evidence_path or DEFAULT_EVIDENCE
    payload = _load_json(path)
    if not payload:
        return {
            **audit,
            "allowed": False,
            "reason": "profitability_release_evidence_missing",
            "evidence_path": str(path),
        }

    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else payload
    gate = evaluate_release_gate(evidence, pcfg)
    allowed_setups = payload.get("allowed_setups")
    setup_allowed = True
    if isinstance(allowed_setups, list):
        setup_allowed = setup_id in {str(value) for value in allowed_setups}

    allowed = bool(gate.get("live_allowed")) and setup_allowed
    reason = "profitability_release_passed" if allowed else (
        "setup_not_released" if not setup_allowed else "profitability_release_blocked"
    )
    return {
        **audit,
        "allowed": allowed,
        "reason": reason,
        "gate": gate,
        "evidence_path": str(path),
        "setup_allowed": setup_allowed,
    }


__all__ = ["evaluate_strategy_release"]
