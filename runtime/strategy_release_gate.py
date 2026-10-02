from __future__ import annotations

import hashlib
import json
from typing import Any


PAPER_MODES = {"paper", "sim", "dry", "dry-run"}


def _canonical_allowed_hash(values: list[str]) -> str:
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def evaluate_strategy_release(
    *,
    item: dict[str, Any],
    decision: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate release authorization without creating exchange side effects.

    Paper/sim modes are research modes and are never blocked by the release
    allow-list. Any side-effect capable mode is fail-closed.
    """
    mode = str(item.get("runtime_mode") or item.get("mode") or "paper").lower()
    setup_id = str(decision.get("setup_id") or "")
    release = config.get("strategy_release") if isinstance(config.get("strategy_release"), dict) else {}

    allowed_raw = release.get("allowed_setup_ids")
    allowed_setup_ids = [str(v) for v in allowed_raw] if isinstance(allowed_raw, list) else []
    configured_hash = str(release.get("allowed_setup_ids_hash") or "")
    calculated_hash = _canonical_allowed_hash(allowed_setup_ids)
    hash_valid = bool(configured_hash) and configured_hash == calculated_hash

    audit: dict[str, Any] = {
        "mode": mode,
        "setup_id": setup_id,
        "requested_release": str(release.get("requested_release") or ""),
        "allow_live": bool(release.get("allow_live", False)),
        "allow_testnet": bool(release.get("allow_testnet", False)),
        "allowed_runtime_modes": [str(v).lower() for v in (release.get("allowed_runtime_modes") or [])],
        "allowed_setup_ids_hash_valid": hash_valid,
        "edge_validated": bool((decision.get("edge_contract") or {}).get("validated") is True),
        "evidence_hash_present": bool(str(release.get("required_strategy_oos_evidence_hash") or "")),
        "policy_hash_present": bool(str(release.get("required_strategy_policy_hash") or "")),
    }

    if mode in PAPER_MODES:
        return {**audit, "allowed": True, "reason": "non_side_effect_research_mode"}

    allowed_modes = set(audit["allowed_runtime_modes"])
    if mode not in allowed_modes:
        return {**audit, "allowed": False, "reason": "runtime_mode_not_released"}

    if mode in {"testnet", "demo", "sandbox"}:
        if not audit["allow_testnet"]:
            return {**audit, "allowed": False, "reason": "testnet_release_disabled"}
    elif not audit["allow_live"]:
        return {**audit, "allowed": False, "reason": "live_release_disabled"}

    if not hash_valid:
        return {**audit, "allowed": False, "reason": "allowed_setup_ids_hash_mismatch"}
    if not setup_id or setup_id not in set(allowed_setup_ids):
        return {**audit, "allowed": False, "reason": "setup_not_in_release_allowlist"}
    if not audit["evidence_hash_present"]:
        return {**audit, "allowed": False, "reason": "oos_evidence_hash_missing"}
    if not audit["policy_hash_present"]:
        return {**audit, "allowed": False, "reason": "strategy_policy_hash_missing"}
    if not audit["edge_validated"]:
        return {**audit, "allowed": False, "reason": "validated_edge_missing"}

    return {**audit, "allowed": True, "reason": "released"}


__all__ = ["evaluate_strategy_release"]
