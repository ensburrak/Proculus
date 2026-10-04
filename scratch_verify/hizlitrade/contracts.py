from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from decimal import Decimal

from .models import BinaryMarketSpec


def _normalized_decimal(value: Decimal) -> str:
    normalized = value.normalize()
    if normalized == normalized.to_integral():
        return str(normalized.quantize(Decimal(1)))
    return format(normalized, "f")


def resolution_fingerprint(spec: BinaryMarketSpec) -> str | None:
    """Return a deterministic hash for economically relevant resolution semantics.

    A fingerprint is deliberately unavailable when the contract does not carry
    enough verified information to prove binary-resolution equivalence. The
    hash is independent of venue and market IDs so two venues can match only
    when their normalized economics match.
    """
    oracle = spec.oracle.strip().lower()
    if not oracle or oracle == "unknown":
        return None
    if spec.settlement_rule == "unknown" or spec.expires_at_ms <= 0:
        return None
    if spec.strike_capture_tolerance_ms <= 0 or spec.settlement_capture_tolerance_ms <= 0:
        return None

    if spec.strike_mode == "fixed":
        if spec.strike <= 0:
            return None
        strike: str | None = _normalized_decimal(spec.strike)
    elif spec.strike_mode == "opening_oracle":
        if spec.starts_at_ms is None or spec.oracle_window_seconds is None:
            return None
        strike = "opening_oracle"
    else:
        return None

    payload = {
        "symbol": spec.symbol.strip().upper(),
        "starts_at_ms": spec.starts_at_ms,
        "expires_at_ms": spec.expires_at_ms,
        "oracle": oracle,
        "oracle_window_seconds": spec.oracle_window_seconds,
        "strike_mode": spec.strike_mode,
        "strike": strike,
        "strike_capture_tolerance_ms": spec.strike_capture_tolerance_ms,
        "settlement_rule": spec.settlement_rule,
        "settlement_capture_tolerance_ms": spec.settlement_capture_tolerance_ms,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_resolution_equivalence_groups(specs: tuple[BinaryMarketSpec, ...]) -> None:
    """Reject ambiguous/mismatched enabled specs that share one canonical ID.

    A single spec may represent both YES/NO legs of one venue and therefore does
    not need cross-venue equivalence proof here. The moment two enabled market
    specs share a canonical ID, all of them must have a complete and identical
    normalized resolution fingerprint.
    """
    groups: dict[str, list[BinaryMarketSpec]] = defaultdict(list)
    for spec in specs:
        if spec.enabled:
            groups[spec.canonical_id].append(spec)

    for canonical_id, group in groups.items():
        if len(group) < 2:
            continue
        fingerprints = [resolution_fingerprint(spec) for spec in group]
        if any(value is None for value in fingerprints):
            raise ValueError(
                f"canonical_id={canonical_id!r} has unverified resolution semantics"
            )
        if len(set(fingerprints)) != 1:
            identities = ", ".join(
                f"{spec.venue.value}:{spec.market_id}" for spec in group
            )
            raise ValueError(
                f"canonical_id={canonical_id!r} mixes non-equivalent contracts: {identities}"
            )
