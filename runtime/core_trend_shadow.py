from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE_PATH = ROOT / "reports" / "core_trend_shadow_events.jsonl"

DEFAULT_UNIVERSE = {
    "BTC",
    "ETH",
    "SOL",
    "BNB",
    "XRP",
    "DOGE",
    "AAVE",
    "LINK",
    "LTC",
    "BCH",
    "ADA",
}


def _base_symbol(symbol: str) -> str:
    raw = str(symbol or "").upper()
    return raw.split("/", 1)[0].split("-", 1)[0]


def _find_4h_frame(item: dict[str, Any]) -> Any:
    multi = item.get("mtf_data")
    if not isinstance(multi, dict):
        return None
    for key, value in multi.items():
        if str(key).lower() == "4h":
            return value
    return None


def build_core_trend_shadow_decision(
    item: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    policy = (
        config.get("core_trend_shadow")
        if isinstance(config.get("core_trend_shadow"), dict)
        else {}
    )
    strategy_id = str(
        policy.get("strategy_id")
        or "core_trend_4h_ema50_200_voltarget.v1"
    )
    symbol = str(item.get("symbol") or "")
    base = _base_symbol(symbol)
    enabled = bool(policy.get("enabled", False))
    allowed_universe = {
        str(value).upper()
        for value in (policy.get("universe") or DEFAULT_UNIVERSE)
        if str(value).strip()
    }

    policy_payload = {
        "strategy_id": strategy_id,
        "universe": sorted(allowed_universe),
        "ema_fast": int(policy.get("ema_fast", 50) or 50),
        "ema_slow": int(policy.get("ema_slow", 200) or 200),
        "vol_window_bars": int(policy.get("vol_window_bars", 180) or 180),
        "min_closed_bars": int(policy.get("min_closed_bars", 220) or 220),
        "target_annualized_vol": float(
            policy.get("target_annualized_vol", 0.20) or 0.20
        ),
        "per_symbol_exposure_cap": float(
            policy.get("per_symbol_exposure_cap", 1.0) or 1.0
        ),
    }
    policy_fingerprint = hashlib.sha256(
        json.dumps(
            policy_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    base_payload: dict[str, Any] = {
        "strategy_id": strategy_id,
        "policy_fingerprint_sha256": policy_fingerprint,
        "symbol": symbol,
        "timeframe": "4h",
        "shadow_only": True,
        "order_authorized": False,
        "action": "observe",
        "desired_direction": "neutral",
        "target_exposure": 0.0,
        "reason": "",
    }

    if not enabled:
        return {**base_payload, "status": "disabled", "reason": "shadow_lane_disabled"}
    if base not in allowed_universe:
        return {
            **base_payload,
            "status": "not_eligible",
            "reason": "symbol_outside_shadow_universe",
        }

    frame = _find_4h_frame(item)
    if frame is None or getattr(frame, "empty", True):
        return {
            **base_payload,
            "status": "insufficient_data",
            "reason": "missing_4h_frame",
        }

    close_col = "close"
    if close_col not in getattr(frame, "columns", []):
        return {
            **base_payload,
            "status": "insufficient_data",
            "reason": "missing_4h_close",
        }

    try:
        close = frame[close_col].astype(float)
    except Exception:
        return {
            **base_payload,
            "status": "insufficient_data",
            "reason": "invalid_4h_close",
        }

    # Never treat the newest 4h row as closed. The research contract is
    # closed 4h candle -> next 4h interval exposure.
    closed = close.iloc[:-1]
    min_closed_bars = int(policy_payload["min_closed_bars"])
    if len(closed) < min_closed_bars:
        return {
            **base_payload,
            "status": "insufficient_data",
            "reason": "not_enough_closed_4h_bars",
            "closed_bars": int(len(closed)),
        }

    ema_fast = int(policy_payload["ema_fast"])
    ema_slow = int(policy_payload["ema_slow"])
    vol_window = int(policy_payload["vol_window_bars"])
    target_vol = float(policy_payload["target_annualized_vol"])
    exposure_cap = float(policy_payload["per_symbol_exposure_cap"])

    fast = closed.ewm(span=ema_fast, adjust=False, min_periods=ema_fast).mean()
    slow = closed.ewm(span=ema_slow, adjust=False, min_periods=ema_slow).mean()
    returns = closed.pct_change()
    ann_vol = (
        returns.rolling(vol_window, min_periods=max(90, vol_window // 2))
        .std(ddof=0)
        * math.sqrt(6 * 365)
    )

    try:
        fast_last = float(fast.iloc[-1])
        slow_last = float(slow.iloc[-1])
        vol_last = float(ann_vol.iloc[-1])
        close_last = float(closed.iloc[-1])
    except (TypeError, ValueError, IndexError):
        return {
            **base_payload,
            "status": "insufficient_data",
            "reason": "indicator_not_ready",
        }

    if not all(math.isfinite(value) for value in (fast_last, slow_last, vol_last, close_last)):
        return {
            **base_payload,
            "status": "insufficient_data",
            "reason": "indicator_not_ready",
        }
    if vol_last <= 0:
        return {
            **base_payload,
            "status": "insufficient_data",
            "reason": "invalid_realized_volatility",
        }

    direction = "long" if fast_last > slow_last else "short" if fast_last < slow_last else "neutral"
    scale = min(exposure_cap, max(0.0, target_vol / vol_last))
    signed_exposure = scale if direction == "long" else -scale if direction == "short" else 0.0

    closed_timestamp = None
    if "timestamp" in getattr(frame, "columns", []):
        try:
            closed_timestamp = str(frame["timestamp"].iloc[-2])
        except Exception:
            closed_timestamp = None

    entry_reference_price = None
    if "open" in getattr(frame, "columns", []):
        try:
            candidate_open = float(frame["open"].iloc[-1])
            if math.isfinite(candidate_open) and candidate_open > 0:
                entry_reference_price = candidate_open
        except (TypeError, ValueError, IndexError):
            entry_reference_price = None

    return {
        **base_payload,
        "status": "shadow_signal",
        "desired_direction": direction,
        "target_exposure": round(float(signed_exposure), 8),
        "reason": "closed_4h_ema_trend_with_vol_target",
        "closed_bar_timestamp": closed_timestamp,
        "closed_bars": int(len(closed)),
        "close": close_last,
        "entry_reference_price": entry_reference_price,
        "ema_fast": fast_last,
        "ema_slow": slow_last,
        "realized_vol_annualized": vol_last,
        "target_annualized_vol": target_vol,
        "research_contract": {
            "retrospective_holdout_reused_for_promotion": False,
            "forward_shadow_required": True,
            "live_release": False,
        },
    }


def _event_identity(payload: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(payload.get("strategy_id") or ""),
        str(payload.get("policy_fingerprint_sha256") or ""),
        str(payload.get("symbol") or ""),
        str(payload.get("closed_bar_timestamp") or ""),
    )


def append_core_trend_shadow_event(
    decision: dict[str, Any],
    *,
    destination: Path = DEFAULT_EVIDENCE_PATH,
) -> bool:
    """Append one unique closed-bar shadow observation.

    Runtime loops may execute multiple times inside the same 4h candle. Evidence
    counts must therefore be keyed by strategy + policy fingerprint + symbol + last closed 4h candle,
    otherwise repeated observations would inflate sample size and confidence.
    """
    identity = _event_identity(decision)
    if not all(identity):
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            for line in reversed(destination.read_text(encoding="utf-8").splitlines()[-256:]):
                if not line.strip():
                    continue
                try:
                    previous = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(previous, dict) and _event_identity(previous) == identity:
                    return False
        except OSError:
            raise

    payload = {
        "recorded_at": datetime.now(UTC).isoformat(),
        **dict(decision),
    }
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    return True


__all__ = [
    "append_core_trend_shadow_event",
    "build_core_trend_shadow_decision",
]
