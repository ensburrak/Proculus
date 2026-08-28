# -*- coding: utf-8 -*-
"""
chatgpt_decision_layer.py
-------------------------

ChatGPT Decision Layer (3. katman)
- 2. katmandan gecen sinyali alir (filter result)
- Nihai karari uretir: enter/skip/modify + TP/SL/LEV

Bu modul, ChatGPT API'sini kullanarak trading sinyallerini valuelendirir
ve nihai karar verir. Iki asamali bir durationc izler:

1. Filter asamasi (gpt-5-mini): Hizli ve ucuz on filtre
2. Decision asamasi (gpt-5-mini): Detayli analiz ve karar

CHANGELOG:
- v1.0: Initial version
- v1.1: Fixed import to work both as module and standalone
- v1.2: Added fallback when ChatGPT is disabled
- v1.3: Added timeout and retry logic
This module remains a verification helper for ad-hoc or legacy workflows.
It is not the official runtime trade-decision pipeline. The production
decision authority is controller_async.decide_batch via
decision.official_pipeline.process_symbol_decision.
"""

from __future__ import annotations


from core.exceptions import BEST_EFFORT_EXCEPTIONS
import os
from typing import Dict, Any, Optional

# Flexible import: works both as package module and standalone script
try:
    from .chatgpt_client import ChatGPTClient
except ImportError:
    try:
        from chatgpt_client import ChatGPTClient
    except ImportError:
        # Fallback: create a dummy client if chatgpt_client not available
        ChatGPTClient = None  # type: ignore


class DecisionLayerError(Exception):
    """Custom exception for decision layer errors."""
    pass


def _advisory_only_result(decision: Dict[str, Any], *, reason_prefix: str) -> Dict[str, Any]:
    """Return a fail-closed advisory payload for this legacy helper."""
    payload = dict(decision or {})
    original_reason = str(payload.get("reason") or "").strip()
    payload["advisory_action"] = payload.get("action")
    payload["advisory_direction"] = payload.get("direction")
    payload["advisory_tp"] = payload.get("tp")
    payload["advisory_sl"] = payload.get("sl")
    payload["advisory_lev"] = payload.get("lev")
    payload["action"] = "skip"
    payload["tp"] = None
    payload["sl"] = None
    payload["lev"] = None
    payload["llm_role"] = "advisory_only"
    payload["decision_authority"] = "suppressed_legacy_advisory_only"
    payload["reason"] = (
        f"{reason_prefix}: executable fields suppressed; {original_reason}"
        if original_reason
        else f"{reason_prefix}: executable fields suppressed"
    )
    return payload


def _get_client() -> Optional[Any]:
    """
    Get or create ChatGPT client instance.
    Returns None if ChatGPT is disabled or unavailable.
    """
    # Check if ChatGPT is disabled via environment
    if os.getenv("CHATGPT_DISABLE", "").lower() in ("1", "true", "yes"):
        return None
    
    if ChatGPTClient is None:
        return None
    
    try:
        return ChatGPTClient()
    except BEST_EFFORT_EXCEPTIONS:
        return None


# Lazy initialization of client
_client: Optional[Any] = None
_client_initialized: bool = False


def get_client() -> Optional[Any]:
    """Get the singleton ChatGPT client."""
    global _client, _client_initialized
    if not _client_initialized:
        _client = _get_client()
        _client_initialized = True
    return _client


def _apply_local_rules(
    snapshot: Dict[str, Any],
    base_signal: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Z-08: Apply simple deterministic rules BEFORE calling LLM.

    For clear-cut signals (extreme RSI, etc.) there is no need to spend
    ~0.5s and ~$0.002 on an LLM call.  Only ambiguous situations are
    forwarded to the LLM.

    Returns a decision dict if a local rule matched, or None if the
    LLM should be consulted.
    """
    rsi = snapshot.get("rsi") or snapshot.get("rsi_14")
    adx = snapshot.get("adx")
    confidence = base_signal.get("confidence", 0.5)

    try:
        rsi_val = float(rsi) if rsi is not None else None
    except (TypeError, ValueError):
        rsi_val = None

    try:
        adx_val = float(adx) if adx is not None else None
    except (TypeError, ValueError):
        adx_val = None

    # Rule 1: Extremely oversold + strong trend → definitive long signal
    if rsi_val is not None and rsi_val < 20 and adx_val is not None and adx_val > 25:
        return {
            "action": "enter",
            "direction": "long",
            "tp": base_signal.get("tp"),
            "sl": base_signal.get("sl"),
            "lev": base_signal.get("lev") or base_signal.get("leverage"),
            "reason": f"Local rule: RSI={rsi_val:.1f} extremely oversold with ADX={adx_val:.1f} (strong trend)",
            "confidence": min(confidence * 1.1, 0.85),
            "filter_passed": True,
            "local_rule": True,
        }

    # Rule 2: Extremely overbought + strong trend → definitive short signal
    if rsi_val is not None and rsi_val > 80 and adx_val is not None and adx_val > 25:
        return {
            "action": "enter",
            "direction": "short",
            "tp": base_signal.get("tp"),
            "sl": base_signal.get("sl"),
            "lev": base_signal.get("lev") or base_signal.get("leverage"),
            "reason": f"Local rule: RSI={rsi_val:.1f} extremely overbought with ADX={adx_val:.1f} (strong trend)",
            "confidence": min(confidence * 1.1, 0.85),
            "filter_passed": True,
            "local_rule": True,
        }

    # Rule 3: Very low confidence → skip without LLM call
    if confidence < 0.30:
        return {
            "action": "skip",
            "tp": None,
            "sl": None,
            "lev": None,
            "reason": f"Local rule: confidence={confidence:.2f} too low, skipping LLM call",
            "confidence": confidence,
            "filter_passed": False,
            "local_rule": True,
        }

    # No definitive local rule matched → forward to LLM
    return None


def verify_and_revise(
    snapshot: Dict[str, Any],
    base_signal: Dict[str, Any],
    timeout: float = 30.0
) -> Dict[str, Any]:
    """
    High-level pipeline for signal verification and revision.

    1) Z-08: Check local deterministic rules first (0 cost, 0 latency)
    2) Filter (4o-mini) -> allow?
    3) If allowed -> final decision (4o)

    Args:
        snapshot: Current market snapshot containing price, indicators, etc.
        base_signal: Base trading signal from technical/AI analysis.
        timeout: Maximum time to wait for API response.

    Returns:
        Decision dictionary with keys:
        - action: "enter", "skip", or "modify"
        - tp: Take profit price (or None)
        - sl: Stop loss price (or None)
        - lev: Recommended leverage (or None)
        - reason: Explanation for the decision
        - confidence: Confidence score (0-1)
    """
    # Z-08: Try local rules first — skip LLM for obvious signals
    local_decision = _apply_local_rules(snapshot, base_signal)
    if local_decision is not None:
        if str(local_decision.get("action") or "").strip().lower() in {"enter", "modify", "long", "short"}:
            return _advisory_only_result(local_decision, reason_prefix="Legacy decision layer is advisory-only")
        return local_decision

    client = get_client()

    # Fail closed when the LLM verifier is unavailable.
    if client is None:
        return {
            "action": "skip",
            "tp": None,
            "sl": None,
            "lev": None,
            "reason": "Decision layer unavailable: ChatGPT client is not available",
            "confidence": 0.0,
            "filter_passed": False,
            "error": "client_unavailable",
        }
    
    try:
        # Step 1: Filter with fast model
        filtered = client.filter_signal(snapshot, base_signal)
        
        if not filtered.get("allow"):
            return {
                "action": "skip",
                "tp": None,
                "sl": None,
                "lev": None,
                "reason": f"Filtered: {filtered.get('risk_flag', 'unknown')} - {filtered.get('reason', 'No reason provided')}",
                "confidence": 0.0,
                "filter_passed": False
            }
        
        # Step 2: Final decision with heavy model
        final_decision = client.decide_trade(snapshot, filtered)
        
        # Sanity check on action
        valid_actions = ("enter", "modify", "skip", "hold")
        if final_decision.get("action") not in valid_actions:
            final_decision["action"] = "skip"
            final_decision["reason"] = f"Invalid action normalized to skip: {final_decision.get('action')}"
        
        # Ensure all required fields exist
        final_decision.setdefault("tp", None)
        final_decision.setdefault("sl", None)
        final_decision.setdefault("lev", None)
        final_decision.setdefault("reason", "")
        final_decision.setdefault("confidence", 0.5)
        final_decision["filter_passed"] = True
        
        return _advisory_only_result(final_decision, reason_prefix="Legacy ChatGPT decision layer is advisory-only")
        
    except BEST_EFFORT_EXCEPTIONS as e:
        # On any error, return skip with error info
        return {
            "action": "skip",
            "tp": None,
            "sl": None,
            "lev": None,
            "reason": f"Decision layer error: {str(e)}",
            "confidence": 0.0,
            "filter_passed": False,
            "error": str(e)
        }


def quick_filter(
    snapshot: Dict[str, Any], 
    base_signal: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Quick filter using only the fast model (4o-mini).
    Useful for pre-screening many signals quickly.
    
    Args:
        snapshot: Current market snapshot.
        base_signal: Base trading signal.
        
    Returns:
        Filter result with 'allow' boolean and 'reason'.
    """
    client = get_client()
    
    if client is None:
        # If client not available, allow by default but flag it
        return {
            "allow": True,
            "reason": "ChatGPT unavailable - allowing by default",
            "risk_flag": None,
            "fallback": True
        }
    
    try:
        return client.filter_signal(snapshot, base_signal)
    except BEST_EFFORT_EXCEPTIONS as e:
        return {
            "allow": False,
            "reason": f"Filter error: {str(e)}",
            "risk_flag": "error",
            "error": str(e)
        }


def reset_client() -> None:
    """
    Reset the client instance.
    Useful for testing or when API credentials change.
    """
    global _client, _client_initialized
    _client = None
    _client_initialized = False


# For backwards compatibility
def get_decision(snapshot: Dict[str, Any], base_signal: Dict[str, Any]) -> Dict[str, Any]:
    """Alias for verify_and_revise for backwards compatibility."""
    return verify_and_revise(snapshot, base_signal)
