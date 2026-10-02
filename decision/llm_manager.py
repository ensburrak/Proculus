from __future__ import annotations

TRADING_SYSTEM_PROMPT = (
    "You are an advisory risk-analysis layer. Never choose or flip trade "
    "direction. Deterministic strategy experts own direction."
)


def is_llm_disable_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(token in text for token in ("api key", "unauthorized", "forbidden", "quota", "disabled"))
