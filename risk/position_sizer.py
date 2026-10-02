from __future__ import annotations

from typing import Any


def _is_micro_account(balance: Any) -> bool:
    try:
        return float(balance) < 250.0
    except (TypeError, ValueError):
        return False


def calculate_kelly_fraction(win_rate: float, reward_risk: float, cap: float = 0.25) -> float:
    if reward_risk <= 0:
        return 0.0
    q = 1.0 - float(win_rate)
    kelly = float(win_rate) - q / float(reward_risk)
    return max(0.0, min(float(cap), kelly))


def calculate_position_size(
    balance: float,
    risk_pct: float = 0.01,
    entry_price: float | None = None,
    stop_price: float | None = None,
    risk_scale: float = 1.0,
    **kwargs: Any,
) -> float:
    del kwargs
    balance = max(0.0, float(balance))
    risk_budget = balance * max(0.0, min(0.05, float(risk_pct))) * max(0.0, min(1.0, float(risk_scale)))
    if entry_price is None or stop_price is None:
        return risk_budget
    distance = abs(float(entry_price) - float(stop_price))
    if distance <= 0:
        return 0.0
    return risk_budget / distance


def apply_dynamic_wallet_cap(size: float, balance: float, price: float, max_wallet_pct: float = 0.20, **kwargs: Any) -> float:
    del kwargs
    if price <= 0:
        return 0.0
    cap_units = max(0.0, float(balance)) * max(0.0, min(1.0, float(max_wallet_pct))) / float(price)
    return max(0.0, min(float(size), cap_units))


def apply_micro_account_logic(size: float, balance: float, **kwargs: Any) -> float:
    del kwargs
    return float(size) * (0.5 if _is_micro_account(balance) else 1.0)


def check_trend_alignment(side: str, ema_fast: Any, ema_slow: Any, **kwargs: Any) -> bool:
    del kwargs
    try:
        fast, slow = float(ema_fast), float(ema_slow)
    except (TypeError, ValueError):
        return False
    return fast > slow if str(side).lower() in {"long", "buy"} else fast < slow


def get_entry_levels(entry_price: float, **kwargs: Any) -> list[float]:
    del kwargs
    return [float(entry_price)]


def get_exit_levels(entry_price: float, side: str = "long", reward_risk: float = 2.0, stop_distance: float | None = None, **kwargs: Any) -> dict[str, float]:
    del kwargs
    dist = abs(float(stop_distance or float(entry_price) * 0.01))
    if str(side).lower() in {"long", "buy"}:
        return {"stop_loss": float(entry_price) - dist, "take_profit": float(entry_price) + dist * float(reward_risk)}
    return {"stop_loss": float(entry_price) + dist, "take_profit": float(entry_price) - dist * float(reward_risk)}
