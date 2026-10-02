from __future__ import annotations

from typing import Any


def _clamp_to_tick(price: float, tick_size: float | None = None) -> float:
    if not tick_size or tick_size <= 0:
        return float(price)
    return round(float(price) / float(tick_size)) * float(tick_size)


def compute_stop_loss(
    side: str,
    entry_price: float,
    atr: float | None = None,
    atr_mult: float = 1.5,
    tick_size: float | None = None,
    percent_fallback: float = 0.01,
    include_fees: bool = False,
    taker_fee: float = 0.0,
    **kwargs: Any,
) -> float:
    del kwargs
    distance = abs(float(atr)) * float(atr_mult) if atr is not None else abs(float(entry_price)) * float(percent_fallback)
    if include_fees:
        distance += abs(float(entry_price)) * max(0.0, float(taker_fee))
    price = float(entry_price) - distance if str(side).lower() in {"long", "buy"} else float(entry_price) + distance
    return _clamp_to_tick(price, tick_size)


def calculate_adaptive_stop_loss(*args: Any, **kwargs: Any) -> float:
    return compute_stop_loss(*args, **kwargs)


def calculate_adaptive_take_profit(side: str, entry_price: float, atr: float | None = None, atr_mult: float = 2.5, **kwargs: Any) -> float:
    del kwargs
    distance = abs(float(atr or float(entry_price) * 0.01)) * float(atr_mult)
    return float(entry_price) + distance if str(side).lower() in {"long", "buy"} else float(entry_price) - distance


def calculate_dynamic_tp_sl(side: str, entry_price: float, atr: float | None = None, **kwargs: Any) -> dict[str, float]:
    return {
        "stop_loss": compute_stop_loss(side, entry_price, atr=atr, **kwargs),
        "take_profit": calculate_adaptive_take_profit(side, entry_price, atr=atr, **kwargs),
    }


def calculate_volatility_percentile(*args: Any, **kwargs: Any) -> float:
    del args, kwargs
    return 0.5


def adjust_tp_for_costs(tp: float, entry_price: float, side: str, fee_rate: float = 0.0005, **kwargs: Any) -> float:
    del kwargs
    cost = abs(float(entry_price)) * max(0.0, float(fee_rate))
    return float(tp) + cost if str(side).lower() in {"long", "buy"} else float(tp) - cost


def manage_trailing_stop(current_stop: float, price: float, side: str, trail_pct: float = 0.01, **kwargs: Any) -> float:
    del kwargs
    candidate = float(price) * (1.0 - trail_pct) if str(side).lower() in {"long", "buy"} else float(price) * (1.0 + trail_pct)
    return max(float(current_stop), candidate) if str(side).lower() in {"long", "buy"} else min(float(current_stop), candidate)
