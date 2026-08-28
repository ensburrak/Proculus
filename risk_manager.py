# -*- coding: utf-8 -*-
"""Backward-compatible facade for legacy ``risk_manager`` imports."""

from __future__ import annotations

from risk.portfolio_guard import (
    PortfolioRiskDecision,
    _load_portfolio_risk_limit_pct,
    check_portfolio_level_risk,
    evaluate_portfolio_risk,
)
from risk.position_sizer import (
    _is_micro_account,
    apply_dynamic_wallet_cap,
    apply_micro_account_logic,
    calculate_kelly_fraction,
    calculate_position_size,
    check_trend_alignment,
    get_entry_levels,
    get_exit_levels,
)
from risk.stop_manager import (
    _clamp_to_tick,
    adjust_tp_for_costs,
    calculate_adaptive_stop_loss,
    calculate_adaptive_take_profit,
    calculate_dynamic_tp_sl,
    calculate_volatility_percentile,
    compute_stop_loss,
    manage_trailing_stop,
)
from risk.tiering import (
    CONFIG_PATH,
    MAX_PORTFOLIO_RISK_PERCENTAGE,
    RISK_TIERS,
    WALLET_CAP_TIERS,
    _get_risk_scale,
    calculate_tiered_leverage_and_allocation,
)
from risk.volatility import VOLATILITY_TIERS, adjust_risk_for_volatility, classify_volatility

__all__ = [
    "CONFIG_PATH",
    "MAX_PORTFOLIO_RISK_PERCENTAGE",
    "PortfolioRiskDecision",
    "RISK_TIERS",
    "VOLATILITY_TIERS",
    "WALLET_CAP_TIERS",
    "_clamp_to_tick",
    "_get_risk_scale",
    "_is_micro_account",
    "_load_portfolio_risk_limit_pct",
    "adjust_risk_for_volatility",
    "adjust_tp_for_costs",
    "apply_dynamic_wallet_cap",
    "apply_micro_account_logic",
    "calculate_adaptive_stop_loss",
    "calculate_adaptive_take_profit",
    "calculate_dynamic_tp_sl",
    "calculate_kelly_fraction",
    "calculate_position_size",
    "calculate_tiered_leverage_and_allocation",
    "calculate_volatility_percentile",
    "check_portfolio_level_risk",
    "check_trend_alignment",
    "classify_volatility",
    "compute_stop_loss",
    "evaluate_portfolio_risk",
    "get_entry_levels",
    "get_exit_levels",
    "manage_trailing_stop",
]
