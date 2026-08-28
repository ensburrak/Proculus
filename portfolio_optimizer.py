# -*- coding: utf-8 -*-
from __future__ import annotations

"""
portfolio_optimizer.py
======================
Portfolio-level sizing and multi-objective optimisation helpers.
"""


from core.exceptions import BEST_EFFORT_EXCEPTIONS
import json
import logging
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from atomic_io import atomic_write_json, safe_read_json
from core.leverage_policy import clamp_entry_leverage
from data.parquet_storage import read_ohlcv_parquet

try:
    import optuna  # type: ignore
except ImportError:
    optuna = None

log = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent
CONFIG_FILE = ROOT_DIR / "config.json"
BACKTEST_CACHE_DIR = ROOT_DIR / "data" / "backtest" / "ohlcv"

PRESET_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "balanced": {"min_sharpe": 1.5, "max_drawdown_pct": 15.0, "min_win_rate_pct": 55.0},
    "aggressive": {"min_sharpe": 2.0, "max_drawdown_pct": 25.0, "min_win_rate_pct": 50.0},
    "conservative": {"min_sharpe": 1.0, "max_drawdown_pct": 8.0, "min_win_rate_pct": 60.0},
}

DEFAULT_AUTO_SELECTION_CFG: Dict[str, Any] = {
    "enabled": True,
    "min_total_trades": 12,
    "require_positive_total_pnl_pct": True,
    "weights": {
        "total_pnl_pct": 0.10,
        "sharpe_ratio": 1.20,
        "sortino_ratio": 0.45,
        "profit_factor": 0.35,
        "win_rate_pct": 0.015,
        "max_drawdown_pct": 0.12,
        "trade_count_bonus": 0.015,
        "trade_count_penalty": 0.18,
    },
}


@dataclass
class ParetoCandidate:
    params: Dict[str, Any]
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "params": dict(self.params),
            "sharpe_ratio": round(self.sharpe_ratio, 6),
            "max_drawdown_pct": round(self.max_drawdown_pct, 6),
            "win_rate_pct": round(self.win_rate_pct, 6),
            "score": round(self.score, 6),
            "metadata": dict(self.metadata),
        }


def _load_optimizer_cfg() -> Dict[str, Any]:
    if not CONFIG_FILE.exists():
        return {}
    try:
        payload = safe_read_json(CONFIG_FILE, default={})
    except (json.JSONDecodeError, OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    section = payload.get("portfolio_optimization")
    return section if isinstance(section, dict) else {}


def _load_auto_selection_cfg() -> Dict[str, Any]:
    multi_objective = _load_optimizer_cfg().get("multi_objective")
    if not isinstance(multi_objective, dict):
        multi_objective = {}
    raw = multi_objective.get("auto_selection")
    if not isinstance(raw, dict):
        raw = {}
    weights = raw.get("weights")
    if not isinstance(weights, dict):
        weights = {}
    default_weights = DEFAULT_AUTO_SELECTION_CFG["weights"]
    resolved_weights = {
        key: max(0.0, _safe_float(weights.get(key), default))
        for key, default in default_weights.items()
    }
    total_weight = sum(resolved_weights.values())
    if total_weight <= 0.0:
        resolved_weights = dict(default_weights)
        total_weight = sum(resolved_weights.values())
    normalized_weights = {key: value / total_weight for key, value in resolved_weights.items()}
    return {
        "enabled": bool(raw.get("enabled", DEFAULT_AUTO_SELECTION_CFG["enabled"])),
        "min_total_trades": max(0, _safe_int(raw.get("min_total_trades"), DEFAULT_AUTO_SELECTION_CFG["min_total_trades"])),
        "require_positive_total_pnl_pct": bool(
            raw.get(
                "require_positive_total_pnl_pct",
                DEFAULT_AUTO_SELECTION_CFG["require_positive_total_pnl_pct"],
            )
        ),
        "weights": normalized_weights,
    }


def _load_recent_trades(limit: int = 50) -> List[Dict[str, Any]]:
    try:
        from execution.trade_logger import get_recent_trades

        trades = list(get_recent_trades(limit=limit))
        trades.reverse()
        return trades
    except BEST_EFFORT_EXCEPTIONS as exc:
        log.debug("recent trade load failed: %s", exc)
        return []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _read_root_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    path = config_path or CONFIG_FILE
    if not path.exists():
        return {}
    try:
        payload = safe_read_json(path, default={})
    except (json.JSONDecodeError, OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_root_config(config: Dict[str, Any], config_path: Optional[Path] = None) -> None:
    path = config_path or CONFIG_FILE
    if not atomic_write_json(path, config):
        raise OSError(f"atomic config write failed: {path}")


def _normalize_symbol_key(value: str) -> str:
    cleaned = str(value or "").upper().replace(":USDT", "").replace("/", "_").replace(":", "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def _iter_backtest_cache_paths(timeframe: str = "15m") -> List[Path]:
    if not BACKTEST_CACHE_DIR.exists():
        return []
    config = _read_root_config()
    trade_params = config.get("trade_parameters", {}) if isinstance(config, dict) else {}
    preferred_symbols = []
    if isinstance(trade_params, dict):
        preferred_symbols.extend(trade_params.get("always_on_symbols", []) or [])
        preferred_symbols.extend(trade_params.get("symbols", []) or [])
    preferred_keys = {_normalize_symbol_key(sym) for sym in preferred_symbols if str(sym).strip()}
    all_paths = list(BACKTEST_CACHE_DIR.glob(f"*_{timeframe}.parquet"))

    def _priority(path: Path) -> Tuple[int, str]:
        stem = path.stem
        symbol_key = _normalize_symbol_key(stem.rsplit(f"_{timeframe}", 1)[0])
        preferred_rank = 0 if symbol_key in preferred_keys else 1
        return preferred_rank, stem

    all_paths.sort(key=_priority)
    return all_paths


def _load_backtest_frame(path: Path) -> Optional[Any]:
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        return None

    frame = None
    try:
        frame = read_ohlcv_parquet(path)
    except BEST_EFFORT_EXCEPTIONS:
        try:
            frame = pd.read_csv(path)
        except BEST_EFFORT_EXCEPTIONS:
            return None
    if frame is None or getattr(frame, "empty", True):
        return None

    columns = [str(col).lower() for col in frame.columns]
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    if not set(required).issubset(set(columns)):
        if len(frame.columns) >= 6:
            frame = frame.iloc[:, :6].copy()
            frame.columns = required
        else:
            return None
    else:
        frame = frame.rename(columns={col: str(col).lower() for col in frame.columns})

    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = frame[column].astype(float)
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    if frame.empty:
        return None
    if float(frame["close"].abs().sum()) <= 0.0:
        return None
    if "timestamp" in frame.columns:
        try:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        except BEST_EFFORT_EXCEPTIONS:
            pass
    return frame


def _frame_to_ohlcv(frame: Any, max_rows: int = 3000) -> List[List[float]]:
    rows: List[List[float]] = []
    if frame is None or getattr(frame, "empty", True):
        return rows
    trimmed = frame.tail(max_rows)
    for item in trimmed.itertuples(index=False):
        timestamp = getattr(item, "timestamp", None)
        if hasattr(timestamp, "timestamp"):
            ts_value = int(timestamp.timestamp() * 1000)
        else:
            ts_value = len(rows)
        rows.append([
            ts_value,
            float(getattr(item, "open")),
            float(getattr(item, "high")),
            float(getattr(item, "low")),
            float(getattr(item, "close")),
            float(getattr(item, "volume", 0.0)),
        ])
    return rows


def _build_strategy_fn(params: Dict[str, Any]) -> Callable[[List[List[float]], Dict[str, Any]], Dict[str, Any]]:
    ma_short = max(5, _safe_int(params.get("ma_short"), 20))
    ma_long = max(ma_short + 1, _safe_int(params.get("ma_long"), 50))
    cooldown_bars = max(0, _safe_int(params.get("cooldown_min"), 15))
    last_entry_index = -10**9

    def _strategy_fn(ohlcv_slice: List[List[float]], _indicators: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal last_entry_index
        if len(ohlcv_slice) < ma_long + 2:
            return {"action": "hold"}

        idx = len(ohlcv_slice) - 1
        if idx - last_entry_index < cooldown_bars:
            return {"action": "hold"}

        closes = [float(row[4]) for row in ohlcv_slice]
        short_prev = sum(closes[-ma_short - 1:-1]) / ma_short
        short_now = sum(closes[-ma_short:]) / ma_short
        long_prev = sum(closes[-ma_long - 1:-1]) / ma_long
        long_now = sum(closes[-ma_long:]) / ma_long

        if short_prev <= long_prev and short_now > long_now:
            last_entry_index = idx
            return {"action": "long", "symbol": "PARETO"}
        if short_prev >= long_prev and short_now < long_now:
            last_entry_index = idx
            return {"action": "short", "symbol": "PARETO"}
        return {"action": "hold"}

    return _strategy_fn


def _evaluate_backtest_candidate(params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        from backtesting.engine import BacktestConfig, BacktestEngine
    except BEST_EFFORT_EXCEPTIONS as exc:
        log.debug("backtest evaluator unavailable: %s", exc)
        return None

    stop_loss_mult = max(0.005, _safe_float(params.get("stop_loss_mult"), 0.02))
    sl_atr_mult = max(0.5, min(5.0, stop_loss_mult * 100.0))
    tp_atr_mult = max(sl_atr_mult + 0.5, sl_atr_mult * 1.8)
    strategy_fn = _build_strategy_fn(params)

    aggregate: List[Dict[str, Any]] = []
    for path in _iter_backtest_cache_paths(timeframe="15m")[:4]:
        frame = _load_backtest_frame(path)
        ohlcv = _frame_to_ohlcv(frame)
        if len(ohlcv) < 300:
            continue
        cfg = BacktestConfig(
            initial_balance=10_000.0,
            commission_pct=0.0004,
            slippage_pct=0.0005,
            max_positions=1,
            default_size_pct=0.10,
            leverage=3,
            sl_atr_mult=sl_atr_mult,
            tp_atr_mult=tp_atr_mult,
            max_hold_bars=max(32, min(192, _safe_int(params.get("cooldown_min"), 15) * 4)),
        )
        result = BacktestEngine(cfg).run(ohlcv, strategy_fn=strategy_fn).to_dict()
        trades = _safe_int(result.get("total_trades"), 0)
        if trades <= 0:
            continue
        aggregate.append(result)

    if not aggregate:
        return None

    weights = [max(1, _safe_int(item.get("total_trades"), 1)) for item in aggregate]
    total_weight = float(sum(weights))
    sharpe = sum(_safe_float(item.get("sharpe_ratio"), 0.0) * weight for item, weight in zip(aggregate, weights)) / total_weight
    drawdown = sum(_safe_float(item.get("max_drawdown_pct"), 0.0) * 100.0 * weight for item, weight in zip(aggregate, weights)) / total_weight
    win_rate = sum(_safe_float(item.get("win_rate"), 0.0) * 100.0 * weight for item, weight in zip(aggregate, weights)) / total_weight
    total_trades = sum(max(0, _safe_int(item.get("total_trades"), 0)) for item in aggregate)
    total_pnl_pct = sum(_safe_float(item.get("total_pnl_pct"), 0.0) * weight for item, weight in zip(aggregate, weights)) / total_weight
    profit_factor = sum(_safe_float(item.get("profit_factor"), 0.0) * weight for item, weight in zip(aggregate, weights)) / total_weight
    sortino_ratio = sum(_safe_float(item.get("sortino_ratio"), 0.0) * weight for item, weight in zip(aggregate, weights)) / total_weight
    return {
        "sharpe_ratio": round(sharpe, 6),
        "max_drawdown_pct": round(drawdown, 6),
        "win_rate_pct": round(win_rate, 6),
        "total_pnl_pct": round(total_pnl_pct, 6),
        "profit_factor": round(profit_factor, 6),
        "sortino_ratio": round(sortino_ratio, 6),
        "evaluation_mode": "local_backtest",
        "symbols_tested": len(aggregate),
        "total_trades": total_trades,
    }


def _evaluate_recent_trade_metrics(params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from execution.trade_logger import get_recent_trades
    except BEST_EFFORT_EXCEPTIONS:
        return {
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 100.0,
            "win_rate_pct": 0.0,
            "evaluation_mode": "empty_fallback",
        }

    trades = list(get_recent_trades(limit=250))
    if not trades:
        return {
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 100.0,
            "win_rate_pct": 0.0,
            "evaluation_mode": "empty_fallback",
        }

    returns: List[float] = []
    wins = 0
    drawdowns: List[float] = []
    for trade in trades:
        pnl_pct = _safe_float(trade.get("pnl_pct"), 0.0)
        returns.append(pnl_pct / 100.0)
        if _safe_float(trade.get("pnl_abs", 0.0), 0.0) > 0:
            wins += 1
        drawdowns.append(abs(_safe_float(trade.get("max_drawdown_pct"), 0.0)))

    mean_ret = sum(returns) / max(1, len(returns))
    variance = sum((item - mean_ret) ** 2 for item in returns) / max(1, len(returns) - 1)
    std_ret = math.sqrt(max(variance, 0.0))
    sharpe = (mean_ret / std_ret * math.sqrt(252)) if std_ret > 1e-10 else 0.0
    win_rate = wins / max(1, len(trades)) * 100.0
    avg_drawdown = sum(drawdowns) / max(1, len(drawdowns))

    ma_gap = abs(_safe_int(params.get("ma_long"), 50) - _safe_int(params.get("ma_short"), 20))
    structural_score = 1.0 - min(0.35, abs(ma_gap - 30.0) / 150.0)
    stop_score = 1.0 - min(0.25, abs(_safe_float(params.get("stop_loss_mult"), 0.02) - 0.02) / 0.08)
    cooldown_score = 1.0 - min(0.20, abs(_safe_float(params.get("cooldown_min"), 15.0) - 15.0) / 120.0)
    quality_multiplier = max(0.65, structural_score * stop_score * cooldown_score)

    return {
        "sharpe_ratio": round(sharpe * quality_multiplier, 6),
        "max_drawdown_pct": round(avg_drawdown / max(quality_multiplier, 0.65), 6),
        "win_rate_pct": round(win_rate * quality_multiplier, 6),
        "evaluation_mode": "recent_trade_history_fallback",
        "sample_size": len(trades),
    }


def get_stored_pareto_frontier(config_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    config = _read_root_config(config_path)
    portfolio = config.get("portfolio_optimization", {}) if isinstance(config, dict) else {}
    multi_objective = portfolio.get("multi_objective", {}) if isinstance(portfolio, dict) else {}
    frontier = multi_objective.get("pareto_frontier", []) if isinstance(multi_objective, dict) else []
    return list(frontier) if isinstance(frontier, list) else []


def apply_selected_candidate(
    *,
    candidate_index: Optional[int] = None,
    preset: Optional[str] = None,
    candidate: Optional[Dict[str, Any]] = None,
    config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    config = _read_root_config(config_path)
    portfolio = config.setdefault("portfolio_optimization", {})
    if not isinstance(portfolio, dict):
        portfolio = {}
        config["portfolio_optimization"] = portfolio
    multi_objective = portfolio.setdefault("multi_objective", {})
    if not isinstance(multi_objective, dict):
        multi_objective = {}
        portfolio["multi_objective"] = multi_objective

    frontier = multi_objective.get("pareto_frontier", [])
    if candidate is None:
        if isinstance(frontier, list) and frontier:
            if candidate_index is None:
                candidate_index = 0
            if candidate_index < 0 or candidate_index >= len(frontier):
                raise IndexError("candidate_index out of range")
            raw_candidate = frontier[candidate_index]
            candidate = dict(raw_candidate) if isinstance(raw_candidate, dict) else {}
        else:
            selected = multi_objective.get("selected_candidate")
            candidate = dict(selected) if isinstance(selected, dict) else {}
    if not candidate:
        raise ValueError("No Pareto candidate available to apply")

    params = candidate.get("params") if isinstance(candidate.get("params"), dict) else {}
    ma_short = max(5, _safe_int(params.get("ma_short"), _safe_int(config.get("ma_short"), 20)))
    ma_long = max(ma_short + 1, _safe_int(params.get("ma_long"), _safe_int(config.get("ma_long"), 50)))
    stop_loss_mult = max(0.001, _safe_float(params.get("stop_loss_mult"), _safe_float(config.get("stop_loss_mult"), 0.02)))
    cooldown = max(0, _safe_int(params.get("cooldown_min"), _safe_int(config.get("trade_cooldown_min"), 15)))

    config["ma_short"] = ma_short
    config["ma_long"] = ma_long
    config["stop_loss_mult"] = stop_loss_mult
    config["trade_cooldown_min"] = cooldown

    trade_parameters = config.setdefault("trade_parameters", {})
    if isinstance(trade_parameters, dict):
        trade_parameters["trade_cooldown_min"] = cooldown
    trading = config.setdefault("trading", {})
    if isinstance(trading, dict):
        trading["cooldown_minutes"] = cooldown

    multi_objective["enabled"] = True
    multi_objective["selected_candidate"] = candidate
    multi_objective["recommended_candidate"] = candidate
    if candidate_index is not None:
        multi_objective["selected_candidate_index"] = int(candidate_index)
    if preset:
        multi_objective["active_preset"] = str(preset).lower()
    multi_objective["selection_required"] = False

    _write_root_config(config, config_path)
    return candidate


def kelly_fraction(p: float, r: float = 1.0) -> float:
    """Classical Kelly fraction in the [0, 1] range."""
    try:
        p = float(p)
        r = float(r)
        if r <= 0:
            return 0.0
        f = (p * (r + 1.0) - 1.0) / r
        return max(0.0, min(1.0, f))
    except BEST_EFFORT_EXCEPTIONS:
        return 0.0


def _estimate_risk_reward(
    recent_trades: Optional[List[Dict[str, Any]]] = None,
    default: float = 1.0,
) -> float:
    if not recent_trades or len(recent_trades) < 5:
        return default
    wins = []
    losses = []
    for trade in recent_trades:
        pnl = _safe_float(trade.get("pnl_abs", trade.get("pnl", 0.0)), 0.0)
        if pnl > 0:
            wins.append(pnl)
        elif pnl < 0:
            losses.append(abs(pnl))
    if not wins or not losses:
        return default
    avg_win = sum(wins) / len(wins)
    avg_loss = sum(losses) / len(losses)
    if avg_loss <= 1e-10:
        return default
    return max(0.25, min(5.0, avg_win / avg_loss))


def _load_dynamic_kelly_cfg() -> Dict[str, Any]:
    top = _load_optimizer_cfg().get("dynamic_kelly")
    if not isinstance(top, dict):
        top = {}
    return {
        "enabled": bool(top.get("enabled", False)),
        "rolling_window_trades": int(top.get("rolling_window_trades", 50) or 50),
        "fractional_multiplier": float(top.get("fractional_multiplier", 0.25) or 0.25),
        "stop_trading_below": float(top.get("stop_trading_below", 0.01) or 0.01),
        "increase_smoothing": float(top.get("increase_smoothing", 0.35) or 0.35),
    }


def build_dynamic_kelly_profile(
    recent_trades: Optional[List[Dict[str, Any]]] = None,
    *,
    current_alloc: Optional[float] = None,
) -> Dict[str, Any]:
    cfg = _load_dynamic_kelly_cfg()
    window = max(10, int(cfg["rolling_window_trades"]))
    # Keep helper semantics deterministic for callers that do not explicitly
    # provide live trade history. Runtime code can still opt in by passing it.
    trades = list(recent_trades or [])[-window:]
    if not cfg["enabled"] or len(trades) < 5:
        return {
            "enabled": bool(cfg["enabled"]),
            "sample_size": len(trades),
            "raw_kelly": 0.0,
            "previous_raw_kelly": 0.0,
            "fractional_kelly": 0.0,
            "target_fraction": 0.0,
            "adjusted_fraction": current_alloc if current_alloc is not None else 0.0,
            "should_halt": False,
            "direction": "flat",
            "upside_applied": False,
        }

    win_rate = sum(1 for trade in trades if _safe_float(trade.get("pnl_abs", trade.get("pnl", 0.0)), 0.0) > 0.0) / len(trades)
    rr = _estimate_risk_reward(trades, 1.0)
    raw_kelly = kelly_fraction(win_rate, rr)
    fractional = raw_kelly * float(cfg["fractional_multiplier"])

    previous_window = trades[:-1]
    previous_raw = 0.0
    if len(previous_window) >= 5:
        prev_win_rate = sum(
            1 for trade in previous_window if _safe_float(trade.get("pnl_abs", trade.get("pnl", 0.0)), 0.0) > 0.0
        ) / len(previous_window)
        previous_raw = kelly_fraction(prev_win_rate, _estimate_risk_reward(previous_window, rr))

    if raw_kelly < float(cfg["stop_trading_below"]):
        adjusted = 0.0
        should_halt = True
    elif current_alloc is None:
        adjusted = fractional
        should_halt = False
    elif raw_kelly >= previous_raw:
        adjusted = current_alloc + (fractional - current_alloc) * float(cfg["increase_smoothing"])
        should_halt = False
    else:
        adjusted = min(current_alloc, fractional)
        should_halt = False

    direction = "flat"
    if raw_kelly > previous_raw + 1e-9:
        direction = "up"
    elif raw_kelly < previous_raw - 1e-9:
        direction = "down"

    return {
        "enabled": True,
        "sample_size": len(trades),
        "win_rate": round(win_rate, 6),
        "risk_reward": round(rr, 6),
        "raw_kelly": round(raw_kelly, 6),
        "previous_raw_kelly": round(previous_raw, 6),
        "fractional_kelly": round(fractional, 6),
        "target_fraction": round(fractional, 6),
        "adjusted_fraction": round(max(0.0, adjusted), 6),
        "should_halt": should_halt,
        "direction": direction,
        "upside_applied": bool(
            current_alloc is not None and direction == "up" and max(0.0, adjusted) > float(current_alloc)
        ),
    }


def _portfolio_correlation_penalty(
    symbol: str,
    open_positions: Optional[List[Dict[str, Any]]] = None,
    correlations: Optional[Dict[str, Dict[str, float]]] = None,
) -> float:
    if not open_positions or not correlations:
        return 1.0
    penalties = []
    for pos in open_positions:
        pos_sym = str(pos.get("symbol", ""))
        corr = _safe_float(correlations.get(symbol, {}).get(pos_sym, 0.0), 0.0)
        if abs(corr) > 0.7:
            penalties.append(1.0 - (abs(corr) - 0.7) / 0.3 * 0.5)
    if not penalties:
        return 1.0
    return max(0.5, min(1.0, sum(penalties) / len(penalties)))


def optimize_leverage_and_allocation(
    master_conf: float,
    current_leverage: int,
    current_alloc: float,
    risk_reward: float = 1.0,
    max_leverage: int = 1,
    min_leverage: int = 1,
    max_alloc: float = 0.40,
    min_alloc: float = 0.05,
    recent_trades: Optional[List[Dict[str, Any]]] = None,
    open_positions: Optional[List[Dict[str, Any]]] = None,
    correlations: Optional[Dict[str, Dict[str, float]]] = None,
    symbol: str = "",
) -> Tuple[int, float]:
    """Refine runtime leverage/allocation with rolling Kelly and portfolio penalties."""
    try:
        max_leverage = clamp_entry_leverage(max_leverage)
        min_leverage = min(int(min_leverage or 1), max_leverage)
        p = max(0.0, min(1.0, float(master_conf)))
        trades = list(recent_trades or [])
        rr = _estimate_risk_reward(trades, risk_reward)
        kelly_profile = build_dynamic_kelly_profile(trades, current_alloc=current_alloc)
        if kelly_profile.get("enabled") and kelly_profile.get("should_halt"):
            log.warning("[PORTFOLIO] Dynamic Kelly halted trading: raw_kelly=%.4f", kelly_profile["raw_kelly"])
            return 0, 0.0

        raw_fraction = kelly_fraction(p, rr)
        fractional = raw_fraction * 0.25
        if kelly_profile.get("enabled"):
            target_alloc = float(kelly_profile.get("adjusted_fraction", fractional) or fractional)
        else:
            target_alloc = fractional

        corr_mult = _portfolio_correlation_penalty(symbol, open_positions, correlations)
        raw_alloc = min_alloc + target_alloc * (max_alloc - min_alloc) * corr_mult
        final_alloc = raw_alloc if current_alloc <= 0 else (raw_alloc + current_alloc) / 2.0

        lev_scale = 0.75 + (p * 0.5)
        if kelly_profile.get("direction") == "down":
            lev_scale *= 0.8
        elif kelly_profile.get("direction") == "up":
            lev_scale *= 1.05
        new_lev = int(round(current_leverage * lev_scale)) if current_leverage > 0 else int(round(max_leverage * p))

        final_lev = max(0, min(max_leverage, max(min_leverage, new_lev))) if final_alloc > 0 else 0
        final_alloc = max(0.0, min(max_alloc, final_alloc))
        if final_alloc < min_alloc and final_alloc > 0:
            final_alloc = min_alloc
        return final_lev, final_alloc
    except BEST_EFFORT_EXCEPTIONS as exc:
        log.debug("optimize_leverage_and_allocation failed: %s", exc)
        return current_leverage, current_alloc


def _score_multi_objective_metrics(metrics: Dict[str, Any]) -> float:
    cfg = _load_auto_selection_cfg()
    weights = cfg["weights"]
    total_pnl_pct = _safe_float(metrics.get("total_pnl_pct"), 0.0)
    sharpe = _safe_float(metrics.get("sharpe_ratio"), 0.0)
    sortino = _safe_float(metrics.get("sortino_ratio"), sharpe)
    profit_factor = _safe_float(metrics.get("profit_factor"), 1.0)
    drawdown = abs(_safe_float(metrics.get("max_drawdown_pct"), 0.0))
    if drawdown <= 1.0:
        drawdown *= 100.0
    win_rate = _safe_float(metrics.get("win_rate_pct", metrics.get("win_rate", 0.0)), 0.0)
    if win_rate <= 1.0:
        win_rate *= 100.0
    total_trades = max(0, _safe_int(metrics.get("total_trades"), 0))

    score = 0.0
    score += max(-50.0, min(150.0, total_pnl_pct)) * weights["total_pnl_pct"]
    score += max(-2.0, min(6.0, sharpe)) * weights["sharpe_ratio"]
    score += max(-2.0, min(8.0, sortino)) * weights["sortino_ratio"]
    score += max(0.5, min(4.0, profit_factor)) * weights["profit_factor"]
    score += max(0.0, min(100.0, win_rate)) * weights["win_rate_pct"]
    score -= max(0.0, min(60.0, drawdown)) * weights["max_drawdown_pct"]
    score += min(float(total_trades), 80.0) * weights["trade_count_bonus"]
    if total_trades < int(cfg["min_total_trades"]):
        score -= (int(cfg["min_total_trades"]) - total_trades) * weights["trade_count_penalty"]
    if cfg["require_positive_total_pnl_pct"] and total_pnl_pct < 0.0:
        score -= abs(total_pnl_pct) * weights["total_pnl_pct"] * 1.5
    return score


def _is_viable_auto_selection_candidate(candidate: ParetoCandidate) -> bool:
    cfg = _load_auto_selection_cfg()
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    has_pnl = "total_pnl_pct" in metadata
    has_trades = "total_trades" in metadata
    total_pnl_pct = _safe_float(metadata.get("total_pnl_pct"), 0.0)
    total_trades = max(0, _safe_int(metadata.get("total_trades"), 0))
    # Negative realized edge is never auto-selectable once pnl evidence exists.
    if has_pnl and total_pnl_pct <= 0.0:
        return False
    if not cfg["enabled"]:
        return True
    if cfg["require_positive_total_pnl_pct"] and has_pnl and total_pnl_pct <= 0.0:
        return False
    if has_trades and total_trades < int(cfg["min_total_trades"]):
        return False
    return True


def _candidate_from_metrics(params: Dict[str, Any], metrics: Dict[str, Any]) -> ParetoCandidate:
    sharpe = _safe_float(metrics.get("sharpe_ratio"), 0.0)
    drawdown = abs(_safe_float(metrics.get("max_drawdown_pct"), 0.0))
    if drawdown <= 1.0:
        drawdown *= 100.0
    win_rate = _safe_float(metrics.get("win_rate_pct", metrics.get("win_rate", 0.0)), 0.0)
    if win_rate <= 1.0:
        win_rate *= 100.0
    score = _score_multi_objective_metrics(metrics)
    metadata = {
        str(key): value
        for key, value in metrics.items()
        if key not in {"sharpe_ratio", "max_drawdown_pct", "win_rate_pct", "win_rate", "params", "score"}
        and (value is None or isinstance(value, (str, int, float, bool, list, dict)))
    }
    return ParetoCandidate(
        params=dict(params),
        sharpe_ratio=sharpe,
        max_drawdown_pct=drawdown,
        win_rate_pct=win_rate,
        score=score,
        metadata=metadata,
    )


def is_pareto_dominated(candidate: ParetoCandidate, others: Iterable[ParetoCandidate]) -> bool:
    for other in others:
        if other is candidate:
            continue
        no_worse = (
            other.sharpe_ratio >= candidate.sharpe_ratio
            and other.max_drawdown_pct <= candidate.max_drawdown_pct
            and other.win_rate_pct >= candidate.win_rate_pct
        )
        strictly_better = (
            other.sharpe_ratio > candidate.sharpe_ratio
            or other.max_drawdown_pct < candidate.max_drawdown_pct
            or other.win_rate_pct > candidate.win_rate_pct
        )
        if no_worse and strictly_better:
            return True
    return False


def pareto_frontier(candidates: Iterable[ParetoCandidate | Dict[str, Any]]) -> List[ParetoCandidate]:
    normalized = [
        item if isinstance(item, ParetoCandidate) else _candidate_from_metrics(item.get("params", {}), item)
        for item in candidates
    ]
    frontier = [candidate for candidate in normalized if not is_pareto_dominated(candidate, normalized)]
    frontier.sort(key=lambda item: (-item.score, -item.sharpe_ratio, item.max_drawdown_pct, -item.win_rate_pct))
    return frontier


def select_pareto_candidate(
    candidates: Iterable[ParetoCandidate | Dict[str, Any]],
    preset: str = "balanced",
) -> Optional[ParetoCandidate]:
    frontier = pareto_frontier(candidates)
    if not frontier:
        return None
    thresholds = PRESET_THRESHOLDS.get(str(preset).lower(), PRESET_THRESHOLDS["balanced"])
    eligible = [
        candidate
        for candidate in frontier
        if candidate.sharpe_ratio >= thresholds["min_sharpe"]
        and candidate.max_drawdown_pct <= thresholds["max_drawdown_pct"]
        and candidate.win_rate_pct >= thresholds["min_win_rate_pct"]
    ]
    pool = eligible or frontier
    viable_pool = [candidate for candidate in pool if _is_viable_auto_selection_candidate(candidate)]
    if viable_pool:
        pool = viable_pool
    elif any(
        isinstance(candidate.metadata, dict)
        and ("total_pnl_pct" in candidate.metadata or "total_trades" in candidate.metadata)
        for candidate in pool
    ):
        return None
    return max(
        pool,
        key=lambda item: (
            item.score,
            _safe_float(item.metadata.get("total_pnl_pct"), 0.0),
            -item.max_drawdown_pct,
            item.sharpe_ratio,
            item.win_rate_pct,
        ),
    )


def _is_verified_multi_objective_metrics(metrics: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(metrics, dict) or not metrics:
        return False
    if bool(metrics.get("verified", False)):
        return True
    return str(metrics.get("evaluation_mode", "")).lower() == "local_backtest"


def _eligible_pareto_candidates(
    candidates: Iterable[ParetoCandidate | Dict[str, Any]],
    preset: str,
) -> List[ParetoCandidate]:
    frontier = pareto_frontier(candidates)
    thresholds = PRESET_THRESHOLDS.get(str(preset).lower(), PRESET_THRESHOLDS["balanced"])
    return [
        candidate
        for candidate in frontier
        if candidate.sharpe_ratio >= thresholds["min_sharpe"]
        and candidate.max_drawdown_pct <= thresholds["max_drawdown_pct"]
        and candidate.win_rate_pct >= thresholds["min_win_rate_pct"]
    ]


def _default_multi_objective_evaluator(params: Dict[str, Any]) -> Dict[str, Any]:
    backtest_metrics = _evaluate_backtest_candidate(params)
    if backtest_metrics is not None:
        return backtest_metrics
    return {
        "evaluation_mode": "unverified_no_local_backtest",
        "reason": "local_backtest_unavailable",
    }


def run_multi_objective_optimization(
    n_trials: int = 50,
    *,
    preset: str = "balanced",
    evaluator: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if n_trials <= 0:
        return {}
    evaluate = evaluator or _default_multi_objective_evaluator
    candidates: List[ParetoCandidate] = []
    invalid_trials = 0
    preset_key = str(preset).lower()

    def _sample_params(trial_like: Any) -> Dict[str, Any]:
        return {
            "C": float(trial_like.suggest_float("C", 1e-3, 1e2, log=True)),
            "stop_loss_mult": float(trial_like.suggest_float("stop_loss_mult", 0.005, 0.05)),
            "ma_short": int(trial_like.suggest_int("ma_short", 10, 100)),
            "ma_long": int(trial_like.suggest_int("ma_long", 20, 200)),
            "cooldown_min": int(trial_like.suggest_int("cooldown_min", 1, 60)),
        }

    def _evaluate_candidate(params: Dict[str, Any]) -> Optional[ParetoCandidate]:
        metrics = evaluate(params)
        if not _is_verified_multi_objective_metrics(metrics):
            return None
        candidate = _candidate_from_metrics(params, metrics)
        candidates.append(candidate)
        return candidate

    if optuna is not None:
        sampler = optuna.samplers.NSGAIISampler()
        study = optuna.create_study(
            directions=["maximize", "minimize", "maximize"],
            sampler=sampler,
        )

        def _objective(trial: "optuna.Trial") -> Tuple[float, float, float]:
            nonlocal invalid_trials
            params = _sample_params(trial)
            candidate = _evaluate_candidate(params)
            if candidate is None:
                invalid_trials += 1
                return 0.0, 10_000.0, 0.0
            return candidate.sharpe_ratio, candidate.max_drawdown_pct, candidate.win_rate_pct

        study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)
    else:
        for _ in range(n_trials):
            class _TrialShim:
                def suggest_float(self, _name: str, low: float, high: float, log: bool = False) -> float:
                    if log:
                        return math.exp(random.uniform(math.log(low), math.log(high)))
                    return random.uniform(low, high)

                def suggest_int(self, _name: str, low: int, high: int) -> int:
                    return random.randint(low, high)

            params = _sample_params(_TrialShim())
            candidate = _evaluate_candidate(params)
            if candidate is None:
                invalid_trials += 1

    frontier = pareto_frontier(candidates)
    recommended = select_pareto_candidate(frontier, preset=preset)
    eligible = _eligible_pareto_candidates(frontier, preset_key)
    selected = recommended.to_dict() if recommended is not None else None
    result = {
        "preset": preset_key,
        "selected": selected,
        "recommended": recommended.to_dict() if recommended is not None else None,
        "pareto_frontier": [candidate.to_dict() for candidate in frontier],
        "preset_eligible_frontier": [candidate.to_dict() for candidate in eligible],
        "candidate_count": len(candidates),
        "verified_candidate_count": len(candidates),
        "invalid_trial_count": invalid_trials,
        "selection_required": False,
        "objective_source": "local_backtest_only",
        "solver": "nsga2" if optuna is not None else "random_search_fallback",
    }
    return result


__all__ = [
    "PRESET_THRESHOLDS",
    "ParetoCandidate",
    "apply_selected_candidate",
    "build_dynamic_kelly_profile",
    "get_stored_pareto_frontier",
    "kelly_fraction",
    "optimize_leverage_and_allocation",
    "pareto_frontier",
    "run_multi_objective_optimization",
    "select_pareto_candidate",
]

