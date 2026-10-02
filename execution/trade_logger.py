from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

from db_utils import DB_PATH, initialise_db, insert_trade


def log_closed_trade(record: dict[str, Any]) -> dict[str, Any]:
    initialise_db()
    payload = dict(record)
    if not payload.get("timestamp_close"):
        payload["timestamp_close"] = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(str(DB_PATH)) as conn:
        insert_trade(conn, payload)
        conn.commit()
    return payload


def _row_dicts(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
    columns = [str(row[0]) for row in cursor.description or []]
    return [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]


def get_recent_trades(limit: int = 250) -> list[dict[str, Any]]:
    initialise_db()
    limit = max(1, min(int(limit), 10_000))
    with sqlite3.connect(str(DB_PATH)) as conn:
        cursor = conn.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return _row_dicts(cursor)


def get_daily_realized_pnl(day: str | None = None) -> float:
    initialise_db()
    day_key = day or datetime.now(timezone.utc).date().isoformat()
    with sqlite3.connect(str(DB_PATH)) as conn:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(pnl_usd), 0.0)
            FROM trades
            WHERE substr(COALESCE(timestamp_close, ''), 1, 10) = ?
            """,
            (day_key,),
        ).fetchone()
    try:
        return float(row[0] if row else 0.0)
    except (TypeError, ValueError):
        return 0.0


def calculate_portfolio_metrics(limit: int = 1000) -> dict[str, float]:
    trades = list(reversed(get_recent_trades(limit=limit)))
    if not trades:
        return {
            "expectancy": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "trade_count": 0.0,
            "realized_pnl": 0.0,
            "max_drawdown_usd": 0.0,
        }

    pnl_values: list[float] = []
    return_values: list[float] = []
    for trade in trades:
        try:
            pnl_values.append(float(trade.get("pnl_usd") or 0.0))
        except (TypeError, ValueError):
            pnl_values.append(0.0)
        try:
            return_values.append(float(trade.get("pnl_pct") or 0.0))
        except (TypeError, ValueError):
            return_values.append(0.0)

    wins = [value for value in pnl_values if value > 0.0]
    losses = [value for value in pnl_values if value < 0.0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for pnl in pnl_values:
        equity += pnl
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)

    return {
        "expectancy": sum(return_values) / max(1, len(return_values)),
        "win_rate": len(wins) / max(1, len(pnl_values)),
        "profit_factor": profit_factor,
        "trade_count": float(len(pnl_values)),
        "realized_pnl": sum(pnl_values),
        "max_drawdown_usd": max_drawdown,
    }


def refresh_performance_aggregates() -> dict[str, Any]:
    metrics = calculate_portfolio_metrics()
    return {
        **metrics,
        "daily_realized_pnl": get_daily_realized_pnl(),
        "recent_trades": int(metrics.get("trade_count", 0.0)),
    }


__all__ = [
    "calculate_portfolio_metrics",
    "get_daily_realized_pnl",
    "get_recent_trades",
    "log_closed_trade",
    "refresh_performance_aggregates",
]
