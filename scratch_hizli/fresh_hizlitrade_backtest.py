from __future__ import annotations

import argparse
import bisect
import json
import math
import statistics
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hizlitrade.alpha_validation import (
    AdaptivePolicyConfig,
    evaluate_policy_stages,
    fit_policy,
)
from hizlitrade.oos import (
    SettledTrade,
    aggregate_markets,
    purged_market_walk_forward_folds,
)
from hizlitrade.validation import trade_return_summary

ASSETS = {
    "btc": ("BTCUSD", "BTC-USDT"),
    "eth": ("ETHUSD", "ETH-USDT"),
    "sol": ("SOLUSD", "SOL-USDT"),
    "xrp": ("XRPUSD", "XRP-USDT"),
    "doge": ("DOGEUSD", "DOGE-USDT"),
    "bnb": ("BNBUSD", "BNB-USDT"),
    "zec": ("ZECUSD", "ZEC-USDT"),
    "hype": ("HYPEUSD", "HYPE-USDT"),
}
DURATIONS = (5, 15)
DEFAULT_FEE_RATE = 0.07
DEFAULT_FEE_EXPONENT = 1.0
MIN_NET_EDGE = 0.025
SIGNAL_SLIPPAGE = 0.003
SAFETY_MARGIN = 0.005
ORDER_NOTIONAL = 2.50
START_BALANCE = 250.0
MAX_OPEN_EXPOSURE_PCT = 0.20
MAX_MARKET_EXPOSURE_PCT = 0.05
MAX_SYMBOL_EXPOSURE_PCT = 0.10
MAX_CONCURRENT_POSITIONS = 8
MAX_DAILY_LOSS_PCT = 0.05
MAX_DRAWDOWN_PCT = 0.10
MIN_TIME_TO_EXPIRY_S = 3
COOLDOWN_S = 5
ALLOW_MARKET_PYRAMIDING = False
VOL_WINDOW_SECONDS = 600
VOL_MIN_SAMPLES = 8
VOL_FLOOR = 1e-7

SCENARIOS = {
    "history_price_optimistic": 0.0,
    "configured_0.003_per_share": 0.003,
}


def fetch_json(url: str, attempts: int = 5) -> Any:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "hizlitrade-fresh-backtest/1.0"},
            )
            with urllib.request.urlopen(req, timeout=25) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last = exc
            time.sleep(0.25 * (attempt + 1))
    raise RuntimeError(f"request failed after {attempts} attempts: {url}: {last}")


def parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def gamma_market_for_slug(slug: str) -> dict[str, Any] | None:
    url = "https://gamma-api.polymarket.com/events?" + urllib.parse.urlencode({"slug": slug})
    data = fetch_json(url)
    if not isinstance(data, list) or not data:
        return None
    markets = data[0].get("markets") or []
    if not isinstance(markets, list):
        return None
    for market in markets:
        if isinstance(market, dict) and str(market.get("slug") or "") == slug:
            return market
    return None


def clob_fee_schedule(market: dict[str, Any]) -> tuple[float, float, str]:
    """Resolve the market's published fee schedule, falling back explicitly."""
    if market.get("feesEnabled") is False:
        return 0.0, 1.0, "gamma_fee_free"
    condition_id = str(
        market.get("conditionId")
        or market.get("condition_id")
        or ""
    )
    if condition_id:
        try:
            info = fetch_json(
                "https://clob.polymarket.com/clob-markets/"
                + urllib.parse.quote(condition_id, safe="")
            )
            fd = info.get("fd") if isinstance(info, dict) else None
            if isinstance(fd, dict):
                rate = float(fd.get("r") or 0.0)
                exponent = float(fd.get("e") if fd.get("e") is not None else 1.0)
                if (
                    math.isfinite(rate)
                    and rate >= 0
                    and math.isfinite(exponent)
                    and exponent >= 0
                ):
                    return rate, exponent, "clob_market_fd"
        except Exception:
            pass
    return DEFAULT_FEE_RATE, DEFAULT_FEE_EXPONENT, "default_crypto_fallback"


def settled_winner(market: dict[str, Any]) -> str | None:
    outcomes = [str(x).strip().lower() for x in parse_json_list(market.get("outcomes"))]
    prices_raw = parse_json_list(market.get("outcomePrices"))
    if len(outcomes) != 2 or len(prices_raw) != 2:
        return None
    try:
        prices = [float(x) for x in prices_raw]
    except (TypeError, ValueError):
        return None
    if max(prices) < 0.99 or min(prices) > 0.01:
        return None
    winner = outcomes[prices.index(max(prices))]
    if winner in {"up", "yes"}:
        return "YES"
    if winner in {"down", "no"}:
        return "NO"
    return None


def instrument_history(instrument: str) -> list[tuple[int, float]]:
    params = urllib.parse.urlencode({"market": instrument, "interval": "max", "fidelity": 1})
    data = fetch_json("https://clob.polymarket.com/prices-history?" + params)
    rows = data.get("history") if isinstance(data, dict) else None
    out: list[tuple[int, float]] = []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                ts = int(row["t"])
                price = float(row["p"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 < price < 1:
                out.append((ts, price))
    return sorted(dict(out).items())


def okx_spot_history(inst_id: str, cutoff_s: int) -> list[tuple[int, float]]:
    rows: dict[int, float] = {}
    after: str | None = None
    stagnant = 0
    while True:
        params: dict[str, Any] = {"instId": inst_id, "bar": "1m", "limit": "300"}
        if after is not None:
            params["after"] = after
        url = "https://www.okx.com/api/v5/market/history-candles?" + urllib.parse.urlencode(params)
        payload = fetch_json(url)
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or not data:
            break
        batch_min: int | None = None
        for raw in data:
            if not isinstance(raw, list) or len(raw) < 9:
                continue
            try:
                ts_ms = int(raw[0])
                open_px = float(raw[1])
                confirm = str(raw[8])
            except (TypeError, ValueError):
                continue
            if confirm != "1" or open_px <= 0:
                continue
            ts = ts_ms // 1000
            rows[ts] = open_px
            batch_min = ts if batch_min is None else min(batch_min, ts)
        if batch_min is None or batch_min <= cutoff_s:
            break
        if after == str(batch_min * 1000):
            stagnant += 1
            if stagnant >= 2:
                break
        else:
            stagnant = 0
        after = str(batch_min * 1000)
        time.sleep(0.08)
    return sorted((ts, px) for ts, px in rows.items() if ts >= cutoff_s)


@dataclass
class SpotSeries:
    times: list[int]
    prices: list[float]
    sigmas: list[float]

    @classmethod
    def build(cls, rows: list[tuple[int, float]]) -> "SpotSeries":
        times = [x[0] for x in rows]
        prices = [x[1] for x in rows]
        sigmas: list[float] = []
        returns: deque[tuple[int, float]] = deque()
        prev_t: int | None = None
        prev_p: float | None = None
        for ts, price in rows:
            if prev_t is not None and prev_p is not None and price > 0 and prev_p > 0:
                dt = ts - prev_t
                if dt > 0:
                    returns.append((ts, math.log(price / prev_p) / math.sqrt(dt)))
            cutoff = ts - VOL_WINDOW_SECONDS
            while returns and returns[0][0] < cutoff:
                returns.popleft()
            values = [value for _return_ts, value in returns]
            if len(values) < VOL_MIN_SAMPLES:
                sigma = VOL_FLOOR
            else:
                sigma = max(statistics.stdev(values), VOL_FLOOR)
            sigmas.append(sigma)
            prev_t, prev_p = ts, price
        return cls(times, prices, sigmas)

    def at(self, ts: int) -> tuple[float, float] | None:
        idx = bisect.bisect_right(self.times, ts) - 1
        if idx < 0:
            return None
        return self.prices[idx], self.sigmas[idx]


@dataclass
class Market:
    slug: str
    market_id: str
    symbol: str
    start: int
    end: int
    winner: str
    yes_instrument: str
    no_instrument: str
    yes_history: list[tuple[int, float]]
    no_history: list[tuple[int, float]]
    volume: float | None
    fee_rate: float
    fee_exponent: float
    fee_source: str


@dataclass
class Candidate:
    ts: int
    market_slug: str
    market_id: str
    symbol: str
    end: int
    outcome: str
    quote_price: float
    fair: float
    net_edge: float
    winner: str
    history: list[tuple[int, float]]
    spot_price: float
    strike: float
    basis_bps: float
    momentum_1m_bps: float | None
    momentum_3m_bps: float | None
    market_duration_ms: float
    fee_rate: float
    fee_exponent: float


@dataclass
class Position:
    market_slug: str
    market_id: str
    symbol: str
    outcome: str
    end: int
    winner: str
    shares: float
    cost_basis: float
    fill_price: float
    fee_per_share: float
    entry_ts: int
    history: list[tuple[int, float]]


def fee_per_share(
    price: float,
    *,
    rate: float = DEFAULT_FEE_RATE,
    exponent: float = DEFAULT_FEE_EXPONENT,
) -> float:
    base = max(price * (1.0 - price), 0.0)
    return rate * (base**exponent) if rate > 0 and base > 0 else 0.0


def fair_yes(spot: float, strike: float, sigma: float, remaining_s: float) -> float:
    if strike <= 0 or spot <= 0:
        return 0.5
    remaining_s = max(remaining_s, 0.001)
    denom = max(sigma * math.sqrt(remaining_s), 1e-9)
    z = (
        math.log(spot / strike)
        - 0.5 * sigma * sigma * remaining_s
    ) / denom
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return min(max(cdf, 0.001), 0.999)


def latest_price(history: list[tuple[int, float]], ts: int) -> float | None:
    times = [x[0] for x in history]
    idx = bisect.bisect_right(times, ts) - 1
    return history[idx][1] if idx >= 0 else None


def market_candidates(market: Market, spot: SpotSeries) -> list[Candidate]:
    strike_point = spot.at(market.start)
    if strike_point is None:
        return []
    strike = strike_point[0]
    yes = [(t, p) for t, p in market.yes_history if market.start <= t < market.end]
    no = [(t, p) for t, p in market.no_history if market.start <= t < market.end]
    if not yes or not no:
        return []
    times = sorted({t for t, _ in yes} | {t for t, _ in no})
    out: list[Candidate] = []
    for ts in times:
        if market.end - ts < MIN_TIME_TO_EXPIRY_S:
            continue
        point = spot.at(ts)
        if point is None:
            continue
        spot_px, sigma = point
        one_minute = spot.at(ts - 60)
        three_minutes = spot.at(ts - 180)
        momentum_1m_bps = (
            (spot_px / one_minute[0] - 1.0) * 10_000
            if one_minute is not None and one_minute[0] > 0
            else None
        )
        momentum_3m_bps = (
            (spot_px / three_minutes[0] - 1.0) * 10_000
            if three_minutes is not None and three_minutes[0] > 0
            else None
        )
        basis_bps = (spot_px / strike - 1.0) * 10_000
        p_yes = fair_yes(spot_px, strike, sigma, market.end - ts)
        for outcome, history, fair in (
            ("YES", yes, p_yes),
            ("NO", no, 1.0 - p_yes),
        ):
            quote = latest_price(history, ts)
            if quote is None or not (0 < quote < 1):
                continue
            fee = fee_per_share(
                quote,
                rate=market.fee_rate,
                exponent=market.fee_exponent,
            )
            net = fair - quote - fee - SIGNAL_SLIPPAGE - SAFETY_MARGIN
            if net >= MIN_NET_EDGE:
                out.append(
                    Candidate(
                        ts=ts,
                        market_slug=market.slug,
                        market_id=market.market_id,
                        symbol=market.symbol,
                        end=market.end,
                        outcome=outcome,
                        quote_price=quote,
                        fair=fair,
                        net_edge=net,
                        winner=market.winner,
                        history=history,
                        spot_price=spot_px,
                        strike=strike,
                        basis_bps=basis_bps,
                        momentum_1m_bps=momentum_1m_bps,
                        momentum_3m_bps=momentum_3m_bps,
                        market_duration_ms=float((market.end - market.start) * 1000),
                        fee_rate=market.fee_rate,
                        fee_exponent=market.fee_exponent,
                    )
                )
    return out


def simulate(candidates: list[Candidate], execution_slippage: float) -> dict[str, Any]:
    cash = START_BALANCE
    start_of_day_equity = START_BALANCE
    risk_day: str | None = None
    peak_equity = START_BALANCE
    positions: list[Position] = []
    settlements: list[dict[str, Any]] = []
    rejections: Counter[str] = Counter()
    last_trade: dict[tuple[str, str], int] = {}

    def settle_due(now: int) -> None:
        nonlocal cash, positions
        keep: list[Position] = []
        for pos in positions:
            if pos.end > now:
                keep.append(pos)
                continue
            payout = pos.shares if pos.outcome == pos.winner else 0.0
            cash += payout
            pnl = payout - pos.cost_basis
            settlements.append(
                {
                    "market": pos.market_slug,
                    "symbol": pos.symbol,
                    "outcome": pos.outcome,
                    "winner": pos.winner,
                    "entry_ts": pos.entry_ts,
                    "fill_price": pos.fill_price,
                    "fee_per_share": pos.fee_per_share,
                    "shares": pos.shares,
                    "pnl": pnl,
                }
            )
        positions = keep

    def snapshot(now: int) -> tuple[float, float, dict[str, float], dict[str, float], int]:
        nonlocal peak_equity
        market_value = 0.0
        market_exposure: dict[str, float] = defaultdict(float)
        symbol_exposure: dict[str, float] = defaultdict(float)
        unique_positions: set[tuple[str, str]] = set()
        open_exposure = 0.0
        for pos in positions:
            mark = latest_price(pos.history, now)
            if mark is None:
                mark = 0.0
            market_value += pos.shares * mark
            open_exposure += pos.cost_basis
            market_exposure[pos.market_id] += pos.cost_basis
            symbol_exposure[pos.symbol] += pos.cost_basis
            unique_positions.add((pos.market_id, pos.outcome))
        equity = cash + market_value
        peak_equity = max(peak_equity, equity)
        return equity, open_exposure, market_exposure, symbol_exposure, len(unique_positions)

    ordered = sorted(
        candidates,
        key=lambda x: (x.ts, x.market_slug, 0 if x.outcome == "YES" else 1),
    )
    for c in ordered:
        settle_due(c.ts)
        equity, open_exposure, market_exp, symbol_exp, position_count = snapshot(c.ts)
        current_risk_day = datetime.fromtimestamp(c.ts, tz=UTC).date().isoformat()
        if risk_day != current_risk_day:
            risk_day = current_risk_day
            start_of_day_equity = equity
        key = (c.market_id, c.outcome)
        if c.ts - last_trade.get(key, -10**18) < COOLDOWN_S:
            rejections["cooldown"] += 1
            continue

        if equity <= 0 or cash <= 0:
            rejections["no_capital"] += 1
            continue
        daily_loss = max(start_of_day_equity - equity, 0.0)
        if daily_loss / start_of_day_equity >= MAX_DAILY_LOSS_PCT:
            rejections["daily_loss_kill"] += 1
            continue
        drawdown = max(peak_equity - equity, 0.0)
        if peak_equity > 0 and drawdown / peak_equity >= MAX_DRAWDOWN_PCT:
            rejections["drawdown_kill"] += 1
            continue
        worst_cash = cash - ORDER_NOTIONAL
        worst_daily = max(start_of_day_equity - worst_cash, 0.0)
        if worst_daily / start_of_day_equity > MAX_DAILY_LOSS_PCT:
            rejections["daily_loss_budget_at_risk"] += 1
            continue
        worst_dd = max(peak_equity - worst_cash, 0.0)
        if peak_equity > 0 and worst_dd / peak_equity > MAX_DRAWDOWN_PCT:
            rejections["drawdown_budget_at_risk"] += 1
            continue
        if position_count >= MAX_CONCURRENT_POSITIONS:
            rejections["max_concurrent"] += 1
            continue
        if open_exposure + ORDER_NOTIONAL > equity * MAX_OPEN_EXPOSURE_PCT:
            rejections["open_exposure"] += 1
            continue
        current_market_exposure = market_exp.get(c.market_id, 0.0)
        if current_market_exposure > 0 and not ALLOW_MARKET_PYRAMIDING:
            rejections["market_pyramiding_disabled"] += 1
            continue
        if current_market_exposure + ORDER_NOTIONAL > equity * MAX_MARKET_EXPOSURE_PCT:
            rejections["market_exposure"] += 1
            continue
        if symbol_exp.get(c.symbol, 0.0) + ORDER_NOTIONAL > equity * MAX_SYMBOL_EXPOSURE_PCT:
            rejections["symbol_exposure"] += 1
            continue
        if ORDER_NOTIONAL > cash:
            rejections["cash"] += 1
            continue

        fill = min(c.quote_price + execution_slippage, 0.999)
        if fill > c.quote_price + SIGNAL_SLIPPAGE + 1e-12:
            rejections["fok_price_limit"] += 1
            continue
        actual_fee = fee_per_share(
            fill,
            rate=c.fee_rate,
            exponent=c.fee_exponent,
        )
        actual_net = c.fair - fill - actual_fee - SAFETY_MARGIN
        if actual_net < MIN_NET_EDGE:
            rejections["edge_decayed_before_execution"] += 1
            continue
        effective = fill + actual_fee
        shares = ORDER_NOTIONAL / effective
        cash -= ORDER_NOTIONAL
        positions.append(
            Position(
                market_slug=c.market_slug,
                market_id=c.market_id,
                symbol=c.symbol,
                outcome=c.outcome,
                end=c.end,
                winner=c.winner,
                shares=shares,
                cost_basis=ORDER_NOTIONAL,
                fill_price=fill,
                fee_per_share=actual_fee,
                entry_ts=c.ts,
                history=c.history,
            )
        )
        last_trade[key] = c.ts

    settle_due(10**18)
    wins = [x for x in settlements if x["pnl"] > 0]
    losses = [x for x in settlements if x["pnl"] <= 0]
    gross_profit = sum(x["pnl"] for x in wins)
    gross_loss = -sum(x["pnl"] for x in losses)
    total_pnl = sum(x["pnl"] for x in settlements)
    deployed = ORDER_NOTIONAL * len(settlements)
    by_symbol: dict[str, list[float]] = defaultdict(list)
    for x in settlements:
        by_symbol[x["symbol"]].append(x["pnl"])
    return {
        "execution_slippage_per_share": execution_slippage,
        "trades": len(settlements),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(settlements) if settlements else 0.0,
        "total_pnl_usd": total_pnl,
        "final_cash_usd": cash,
        "roi_on_deployed_capital": total_pnl / deployed if deployed else None,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else None,
        "avg_pnl_per_trade_usd": total_pnl / len(settlements) if settlements else None,
        "rejections": dict(sorted(rejections.items())),
        "by_symbol": {
            symbol: {
                "trades": len(values),
                "pnl_usd": sum(values),
            }
            for symbol, values in sorted(by_symbol.items())
        },
        "sample_settlements": settlements[:20],
    }


def shadow_samples(
    candidates: list[Candidate],
    execution_slippage: float,
) -> list[SettledTrade]:
    """Build one execution-valid counterfactual sample per market/outcome."""
    samples: list[SettledTrade] = []
    seen: set[tuple[str, str]] = set()
    for candidate in sorted(
        candidates,
        key=lambda item: (item.ts, item.market_id, item.outcome),
    ):
        key = (candidate.market_id, candidate.outcome)
        if key in seen:
            continue
        fill = min(candidate.quote_price + execution_slippage, 0.999)
        if fill > candidate.quote_price + SIGNAL_SLIPPAGE + 1e-12:
            continue
        actual_fee = fee_per_share(
            fill,
            rate=candidate.fee_rate,
            exponent=candidate.fee_exponent,
        )
        actual_net = candidate.fair - fill - actual_fee - SAFETY_MARGIN
        if actual_net < MIN_NET_EDGE:
            continue
        effective = fill + actual_fee
        if effective <= 0:
            continue
        shares = ORDER_NOTIONAL / effective
        payout = shares if candidate.outcome == candidate.winner else 0.0
        pnl = payout - ORDER_NOTIONAL
        samples.append(
            SettledTrade(
                market_id=candidate.market_id,
                instrument=f"{candidate.market_id}:{candidate.outcome}",
                symbol=candidate.symbol,
                signal_ts_ns=candidate.ts * 1_000_000_000,
                settled_ts_ns=candidate.end * 1_000_000_000,
                cost_basis_usd=ORDER_NOTIONAL,
                realized_pnl_usd=pnl,
                shares=shares,
                fair_probability=candidate.fair,
                won=candidate.outcome == candidate.winner,
                momentum_1s_bps=candidate.momentum_1m_bps,
                oracle_basis_bps=candidate.basis_bps,
                prediction_spread=None,
                outcome=candidate.outcome,
                momentum_250ms_bps=None,
                momentum_3s_bps=candidate.momentum_3m_bps,
                book_imbalance=None,
                time_to_expiry_ms=float((candidate.end - candidate.ts) * 1000),
                net_edge=actual_net,
                signal_executable_price=candidate.quote_price,
                market_duration_ms=candidate.market_duration_ms,
            )
        )
        seen.add(key)
    return samples


def _proxy_stage_summary(rows: list[SettledTrade]) -> dict[str, Any]:
    grouped: dict[str, list[SettledTrade]] = defaultdict(list)
    for row in rows:
        grouped[row.market_id].append(row)
    market_returns: list[float] = []
    for market_rows in grouped.values():
        cost = sum(row.cost_basis_usd for row in market_rows)
        pnl = sum(row.realized_pnl_usd for row in market_rows)
        market_returns.append(pnl / cost if cost > 0 else 0.0)
    trade_returns = [row.realized_return for row in rows]
    return {
        "trades": len(rows),
        "unique_markets": len(grouped),
        "wins": sum(row.realized_pnl_usd > 0 for row in rows),
        "win_rate": (
            sum(row.realized_pnl_usd > 0 for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "realized_pnl_usd": sum(row.realized_pnl_usd for row in rows),
        "cost_basis_usd": sum(row.cost_basis_usd for row in rows),
        "return_stats": trade_return_summary(market_returns),
        "trade_return_stats": trade_return_summary(trade_returns),
    }


def adaptive_proxy_backtest(samples: list[SettledTrade]) -> dict[str, Any]:
    markets = aggregate_markets(tuple(samples))
    market_count = len(markets)
    if market_count < 50:
        return {
            "qualified_for_diagnostic": False,
            "blockers": ["fewer_than_50_execution_valid_markets"],
            "market_count": market_count,
            "sample_count": len(samples),
            "folds": [],
        }

    min_train = max(30, min(100, market_count // 2))
    remaining = market_count - min_train
    test_size = max(10, min(50, max(remaining // 3, 10)))
    folds = purged_market_walk_forward_folds(
        markets,
        min_train_size=min_train,
        test_size=test_size,
        step=test_size,
    )
    config = AdaptivePolicyConfig(
        min_symbol_train_markets=max(5, min_train // 8),
        min_setup_train_markets=max(5, min_train // 8),
        min_session_train_markets=max(5, min_train // 8),
        min_ml_train_trades=max(30, min_train // 2),
        min_final_train_markets=max(15, min_train // 5),
        stability_windows=3,
        min_positive_stability_windows=2,
    )
    fold_reports: list[dict[str, Any]] = []
    accumulated: dict[str, dict[tuple[str, str, int], SettledTrade]] = defaultdict(dict)

    by_id = {row.market_id: row for row in markets}
    for index, fold in enumerate(folds):
        train_rows = tuple(by_id[market_id] for market_id in fold.train_market_ids)
        test_rows = tuple(by_id[market_id] for market_id in fold.test_market_ids)
        train_ids = set(fold.train_market_ids)
        test_ids = set(fold.test_market_ids)
        if train_ids & test_ids:
            raise RuntimeError("proxy walk-forward market leakage detected")
        if train_rows and max(row.settled_ts_ns for row in train_rows) >= fold.test_start_ts_ns:
            raise RuntimeError("proxy walk-forward used a future settlement label")
        train = [row for row in samples if row.market_id in train_ids]
        test = [row for row in samples if row.market_id in test_ids]
        policy = fit_policy(train, config)
        stages = evaluate_policy_stages(test, policy, config)
        for stage, rows in stages.items():
            for row in rows:
                accumulated[stage][(row.market_id, row.instrument, row.signal_ts_ns)] = row
        fold_reports.append(
            {
                "fold": index,
                "train_markets": len(train_ids),
                "test_markets": len(test_ids),
                "purged_train_markets": fold.purged_train_markets,
                "train_last_settled_ts_ns": fold.train_last_settled_ts_ns,
                "train_end_ts_ns": fold.train_end_ts_ns,
                "test_start_ts_ns": fold.test_start_ts_ns,
                "selected_symbols": list(policy.selected_symbols),
                "selected_setup_families": list(policy.selected_setup_families),
                "selected_sessions": list(policy.selected_sessions),
                "selected_regimes": list(policy.selected_regimes),
                "selection_mode": policy.selection_mode,
                "min_consensus": policy.min_consensus,
                "min_net_edge": policy.min_net_edge,
                "max_prediction_spread": policy.max_prediction_spread,
                "ml_threshold": (
                    policy.ml_model.threshold if policy.ml_model is not None else None
                ),
                "alpha_evidence": {
                    "qualified": policy.alpha_evidence_qualified,
                    "reason": policy.alpha_evidence_reason,
                    "train_markets": policy.alpha_train_markets,
                    "train_mean_return": policy.alpha_train_mean_return,
                    "train_ci95_low": policy.alpha_train_ci95_low,
                    "stability_positive_windows": policy.alpha_stability_positive_windows,
                    "stability_total_windows": policy.alpha_stability_total_windows,
                },
                "stages": {
                    stage: _proxy_stage_summary(rows)
                    for stage, rows in stages.items()
                },
            }
        )

    combined = {
        stage: _proxy_stage_summary(list(rows.values()))
        for stage, rows in accumulated.items()
    }
    return {
        "qualified_for_diagnostic": bool(folds),
        "market_count": market_count,
        "sample_count": len(samples),
        "min_train_markets": min_train,
        "test_markets": test_size,
        "alpha_gate_parameters": {
            "min_final_train_markets": config.min_final_train_markets,
            "stability_windows": config.stability_windows,
            "min_positive_stability_windows": config.min_positive_stability_windows,
        },
        "folds": fold_reports,
        "combined_oos_ablation": combined,
        "notes": [
            "Every policy is fitted only on prior markets whose settlements were observable before the next OOS test signal.",
            "The proxy maps OKX 1m/3m momentum into the short-horizon momentum feature slots because sub-second historical spot is unavailable.",
            "Historical CLOB L2 spread and book imbalance are unavailable and are not fabricated.",
            "The final alpha-evidence gate abstains when the train-selected portfolio lacks a positive lower 95% confidence bound or chronological stability.",
            "A NO_TRADE result is treated as risk control, not as profitable OOS evidence.",
            "This diagnostic cannot satisfy the live-promotion replay or microstructure evidence gates.",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=12)
    ap.add_argument("--output", default="fresh_hizlitrade_backtest.json")
    args = ap.parse_args()

    now = int(time.time())
    slugs: list[tuple[str, str, int, int]] = []
    for asset, (symbol, _inst) in ASSETS.items():
        for minutes in DURATIONS:
            step = minutes * 60
            anchor = (now // step) * step
            count = max(1, args.hours * 60 // minutes)
            for i in range(1, count + 1):
                start = anchor - i * step
                slugs.append((asset, symbol, minutes, start))

    discovered: list[tuple[tuple[str, str, int, int], dict[str, Any]]] = []
    discovery_failures: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(gamma_market_for_slug, f"{asset}-updown-{minutes}m-{start}"):
            (asset, symbol, minutes, start)
            for asset, symbol, minutes, start in slugs
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                market = future.result()
            except Exception:
                discovery_failures["request_error"] += 1
                continue
            if market is None:
                discovery_failures["not_found"] += 1
                continue
            winner = settled_winner(market)
            if winner is None or not bool(market.get("closed")):
                discovery_failures["not_settled"] += 1
                continue
            discovered.append((key, market))

    # Fetch enough spot warmup for the runtime-parity 600-second volatility window.
    cutoff = now - (args.hours + 1) * 3600
    spots: dict[str, SpotSeries] = {}
    for _asset, (symbol, inst_id) in ASSETS.items():
        rows = okx_spot_history(inst_id, cutoff)
        if rows:
            spots[symbol] = SpotSeries.build(rows)

    markets: list[Market] = []
    history_failures: Counter[str] = Counter()

    def build_market(item: tuple[tuple[str, str, int, int], dict[str, Any]]) -> Market | None:
        (asset, symbol, minutes, start), row = item
        instruments = [str(x) for x in parse_json_list(row.get("clobTokenIds"))]
        if len(instruments) != 2:
            return None
        winner = settled_winner(row)
        if winner is None:
            return None
        yes_h = instrument_history(instruments[0])
        no_h = instrument_history(instruments[1])
        fee_rate, fee_exponent, fee_source = clob_fee_schedule(row)
        try:
            volume = float(row.get("volume")) if row.get("volume") is not None else None
        except (TypeError, ValueError):
            volume = None
        return Market(
            slug=f"{asset}-updown-{minutes}m-{start}",
            market_id=str(row.get("id") or ""),
            symbol=symbol,
            start=start,
            end=start + minutes * 60,
            winner=winner,
            yes_instrument=instruments[0],
            no_instrument=instruments[1],
            yes_history=yes_h,
            no_history=no_h,
            volume=volume,
            fee_rate=fee_rate,
            fee_exponent=fee_exponent,
            fee_source=fee_source,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(build_market, item): item for item in discovered}
        for future in as_completed(futures):
            try:
                market = future.result()
            except Exception:
                history_failures["request_error"] += 1
                continue
            if market is None:
                history_failures["invalid_market"] += 1
                continue
            if not market.yes_history or not market.no_history:
                history_failures["missing_price_history"] += 1
                continue
            if market.symbol not in spots:
                history_failures["missing_spot_history"] += 1
                continue
            markets.append(market)

    candidates: list[Candidate] = []
    market_candidate_counts: dict[str, int] = {}
    for market in markets:
        created = market_candidates(market, spots[market.symbol])
        candidates.extend(created)
        market_candidate_counts[market.slug] = len(created)

    scenario_results = {
        name: simulate(candidates, slippage)
        for name, slippage in SCENARIOS.items()
    }
    research_samples = shadow_samples(
        candidates,
        SCENARIOS["configured_0.003_per_share"],
    )
    adaptive_diagnostic = adaptive_proxy_backtest(research_samples)
    report = {
        "generated_at_epoch": int(time.time()),
        "method": "fresh_public_historical_proxy_backtest",
        "old_backtest_results_used": False,
        "scope": {
            "hours": args.hours,
            "assets": list(ASSETS),
            "durations_minutes": list(DURATIONS),
            "requested_market_slots": len(slugs),
            "settled_markets_discovered": len(discovered),
            "markets_with_complete_history": len(markets),
            "candidate_signals": len(candidates),
        },
        "strategy_parameters": {
            "fee_model": "per-market CLOB fd.r/fd.e with explicit crypto fallback",
            "default_fee_rate_fallback": DEFAULT_FEE_RATE,
            "default_fee_exponent_fallback": DEFAULT_FEE_EXPONENT,
            "fee_schedule_sources": dict(
                sorted(Counter(market.fee_source for market in markets).items())
            ),
            "min_net_edge": MIN_NET_EDGE,
            "signal_slippage_per_share": SIGNAL_SLIPPAGE,
            "safety_margin": SAFETY_MARGIN,
            "order_notional_usd": ORDER_NOTIONAL,
            "paper_balance_usd": START_BALANCE,
            "max_daily_loss_pct": MAX_DAILY_LOSS_PCT,
            "max_drawdown_pct": MAX_DRAWDOWN_PCT,
            "max_open_exposure_pct": MAX_OPEN_EXPOSURE_PCT,
            "max_market_exposure_pct": MAX_MARKET_EXPOSURE_PCT,
            "max_symbol_exposure_pct": MAX_SYMBOL_EXPOSURE_PCT,
            "allow_market_pyramiding": ALLOW_MARKET_PYRAMIDING,
            "volatility_window_seconds": VOL_WINDOW_SECONDS,
            "volatility_min_samples": VOL_MIN_SAMPLES,
            "risk_day_timezone": "UTC",
            "daily_loss_baseline": "reset to current equity on each UTC date change",
        },
        "data_sources": {
            "prediction_market": "Polymarket Gamma + CLOB prices-history",
            "spot_proxy": "OKX 1m historical candle open",
            "strike_proxy": "OKX 1m open at the contract slug start epoch",
        },
        "limitations": [
            "Historical CLOB prices-history does not expose historical L2 depth or exact executable ask; this is not promotion-quality frozen replay.",
            "Chainlink 60s opening TWAP is proxied by OKX 1m open at the same start timestamp.",
            "One-minute historical sampling cannot reproduce the live 100ms execution latency or sub-minute feed ordering.",
            "The full currently configured eight-symbol crypto reference universe is queried; missing or unsettled recurring Polymarket slots remain explicit discovery failures rather than being silently excluded from requested scope.",
            "The exact HIZLITRADE fair-value equation and current signal/risk thresholds are used, but the market-data microstructure is necessarily approximated.",
            "Fee schedules are resolved per settled market from CLOB fd metadata when available; any fallback is counted explicitly in strategy_parameters.fee_schedule_sources.",
        ],
        "discovery_failures": dict(discovery_failures),
        "history_failures": dict(history_failures),
        "market_candidate_counts": market_candidate_counts,
        "scenarios": scenario_results,
        "shadow_research_samples": len(research_samples),
        "adaptive_proxy": adaptive_diagnostic,
    }
    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("=== FRESH HIZLITRADE BACKTEST ===")
    print(json.dumps(report["scope"], indent=2))
    for name, result in scenario_results.items():
        print(
            name,
            "trades=", result["trades"],
            "win_rate=", round(result["win_rate"] * 100, 2),
            "pnl=", round(result["total_pnl_usd"], 4),
            "final_cash=", round(result["final_cash_usd"], 4),
            "roi_deployed=", None if result["roi_on_deployed_capital"] is None else round(result["roi_on_deployed_capital"] * 100, 3),
            "profit_factor=", None if result["profit_factor"] is None else round(result["profit_factor"], 3),
        )
        print("rejections=", result["rejections"])
        print("by_symbol=", result["by_symbol"])
    print("shadow_research_samples=", len(research_samples))
    for stage, summary in adaptive_diagnostic.get("combined_oos_ablation", {}).items():
        print(
            "adaptive",
            stage,
            "trades=", summary["trades"],
            "markets=", summary["unique_markets"],
            "win_rate=", round(summary["win_rate"] * 100, 2),
            "pnl=", round(summary["realized_pnl_usd"], 4),
            "mean_return=", summary["return_stats"].get("mean"),
            "ci95_low=", summary["return_stats"].get("mean_ci95_low"),
        )


if __name__ == "__main__":
    main()
