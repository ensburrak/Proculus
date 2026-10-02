from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import statistics
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hizlitrade.alpha_validation import (
    AdaptivePolicyConfig,
    PortfolioReplayConfig,
    evaluate_policy_stages,
    fit_policy,
    replay_portfolio,
)
from hizlitrade.edge_quality import EdgeQualityModel, sequential_edge_quality_filter
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
FEE_RATE = 0.07
MIN_NET_EDGE = 0.025
SIGNAL_SLIPPAGE = 0.003
SAFETY_MARGIN = 0.005
ORDER_NOTIONAL = 2.50
START_BALANCE = 100.0
MAX_OPEN_EXPOSURE_PCT = 0.20
MAX_MARKET_EXPOSURE_PCT = 0.05
MAX_SYMBOL_EXPOSURE_PCT = 0.10
MAX_CONCURRENT_POSITIONS = 8
MAX_DAILY_LOSS_PCT = 0.05
MAX_DRAWDOWN_PCT = 0.10
MIN_TIME_TO_EXPIRY_S = 3
COOLDOWN_S = 5
VOL_WINDOW_SECONDS = 300
VOL_PROXY_BAR_SECONDS = 60
VOL_RETURN_CLIP_BPS = 100.0
VOL_FLOOR = 1e-7
DIVERSIFIED_RISK_SLOTS = 8

SCENARIOS = {
    "history_price_optimistic": {"slippage": 0.0, "order_notional": 2.50},
    "configured_0.003_per_share": {"slippage": 0.003, "order_notional": 2.50},
    "configured_0.003_small_notional": {"slippage": 0.003, "order_notional": 1.00},
    "stress_0.005_per_share": {"slippage": 0.005, "order_notional": 2.50},
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


def post_json(url: str, payload: dict[str, Any], attempts: int = 5) -> Any:
    last: Exception | None = None
    body = json.dumps(payload).encode("utf-8")
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "hizlitrade-fresh-backtest/1.0",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
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


def _cache_path(cache_dir: Path | None, namespace: str, key: str) -> Path | None:
    if cache_dir is None:
        return None
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return cache_dir / namespace / f"{digest}.json"


def _read_cache(cache_dir: Path | None, namespace: str, key: str) -> Any | None:
    path = _cache_path(cache_dir, namespace, key)
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("key") != key:
        return None
    return payload.get("value")


def _write_cache(cache_dir: Path | None, namespace: str, key: str, value: Any) -> None:
    path = _cache_path(cache_dir, namespace, key)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps({"key": key, "value": value}, separators=(",", ":")),
        encoding="utf-8",
    )
    tmp.replace(path)


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


def gamma_markets_for_slugs(
    slugs: set[str],
    *,
    batch_size: int = 40,
    cache_dir: Path | None = None,
    workers: int = 12,
) -> dict[str, dict[str, Any]]:
    """Fetch exact Gamma market slugs with bounded concurrency and settled cache.

    Only markets that are already closed and have a final binary winner are
    persisted. That makes cached Gamma rows immutable research evidence rather
    than a stale snapshot of a still-live market.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")

    targets = tuple(sorted({slug for slug in slugs if slug}))
    found: dict[str, dict[str, Any]] = {}

    for slug in targets:
        cached = _read_cache(cache_dir, "gamma", slug)
        if (
            isinstance(cached, dict)
            and str(cached.get("slug") or "") == slug
            and bool(cached.get("closed"))
            and settled_winner(cached) is not None
        ):
            found[slug] = cached

    remaining = tuple(slug for slug in targets if slug not in found)
    batches = [
        remaining[offset : offset + batch_size]
        for offset in range(0, len(remaining), batch_size)
    ]

    def fetch_batch(batch: tuple[str, ...]) -> dict[str, dict[str, Any]]:
        params: list[tuple[str, str]] = [("slug", slug) for slug in batch]
        params.append(("limit", str(max(len(batch), 1))))
        url = "https://gamma-api.polymarket.com/events?" + urllib.parse.urlencode(params)
        try:
            data = fetch_json(url)
        except Exception:
            data = []
        batch_found: dict[str, dict[str, Any]] = {}
        if isinstance(data, list):
            batch_set = set(batch)
            for event in data:
                if not isinstance(event, dict):
                    continue
                markets = event.get("markets") or []
                if not isinstance(markets, list):
                    continue
                for market in markets:
                    if not isinstance(market, dict):
                        continue
                    slug = str(market.get("slug") or "")
                    if slug in batch_set:
                        batch_found[slug] = market
        return batch_found

    if batches:
        with ThreadPoolExecutor(max_workers=min(workers, len(batches))) as pool:
            futures = {pool.submit(fetch_batch, batch): batch for batch in batches}
            for future in as_completed(futures):
                try:
                    found.update(future.result())
                except Exception:
                    continue

    missing = [slug for slug in remaining if slug not in found]
    if missing:
        with ThreadPoolExecutor(max_workers=min(workers, len(missing))) as pool:
            futures = {
                pool.submit(gamma_market_for_slug, slug): slug
                for slug in missing
            }
            for future in as_completed(futures):
                slug = futures[future]
                try:
                    market = future.result()
                except Exception:
                    continue
                if market is not None:
                    found[slug] = market

    for slug, market in found.items():
        if bool(market.get("closed")) and settled_winner(market) is not None:
            _write_cache(cache_dir, "gamma", slug, market)
    return found

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


def _history_points(rows: object) -> list[tuple[int, float]]:
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


def instrument_history(instrument: str) -> list[tuple[int, float]]:
    params = urllib.parse.urlencode({"market": instrument, "interval": "max", "fidelity": 1})
    data = fetch_json("https://clob.polymarket.com/prices-history?" + params)
    rows = data.get("history") if isinstance(data, dict) else None
    return _history_points(rows)


def batch_instrument_histories(
    instruments: set[str],
    *,
    start_ts: int,
    end_ts: int,
    batch_size: int = 20,
    cache_dir: Path | None = None,
    workers: int = 12,
) -> dict[str, list[tuple[int, float]]]:
    """Fetch CLOB histories in parallel batches with immutable token caching.

    Callers provide token IDs from already-settled markets, so a non-empty
    historical token series is immutable and safe to reuse across wider audits.
    Missing or malformed rows are never synthesized.
    """
    if batch_size <= 0 or batch_size > 20:
        raise ValueError("batch_size must be between 1 and 20")
    if end_ts <= start_ts:
        raise ValueError("end_ts must be greater than start_ts")
    if workers <= 0:
        raise ValueError("workers must be positive")

    targets = tuple(sorted({token for token in instruments if token}))
    found: dict[str, list[tuple[int, float]]] = {}

    for token in targets:
        cached = _read_cache(cache_dir, "clob", token)
        if isinstance(cached, list):
            points = _history_points(
                [
                    {"t": row[0], "p": row[1]}
                    for row in cached
                    if isinstance(row, list | tuple) and len(row) == 2
                ]
            )
            if points:
                found[token] = points

    remaining = tuple(token for token in targets if token not in found)
    batches = [
        remaining[offset : offset + batch_size]
        for offset in range(0, len(remaining), batch_size)
    ]
    url = "https://clob.polymarket.com/batch-prices-history"

    def fetch_batch(batch: tuple[str, ...]) -> dict[str, list[tuple[int, float]]]:
        try:
            payload = post_json(
                url,
                {
                    "markets": list(batch),
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "fidelity": 1,
                },
            )
        except Exception:
            payload = {}
        history = payload.get("history") if isinstance(payload, dict) else None
        batch_found: dict[str, list[tuple[int, float]]] = {}
        if isinstance(history, dict):
            for token in batch:
                points = _history_points(history.get(token))
                if points:
                    batch_found[token] = points
        return batch_found

    if batches:
        with ThreadPoolExecutor(max_workers=min(workers, len(batches))) as pool:
            futures = {pool.submit(fetch_batch, batch): batch for batch in batches}
            for future in as_completed(futures):
                try:
                    found.update(future.result())
                except Exception:
                    continue

    missing = [token for token in remaining if token not in found]
    if missing:
        with ThreadPoolExecutor(max_workers=min(workers, len(missing))) as pool:
            futures = {
                pool.submit(instrument_history, token): token
                for token in missing
            }
            for future in as_completed(futures):
                token = futures[future]
                try:
                    points = future.result()
                except Exception:
                    continue
                if points:
                    found[token] = points

    for token, points in found.items():
        if points:
            _write_cache(cache_dir, "clob", token, [[ts, price] for ts, price in points])
    return found

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
        proxy_return_points = max(2, VOL_WINDOW_SECONDS // VOL_PROXY_BAR_SECONDS)
        returns: deque[float] = deque(maxlen=proxy_return_points)
        prev_t: int | None = None
        prev_p: float | None = None
        for ts, price in rows:
            if prev_t is not None and prev_p is not None and price > 0 and prev_p > 0:
                dt = ts - prev_t
                if dt > 0:
                    normalized = math.log(price / prev_p) / math.sqrt(dt)
                    clip = VOL_RETURN_CLIP_BPS / 10_000.0
                    returns.append(max(min(normalized, clip), -clip))
            if len(returns) < 2:
                sigma = VOL_FLOOR
            else:
                sigma = max(statistics.stdev(returns), VOL_FLOOR)
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
    moneyness_bps: float
    momentum_1m_bps: float | None
    momentum_3m_bps: float | None
    market_duration_ms: float


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


def fee_per_share(price: float) -> float:
    return FEE_RATE * max(price * (1.0 - price), 0.0)


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

    # A historical price observation is usable only at its own recorded
    # timestamp. Previously, taking the union of YES/NO timestamps and carrying
    # the other outcome's last price forward could manufacture an apparent edge
    # from a stale historical observation. Live HIZLITRADE rejects stale quotes,
    # so the proxy must not silently forward-fill them either.
    out: list[Candidate] = []
    for outcome, history in (("YES", yes), ("NO", no)):
        for ts, quote in history:
            if market.end - ts < MIN_TIME_TO_EXPIRY_S:
                continue
            if not (0 < quote < 1):
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
            moneyness_bps = (spot_px / strike - 1.0) * 10_000
            p_yes = fair_yes(spot_px, strike, sigma, market.end - ts)
            fair = p_yes if outcome == "YES" else 1.0 - p_yes
            fee = fee_per_share(quote)
            net = fair - quote - fee - SIGNAL_SLIPPAGE - SAFETY_MARGIN
            if net < MIN_NET_EDGE:
                continue
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
                    moneyness_bps=moneyness_bps,
                    momentum_1m_bps=momentum_1m_bps,
                    momentum_3m_bps=momentum_3m_bps,
                    market_duration_ms=float((market.end - market.start) * 1000),
                )
            )
    return out


def simulate(
    candidates: list[Candidate],
    execution_slippage: float,
    *,
    order_notional: float = ORDER_NOTIONAL,
) -> dict[str, Any]:
    cash = START_BALANCE
    start_equity = START_BALANCE
    peak_equity = START_BALANCE
    risk_day: int | None = None
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
        day = c.ts // 86_400
        if risk_day != day:
            risk_day = day
            start_equity = snapshot(c.ts)[0]
        key = (c.market_id, c.outcome)
        if c.ts - last_trade.get(key, -10**18) < COOLDOWN_S:
            rejections["cooldown"] += 1
            continue

        equity, open_exposure, market_exp, symbol_exp, position_count = snapshot(c.ts)
        if equity <= 0 or cash <= 0:
            rejections["no_capital"] += 1
            continue
        daily_loss = max(start_equity - equity, 0.0)
        if daily_loss / start_equity >= MAX_DAILY_LOSS_PCT:
            rejections["daily_loss_kill"] += 1
            continue
        drawdown = max(peak_equity - equity, 0.0)
        if peak_equity > 0 and drawdown / peak_equity >= MAX_DRAWDOWN_PCT:
            rejections["drawdown_kill"] += 1
            continue
        worst_cash = cash - order_notional
        worst_daily = max(start_equity - worst_cash, 0.0)
        if worst_daily / start_equity > MAX_DAILY_LOSS_PCT:
            rejections["daily_loss_budget_at_risk"] += 1
            continue
        worst_dd = max(peak_equity - worst_cash, 0.0)
        if peak_equity > 0 and worst_dd / peak_equity > MAX_DRAWDOWN_PCT:
            rejections["drawdown_budget_at_risk"] += 1
            continue
        if position_count >= MAX_CONCURRENT_POSITIONS:
            rejections["max_concurrent"] += 1
            continue
        if open_exposure + order_notional > equity * MAX_OPEN_EXPOSURE_PCT:
            rejections["open_exposure"] += 1
            continue
        if market_exp.get(c.market_id, 0.0) + order_notional > equity * MAX_MARKET_EXPOSURE_PCT:
            rejections["market_exposure"] += 1
            continue
        if symbol_exp.get(c.symbol, 0.0) + order_notional > equity * MAX_SYMBOL_EXPOSURE_PCT:
            rejections["symbol_exposure"] += 1
            continue
        if order_notional > cash:
            rejections["cash"] += 1
            continue

        fill = min(c.quote_price + execution_slippage, 0.999)
        if fill > c.quote_price + SIGNAL_SLIPPAGE + 1e-12:
            rejections["fok_price_limit"] += 1
            continue
        actual_fee = fee_per_share(fill)
        actual_net = c.fair - fill - actual_fee - SAFETY_MARGIN
        if actual_net < MIN_NET_EDGE:
            rejections["edge_decayed_before_execution"] += 1
            continue
        effective = fill + actual_fee
        shares = order_notional / effective
        cash -= order_notional
        positions.append(
            Position(
                market_slug=c.market_slug,
                market_id=c.market_id,
                symbol=c.symbol,
                outcome=c.outcome,
                end=c.end,
                winner=c.winner,
                shares=shares,
                cost_basis=order_notional,
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
    deployed = order_notional * len(settlements)
    by_symbol: dict[str, list[float]] = defaultdict(list)
    for x in settlements:
        by_symbol[x["symbol"]].append(x["pnl"])
    return {
        "execution_slippage_per_share": execution_slippage,
        "order_notional_usd": order_notional,
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
        actual_fee = fee_per_share(fill)
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
                oracle_basis_bps=None,
                prediction_spread=None,
                outcome=candidate.outcome,
                momentum_250ms_bps=None,
                momentum_3s_bps=candidate.momentum_3m_bps,
                book_imbalance=None,
                time_to_expiry_ms=float((candidate.end - candidate.ts) * 1000),
                signal_net_edge=actual_net,
                spot_dispersion_bps=None,
                volatility_sigma_per_sqrt_second=None,
                signal_executable_price=candidate.quote_price,
                market_duration_ms=candidate.market_duration_ms,
            )
        )
        seen.add(key)
    return samples


def _proxy_stage_summary(rows: list[SettledTrade]) -> dict[str, Any]:
    returns = [row.realized_return for row in rows]
    grouped: dict[str, list[SettledTrade]] = defaultdict(list)
    for row in rows:
        grouped[row.market_id].append(row)
    market_returns = []
    for market_rows in grouped.values():
        cost = sum(row.cost_basis_usd for row in market_rows)
        pnl = sum(row.realized_pnl_usd for row in market_rows)
        market_returns.append(pnl / cost if cost > 0 else 0.0)
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
        "return_stats": trade_return_summary(returns),
        "market_return_stats": trade_return_summary(market_returns),
    }


def _proxy_portfolio_summary(
    rows: list[SettledTrade],
    *,
    risk_budget_slots: int | None = None,
) -> dict[str, Any]:
    result = replay_portfolio(
        rows,
        PortfolioReplayConfig(
            initial_balance=START_BALANCE,
            max_open_exposure_pct=MAX_OPEN_EXPOSURE_PCT,
            max_market_exposure_pct=MAX_MARKET_EXPOSURE_PCT,
            max_symbol_exposure_pct=MAX_SYMBOL_EXPOSURE_PCT,
            max_concurrent_positions=MAX_CONCURRENT_POSITIONS,
            max_daily_loss_pct=MAX_DAILY_LOSS_PCT,
            max_drawdown_pct=MAX_DRAWDOWN_PCT,
            risk_budget_slots=risk_budget_slots,
        ),
    )
    summary = _proxy_stage_summary(list(result.accepted_trades))
    return {
        **summary,
        "final_cash_usd": result.final_cash,
        "portfolio_realized_pnl_usd": result.realized_pnl_usd,
        "max_drawdown_pct": result.max_drawdown_pct,
        "rejections": result.rejection_counts,
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
    market_by_id = {row.market_id: row for row in markets}
    config = AdaptivePolicyConfig(
        min_symbol_train_markets=max(5, min_train // 8),
        min_setup_train_markets=max(5, min_train // 8),
        min_ml_train_trades=max(30, min_train // 2),
    )
    fold_reports: list[dict[str, Any]] = []
    accumulated: dict[str, dict[tuple[str, str, int], SettledTrade]] = defaultdict(dict)

    for index, fold in enumerate(folds):
        train_rows = tuple(market_by_id[market_id] for market_id in fold.train_market_ids)
        test_rows = tuple(market_by_id[market_id] for market_id in fold.test_market_ids)
        train_ids = set(fold.train_market_ids)
        test_ids = set(fold.test_market_ids)
        if train_ids & test_ids:
            raise RuntimeError("proxy walk-forward market leakage detected")
        if train_rows and max(row.settled_ts_ns for row in train_rows) >= fold.test_start_ts_ns:
            raise RuntimeError("future settlement label leaked into proxy train window")
        train = [row for row in samples if row.market_id in train_ids]
        test = [row for row in samples if row.market_id in test_ids]
        policy = fit_policy(train, config)
        stages = evaluate_policy_stages(test, policy, config)
        for stage, rows in stages.items():
            bucket = accumulated[stage]
            for row in rows:
                bucket[(row.market_id, row.instrument, row.signal_ts_ns)] = row
        fold_reports.append(
            {
                "fold": index,
                "train_markets": len(train_ids),
                "test_markets": len(test_ids),
                "purged_train_markets": fold.purged_train_markets,
                "train_end_ts_ns": fold.train_end_ts_ns,
                "train_last_settled_ts_ns": fold.train_last_settled_ts_ns,
                "test_start_ts_ns": fold.test_start_ts_ns,
                "selected_symbols": list(policy.selected_symbols),
                "symbol_gate_available": policy.symbol_gate_available,
                "selected_setup_families": list(policy.selected_setup_families),
                "setup_gate_available": policy.setup_gate_available,
                "selected_regimes": list(policy.selected_regimes),
                "regime_gate_available": policy.regime_gate_available,
                "min_consensus": policy.min_consensus,
                "consensus_gate_available": policy.consensus_gate_available,
                "consensus_no_trade": policy.consensus_no_trade,
                "min_net_edge": policy.min_net_edge,
                "max_prediction_spread": policy.max_prediction_spread,
                "execution_gate_available": policy.execution_gate_available,
                "execution_no_trade": policy.execution_no_trade,
                "ml_gate_available": policy.ml_gate_available,
                "ml_no_trade": policy.ml_no_trade,
                "strict_edge_evidence_passed": policy.strict_edge_evidence_passed,
                "ml_threshold": (
                    policy.ml_model.threshold if policy.ml_model is not None else None
                ),
                "stages": {
                    stage: _proxy_stage_summary(rows)
                    for stage, rows in stages.items()
                },
                "portfolio_stages": {
                    stage: _proxy_portfolio_summary(rows)
                    for stage, rows in stages.items()
                },
                "diversified_portfolio_stages": {
                    stage: _proxy_portfolio_summary(
                        rows,
                        risk_budget_slots=DIVERSIFIED_RISK_SLOTS,
                    )
                    for stage, rows in stages.items()
                },
            }
        )

    combined_rows = {
        stage: list(rows.values())
        for stage, rows in accumulated.items()
    }
    combined = {
        stage: _proxy_stage_summary(rows)
        for stage, rows in combined_rows.items()
    }
    combined_portfolio = {
        stage: _proxy_portfolio_summary(rows)
        for stage, rows in combined_rows.items()
    }
    combined_diversified_portfolio = {
        stage: _proxy_portfolio_summary(
            rows,
            risk_budget_slots=DIVERSIFIED_RISK_SLOTS,
        )
        for stage, rows in combined_rows.items()
    }
    return {
        "qualified_for_diagnostic": bool(folds),
        "market_count": market_count,
        "sample_count": len(samples),
        "min_train_markets": min_train,
        "test_markets": test_size,
        "folds": fold_reports,
        "combined_oos_ablation": combined,
        "combined_oos_portfolio_ablation": combined_portfolio,
        "combined_oos_diversified_portfolio_ablation": combined_diversified_portfolio,
        "diversified_risk_slots": DIVERSIFIED_RISK_SLOTS,
        "notes": [
            "Every policy is fitted only on prior settled markets and frozen for the next OOS fold.",
            "Training labels that settle at or after the next test start are purged.",
            "Every stage is replayed again under the current paper-account risk envelope.",
            "A separate research-only portfolio replay divides the unchanged hard daily-loss budget across eight worst-case slots; it does not relax the daily loss cap.",
            "The proxy maps OKX 1m/3m momentum into the short-horizon momentum feature slots because sub-second historical spot is unavailable.",
            "Historical CLOB L2 spread and book imbalance are unavailable and are not fabricated.",
            "This diagnostic cannot satisfy the live-promotion replay or microstructure evidence gates.",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=12)
    ap.add_argument("--output", default="fresh_hizlitrade_backtest.json")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--cache-dir", default=".cache/hizlitrade-audit")
    args = ap.parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    cache_dir = Path(args.cache_dir)

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
    keys_by_slug = {
        f"{asset}-updown-{minutes}m-{start}": (asset, symbol, minutes, start)
        for asset, symbol, minutes, start in slugs
    }
    gamma_markets = gamma_markets_for_slugs(
        set(keys_by_slug),
        cache_dir=cache_dir,
        workers=args.workers,
    )
    for slug, key in keys_by_slug.items():
        market = gamma_markets.get(slug)
        if market is None:
            discovery_failures["not_found"] += 1
            continue
        winner = settled_winner(market)
        if winner is None or not bool(market.get("closed")):
            discovery_failures["not_settled"] += 1
            continue
        discovered.append((key, market))

    # Fetch one hour of warmup beyond the requested horizon. Historical OKX
    # proxy bars are 60s, so a 300s runtime window is approximated with five
    # time-normalized 1m returns; sub-second runtime parity is explicitly
    # unavailable in this public historical diagnostic.
    cutoff = now - (args.hours + 1) * 3600
    spots: dict[str, SpotSeries] = {}
    for _asset, (symbol, inst_id) in ASSETS.items():
        rows = okx_spot_history(inst_id, cutoff)
        if rows:
            spots[symbol] = SpotSeries.build(rows)

    markets: list[Market] = []
    history_failures: Counter[str] = Counter()
    history_tokens = {
        str(token)
        for _key, row in discovered
        for token in parse_json_list(row.get("clobTokenIds"))
        if str(token)
    }
    prediction_histories = batch_instrument_histories(
        history_tokens,
        start_ts=cutoff,
        end_ts=now,
        cache_dir=cache_dir,
        workers=args.workers,
    )

    def build_market(item: tuple[tuple[str, str, int, int], dict[str, Any]]) -> Market | None:
        (asset, symbol, minutes, start), row = item
        instruments = [str(x) for x in parse_json_list(row.get("clobTokenIds"))]
        if len(instruments) != 2:
            return None
        winner = settled_winner(row)
        if winner is None:
            return None
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
            yes_history=prediction_histories.get(instruments[0], []),
            no_history=prediction_histories.get(instruments[1], []),
            volume=volume,
        )

    for item in discovered:
        market = build_market(item)
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
        name: simulate(
            candidates,
            float(scenario["slippage"]),
            order_notional=float(scenario["order_notional"]),
        )
        for name, scenario in SCENARIOS.items()
    }
    research_samples = shadow_samples(
        candidates,
        float(SCENARIOS["configured_0.003_per_share"]["slippage"]),
    )
    adaptive_diagnostic = adaptive_proxy_backtest(research_samples)
    edge_quality_result = sequential_edge_quality_filter(
        research_samples,
        model=EdgeQualityModel(
            min_calibration_bin_samples=30,
            min_symbol_samples=30,
            min_setup_samples=30,
        ),
        min_net_edge=MIN_NET_EDGE,
    )
    edge_quality_rows = list(edge_quality_result.accepted_trades)
    edge_quality_diagnostic = {
        "method": "sequential_realized_settlement_edge_quality",
        "accepted": _proxy_stage_summary(edge_quality_rows),
        "portfolio": _proxy_portfolio_summary(edge_quality_rows),
        "diversified_portfolio": _proxy_portfolio_summary(
            edge_quality_rows,
            risk_budget_slots=DIVERSIFIED_RISK_SLOTS,
        ),
        "rejections": edge_quality_result.rejection_counts,
        "calibration_applied": edge_quality_result.calibration_applied,
        "observed_settlements": edge_quality_result.observed_settlements,
        "parameters": {
            "min_calibration_bin_samples": 30,
            "min_symbol_samples": 30,
            "min_setup_samples": 30,
            "min_net_edge": MIN_NET_EDGE,
        },
        "notes": [
            "Each signal sees only settlements whose labels were observable strictly before that signal timestamp.",
            "Calibration is one-sided and can only reduce fair probability.",
            "Negative symbol/setup vetoes require a negative one-sided 95% upper confidence bound.",
            "Counterfactual research settlements remain training evidence even when the deployment policy would have rejected the trade, avoiding a self-confirming blind spot.",
        ],
    }
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
            "prediction_history_tokens_requested": len(history_tokens),
            "prediction_history_tokens_loaded": len(prediction_histories),
            "candidate_signals": len(candidates),
            "audit_workers": args.workers,
            "cache_dir": str(cache_dir),
        },
        "strategy_parameters": {
            "fee_rate": FEE_RATE,
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
            "volatility_window_seconds": VOL_WINDOW_SECONDS,
            "volatility_proxy_bar_seconds": VOL_PROXY_BAR_SECONDS,
            "volatility_return_clip_bps": VOL_RETURN_CLIP_BPS,
        },
        "data_sources": {
            "prediction_market": "Polymarket Gamma + CLOB prices-history",
            "spot_proxy": "OKX 1m historical candle open",
            "strike_proxy": "OKX 1m open at the contract slug start epoch",
        },
        "limitations": [
            "Historical CLOB prices-history does not expose historical L2 depth or exact executable ask; this is not promotion-quality frozen replay.",
            "Historical prediction prices are never forward-filled across the opposite outcome's timestamps; only an instrument's own recorded observations can create candidates.",
            "The simulated daily-loss baseline resets by UTC day to match PaperBroker semantics.",
            "Chainlink 60s opening TWAP is proxied by OKX 1m open at the same start timestamp.",
            "One-minute historical sampling cannot reproduce the live 100ms execution latency or sub-minute feed ordering.",
            "The current eight-asset research universe is requested, but only markets with complete public history are scored.",
            "Historical current Chainlink 60s TWAP snapshots are unavailable in this proxy; oracle_basis_bps is left missing rather than substituting spot-vs-strike moneyness.",
            "Runtime uses 1s fixed volatility buckets; this proxy uses time-normalized 1m OKX bars over the same 300s window and cannot reproduce sub-second microstructure.",
            "The exact HIZLITRADE fair-value equation and current signal/risk thresholds are used, but the market-data microstructure is necessarily approximated.",
        ],
        "discovery_failures": dict(discovery_failures),
        "history_failures": dict(history_failures),
        "market_candidate_counts": market_candidate_counts,
        "scenarios": scenario_results,
        "shadow_research_samples": len(research_samples),
        "adaptive_proxy": adaptive_diagnostic,
        "edge_quality_proxy": edge_quality_diagnostic,
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
    quality_summary = edge_quality_diagnostic["accepted"]
    quality_portfolio = edge_quality_diagnostic["portfolio"]
    print(
        "edge_quality",
        "trades=", quality_summary["trades"],
        "markets=", quality_summary["unique_markets"],
        "win_rate=", round(quality_summary["win_rate"] * 100, 2),
        "pnl=", round(quality_summary["realized_pnl_usd"], 4),
        "market_ci95_low=", quality_summary["market_return_stats"].get("mean_ci95_low"),
        "rejections=", edge_quality_diagnostic["rejections"],
        "calibration_applied=", edge_quality_diagnostic["calibration_applied"],
        "observed_settlements=", edge_quality_diagnostic["observed_settlements"],
    )
    print(
        "edge_quality_portfolio",
        "trades=", quality_portfolio["trades"],
        "pnl=", round(quality_portfolio["portfolio_realized_pnl_usd"], 4),
        "final_cash=", round(quality_portfolio["final_cash_usd"], 4),
        "max_dd=", round(quality_portfolio["max_drawdown_pct"] * 100, 3),
        "market_pf=", quality_portfolio["market_return_stats"].get("profit_factor"),
        "rejections=", quality_portfolio["rejections"],
    )
    for stage, summary in adaptive_diagnostic.get("combined_oos_ablation", {}).items():
        print(
            "adaptive",
            stage,
            "trades=", summary["trades"],
            "markets=", summary["unique_markets"],
            "win_rate=", round(summary["win_rate"] * 100, 2),
            "pnl=", round(summary["realized_pnl_usd"], 4),
            "mean_return=", summary["return_stats"].get("mean"),
            "market_ci95_low=", summary["market_return_stats"].get("mean_ci95_low"),
        )
    for stage, summary in adaptive_diagnostic.get(
        "combined_oos_diversified_portfolio_ablation", {}
    ).items():
        print(
            "adaptive_diversified_portfolio",
            stage,
            "trades=", summary["trades"],
            "markets=", summary["unique_markets"],
            "win_rate=", round(summary["win_rate"] * 100, 2),
            "pnl=", round(summary["portfolio_realized_pnl_usd"], 4),
            "final_cash=", round(summary["final_cash_usd"], 4),
            "max_dd=", round(summary["max_drawdown_pct"] * 100, 3),
            "market_pf=", summary["market_return_stats"].get("profit_factor"),
            "market_ci95_low=", summary["market_return_stats"].get("mean_ci95_low"),
            "rejections=", summary["rejections"],
        )

    for stage, summary in adaptive_diagnostic.get(
        "combined_oos_portfolio_ablation", {}
    ).items():
        print(
            "adaptive_portfolio",
            stage,
            "trades=", summary["trades"],
            "markets=", summary["unique_markets"],
            "win_rate=", round(summary["win_rate"] * 100, 2),
            "pnl=", round(summary["portfolio_realized_pnl_usd"], 4),
            "final_cash=", round(summary["final_cash_usd"], 4),
            "max_dd=", round(summary["max_drawdown_pct"] * 100, 3),
            "market_pf=", summary["market_return_stats"].get("profit_factor"),
            "market_ci95_low=", summary["market_return_stats"].get("mean_ci95_low"),
            "rejections=", summary["rejections"],
        )


if __name__ == "__main__":
    main()
