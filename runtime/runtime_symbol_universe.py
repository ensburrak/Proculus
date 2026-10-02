from __future__ import annotations

import inspect
import time
from typing import Any, Mapping


async def _await_maybe(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _f(value: Any, default: float = 0.0) -> float:
    try:
        parsed=float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if parsed == parsed else float(default)


def _normalize_always_on(raw: Any) -> list[str]:
    out=[]
    for value in raw or []:
        token=str(value or "").strip()
        if not token:
            continue
        if ":" not in token and token.endswith("/USDT"):
            token=f"{token}:USDT"
        if token not in out:
            out.append(token)
    return out


def _ticker_for(tickers: Mapping[str, Any], symbol: str, market: Mapping[str, Any]) -> Mapping[str, Any]:
    direct=tickers.get(symbol)
    if isinstance(direct, Mapping):
        return direct
    market_id=str(market.get("id") or "")
    direct=tickers.get(market_id)
    return direct if isinstance(direct, Mapping) else {}


def _approx_quote_volume(ticker: Mapping[str, Any], last: float) -> float:
    quote=_f(ticker.get("quoteVolume"))
    if quote>0:
        return quote
    base=_f(ticker.get("baseVolume"))
    if base>0 and last>0:
        return base*last
    info=ticker.get("info") if isinstance(ticker.get("info"),Mapping) else {}
    vol_ccy=_f(info.get("volCcy24h"))
    return vol_ccy*last if vol_ccy>0 and last>0 else 0.0


def _listing_days(market: Mapping[str, Any], now_ms: int) -> float:
    info=market.get("info") if isinstance(market.get("info"),Mapping) else {}
    raw=info.get("listTime") or market.get("listTime")
    try:
        listed=int(raw or 0)
    except (TypeError, ValueError):
        listed=0
    if listed<=0:
        return 0.0
    return max(0.0,(now_ms-listed)/86_400_000.0)


def _funding_abs(funding_map: Mapping[str, Any], symbol: str, market: Mapping[str, Any]) -> float | None:
    row=funding_map.get(symbol)
    if not isinstance(row,Mapping):
        row=funding_map.get(str(market.get("id") or ""))
    if not isinstance(row,Mapping):
        return None
    value=row.get("fundingRate")
    if value is None:
        info=row.get("info") if isinstance(row.get("info"),Mapping) else {}
        value=info.get("fundingRate")
    try:
        return abs(float(value))
    except (TypeError, ValueError):
        return None


async def resolve_runtime_symbols(exchange: Any, config: Mapping[str, Any] | None) -> list[str]:
    cfg=dict(config or {})
    trade=cfg.get("trade_parameters") if isinstance(cfg.get("trade_parameters"),dict) else {}
    filters=cfg.get("symbol_filters") if isinstance(cfg.get("symbol_filters"),dict) else {}
    explicit=[str(v) for v in (trade.get("symbols") or []) if str(v or "").strip()]
    if explicit:
        return explicit

    always_on=_normalize_always_on(trade.get("always_on_symbols") or ["BTC/USDT","ETH/USDT","SOL/USDT"])
    if str(trade.get("symbol_source") or "").lower()!="okx_swap_all":
        return always_on

    try:
        markets=await _await_maybe(exchange.load_markets())
        tickers=await _await_maybe(exchange.fetch_tickers())
    except (AttributeError, RuntimeError, OSError, TypeError, ValueError):
        return always_on

    if not isinstance(markets,Mapping) or not isinstance(tickers,Mapping):
        return always_on

    quote=str(trade.get("symbol_quote") or "USDT").upper()
    excluded={str(v).upper() for v in (trade.get("symbol_exclude_bases") or [])}
    prefixes=tuple(str(v).upper() for v in (trade.get("symbol_exclude_prefixes") or []))
    min_volume=float(filters.get("min_24h_volume") or 50_000_000.0)
    max_spread_fraction=float(filters.get("max_spread_pct") or 0.05)/100.0
    min_listing_days=float(filters.get("min_listing_days") or 90.0)
    max_funding_abs=float(filters.get("max_funding_rate_abs") or 0.001)
    max_symbols=int((cfg.get("performance") or {}).get("max_symbols_per_loop",0) or 0)
    now_ms=int(time.time()*1000)

    prelim: list[tuple[str,Mapping[str,Any],float]]=[]
    for key,raw_market in markets.items():
        if not isinstance(raw_market,Mapping):
            continue
        market=raw_market
        symbol=str(market.get("symbol") or key or "")
        base=str(market.get("base") or "").upper()
        if not symbol or not base:
            continue
        if market.get("active") is False:
            continue
        if not bool(market.get("swap")):
            continue
        if market.get("linear") is False:
            continue
        if str(market.get("quote") or "").upper()!=quote:
            continue
        settle=str(market.get("settle") or quote).upper()
        if settle!=quote:
            continue
        if base in excluded or any(base.startswith(prefix) for prefix in prefixes):
            continue
        if _listing_days(market,now_ms)<min_listing_days:
            continue

        ticker=_ticker_for(tickers,symbol,market)
        bid=_f(ticker.get("bid")); ask=_f(ticker.get("ask")); last=_f(ticker.get("last"))
        if min(bid,ask,last)<=0:
            continue
        mid=(bid+ask)/2.0
        spread=(ask-bid)/mid if mid>0 else 999.0
        if spread>max_spread_fraction:
            continue
        volume=_approx_quote_volume(ticker,last)
        if volume<min_volume:
            continue
        prelim.append((symbol,market,volume))

    funding_map: Mapping[str,Any]={}
    fetch_rates=getattr(exchange,"fetch_funding_rates",None)
    if callable(fetch_rates) and prelim:
        try:
            symbols=[symbol for symbol,_,_ in prelim]
            payload=await _await_maybe(fetch_rates(symbols))
            if isinstance(payload,Mapping):
                funding_map=payload
        except (RuntimeError,OSError,TypeError,ValueError,AttributeError):
            funding_map={}

    eligible=[]
    for symbol,market,volume in prelim:
        funding=_funding_abs(funding_map,symbol,market)
        if funding is not None and funding>max_funding_abs:
            continue
        eligible.append((symbol,volume))

    eligible.sort(key=lambda row:(-row[1],row[0]))
    symbols=[symbol for symbol,_ in eligible]
    if max_symbols>0:
        symbols=symbols[:max_symbols]

    # Always-on majors are safety fallbacks, not a bypass around a healthy
    # discovered universe. Add only when they exist in the loaded swap markets.
    market_symbols={str(m.get("symbol") or k) for k,m in markets.items() if isinstance(m,Mapping)}
    for symbol in always_on:
        if symbol in market_symbols and symbol not in symbols:
            symbols.append(symbol)

    return symbols or always_on


__all__=["resolve_runtime_symbols"]
