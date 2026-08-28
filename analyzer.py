# -*- coding: utf-8 -*-
from __future__ import annotations

from core.exceptions import BEST_EFFORT_EXCEPTIONS
import ccxt
import ccxt.async_support as ccxt_async
import pandas as pd
import talib as ta
import numpy as np
import time
import asyncio
import inspect
import logging
import os
import weakref

from analysis.indicator_config import ANALYZER_PERIODS

# Logger kurulumu
log = logging.getLogger(__name__)
_ANALYZER_PERIODS = dict(ANALYZER_PERIODS)


def _timeframe_seconds(exchange, timeframe: str) -> int:
    """Resolve timeframe duration even when an exchange wrapper lacks ccxt helpers."""
    try:
        parser = getattr(exchange, "parse_timeframe", None)
        if callable(parser):
            parsed = int(float(parser(timeframe)))
            if parsed > 0:
                return parsed
    except BEST_EFFORT_EXCEPTIONS:
        pass
    raw = str(timeframe or "").strip().lower()
    if len(raw) < 2:
        return 60
    multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    try:
        return max(1, int(raw[:-1]) * multipliers[raw[-1]])
    except (KeyError, TypeError, ValueError):
        return 60

def _env_flag(key: str, default: str = "0") -> bool:
    return str(os.getenv(key, default)).strip().lower() in ("1", "true", "yes", "y", "on")

DEBUG_MTF = _env_flag("DEBUG_MTF", "0") or _env_flag("BOT_DEBUG", "0")
ENABLE_BINANCE_FALLBACK = _env_flag("ENABLE_BINANCE_FALLBACK", "1")  # Default: ON

# ---------------------------------------------------------------------------
# BINANCE FALLBACK - OKX API hatalarında yedek veri kaynağı
# ---------------------------------------------------------------------------
_binance_sync: ccxt.binance = None
_binance_async: ccxt_async.binance = None
_binance_async_lock: asyncio.Lock | None = None
_binance_async_clients_by_loop: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, ccxt_async.binance] | None = None
_binance_async_locks_by_loop: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] | None = None


def _get_binance_async_lock() -> asyncio.Lock:
    global _binance_async_lock, _binance_async_locks_by_loop
    loop = asyncio.get_running_loop()
    if _binance_async_locks_by_loop is None:
        _binance_async_locks_by_loop = weakref.WeakKeyDictionary()
    lock = _binance_async_locks_by_loop.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _binance_async_locks_by_loop[loop] = lock
    _binance_async_lock = lock
    return lock


def _get_binance_async_client(loop: asyncio.AbstractEventLoop) -> ccxt_async.binance | None:
    global _binance_async, _binance_async_clients_by_loop
    if _binance_async_clients_by_loop is None:
        _binance_async_clients_by_loop = weakref.WeakKeyDictionary()
    client = _binance_async_clients_by_loop.get(loop)
    _binance_async = client
    return client


def _set_binance_async_client(loop: asyncio.AbstractEventLoop, client: ccxt_async.binance) -> None:
    global _binance_async, _binance_async_clients_by_loop
    if _binance_async_clients_by_loop is None:
        _binance_async_clients_by_loop = weakref.WeakKeyDictionary()
    _binance_async_clients_by_loop[loop] = client
    _binance_async = client


async def _close_binance_async_exchange(client: object | None) -> None:
    if client is None:
        return
    close = getattr(client, "close", None)
    if not callable(close):
        return
    result = close()
    if inspect.isawaitable(result):
        await result


def normalize_symbol_for_binance_fallback(symbol: str) -> str:
    """Normalize OKX-style symbols into Binance-compatible market keys."""
    normalized = str(symbol or "").strip().upper().replace(" ", "")
    if not normalized:
        return ""
    if ":" in normalized:
        normalized = normalized.split(":", 1)[0]
    if normalized.endswith("-SWAP"):
        normalized = normalized[:-5]
    if "/" in normalized:
        return normalized
    if "-" in normalized:
        parts = normalized.split("-")
        if len(parts) >= 2 and parts[0] and parts[1]:
            return f"{parts[0]}/{parts[1]}"
    for quote in ("USDT", "USDC", "BUSD", "USD"):
        if normalized.endswith(quote) and len(normalized) > len(quote):
            return f"{normalized[:-len(quote)]}/{quote}"
    return normalized


def binance_supports_symbol(markets: dict | None, symbol: str) -> bool:
    if not isinstance(markets, dict) or not markets:
        return False
    normalized = normalize_symbol_for_binance_fallback(symbol)
    if not normalized:
        return False
    candidates = {
        normalized,
        f"{normalized}:USDT",
        f"{normalized}:USDC",
        f"{normalized}:BUSD",
    }
    return any(candidate in markets for candidate in candidates)

def get_binance_sync():
    """Singleton Binance sync client."""
    global _binance_sync
    if _binance_sync is None:
        _binance_sync = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}  # Futures için
        })
        try:
            _binance_sync.load_markets()
            log.info("[BINANCE_FALLBACK] Binance sync client initialized")
        except BEST_EFFORT_EXCEPTIONS as e:
            log.warning(f"[BINANCE_FALLBACK] Binance init failed: {e}")
    return _binance_sync

async def get_binance_async() -> ccxt_async.binance | None:
    """Singleton Binance async client."""
    loop = asyncio.get_running_loop()
    async with _get_binance_async_lock():
        client = _get_binance_async_client(loop)
        if client is None:
            client = ccxt_async.binance({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                    'adjustForTimeDifference': True,
                }
            })
            try:
                await client.load_markets()
                log.info("[BINANCE_FALLBACK] Binance async client initialized")
            except asyncio.CancelledError:
                await _close_binance_async_exchange(client)
                raise
            except Exception as e:
                await _close_binance_async_exchange(client)
                if not isinstance(e, BEST_EFFORT_EXCEPTIONS):
                    raise
                log.warning(f"[BINANCE_FALLBACK] Binance async init failed: {e}")
                return None
            _set_binance_async_client(loop, client)
        return client


async def close_binance_async_client() -> None:
    """Close the fallback client owned by the current runtime event loop."""
    global _binance_async, _binance_async_clients_by_loop

    loop = asyncio.get_running_loop()
    async with _get_binance_async_lock():
        client = None
        if _binance_async_clients_by_loop is not None:
            client = _binance_async_clients_by_loop.pop(loop, None)
        if _binance_async is client:
            _binance_async = None
    await _close_binance_async_exchange(client)

def _convert_symbol_for_binance(symbol: str) -> str:
    """OKX sembolünü Binance formatına çevir (BTC/USDT → BTCUSDT)."""
    return normalize_symbol_for_binance_fallback(symbol)


def _resolve_symbol(exchange, symbol: str):
    """
    OKX (ve diğer borsalar) için sembolü gerçek market anahtarına çevirir.
    Örnek:
      - "BTC/USDT" spot veya swap sembolü varsa onu kullanır
      - "BTC/USDT:USDT" / "BTC/USDT:USDC" varyantlarını dener
      - OKX swap id'si "BTC-USDT-SWAP" olan marketi bulup onun symbol alanını döndürür
    """
    mkts = exchange.markets or {}
    
    # [DEBUG] İlk 3 çağrı için print
    if not hasattr(_resolve_symbol, '_debug_count'):
        _resolve_symbol._debug_count = 0
    if _resolve_symbol._debug_count < 3:
        log.debug("[RESOLVE_DEBUG] exchange.markets sayısı: %d", len(mkts))
        if len(mkts) == 0:
            log.error("[RESOLVE_CRITICAL] MARKETS BOŞ! load_markets() çağrılmamış olabilir!")
        _resolve_symbol._debug_count += 1

    # 1) Doğrudan sembol
    if symbol in mkts:
        return symbol

    # 2) OKX tarzı "BTC/USDT:USDT" ve "BTC/USDT:USDC"
    for alt in (f"{symbol}:USDT", f"{symbol}:USDC"):
        if alt in mkts:
            return alt

    # 3) OKX SWAP id → symbol eşleşmesi (BTC/USDT → BTC-USDT-SWAP)
    try:
        target_id = f"{symbol.replace('/', '-')}-SWAP"
        for m in mkts.values():
            if m.get("id") == target_id:
                sym = m.get("symbol")
                if sym:
                    # symbol anahtar olarak varsa onu kullan
                    if sym in mkts:
                        return sym
                    return sym
    except BEST_EFFORT_EXCEPTIONS:
        pass

    # Hiçbiri bulunamadı
    return None

def _ensure_symbol_exists(exchange, symbol: str):
    """
    Eski fonksiyon korunuyor ama artık _resolve_symbol kullanıyor.
    Sadece varlık kontrolü yapar.
    """
    return _resolve_symbol(exchange, symbol) is not None


def fetch_and_analyze_data(exchange, symbol, timeframe='15m', limit=2000):
    """
    Çoklu borsa uyumlu teknik analiz (20+ indikatör).
    OKX market kontrolü, NaN-tolerans, temiz kolonlar.
    [GÜNCELLEME]: 300 mum limitini aşmak için Pagination Loop eklendi.
    """
    try:
        resolved_symbol = _resolve_symbol(exchange, symbol)
        if not resolved_symbol:
            return pd.DataFrame()

        all_ohlcv = []
        tf_seconds = exchange.parse_timeframe(timeframe)
        since = exchange.milliseconds() - (limit * tf_seconds * 1000)
        since -= (10 * tf_seconds * 1000)

        retry_count = 0
        
        while len(all_ohlcv) < limit:
            try:
                chunk_size = 500 
                ohlcv = exchange.fetch_ohlcv(resolved_symbol, timeframe, since=int(since), limit=chunk_size)
                
                if not ohlcv:
                    break
                
                all_ohlcv.extend(ohlcv)
                last_time = ohlcv[-1][0]
                since = last_time + 1
                
                if last_time >= exchange.milliseconds() - (tf_seconds * 1000):
                    break

                time.sleep(0.1) 
                
                if len(ohlcv) < chunk_size and len(all_ohlcv) < limit:
                    break

            except BEST_EFFORT_EXCEPTIONS as e:
                retry_count += 1
                if retry_count > 3:
                    break
                time.sleep(1)
        
        if not all_ohlcv:
            return pd.DataFrame()

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
        
        if len(df) > limit:
            df = df.iloc[-limit:]
            
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        volume = df['volume'].astype(float)

        # Momentum / Osilatör
        df['RSI'] = ta.RSI(close, timeperiod=int(_ANALYZER_PERIODS['rsi']))
        df['STOCH_K'], df['STOCH_D'] = ta.STOCH(
            high,
            low,
            close,
            int(_ANALYZER_PERIODS['stoch_k']),
            int(_ANALYZER_PERIODS['stoch_d']),
            int(_ANALYZER_PERIODS['stoch_slow']),
        )
        df['WILLR'] = ta.WILLR(high, low, close, timeperiod=int(_ANALYZER_PERIODS['willr']))
        df['MOM'] = ta.MOM(close, timeperiod=int(_ANALYZER_PERIODS['mom']))

        # Trend
        macd, macd_signal, macd_hist = ta.MACD(
            close,
            int(_ANALYZER_PERIODS['macd_fast']),
            int(_ANALYZER_PERIODS['macd_slow']),
            int(_ANALYZER_PERIODS['macd_signal']),
        )
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['MACD_Hist'] = macd_hist
        df['ADX'] = ta.ADX(high, low, close, timeperiod=int(_ANALYZER_PERIODS['adx']))
        df['SAR'] = ta.SAR(
            high,
            low,
            acceleration=float(_ANALYZER_PERIODS['sar_acceleration']),
            maximum=float(_ANALYZER_PERIODS['sar_maximum']),
        )

        # Volatilite
        upper, middle, lower = ta.BBANDS(close, timeperiod=int(_ANALYZER_PERIODS['bollinger']))
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = upper, middle, lower
        df['ATR'] = ta.ATR(high, low, close, timeperiod=int(_ANALYZER_PERIODS['atr']))
        df['NATR'] = ta.NATR(high, low, close, timeperiod=int(_ANALYZER_PERIODS['natr']))

        # Ichimoku
        try:
            period_tenkan = int(_ANALYZER_PERIODS['ichimoku_tenkan'])
            period_kijun = int(_ANALYZER_PERIODS['ichimoku_kijun'])
            period_span_b = int(_ANALYZER_PERIODS['ichimoku_span_b'])
            highest_high_tenkan = high.rolling(window=period_tenkan).max()
            lowest_low_tenkan = low.rolling(window=period_tenkan).min()
            tenkan_sen = (highest_high_tenkan + lowest_low_tenkan) / 2.0
            df['ICHIMOKU_TENKAN'] = tenkan_sen
            highest_high_kijun = high.rolling(window=period_kijun).max()
            lowest_low_kijun = low.rolling(window=period_kijun).min()
            kijun_sen = (highest_high_kijun + lowest_low_kijun) / 2.0
            df['ICHIMOKU_KIJUN'] = kijun_sen
            span_a = ((tenkan_sen + kijun_sen) / 2.0).shift(period_kijun)
            df['ICHIMOKU_SPAN_A'] = span_a
            highest_high_span_b = high.rolling(window=period_span_b).max()
            lowest_low_span_b = low.rolling(window=period_span_b).min()
            span_b = ((highest_high_span_b + lowest_low_span_b) / 2.0).shift(period_kijun)
            df['ICHIMOKU_SPAN_B'] = span_b
        except BEST_EFFORT_EXCEPTIONS:
            df['ICHIMOKU_TENKAN'] = np.nan
            df['ICHIMOKU_KIJUN'] = np.nan
            df['ICHIMOKU_SPAN_A'] = np.nan
            df['ICHIMOKU_SPAN_B'] = np.nan

        # Fibonacci
        try:
            fib_window = 50
            max_high = high.rolling(window=fib_window).max()
            min_low = low.rolling(window=fib_window).min()
            diff = max_high - min_low
            df['FIB_0_382'] = max_high - diff * 0.382
            df['FIB_0_618'] = max_high - diff * 0.618
        except BEST_EFFORT_EXCEPTIONS:
            df['FIB_0_382'] = np.nan
            df['FIB_0_618'] = np.nan

        # Hacim
        df['OBV'] = ta.OBV(close, volume)
        df['AD'] = ta.AD(high, low, close, volume)

        # Hareketli Ortalamalar
        df['SMA_10'] = ta.SMA(close, timeperiod=int(_ANALYZER_PERIODS['sma_fast']))
        df['EMA_20'] = ta.EMA(close, timeperiod=int(_ANALYZER_PERIODS['ema_fast']))
        df['EMA_50'] = ta.EMA(close, timeperiod=int(_ANALYZER_PERIODS['ema_slow']))
        df['WMA_50'] = ta.WMA(close, timeperiod=int(_ANALYZER_PERIODS['wma_fast']))
        df['WMA_100'] = ta.WMA(close, timeperiod=int(_ANALYZER_PERIODS['wma_slow']))
        df['HT_TRENDLINE'] = ta.HT_TRENDLINE(close)

        # NaN temizliği
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.ffill(inplace=True)
        df.dropna(inplace=True)

        try:
            q1, q3 = np.nanpercentile(df['close'], [5, 95])
            if q1 > 0 and q3 > 0 and q3 > q1:
                cap_low = q1 * 0.1
                cap_high = q3 * 10.0
                df = df[(df['close'] >= cap_low) & (df['close'] <= cap_high)]
        except BEST_EFFORT_EXCEPTIONS:
            pass
        
        df = analyze_data_with_zscore(df)
        return df

    except BEST_EFFORT_EXCEPTIONS as e:
        return pd.DataFrame()


def analyze_data_with_zscore(df: pd.DataFrame, window=50, temperature=2.0) -> pd.DataFrame:
    """Z-Score + Temperature Scaling."""
    if df.empty:
        return df

    targets = ['RSI', 'MOM', 'ADX', 'ATR', 'MACD_Hist', 'WILLR']
    
    for col in targets:
        if col not in df.columns:
            continue
        
        roll_mean = df[col].rolling(window=window).mean()
        roll_std = df[col].rolling(window=window).std()
        
        z_col_name = f"{col}_z"
        df[z_col_name] = (df[col] - roll_mean) / (roll_std + 1e-9)
        
        norm_col_name = f"{col}_norm"
        df[norm_col_name] = 1.0 / (1.0 + np.exp(-df[z_col_name] / temperature))
        
        df[z_col_name] = df[z_col_name].fillna(0.0)
        df[norm_col_name] = df[norm_col_name].fillna(0.5)

    return df


def get_multi_timeframe_analysis(exchange, symbol):
    """Çoklu timeframe analiz: 5m, 15m, 1h, 4h"""
    timeframes = ['5m', '15m', '1h', '4h']
    multi_data = {}
    for tf in timeframes:
        df = fetch_and_analyze_data(exchange, symbol, timeframe=tf, limit=2000)
        if df.empty:
            continue
        df.columns = [f"{col}_{tf}" for col in df.columns]
        multi_data[tf] = df
    return multi_data


async def _maybe_await(x):
    """Await helper for ccxt.async_support compatibility."""
    if inspect.isawaitable(x):
        return await x
    return x


def calc_vwap(high, low, close, volume):
    """
    [FIX] Daily Session Anchored VWAP hesaplama yardımcı fonksiyonu.
    Giriş verisi pd.Series olarak gelirse, index'teki timestamp'i gün (session)
    olarak kullanarak gün bazında VWAP hesaplar. Aksi halde düz kümülatif döner.
    """
    try:
        if isinstance(close, pd.Series) and isinstance(close.index, pd.DatetimeIndex):
            # Typical Price
            tp = (high + low + close) / 3.0
            tp_vol = tp * volume
            
            # Günlük periyoda (midnight UTC) göre grupla
            daily_group = tp.index.floor('D')
            
            # Grup bazında cumulative sum
            cum_tp_vol = tp_vol.groupby(daily_group).cumsum()
            cum_vol = volume.groupby(daily_group).cumsum()
            
            # Sifira bolme hatasini engelle
            cum_vol = cum_vol.replace(0, 1)
            return cum_tp_vol / cum_vol
        else:
            # Fallback: Eğer pandas indeksli_değilse (eski numpy bazlı array ise)
            typical_price = (high + low + close) / 3.0
            cumulative_tp_vol = np.cumsum(typical_price * volume)
            cumulative_vol = np.cumsum(volume)
            cumulative_vol = np.where(cumulative_vol == 0, 1, cumulative_vol)
            return cumulative_tp_vol / cumulative_vol
    except BEST_EFFORT_EXCEPTIONS:
        return np.full_like(close, np.nan)



def _compute_indicators_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add the same indicator columns as fetch_and_analyze_data_async to an OHLCV dataframe.

    Expects columns: timestamp, open, high, low, close, volume
    """
    if df is None or df.empty:
        return pd.DataFrame()
    try:
        close = df['close'].astype(float).values
        high = df['high'].astype(float).values
        low = df['low'].astype(float).values
        volume = df['volume'].astype(float).values

        # RSI
        df['RSI'] = ta.RSI(close, timeperiod=int(_ANALYZER_PERIODS['rsi']))

        # MACD
        macd, macd_signal, macd_hist = ta.MACD(
            close,
            fastperiod=int(_ANALYZER_PERIODS['macd_fast']),
            slowperiod=int(_ANALYZER_PERIODS['macd_slow']),
            signalperiod=int(_ANALYZER_PERIODS['macd_signal']),
        )
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['MACD_Hist'] = macd_hist

        # EMA
        df['EMA_20'] = ta.EMA(close, timeperiod=int(_ANALYZER_PERIODS['ema_fast']))
        df['EMA_50'] = ta.EMA(close, timeperiod=int(_ANALYZER_PERIODS['ema_slow']))

        # ATR
        df['ATR'] = ta.ATR(high, low, close, timeperiod=int(_ANALYZER_PERIODS['atr']))
        df['NATR'] = ta.NATR(high, low, close, timeperiod=int(_ANALYZER_PERIODS['natr']))

        # ADX
        df['ADX'] = ta.ADX(high, low, close, timeperiod=int(_ANALYZER_PERIODS['adx']))

        # Bollinger
        upper, middle, lower = ta.BBANDS(
            close,
            timeperiod=int(_ANALYZER_PERIODS['bollinger']),
            nbdevup=2,
            nbdevdn=2,
            matype=0,
        )
        df['BB_Upper'] = upper
        df['BB_Middle'] = middle
        df['BB_Lower'] = lower

        # VWAP
        df['VWAP'] = calc_vwap(high, low, close, volume)

        # Volume-derived confirmations used by v2 experts
        df['OBV'] = ta.OBV(close, volume)
        money_flow_mult = np.divide(
            ((df['close'] - df['low']) - (df['high'] - df['close'])),
            (df['high'] - df['low']).replace(0, np.nan),
        )
        money_flow_mult = money_flow_mult.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        money_flow_vol = money_flow_mult * df['volume']
        cmf_period = 20
        vol_sum = df['volume'].rolling(window=cmf_period).sum().replace(0, np.nan)
        df['CMF'] = (money_flow_vol.rolling(window=cmf_period).sum() / vol_sum).fillna(0.0)

        # pivot
        df['Pivot'] = (df['high'] + df['low'] + df['close']) / 3

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(how='all')

        # Fallback: ensure MACD columns exist even if TA-lib backend is missing or returns NaNs
        try:
            if 'MACD_Hist' not in df.columns or df['MACD_Hist'].isna().all():
                s = pd.Series(df['close'].astype(float).values)
                ema12 = s.ewm(span=12, adjust=False).mean()
                ema26 = s.ewm(span=26, adjust=False).mean()
                macd = ema12 - ema26
                signal = macd.ewm(span=9, adjust=False).mean()
                hist = macd - signal
                df['MACD'] = macd.values
                df['MACD_Signal'] = signal.values
                df['MACD_Hist'] = hist.values
        except BEST_EFFORT_EXCEPTIONS:
            pass
        return df
    except BEST_EFFORT_EXCEPTIONS:
        return df


async def _fetch_ohlcv_paginated(exchange, resolved_symbol: str, timeframe: str, limit: int = 2000):
    """Fetch OHLCV with pagination (since) to reliably fill long lookbacks."""
    import inspect, asyncio, ccxt
    out = []
    remaining = int(limit)
    # OKX maximum per call may be 100 or 300 depending on endpoint/tier.
    # We set chunk to 500. If exchange limits to 100, the loop will adapt 
    # (provided we don't break prematurely).
    chunk = min(500, max(100, limit))
    tf_ms = _timeframe_seconds(exchange, timeframe) * 1000
    try:
        now_ms = int(exchange.milliseconds())
    except BEST_EFFORT_EXCEPTIONS:
        now_ms = int(time.time() * 1000)
    # Start at the requested lookback window, not an extra chunk earlier.
    # CCXT returns candles forward from `since`; starting too far back can
    # fill the requested limit with old candles and make runtime data look
    # stale even when the exchange has current OHLCV.
    since = max(0, now_ms - (remaining * tf_ms))
    tries = 0
    last_seen_ts = None
    while remaining > 0:
        per = min(chunk, remaining)
        tries += 1
        if tries > 30: # fail-safe for infinite loops
            break
        try:
            if inspect.iscoroutinefunction(getattr(exchange, 'fetch_ohlcv', None)):
                data = await exchange.fetch_ohlcv(resolved_symbol, timeframe=timeframe, since=since, limit=per)
            else:
                data = await asyncio.to_thread(exchange.fetch_ohlcv, resolved_symbol, timeframe, since, per)
        except BEST_EFFORT_EXCEPTIONS as e:
            # retry a couple times on network/limit errors
            msg = str(e).lower()
            if isinstance(e, ccxt.BadSymbol) or 'bad symbol' in msg:
                break # Gracefully exit loop, return whatever we have (usually empty) to trigger fallback
            if tries < 4 and ('timeout' in msg or 'rate' in msg or '429' in msg or 'too many' in msg):
                await asyncio.sleep(0.6 * tries)
                continue
            break
        if not data:
            break
        out.extend(data)
        remaining = limit - len(out)
        # advance since by last candle + 1ms to avoid duplicates
        try:
            current_last_ts = int(data[-1][0])
            if last_seen_ts is not None and current_last_ts <= last_seen_ts:
                break
            last_seen_ts = current_last_ts
            since = current_last_ts + 1
        except BEST_EFFORT_EXCEPTIONS:
            since = None
        
        # [CHANGE] Do NOT break just because len(data) < per.
        # Captures cases where we requested 500 but exchange capped at 100.
        # We rely on 'remaining > 0' and 'if not data' to terminate.
        if len(data) == 0:
            break
        # small pacing
        await asyncio.sleep(0)
    # De-duplicate by timestamp
    if not out:
        return []
    seen = set()
    dedup = []
    for row in out:
        try:
            ts = int(row[0])
            if ts in seen:
                continue
            seen.add(ts)
        except BEST_EFFORT_EXCEPTIONS:
            pass
        dedup.append(row)
    dedup.sort(key=lambda r: r[0])
    # keep last 'limit'
    if len(dedup) > limit:
        dedup = dedup[-limit:]
    return dedup

# ==========================================================================
# [FIX] ASYNC VERSİYONU - _resolve_symbol EKLENDİ
# ==========================================================================
async def fetch_and_analyze_data_async(exchange, symbol, timeframe='15m', limit=2000):
    """Async-safe version of fetch_and_analyze_data.

    Works with both sync ccxt exchanges and ccxt.async_support exchanges.
    
    [FIX] _resolve_symbol eklendi - OKX SWAP sembollerini doğru çözümler.
    """
    try:
        # Optional debug counters (guarded)
        if not hasattr(fetch_and_analyze_data_async, '_call_count'):
            fetch_and_analyze_data_async._call_count = 0
        fetch_and_analyze_data_async._call_count += 1

        if fetch_and_analyze_data_async._call_count <= 5 and DEBUG_MTF:
            log.debug('[ASYNC_DEBUG] fetch_and_analyze_data_async çağrıldı: %s, tf=%s', symbol, timeframe)

        # [FIX] Sembolü OKX formatına çevir
        resolved_symbol = _resolve_symbol(exchange, symbol)
        if not resolved_symbol:
            log.warning(f'[ASYNC_FETCH] {symbol} için OKX market bulunamadı, atlanıyor.')
            return pd.DataFrame()

        # Debug log (ilk birkaç sembol için)
        if DEBUG_MTF:
            if not hasattr(fetch_and_analyze_data_async, '_debug_count'):
                fetch_and_analyze_data_async._debug_count = 0
            if fetch_and_analyze_data_async._debug_count < 5:
                log.info(f'[ASYNC_FETCH] {symbol} -> resolved: {resolved_symbol} | tf={timeframe}')
                fetch_and_analyze_data_async._debug_count += 1

        async def _fetch_ohlcv_with_retry():
            max_tries = 4
            delay = 0.6
            for attempt in range(1, max_tries + 1):
                try:
                    return await _fetch_ohlcv_paginated(exchange, resolved_symbol, timeframe=timeframe, limit=limit)
                except BEST_EFFORT_EXCEPTIONS as e:
                    msg = str(e).lower()
                    retryable = (
                        isinstance(e, (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable))
                        or 'timeout' in msg
                        or 'timed out' in msg
                        or 'connection reset' in msg
                        or 'temporarily unavailable' in msg
                        or '429' in msg
                        or 'rate limit' in msg
                        or 'too many requests' in msg
                        or 'frequency limit' in msg
                    )
                    
                    if isinstance(e, ccxt.BadSymbol) or 'bad symbol' in msg:
                        log.info(f"[ASYNC_FETCH] {symbol} OKX BadSymbol. Marking as empty for Binance fallback.")
                        return None # Return None specifically to trigger Binance fallback later

                    if retryable and attempt < max_tries:
                        await asyncio.sleep(delay)
                        delay = min(delay * 1.7, 6.0)
                        continue
                    # Final failure: log once
                    log.warning(f"[ASYNC_FETCH] {symbol} ({timeframe}) OKX fetch failed: {e}")
                    return None
            return None

        ohlcv = await _fetch_ohlcv_with_retry()
        
        # ---------------------------------------------------------------------------
        # BINANCE FALLBACK - OKX veri çekemezse Binance'dan dene
        # ---------------------------------------------------------------------------
        if (ohlcv is None or len(ohlcv) == 0) and ENABLE_BINANCE_FALLBACK:
            try:
                binance_symbol = _convert_symbol_for_binance(symbol)
                binance = await get_binance_async()
                if binance:
                    if binance_supports_symbol(getattr(binance, "markets", None), binance_symbol):
                        log.info(f"[BINANCE_FALLBACK] {symbol} için Binance'dan veri çekiliyor...")
                        ohlcv = await _fetch_ohlcv_paginated(binance, binance_symbol, timeframe=timeframe, limit=limit)
                        if ohlcv and len(ohlcv) > 0:
                            log.info(f"[BINANCE_FALLBACK] {symbol}: {len(ohlcv)} mum Binance'dan alındı")
                    else:
                        log.info(f"[BINANCE_FALLBACK] {symbol}: Binance market desteklemiyor, fallback atlandi")
            except BEST_EFFORT_EXCEPTIONS as e:
                log.warning(f"[BINANCE_FALLBACK] {symbol} Binance fallback da başarısız: {e}")

        if not ohlcv:
            log.warning(f"[ASYNC_FETCH] {symbol} ({timeframe}) boş OHLCV döndü (OKX + Binance)")
            return pd.DataFrame()

        def _compute_ta(ohlcv_data):
            df_comp = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_comp['timestamp'] = pd.to_datetime(df_comp['timestamp'], unit='ms')

            # [FIX] Vwap için set_index gereklidir (DatetimeIndex üzerinden çalışması için)
            df_comp.set_index('timestamp', inplace=True, drop=False)
            
            close_vals = df_comp['close'].astype(float)
            high_vals = df_comp['high'].astype(float)
            low_vals = df_comp['low'].astype(float)
            volume_vals = df_comp['volume'].astype(float)
            
            c_arr = close_vals.values
            h_arr = high_vals.values
            l_arr = low_vals.values
            v_arr = volume_vals.values

            # RSI
            df_comp['RSI'] = ta.RSI(c_arr, timeperiod=14)

            # MACD
            macd, macd_signal, macd_hist = ta.MACD(c_arr, fastperiod=12, slowperiod=26, signalperiod=9)
            df_comp['MACD'] = macd
            df_comp['MACD_Signal'] = macd_signal
            df_comp['MACD_Hist'] = macd_hist

            # EMA
            df_comp['EMA_20'] = ta.EMA(c_arr, timeperiod=20)
            df_comp['EMA_50'] = ta.EMA(c_arr, timeperiod=50)

            # ATR
            df_comp['ATR'] = ta.ATR(h_arr, l_arr, c_arr, timeperiod=14)
            df_comp['NATR'] = ta.NATR(h_arr, l_arr, c_arr, timeperiod=14)
            
            # ADX - trend gücü için önemli
            df_comp['ADX'] = ta.ADX(h_arr, l_arr, c_arr, timeperiod=14)

            # Bollinger
            upper, middle, lower = ta.BBANDS(c_arr, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            df_comp['BB_Upper'] = upper
            df_comp['BB_Middle'] = middle
            df_comp['BB_Lower'] = lower

            # VWAP [FIX] - Artık calc_vwap pd.Series alabilir (Anchored VWAP için DatetimeIndex gerekli)
            df_comp['VWAP'] = calc_vwap(high_vals, low_vals, close_vals, volume_vals)

            # Runtime expert chain confirmations
            df_comp['OBV'] = ta.OBV(c_arr, v_arr)
            spread = (df_comp['high'] - df_comp['low']).replace(0, np.nan)
            mf_mult = (((df_comp['close'] - df_comp['low']) - (df_comp['high'] - df_comp['close'])) / spread)
            mf_mult = mf_mult.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            mf_vol = mf_mult * df_comp['volume']
            cmf_period = 20
            vol_sum = df_comp['volume'].rolling(window=cmf_period).sum().replace(0, np.nan)
            df_comp['CMF'] = (mf_vol.rolling(window=cmf_period).sum() / vol_sum).fillna(0.0)

            # pivot
            df_comp['Pivot'] = (high_vals + low_vals + close_vals) / 3

            # nan cleanup
            df_comp = df_comp.replace([np.inf, -np.inf], np.nan)
            df_comp = df_comp.dropna(how='all')
            
            # Indexi resetle ki orijinal kodla uyumlu kalsın (timestamp kolonu var)
            df_comp.reset_index(drop=True, inplace=True)
            return df_comp

        # [FIX] TA hesaplamalarını thread'e yolla (Event Loop dondurmasını önler)
        df = await asyncio.to_thread(_compute_ta, ohlcv)
        
        # Başarı logu (ilk 3 sembol için)
        if not hasattr(fetch_and_analyze_data_async, '_success_count'):
            fetch_and_analyze_data_async._success_count = 0
        if fetch_and_analyze_data_async._success_count < 3:
            log.info(f"[ASYNC_SUCCESS] {symbol} | {timeframe}: {len(df)} satır, columns={list(df.columns)[:5]}...")
            fetch_and_analyze_data_async._success_count += 1
        
        return df
    except BEST_EFFORT_EXCEPTIONS as e:
        import traceback
        tb = traceback.format_exc()
        if DEBUG_MTF:
            log.error("[ASYNC_ERROR] %s (%s): %s: %s", symbol, timeframe, type(e).__name__, e)
            log.debug("[ASYNC_TRACEBACK] %s", tb[:500])
        log.warning(f"[ASYNC_FETCH_ERROR] {symbol} ({timeframe}) analiz hatası: {e}")
        # Full traceback only in debug mode
        if DEBUG_MTF:
            log.debug(tb)
        return pd.DataFrame()


async def get_multi_timeframe_analysis_async(exchange, symbol, timeframes=None, limit=2000):
    """
    Fetch REAL historical data for multiple timeframes concurrently.
    [MODIFIED] Uses asyncio.gather to fetch distinct 2000-candle histories for 15m, 1h, 4h, 12h.
    No resampling -> better long-term trend accuracy.
    """
    if not hasattr(get_multi_timeframe_analysis_async, '_call_count'):
        get_multi_timeframe_analysis_async._call_count = 0
    get_multi_timeframe_analysis_async._call_count += 1

    # Default set includes 12h for deep trend analysis
    target_tfs = timeframes or ['15m', '1h', '4h', '12h']
    target_tfs = [str(t) for t in target_tfs if t]

    multi_data = {}
    
    # Resolve symbol once
    resolved_symbol = _resolve_symbol(exchange, symbol)
    if not resolved_symbol:
        log.warning(f"[MTF_EMPTY] {symbol}: market resolve failed")
        return {}

    # 1. Prepare tasks for concurrent fetching
    tasks = []
    # We use fetch_and_analyze_data_async directly which already handles:
    # - Pagination (chunk=300 -> 2000 total)
    # - Indicator computation
    # - Error handling
    for tf in target_tfs:
        tasks.append(fetch_and_analyze_data_async(
            exchange=exchange, 
            symbol=symbol, 
            timeframe=tf, 
            limit=limit 
        ))

    # 2. Execute all fetches effectively in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 3. Process results
    for tf, res in zip(target_tfs, results):
        if isinstance(res, Exception):
            if DEBUG_MTF:
                log.warning(f"[MTF_FETCH_ERR] {symbol} {tf}: {res}")
            continue
            
        if res is None or res.empty:
            if tf == '15m': 
                log.warning(f"[MTF_EMPTY] {symbol} {tf} is empty")
            continue

        # Suffix columns (e.g. "RSI_15m", "EMA_1h") to fit the expected multi_data format
        df_suffixed = res.copy()
        df_suffixed.columns = [f"{c}_{tf}" for c in res.columns]
        multi_data[tf] = df_suffixed

    if not multi_data:
        log.warning(f"[MTF_EMPTY] {symbol}: no timeframe data returned")
        return {}

    return multi_data


def check_bbands_squeeze(multi_data, tf='1h', window=50, threshold_percent=0.80):
    if tf not in multi_data or multi_data[tf].empty:
        return False, "HATA: Analiz verisi yok."
    df = multi_data[tf].copy()
    upper_col = f'BB_Upper_{tf}'
    lower_col = f'BB_Lower_{tf}'
    
    if upper_col not in df.columns or lower_col not in df.columns:
        return False, f"BB kolonları ({upper_col}, {lower_col}) bulunamadı."

    df['BB_Width'] = df[upper_col] - df[lower_col]
    rolling_mean_width = df['BB_Width'].rolling(window=window).mean()
    if rolling_mean_width.empty or pd.isna(rolling_mean_width.iloc[-1]):
        return False, "Ortalama genişlik hesaplanamadı."
    current_width = df['BB_Width'].iloc[-1]
    average_width = rolling_mean_width.iloc[-1]
    is_squeezed = current_width < (average_width * threshold_percent)
    if is_squeezed:
        return True, "Bollinger SIKIŞMASI tespit edildi."
    return False, "Sıkışma yok."


def check_smart_take_profit(df, side='long'):
    try:
        macd_hist_col = next((c for c in df.columns if 'MACD_Hist' in c), None)
        
        if not macd_hist_col:
            return False, "MACD_Hist kolonu bulunamadı."

        if len(df) < 2:
            return False, "Yetersiz veri."
            
        last, prev = df.iloc[-1], df.iloc[-2]
        if side == 'long' and last[macd_hist_col] < 0 < prev[macd_hist_col]:
            return True, "MACD Histogram pozitiften negatife döndü → momentum zayıfladı."
        elif side == 'short' and last[macd_hist_col] > 0 > prev[macd_hist_col]:
            return True, "MACD Histogram negatife döndü → momentum zayıfladı."
    except BEST_EFFORT_EXCEPTIONS as e:
        return False, f"Analiz hatası: {e}"
    return False, "Aktif çıkış sinyali yok."


# ==========================================================================
# STANDALONE TEST
# ==========================================================================
if __name__ == "__main__":
    import ccxt
    import os
    from dotenv import load_dotenv
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    load_dotenv()
    
    print("=" * 60)
    print("ANALYZER.PY STANDALONE TEST")
    print("=" * 60)
    
    exchange = ccxt.okx({
        'apiKey': os.getenv('OKX_API_KEY'),
        'secret': os.getenv('OKX_API_SECRET'),
        'password': os.getenv('OKX_API_PASSPHRASE'),
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'},
    })
    
    use_testnet = os.getenv('OKX_USE_TESTNET', 'true').lower() == 'true'
    exchange.set_sandbox_mode(use_testnet)
    
    print(f"[TEST] Sandbox mode: {use_testnet}")
    
    exchange.load_markets()
    print(f"[TEST] {len(exchange.markets)} market yüklendi")
    
    test_symbol = "BTC/USDT"
    resolved = _resolve_symbol(exchange, test_symbol)
    print(f"[TEST] {test_symbol} -> resolved: {resolved}")
    
    if resolved:
        print(f"\n[TEST] Senkron fetch_and_analyze_data testi...")
        df_sync = fetch_and_analyze_data(exchange, test_symbol, timeframe='1h', limit=50)
        print(f"[TEST] Senkron sonuç: {len(df_sync)} satır")
        if not df_sync.empty:
            print(f"[TEST] Kolonlar: {list(df_sync.columns)[:10]}...")
        
        print(f"\n[TEST] Async get_multi_timeframe_analysis_async testi...")
        
        async def test_async():
            result = await get_multi_timeframe_analysis_async(exchange, test_symbol, timeframes=['5m', '15m', '1h', '4h'], limit=100)
            return result
        
        multi_data = asyncio.run(test_async())
        print(f"[TEST] Multi-timeframe sonuç: {len(multi_data)} timeframe")
        for tf, df in multi_data.items():
            print(f"  - {tf}: {len(df)} satır")
        
        if multi_data:
            print("\n✅ ASYNC FONKSİYONLAR ÇALIŞIYOR!")
        else:
            print("\n❌ ASYNC FONKSİYONLAR BOŞ DÖNDÜ!")
    else:
        print(f"\n❌ Sembol çözümlenemedi: {test_symbol}")
    
    print("\n" + "=" * 60)

