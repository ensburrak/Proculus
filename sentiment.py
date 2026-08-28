from core.exceptions import BEST_EFFORT_EXCEPTIONS
import ccxt
import logging
import requests
import json
import time
from pathlib import Path

from atomic_io import safe_write_json, safe_read_json

log = logging.getLogger(__name__)

"""
This module collects various sentiment data sources used by the trading bot.
In addition to OKX funding and open interest data and the Fear & Greed index,
this file now includes a social sentiment component. The social sentiment score
is computed from a local metrics file (metrics/social_sentiment.json) which
aggregates signals such as Twitter mentions, Reddit threads and news sentiment.
Values in that file should be in the range [-1, 1] where -1 is very bearish
and +1 is very bullish. These values are mapped to [0, 1] internally. If the
file is missing or malformed the social sentiment contribution defaults to
neutral (0.5).
"""

# Path to the social sentiment metrics file. If you add your own data
# pipeline for social media sentiment, write the results into this file
# using keys 'tweet_sentiment', 'reddit_sentiment' and 'news_sentiment'.
SOCIAL_SENTIMENT_FILE = Path("metrics/social_sentiment.json")
OI_CACHE_FILE = Path("data/oi_cache.json")

def get_social_sentiment() -> dict:
    """
    Reads social sentiment values from the metrics/social_sentiment.json file.
    The file is expected to contain keys like 'tweet_sentiment',
    'reddit_sentiment' and 'news_sentiment' with values in the range [-1, 1].
    Returns an empty dict if the file does not exist or cannot be parsed.
    """
    try:
        if SOCIAL_SENTIMENT_FILE.exists():
            txt = SOCIAL_SENTIMENT_FILE.read_text(encoding="utf-8").strip()
            if not txt:
                return {}
            data = json.loads(txt)
            if isinstance(data, dict):
                return data
    except BEST_EFFORT_EXCEPTIONS as _exc:
        logging.getLogger(__name__).debug("Suppressed in get_social_sentiment: %s", _exc)
    return {}

def _compute_social_score(data: dict) -> float:
    """
    Converts raw social sentiment values into a 0–1 score.
    Accepts a dictionary with sentiment values keyed by 'tweet_sentiment',
    'reddit_sentiment' and 'news_sentiment'. Each value should be in [-1, 1].
    Values outside that range are clamped. The returned score is the
    arithmetic mean of the mapped values (bearish -1 maps to 0.0, bullish +1
    maps to 1.0). If no valid values are present, returns 0.5 (neutral).
    """
    if not isinstance(data, dict):
        return 0.5
    scores = []
    for key in ("tweet_sentiment", "reddit_sentiment", "news_sentiment"):
        v = data.get(key)
        try:
            f = float(v)
            if f > 1.0:
                f = 1.0
            elif f < -1.0:
                f = -1.0
            scores.append((f + 1.0) / 2.0)
        except BEST_EFFORT_EXCEPTIONS:
            continue
    if scores:
        avg = sum(scores) / len(scores)
        # Clamp to [0,1]
        if avg < 0.0:
            avg = 0.0
        if avg > 1.0:
            avg = 1.0
        return avg
    return 0.5

def normalize_sentiment(value: float) -> float:
    """Normalize any sentiment value to [-1, +1] range.

    Use this as the single normalization point for all sentiment values
    throughout the codebase (Z-32).
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(-1.0, min(1.0, v))


# --- 1. OKX API Verileri ---

def get_okx_funding_and_position_data(exchange, symbol='BTC/USDT'):
    """
    OKX'ten Fonlama Oranini ve open Pozisyonlarin buyuklugunu ceker.
    """
    sentiment_data = {}
    
    # DUZELTME: OKX Swap piyasasi icin dogru symbol formatini olustur.
    # ccxt'nin birlesik formati 'BASE/QUOTE:QUOTE' seklindedir (orn: 'BTC/USDT:USDT')
    swap_symbol = symbol
    if ':' not in symbol and '/USDT' in symbol:
        # Gelen symbol 'BTC/USDT' ise, onu 'BTC/USDT:USDT' formatina cevir.
        swap_symbol = symbol.replace('/USDT', '/USDT:USDT')
    
    try:
        # a) Fonlama Orani (Funding Rate) Cekme - Duzeltilmis symbol ile
        funding_rate_data = exchange.fetch_funding_rate(swap_symbol)
        sentiment_data['funding_rate'] = funding_rate_data.get('fundingRate')
        
    except BEST_EFFORT_EXCEPTIONS as e:
        sentiment_data['funding_rate'] = None
        if 'swap markets' in str(e):
            log.warning("OKX Fonlama Orani (%s) cekilemedi: Swap piyasasi yetkisi eksik olabilir.", swap_symbol)
        else:
            log.warning("OKX Fonlama Orani (%s) cekilemedi: %s", swap_symbol, e)


    # b) open Pozisyon Buyurlugu (Open Interest) Cekme - Duzeltilmis symbol ile
    try:
        open_interest_data = exchange.fetch_open_interest(swap_symbol)
        
        if open_interest_data:
            sentiment_data['open_interest'] = open_interest_data.get('openInterestAmount')
        else:
            sentiment_data['open_interest'] = None
            
    except BEST_EFFORT_EXCEPTIONS as e:
        sentiment_data['open_interest'] = None
        if 'contract markets' in str(e):
            log.warning("OKX open Pozisyon (%s) cekilemedi: Kontrat piyasasi yetkisi eksik olabilir.", swap_symbol)
        else:
            log.warning("OKX open Pozisyon (%s) cekilemedi: %s", swap_symbol, e)
        
    return sentiment_data

# --- 2. Harici Sentiment Verileri ---
def get_fear_greed_index():
    """
    Alternative.me API'sini kullanarak Korku ve Acgozluluk Endeksi'ni ceker.
    """
    URL = "https://api.alternative.me/fng/?limit=1"
    
    try:
        response = requests.get(URL, timeout=10)
        response.raise_for_status() # HTTP errorlarini yakala
        data = response.json()
        
        if data and 'data' in data and data['data']:
            return {
                'fng_value': int(data['data'][0]['value']),
                'fng_class': data['data'][0]['value_classification']
            }
            
    except requests.exceptions.RequestException as e:
        log.warning("Fear & Greed Index API errorsi: %s", e)
        return {'fng_value': 50, 'fng_class': 'Neutral (Default)'}

def _compute_oi_change(current_oi, symbol: str):
    """Compute OI percentage change by comparing against the cached previous value.

    Stores the current OI value (with timestamp) into ``data/oi_cache.json``
    and returns the percentage change relative to the previously cached value.
    Returns ``None`` when there is no previous value to compare against or
    when the current OI is not a valid positive number.
    """
    if current_oi is None:
        return None
    try:
        oi_now = float(current_oi)
        if oi_now <= 0:
            return None
    except (TypeError, ValueError):
        return None

    oi_change = None
    try:
        cache = safe_read_json(OI_CACHE_FILE, default={})
        if not isinstance(cache, dict):
            cache = {}

        prev_entry = cache.get(symbol)
        if isinstance(prev_entry, dict):
            prev_oi = prev_entry.get("oi")
            if prev_oi is not None:
                prev_oi_f = float(prev_oi)
                if prev_oi_f > 0:
                    oi_change = ((oi_now - prev_oi_f) / prev_oi_f) * 100.0

        # Update cache with the current value
        cache[symbol] = {"oi": oi_now, "ts": int(time.time())}
        OI_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        safe_write_json(OI_CACHE_FILE, cache)
    except BEST_EFFORT_EXCEPTIONS as exc:
        log.debug("OI cache read/write error: %s", exc)

    return oi_change


# --- 3. Ana Fonksiyon (Tum Verileri Toplama) ---
def get_combined_sentiment_data(exchange, symbol):
    """
    Gather sentiment data from multiple sources and merge into a single dict.

    This function now aggregates:
      * OKX funding rate and open interest via get_okx_funding_and_position_data.
      * Fear & Greed Index via get_fear_greed_index.
      * Social sentiment via get_social_sentiment (tweet, reddit and news scores).

    Social sentiment values should be in [-1, 1] range in metrics/social_sentiment.json.
    They are mapped to a combined 'social_score' in [0, 1]. If the file is
    missing, the social score defaults to 0.5 and individual keys are None.
    A placeholder 'oi_change' key is also included for future open interest delta.
    """
    log.info("Piyasa duyarliligi (Sentiment) verileri cekiliyor...")

    okx_sentiment = get_okx_funding_and_position_data(exchange, symbol)
    fng_sentiment = get_fear_greed_index() or {}
    social_raw = get_social_sentiment()
    social_score = _compute_social_score(social_raw)

    # Merge all sources into a single dict.  If the Fear & Greed API returns
    # 'fng_value' we also set a legacy 'fear_greed' key for backward
    # compatibility with sentiment scoring logic.
    combined_sentiment = {
        'timestamp': int(time.time()),
        'symbol': symbol,
        **(okx_sentiment or {}),
        **fng_sentiment,
        'tweet_sentiment': social_raw.get('tweet_sentiment') if isinstance(social_raw, dict) else None,
        'reddit_sentiment': social_raw.get('reddit_sentiment') if isinstance(social_raw, dict) else None,
        'news_sentiment': social_raw.get('news_sentiment') if isinstance(social_raw, dict) else None,
        'social_score': social_score,
        'oi_change': _compute_oi_change(okx_sentiment.get('open_interest'), symbol),
    }
    # If a fear & greed value exists under fng_value, copy it to fear_greed
    try:
        if 'fng_value' in combined_sentiment and 'fear_greed' not in combined_sentiment:
            fv = combined_sentiment['fng_value']
            # Ensure numeric
            fv_float = float(fv)
            combined_sentiment['fear_greed'] = fv_float
    except BEST_EFFORT_EXCEPTIONS as _exc:
        logging.getLogger(__name__).debug("Suppressed in get_combined_sentiment_data: %s", _exc)
    return combined_sentiment


def get_lightweight_sentiment(symbol: str = "BTC/USDT") -> dict:
    """Assemble sentiment data from cached files without exchange calls.

    This function is safe to call from the async runtime loop because it
    does not perform any exchange API calls.  It reads:
      - Social sentiment from ``metrics/social_sentiment.json``
      - Fear & Greed index from the REST API (HTTP, non-blocking enough)
      - OI change from ``data/oi_cache.json`` (previously cached)

    The returned dict uses the same key schema expected by
    ``decision.sent_score.sent_score()``:
      fear_greed, funding, oi_change, social_score, tweet_sentiment,
      reddit_sentiment, news_sentiment, social_available.
    """
    senti: dict = {}
    try:
        # Social sentiment (file-based)
        social_raw = get_social_sentiment()
        social_score = _compute_social_score(social_raw)
        senti["tweet_sentiment"] = social_raw.get("tweet_sentiment") if isinstance(social_raw, dict) else None
        senti["reddit_sentiment"] = social_raw.get("reddit_sentiment") if isinstance(social_raw, dict) else None
        senti["news_sentiment"] = social_raw.get("news_sentiment") if isinstance(social_raw, dict) else None
        senti["social_score"] = social_score
        senti["social_available"] = social_score != 0.5 or bool(social_raw)

        # Fear & Greed (HTTP call, but lightweight)
        fng = get_fear_greed_index() or {}
        if fng.get("fng_value") is not None:
            senti["fear_greed"] = float(fng["fng_value"])

        # OI change from cache (no exchange call needed)
        oi_cache = safe_read_json(OI_CACHE_FILE, default={})
        if isinstance(oi_cache, dict):
            entry = oi_cache.get(symbol)
            if isinstance(entry, dict) and entry.get("oi") is not None:
                # The oi_change is only meaningful if there has been a
                # subsequent call to _compute_oi_change that stored a
                # newer value.  We store the last computed change if
                # available; otherwise leave it None so sent_score
                # treats it as unavailable.
                pass  # oi_change is computed by _compute_oi_change; cannot derive here

        # Funding rate: not available without exchange call, leave as None
        senti.setdefault("funding", None)
        senti.setdefault("oi_change", None)
    except BEST_EFFORT_EXCEPTIONS as exc:
        log.debug("get_lightweight_sentiment error: %s", exc)

    return senti
