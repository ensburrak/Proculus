# -*- coding: utf-8 -*-
"""
chatgpt_client.py - Profesyonel ChatGPT İstemcisi
=================================================

[2026-01-14] Profesyonel seviye güncelleme.

Özellikler:
- Gelişmiş hata yönetimi ve retry stratejisi
- Rate limiting (API limitlerine uyum)
- Response caching (maliyet optimizasyonu)
- Structured output parsing
- Token kullanımı takibi
- Timeout yönetimi
- Circuit breaker pattern
"""

from core.exceptions import BEST_EFFORT_EXCEPTIONS
import os
import time
import json
import hashlib
import logging
import threading
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta, timezone

try:
    from runtime_paths import STATE_DIR
except (ImportError, RuntimeError, OSError):
    STATE_DIR = Path("state")


# OpenAI SDK
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    class _OpenAIFallbackModule:
        class APITimeoutError(Exception):
            pass

        class APIConnectionError(Exception):
            pass

        class RateLimitError(Exception):
            pass

        class APIStatusError(Exception):
            def __init__(self, *args, status_code: Optional[int] = None, **kwargs):
                super().__init__(*args)
                self.status_code = status_code

        class APIError(Exception):
            pass

    openai = _OpenAIFallbackModule()
    OPENAI_AVAILABLE = False
    OpenAI = None

# Logger
try:
    from logger import get_logger, log_event
    log = get_logger("chatgpt")
except ImportError:
    log = logging.getLogger("chatgpt")
    log.setLevel(logging.INFO)

    def log_event(logger, level, event, message, **context):
        del event, context
        logger.log(level, message)


# =============================================================================
# CONFIGURATION
# =============================================================================

OPENAI_MODEL = (os.getenv("OPENAI_MODEL", "") or "").strip()
DEFAULT_FILTER_MODEL = os.getenv("OPENAI_FILTER_MODEL", OPENAI_MODEL or "gpt-4o-mini")
DEFAULT_DECISION_MODEL = os.getenv("OPENAI_DECISION_MODEL", OPENAI_MODEL or "gpt-4o-mini")
API_KEY = os.getenv("OPENAI_API_KEY", "")


def _load_api_cost_limit_config() -> dict[str, Any]:
    try:
        import core.config_loader as config_loader

        cfg = config_loader.load_config()
        api_cost_limit = cfg.get("api_cost_limit") if isinstance(cfg, dict) else {}
        return api_cost_limit if isinstance(api_cost_limit, dict) else {}
    except BEST_EFFORT_EXCEPTIONS:
        return {}
    except (ImportError, TypeError, ValueError):
        return {}


def _resolve_daily_budget_config(default: float = 10.0) -> tuple[float, bool]:
    cfg = _load_api_cost_limit_config()
    disable_on_limit = bool(cfg.get("disable_llm_on_limit", True))
    if "CHATGPT_MAX_DAILY_BUDGET_USD" in os.environ:
        try:
            return float(os.getenv("CHATGPT_MAX_DAILY_BUDGET_USD", str(default))), disable_on_limit
        except ValueError:
            return float(default), disable_on_limit
    try:
        return float(cfg.get("daily_max_usd", default)), disable_on_limit
    except (TypeError, ValueError):
        return float(default), disable_on_limit

# Token maliyetleri (1K token başına USD)
TOKEN_COSTS = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-5-mini": {"input": 0.00025, "output": 0.002},
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class APIMetrics:
    """API kullanım metrikleri."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost_usd: float = 0.0
    last_call_time: float = 0.0
    circuit_breaker_trips: int = 0


@dataclass
class CachedResponse:
    """Önbelleğe alınmış yanıt."""
    response: Dict[str, Any]
    timestamp: float
    ttl_seconds: int = 300  # 5 dakika default


# =============================================================================
# PROFESSIONAL CHATGPT CLIENT
# =============================================================================

class ChatGPTClient:
    """
    Profesyonel ChatGPT istemcisi.
    
    Features:
    - Exponential backoff retry
    - Response caching
    - Rate limiting
    - Circuit breaker
    - Token tracking
    """
    
    # Rate limiting
    MIN_CALL_INTERVAL = 0.1  # Minimum 100ms between calls
    MAX_CALLS_PER_MINUTE = 60
    
    # Circuit breaker
    CIRCUIT_BREAKER_THRESHOLD = 5  # 5 ardışık hata
    CIRCUIT_BREAKER_RESET_TIME = 60  # 1 dakika
    
    # Cache
    CACHE_TTL = 300  # 5 dakika
    
    def __init__(
        self,
        filter_model: str = DEFAULT_FILTER_MODEL,
        decision_model: str = DEFAULT_DECISION_MODEL,
        request_timeout: int = 8,
        max_retries: int = 2,
        enable_cache: bool = True,
    ) -> None:
        """
        Args:
            filter_model: Sinyal filtreleme modeli
            decision_model: Karar modeli
            request_timeout: İstek timeout (saniye)
            max_retries: Maksimum retry sayısı
            enable_cache: Önbellek aktif mi
        """
        # Devre dışı bırakma kontrolü
        disable_env = os.getenv("CHATGPT_DISABLE", "0")
        self.enabled = disable_env not in ("1", "true", "True") and OPENAI_AVAILABLE and API_KEY
        
        self.filter_model = filter_model
        self.decision_model = decision_model
        self.max_retries = min(int(max_retries), 2)
        self.request_timeout = min(int(request_timeout), 10, max(1, 25 // (self.max_retries + 1)))
        self.total_request_budget_seconds = self.request_timeout * (self.max_retries + 1)
        self.enable_cache = enable_cache
        
        # Daily cost budget configuration (enforced fallback check)
        self.max_daily_budget_usd, self.disable_llm_on_budget_limit = _resolve_daily_budget_config()
        self._daily_cost_file = STATE_DIR / "llm_daily_cost.json"

        if self.enabled:
            self.client = OpenAI(api_key=API_KEY, timeout=self.request_timeout, max_retries=0)
            log_event(
                log,
                logging.INFO,
                "chatgpt_client_initialized",
                "[ChatGPT] Initialized",
                filter_model=filter_model,
                decision_model=decision_model,
            )
        else:
            self.client = None
            if not OPENAI_AVAILABLE:
                log.warning("[ChatGPT] OpenAI SDK not installed")
            elif not API_KEY:
                log.warning("[ChatGPT] OPENAI_API_KEY not set")
            else:
                log.warning("[ChatGPT] Disabled via environment")
        
        # Metrics
        self.metrics = APIMetrics()
        self._lock = threading.Lock()
        
        # Cache
        self._cache: Dict[str, CachedResponse] = {}
        
        # Circuit breaker
        self._consecutive_errors = 0
        self._circuit_open_until = 0.0
        
        # Rate limiting
        self._call_times: List[float] = []
        self.MAX_TOKENS_PER_MINUTE = int(os.getenv("CHATGPT_MAX_TOKENS_PER_MINUTE", "20000"))
        self._token_usage_window: List[tuple[float, int]] = []
    
    def _get_cache_key(self, model: str, messages: List[Dict]) -> str:
        """Önbellek anahtarı oluştur."""
        content = json.dumps({"model": model, "messages": messages}, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _check_cache(self, cache_key: str) -> Optional[str]:
        """Önbellekten yanıt kontrol et."""
        if not self.enable_cache:
            return None
        
        cached = self._cache.get(cache_key)
        if cached:
            if time.time() - cached.timestamp < cached.ttl_seconds:
                log_event(
                    log,
                    logging.DEBUG,
                    "chatgpt_cache_hit",
                    "[ChatGPT] Cache hit",
                    cache_key_prefix=cache_key[:8],
                )
                return cached.response
            else:
                del self._cache[cache_key]
        return None
    
    def _set_cache(self, cache_key: str, response: str) -> None:
        """Yanıtı önbelleğe al."""
        if self.enable_cache:
            self._cache[cache_key] = CachedResponse(
                response=response,
                timestamp=time.time(),
                ttl_seconds=self.CACHE_TTL
            )
    
    def _check_rate_limit(self) -> bool:
        """Rate limit kontrolü."""
        now = time.time()
        
        # Son 1 dakikadaki çağrıları filtrele
        self._call_times = [t for t in self._call_times if now - t < 60]
        
        if len(self._call_times) >= self.MAX_CALLS_PER_MINUTE:
            log_event(
                log,
                logging.WARNING,
                "chatgpt_rate_limit_reached",
                "[ChatGPT] Rate limit reached",
                calls_last_minute=len(self._call_times),
                limit=self.MAX_CALLS_PER_MINUTE,
            )
            return False
        
        # Minimum interval kontrolü
        if self._call_times and now - self._call_times[-1] < self.MIN_CALL_INTERVAL:
            time.sleep(self.MIN_CALL_INTERVAL)
        
        return True
    
    def _check_circuit_breaker(self) -> bool:
        """Circuit breaker kontrolü."""
        if self._circuit_open_until > time.time():
            log.warning("[ChatGPT] Circuit breaker OPEN - skipping call")
            return False
        return True
    
    def _record_success(self) -> None:
        """Başarılı çağrıyı kaydet."""
        self._consecutive_errors = 0
        self.metrics.successful_calls += 1
    
    def _record_failure(self) -> None:
        """Başarısız çağrıyı kaydet."""
        self._consecutive_errors += 1
        self.metrics.failed_calls += 1
        
        if self._consecutive_errors >= self.CIRCUIT_BREAKER_THRESHOLD:
            self._circuit_open_until = time.time() + self.CIRCUIT_BREAKER_RESET_TIME
            self.metrics.circuit_breaker_trips += 1
            log_event(
                log,
                logging.ERROR,
                "chatgpt_circuit_breaker_tripped",
                "[ChatGPT] Circuit breaker tripped",
                cooldown_seconds=self.CIRCUIT_BREAKER_RESET_TIME,
                consecutive_errors=self._consecutive_errors,
            )
    
    def _track_tokens(self, model: str, tokens_in: int, tokens_out: int) -> None:
        """Token kullanımını takip et."""
        total_tokens = int(tokens_in or 0) + int(tokens_out or 0)
        if not self._check_token_budget(total_tokens):
            raise RuntimeError("chatgpt token-per-minute budget exceeded")
        self.metrics.total_tokens_in += tokens_in
        self.metrics.total_tokens_out += tokens_out
        self._token_usage_window.append((time.time(), total_tokens))
        
        costs = TOKEN_COSTS.get(model, TOKEN_COSTS["gpt-4o-mini"])
        cost = (tokens_in / 1000 * costs["input"]) + (tokens_out / 1000 * costs["output"])
        self.metrics.total_cost_usd += cost
        
        # Persist daily cost
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._lock:
            self._increment_daily_cost(today_str, cost)

    def _check_token_budget(self, next_tokens: int) -> bool:
        """Return False when the per-minute token budget would be exceeded."""
        now = time.time()
        self._token_usage_window = [
            (ts, count) for ts, count in self._token_usage_window if now - float(ts) < 60.0
        ]
        used = sum(int(count) for _, count in self._token_usage_window)
        return used + max(0, int(next_tokens or 0)) <= int(self.MAX_TOKENS_PER_MINUTE)

    @staticmethod
    def _estimate_request_tokens(messages: List[Dict[str, str]], max_output_tokens: int) -> int:
        """Conservative preflight token estimate used before making a paid call."""
        serialized = json.dumps(messages, ensure_ascii=False, default=str)
        input_estimate = max(1, len(serialized.encode("utf-8")))
        return input_estimate + max(0, int(max_output_tokens or 0))

    def _get_daily_cost(self, today_str: str) -> float:
        """Read daily cost from persistent file, resetting if date has changed."""
        if not self._daily_cost_file.exists():
            return 0.0
        try:
            with open(self._daily_cost_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("date") == today_str:
                    return float(data.get("cost_usd", 0.0))
        except Exception as exc:
            log.warning("[ChatGPT] Failed to read daily cost file: %s", exc)
        return 0.0

    def _increment_daily_cost(self, today_str: str, cost: float) -> float:
        """Safely increment the persistent daily cost."""
        current = 0.0
        try:
            self._daily_cost_file.parent.mkdir(parents=True, exist_ok=True)
            if self._daily_cost_file.exists():
                try:
                    with open(self._daily_cost_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if data.get("date") == today_str:
                            current = float(data.get("cost_usd", 0.0))
                except Exception:
                    pass
            
            new_cost = current + cost
            temp_file = self._daily_cost_file.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump({"date": today_str, "cost_usd": new_cost}, f)
            
            if os.path.exists(self._daily_cost_file):
                os.remove(self._daily_cost_file)
            os.rename(temp_file, self._daily_cost_file)
            return new_cost
        except Exception as exc:
            log.warning("[ChatGPT] Failed to write daily cost file: %s", exc)
            return current + cost

    @staticmethod
    def _fail_closed_meta(error_code: str) -> Dict[str, Any]:
        """Return standardized fail-closed metadata for LLM call failures."""
        return {"action": "skip", "error": error_code, "confidence": 0.0}

    def _chat(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.2) -> Tuple[str, Dict]:
        """
        OpenAI Chat API çağrısı.
        
        Returns:
            (response_text, metadata)
        """
        # Devre dışı kontrolü
        if not self.enabled or self.client is None:
            return "", {"skipped": True, "reason": "disabled"}
        
        # Circuit breaker kontrolü
        if not self._check_circuit_breaker():
            return "", {"skipped": True, "reason": "circuit_breaker"}
        
        # Rate limit kontrolü
        if not self._check_rate_limit():
            return "", {"skipped": True, "reason": "rate_limit"}
        
        # Cache kontrolü
        cache_key = self._get_cache_key(model, messages)
        cached = self._check_cache(cache_key)
        if cached:
            return cached, {"cached": True}

        # Daily budget check (persistent fallback)
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._lock:
            current_daily_cost = self._get_daily_cost(today_str)
        
        costs = TOKEN_COSTS.get(model, TOKEN_COSTS["gpt-4o-mini"])
        max_output_tokens = 1000
        estimated_tokens = self._estimate_request_tokens(messages, max_output_tokens)
        estimated_cost = (estimated_tokens / 1000) * max(costs["input"], costs["output"])
        
        if self.disable_llm_on_budget_limit and current_daily_cost + estimated_cost > self.max_daily_budget_usd:
            log_event(
                log,
                logging.WARNING,
                "chatgpt_daily_budget_exceeded",
                "[ChatGPT] Daily cost budget exceeded before API call",
                current_daily_cost=current_daily_cost,
                estimated_cost=estimated_cost,
                limit=self.max_daily_budget_usd,
            )
            return "", {"skipped": True, "reason": "daily_budget"}

        if not self._check_token_budget(estimated_tokens):
            log_event(
                log,
                logging.WARNING,
                "chatgpt_token_budget_preflight_rejected",
                "[ChatGPT] Token budget rejected before API call",
                estimated_tokens=estimated_tokens,
                max_tokens_per_minute=self.MAX_TOKENS_PER_MINUTE,
            )
            return "", {"skipped": True, "reason": "token_budget"}
        
        # API çağrısı (exponential backoff ile)
        self.metrics.total_calls += 1
        self.metrics.last_call_time = time.time()
        self._call_times.append(time.time())
        
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.responses.create(
                    model=model,
                    input=messages,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
                
                if resp and resp.output_text:
                    content = resp.output_text or ""
                    
                    # Token kullanımını kaydet
                    if hasattr(resp, 'usage') and resp.usage:
                        self._track_tokens(
                            model,
                            resp.usage.input_tokens if hasattr(resp.usage, 'input_tokens') else 0,
                            resp.usage.output_tokens if hasattr(resp.usage, 'output_tokens') else 0
                        )
                    
                    # Başarıyı kaydet ve cache'e al
                    self._record_success()
                    self._set_cache(cache_key, content)
                    
                    return content, {
                        "model": model,
                        "tokens_in": resp.usage.input_tokens if (hasattr(resp, 'usage') and resp.usage and hasattr(resp.usage, 'input_tokens')) else 0,
                        "tokens_out": resp.usage.output_tokens if (hasattr(resp, 'usage') and resp.usage and hasattr(resp.usage, 'output_tokens')) else 0,
                        "attempt": attempt + 1
                    }
                
                return "", {"error": "empty_response"}
                
            except openai.APITimeoutError as e:
                self._record_failure()
                log.warning("[LLM] Timeout: %s", e)
                return "", self._fail_closed_meta("llm_timeout")
            except openai.APIConnectionError as e:
                self._record_failure()
                log.warning("[LLM] Connection: %s", e)
                return "", self._fail_closed_meta("llm_connection")
            except openai.RateLimitError as e:
                self._record_failure()
                log.warning("[LLM] Rate limit: %s", e)
                return "", self._fail_closed_meta("llm_rate_limit")
            except openai.APIStatusError as e:
                self._record_failure()
                log.error("[LLM] Status %s: %s", e.status_code, e)
                return "", self._fail_closed_meta("llm_api_status")
            except openai.APIError as e:
                self._record_failure()
                log.error("[LLM] Generic API: %s", e)
                return "", self._fail_closed_meta("llm_api_generic")
            except json.JSONDecodeError as e:
                self._record_failure()
                log.error("[LLM] JSON parse: %s", e)
                return "", self._fail_closed_meta("llm_bad_json")
            except RuntimeError as e:
                if "token-per-minute budget" in str(e):
                    self._record_failure()
                    log.warning("[LLM] Token budget: %s", e)
                    return "", self._fail_closed_meta("llm_token_budget")
                last_error = e
                wait_time = min(2 ** attempt, 10)
                log_event(
                    log,
                    logging.WARNING,
                    "chatgpt_retry_attempt_failed",
                    "[ChatGPT] API attempt failed",
                    attempt=attempt + 1,
                    max_attempts=self.max_retries + 1,
                    wait_seconds=wait_time,
                    error=str(e),
                )
                time.sleep(wait_time)
            except BEST_EFFORT_EXCEPTIONS as e:
                last_error = e
                wait_time = min(2 ** attempt, 10)  # Max 10 saniye bekleme
                log_event(
                    log,
                    logging.WARNING,
                    "chatgpt_retry_attempt_failed",
                    "[ChatGPT] API attempt failed",
                    attempt=attempt + 1,
                    max_attempts=self.max_retries + 1,
                    wait_seconds=wait_time,
                    error=str(e),
                )
                time.sleep(wait_time)
        
        # Tüm denemeler başarısız
        self._record_failure()
        log_event(
            log,
            logging.ERROR,
            "chatgpt_all_retries_failed",
            "[ChatGPT] All retries failed",
            error=str(last_error),
        )
        return "", {"error": "llm_unexpected", "confidence": 0.0}
    
    def _parse_json_response(self, content: str, default: Dict) -> Dict:
        """JSON yanıtını güvenli şekilde parse et."""
        if not content:
            return default
        
        # JSON bloğunu bul
        try:
            # Direkt JSON
            if content.strip().startswith("{"):
                return json.loads(content)
            
            # Markdown code block içinde
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                if end > start:
                    return json.loads(content[start:end].strip())
            
            # Sadece {...} bul
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
                
        except json.JSONDecodeError as e:
            log_event(
                log,
                logging.WARNING,
                "chatgpt_json_parse_error",
                "[ChatGPT] JSON parse error",
                error=str(e),
            )
        
        return default

    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def filter_signal(self, snapshot: Dict[str, Any], base_signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sinyal filtreleme (hafif model).
        
        Returns:
            {"allow": bool, "risk_flag": str, "reason": str, "confidence": float}
        """
        default = {"allow": True, "risk_flag": "unknown", "reason": "fallback", "confidence": 0.5}
        
        sym = snapshot.get("symbol", "UNKNOWN")
        tf = snapshot.get("timeframe", "15m")
        features = snapshot.get("features", {})
        risk = snapshot.get("risk", {})
        base_decision = base_signal.get("decision", "neutral")
        
        sys_prompt = """You are a professional crypto trading risk filter.
Analyze the signal and decide if it should be allowed or rejected.
Be conservative - safety first.

Output JSON with:
- allow: boolean
- risk_flag: "low"|"medium"|"high"|"critical"
- reason: string (max 150 chars)
- confidence: 0.0-1.0"""

        user_prompt = f"""Symbol: {sym}
Timeframe: {tf}
Base Decision: {base_decision}
Features: {json.dumps(features, default=str)}
Risk Info: {json.dumps(risk, default=str)}

Rules:
- Reject if ADX < 15 (no trend)
- Reject if RSI extreme (< 25 or > 75) without confirmation
- Reject if volatility spike > 2x normal
- Allow if all conditions normal

Respond with JSON only."""

        content, meta = self._chat(
            self.filter_model,
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1
        )
        
        if meta.get("skipped") or meta.get("error"):
            reason = str(meta.get("error") or meta.get("reason") or "llm_unavailable")
            log_event(
                log,
                logging.DEBUG,
                "chatgpt_filter_signal_skipped",
                "[ChatGPT] filter_signal skipped",
                meta=dict(meta),
            )
            return {"allow": False, "risk_flag": "critical", "reason": reason, "confidence": 0.0}
        
        result = self._parse_json_response(content, default)
        result.setdefault("confidence", 0.5)
        return result
    
    def decide_trade(self, snapshot: Dict[str, Any], filtered: Dict[str, Any]) -> Dict[str, Any]:
        """
        Trade kararı (ana model).
        
        Returns:
            {"action": str, "direction": str, "confidence": float, "reason": str}
        """
        default = {"action": "skip", "direction": "neutral", "confidence": 0.5, "reason": "fallback"}
        
        sym = snapshot.get("symbol", "UNKNOWN")
        tf = snapshot.get("timeframe", "15m")
        price = snapshot.get("price", 0)
        atr = snapshot.get("features", {}).get("atr", 0)
        rsi = snapshot.get("features", {}).get("rsi", 50)
        macd = snapshot.get("features", {}).get("macd_hist", 0)
        funding = snapshot.get("sentiment", {}).get("funding", 0)
        filt_allow = filtered.get("allow", True)
        filt_conf = filtered.get("confidence", 0.5)
        
        if not filt_allow:
            return {"action": "skip", "direction": "neutral", "confidence": 0.0, "reason": "filtered_out"}
        
        sys_prompt = """You are a professional crypto futures trading decision engine.
Make the final trading decision based on all available data.

Output JSON with:
- action: "enter"|"skip"
- direction: "long"|"short"|"neutral"
- confidence: 0.0-1.0 (your confidence in this decision)
- reason: string (max 200 chars)"""

        user_prompt = f"""Symbol: {sym}
Timeframe: {tf}
Current Price: {price}
ATR: {atr}
RSI: {rsi}
MACD Histogram: {macd}
Funding Rate: {funding}
Filter Confidence: {filt_conf}

Decision Rules:
1. If RSI < 35 and MACD > 0: Consider LONG
2. If RSI > 65 and MACD < 0: Consider SHORT
3. If no clear signal: SKIP
4. Higher confidence for stronger signals

Respond with JSON only."""

        content, meta = self._chat(
            self.decision_model,
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2
        )
        
        if meta.get("skipped") or meta.get("error"):
            reason = str(meta.get("error") or meta.get("reason") or "llm_unavailable")
            log_event(
                log,
                logging.DEBUG,
                "chatgpt_decide_trade_skipped",
                "[ChatGPT] decide_trade skipped",
                meta=dict(meta),
            )
            return {"action": "skip", "direction": "neutral", "confidence": 0.0, "reason": reason}
        
        result = self._parse_json_response(content, default)
        
        # Normalize
        result["action"] = str(result.get("action", "skip")).lower()
        result["direction"] = str(result.get("direction", "neutral")).lower()
        result["confidence"] = float(result.get("confidence", 0.5))
        if result["action"] not in ("enter", "skip"):
            result["action"] = "skip"
        if result["direction"] not in ("long", "short", "neutral"):
            result["direction"] = "neutral"
        
        return result
    
    def get_metrics(self) -> Dict:
        """API kullanım metriklerini döndür."""
        return {
            "total_calls": self.metrics.total_calls,
            "successful_calls": self.metrics.successful_calls,
            "failed_calls": self.metrics.failed_calls,
            "success_rate": self.metrics.successful_calls / max(1, self.metrics.total_calls),
            "total_tokens_in": self.metrics.total_tokens_in,
            "total_tokens_out": self.metrics.total_tokens_out,
            "total_cost_usd": round(self.metrics.total_cost_usd, 4),
            "circuit_breaker_trips": self.metrics.circuit_breaker_trips,
            "cache_size": len(self._cache),
        }
    
    def clear_cache(self) -> int:
        """Önbelleği temizle."""
        count = len(self._cache)
        self._cache.clear()
        return count


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_default_client: Optional[ChatGPTClient] = None


def get_client(**kwargs) -> ChatGPTClient:
    """Default ChatGPT client."""
    global _default_client
    if _default_client is None:
        _default_client = ChatGPTClient(**kwargs)
    return _default_client


def filter_signal(snapshot: Dict, base_signal: Dict) -> Dict:
    """Shortcut for signal filtering."""
    return get_client().filter_signal(snapshot, base_signal)


def decide_trade(snapshot: Dict, filtered: Dict) -> Dict:
    """Shortcut for trade decision."""
    return get_client().decide_trade(snapshot, filtered)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=== Professional ChatGPT Client Test ===")
    client = ChatGPTClient()
    print(f"Enabled: {client.enabled}")
    print(f"Filter Model: {client.filter_model}")
    print(f"Decision Model: {client.decision_model}")
    print(f"Metrics: {client.get_metrics()}")
