# -*- coding: utf-8 -*-
from __future__ import annotations

"""Static implementation generated from former trade quality gate split parts."""


from core.exceptions import BEST_EFFORT_EXCEPTIONS
# --- former _trade_quality_gate_impl_parts/part_01.py ---
# Auto-split from _trade_quality_gate_impl.py, original lines 1-329.
"""
trade_quality_gate.py - GERÇEK Trade Denetleyici
=================================================

[2026-01-20] TAMAMEN YENİDEN YAZILDI

Bu modül AI'ın (Transformer, RL, ChatGPT, DeepSeek) GÖREMEYECEĞI şeyleri
kontrol eder. AI sadece fiyat pattern'lerini görür, bu sistem ise:

1. Exchange Health     - OKX/Binance API sağlıklı mı? Yanıt süresi?
2. Portfolio Risk      - Portföyde aşırı korelasyon var mı?
3. Daily Loss Limit    - Bugün zaten çok kaybettik mi?
4. Margin Health       - Margin kullanımı tehlikeli seviyede mi?
5. Event Calendar      - Yaklaşan FOMC, CPI gibi eventler var mı?
6. Streak Detection    - Aşırı kazanma/kaybetme serisi var mı?

Bu kontroller AI'ın tahmininden BAĞIMSIZ çalışır.
AI "BUY %90 güven" dese bile, eğer:
- OKX API yavaşlıyorsa
- Bugün %5 kaybettiysek
- Margin %80'i geçtiyse
- 2 saat sonra FOMC varsa
Trade ENGELLENİR.

Bu GERÇEK bir denetleyici sistemidir.
"""


import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import asyncio

try:
    from logger import get_logger
    log = get_logger("quality_gate")
except ImportError:
    import logging
    log = logging.getLogger("quality_gate")


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class SafetyCheck:
    """Tek bir güvenlik kontrolünün sonucu."""
    name: str
    passed: bool
    severity: str  # "critical", "warning", "info"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    can_override: bool = False  # Kullanıcı override edebilir mi?


@dataclass
class SafetyResult:
    """Tüm güvenlik kontrollerinin sonucu."""
    approved: bool
    checks: List[SafetyCheck] = field(default_factory=list)
    critical_failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    summary: str = ""


# =============================================================================
# CORRELATION MAP - Hangi coinler birbiriyle korele?
# =============================================================================

_STATIC_CORRELATION_GROUPS = {
    "BTC_CORRELATED": [
        "BTC", "ETH", "SOL", "BNB", "AVAX", "MATIC", "DOT", "ATOM",
        "LINK", "UNI", "AAVE", "LTC", "BCH", "ETC"
    ],
    "MEME_COINS": [
        "DOGE", "SHIB", "PEPE", "FLOKI", "BONK", "WIF", "MEME"
    ],
    "AI_TOKENS": [
        "FET", "AGIX", "OCEAN", "RNDR", "TAO", "ARKM"
    ],
    "DEFI": [
        "UNI", "AAVE", "SUSHI", "CAKE", "CRV", "MKR", "COMP"
    ],
    "LAYER2": [
        "ARB", "OP", "MATIC", "IMX", "STRK", "MANTA"
    ],
    "GAMING": [
        "AXS", "SAND", "MANA", "GALA", "IMX", "ENJ"
    ]
}

def get_correlation_groups() -> Dict[str, List[str]]:
    """
    O8 FIX: Dinamik korelasyon hesaplamasi. Eger cache/matris mevcut degilse
    statik gruplamaya fallback yapar.
    """
    try:
        from state_manager import get_metric
        dynamic_groups = get_metric("dynamic_correlation_groups")
        if isinstance(dynamic_groups, dict) and len(dynamic_groups) > 0:
            return dynamic_groups
    except BEST_EFFORT_EXCEPTIONS:
        pass
    return _STATIC_CORRELATION_GROUPS

# =============================================================================
# MACRO EVENT CALENDAR - Önemli ekonomik olaylar
# =============================================================================

# Bu statik bir liste, gerçek uygulamada API'den çekilmeli
# Format: (datetime_utc, event_name, impact_level)
# Impact: "HIGH", "MEDIUM", "LOW"

def get_upcoming_events() -> List[Dict]:
    """
    Yaklaşan önemli makro ekonomik olayları döndür.
    
    [FIX] Öncelikli olarak state_manager üzerinden canlı macro_data_updater verisini alır.
    Başarısız olursa yerel macro_events.json dosyasına fallback yapar.
    """
    now = datetime.now(timezone.utc)
    events = []

    # 1. Öncelik: state_manager üzerinden güncel canlı macro verisi
    try:
        from state_manager import get_metric
        live_events = get_metric("macro_events")
        if isinstance(live_events, list) and len(live_events) > 0:
            for ev in live_events:
                # ev: {"title": "CPI", "date": "2023-...", "impact": "High"}
                try:
                    # Gelen tarih stringini işle (macro_data_updater formatlarına uyumlu olmalı)
                    ev_time_str = ev.get("date") or ev.get("datetime")
                    if ev_time_str:
                        # Eğer ISO formatına çevrilemiyorsa pass geçer
                        import dateutil.parser as dp
                        ev_time = dp.isoparse(ev_time_str)
                        # Sadece gelecekteki olayları al
                        if ev_time > now:
                            impact_raw = str(ev.get("impact", "HIGH")).upper()
                            events.append({
                                "datetime": ev_time,
                                "name": ev.get("title") or ev.get("name", "Macro Event"),
                                "impact": impact_raw
                            })
                except BEST_EFFORT_EXCEPTIONS as e:
                    log.debug(f"[QUALITY_GATE] Canlı macro event işleme hatası: {e}")
            
            # Eğer canlı veriden event alabildiysek dosyaya bakmaya gerek yok
            if events:
                return events
    except BEST_EFFORT_EXCEPTIONS as e:
        log.debug(f"[QUALITY_GATE] state_manager macro_events alınamadı: {e}")

    # 2. Öncelik: Fallback (Eski json dosyası)
    events_file = Path(__file__).resolve().parent / "config" / "macro_events.json"
    if not events_file.exists():
        events_file = Path(__file__).resolve().parent / "data" / "macro_events.json"
    if events_file.exists():
        try:
            with open(events_file, "r", encoding="utf-8") as f:
                raw_events = json.load(f)
                for ev in raw_events:
                    try:
                        ev_time = datetime.fromisoformat(ev["datetime"])
                        if ev_time > now:
                            events.append({
                                "datetime": ev_time,
                                "name": ev["name"],
                                "impact": ev.get("impact", "HIGH")
                            })
                    except BEST_EFFORT_EXCEPTIONS:
                        pass
        except BEST_EFFORT_EXCEPTIONS as e:
            log.debug(f"Event calendar okunamadı: {e}")

    return events


def _qg_parse_event_time(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value.strip():
        raw = value.strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(raw)
        except BEST_EFFORT_EXCEPTIONS:
            try:
                from dateutil import parser as date_parser  # type: ignore
                dt = date_parser.isoparse(value.strip())
            except BEST_EFFORT_EXCEPTIONS:
                return None
    else:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _qg_event_impact(ev: Dict[str, Any]) -> str:
    impact = str(ev.get("impact") or "").upper()
    if impact in {"HIGH", "MEDIUM", "LOW"}:
        return impact
    try:
        multiplier = float(ev.get("multiplier"))
        if multiplier <= 0.55:
            return "HIGH"
        if multiplier <= 0.75:
            return "MEDIUM"
        return "LOW"
    except BEST_EFFORT_EXCEPTIONS:
        return "MEDIUM"


def _qg_normalise_event(ev: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
    ev_time = _qg_parse_event_time(ev.get("datetime") or ev.get("date") or ev.get("start"))
    if ev_time is None:
        return None
    return {
        "datetime": ev_time,
        "name": ev.get("title") or ev.get("name") or ev.get("event") or "Macro Event",
        "impact": _qg_event_impact(ev),
        "source": source,
        "pre_minutes": int(ev.get("pre_minutes") or 120),
    }


def _qg_load_macro_event_sources() -> Tuple[List[Dict[str, Any]], str, Optional[float]]:
    root = Path(__file__).resolve().parent
    try:
        from state_manager import get_metric
        live_events = get_metric("macro_events")
        if isinstance(live_events, list) and live_events:
            return live_events, "state_manager", None
    except BEST_EFFORT_EXCEPTIONS as e:
        log.debug(f"[QUALITY_GATE] state_manager macro_events alınamadı: {e}")

    cfg_path = root / "config.json"
    try:
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            cfg_events = cfg.get("macro_events")
            if isinstance(cfg_events, list) and cfg_events:
                return cfg_events, "config.json", cfg_path.stat().st_mtime
    except BEST_EFFORT_EXCEPTIONS as e:
        log.debug(f"[QUALITY_GATE] config.json macro_events okunamadı: {e}")

    for events_file in (root / "config" / "macro_events.json", root / "data" / "macro_events.json"):
        if events_file.exists():
            try:
                raw_events = json.loads(events_file.read_text(encoding="utf-8"))
                if isinstance(raw_events, list):
                    return raw_events, str(events_file), events_file.stat().st_mtime
            except BEST_EFFORT_EXCEPTIONS as e:
                log.debug(f"Event calendar okunamadı: {e}")

    try:
        import os
        if os.environ.get("MACRO_API_KEY"):
            from macro_data_updater import update_macro_events
            refreshed = update_macro_events(days_ahead=7)
            if refreshed:
                return refreshed, "macro_data_updater", None
    except BEST_EFFORT_EXCEPTIONS as e:
        log.debug(f"[QUALITY_GATE] macro event refresh başarısız: {e}")

    return [], "unavailable", None


def get_event_calendar_status(lookahead_hours: float = 72.0) -> Dict[str, Any]:
    """Return normalized macro calendar status for safety and shock telemetry."""
    now = datetime.now(timezone.utc)
    raw_events, source, mtime = _qg_load_macro_event_sources()
    events: List[Dict[str, Any]] = []
    for raw in raw_events:
        if not isinstance(raw, dict):
            continue
        event = _qg_normalise_event(raw, source)
        if event and event["datetime"] > now:
            events.append(event)
    events.sort(key=lambda item: item["datetime"])

    next_event = events[0] if events else None
    next_minutes = None
    if next_event:
        next_minutes = (next_event["datetime"] - now).total_seconds() / 60.0

    source_age_hours = None
    if mtime is not None:
        source_age_hours = max(0.0, (time.time() - float(mtime)) / 3600.0)
    stale = source == "unavailable" or (source_age_hours is not None and source_age_hours > 24.0)
    stale_reason = "no_calendar_source" if source == "unavailable" else None
    if stale_reason is None and source_age_hours is not None and source_age_hours > 24.0:
        stale_reason = "calendar_file_older_than_24h"

    high_impact_news = False
    current_event = None
    for event in events:
        minutes = (event["datetime"] - now).total_seconds() / 60.0
        if event["impact"] == "HIGH" and 0.0 <= minutes <= float(event.get("pre_minutes", 120)):
            high_impact_news = True
            current_event = event
            break

    return {
        "source": source,
        "stale": bool(stale),
        "stale_reason": stale_reason,
        "events": events,
        "events_count": len(events),
        "lookahead_hours": float(lookahead_hours),
        "next_event_minutes": next_minutes,
        "next_event_name": next_event.get("name") if next_event else None,
        "next_event_impact": next_event.get("impact") if next_event else None,
        "high_impact_news": high_impact_news,
        "current_event": current_event,
        "source_age_hours": source_age_hours,
    }


def get_upcoming_events() -> List[Dict]:
    """Return upcoming events from the normalized macro calendar status."""
    return get_event_calendar_status().get("events", [])

# --- former _trade_quality_gate_impl_parts/part_02.py ---
# Auto-split from _trade_quality_gate_impl.py, original lines 330-1131.


# =============================================================================
# ANA SINIF: TradeSafetyGuard
# =============================================================================


class TradeSafetyGuard:
    """
    Trade Güvenlik Denetleyicisi.

    AI'ın göremediği kritik faktörleri kontrol eder.
    Bu kontroller başarısız olursa trade ENGELLENİR.
    """

    # Konfigürasyon
    DEFAULT_CONFIG = {
        # Exchange Health
        "max_api_response_ms": 2000,         # Max API yanıt süresi (ms)
        "min_orderbook_depth": 10,            # Minimum order book derinliği

        # Portfolio Risk
        "max_correlated_positions": 2,        # Max aynı grupta açık pozisyon
        "max_total_positions": 5,             # Max toplam açık pozisyon

        # Daily Loss
        "daily_loss_limit_pct": 0.05,         # Günlük max kayıp %5
        "session_loss_limit_pct": 0.03,       # Session max kayıp %3

        # Margin
        "margin_warning_pct": 0.60,           # Margin uyarı seviyesi
        "margin_critical_pct": 0.80,          # Margin kritik seviyesi

        # Event Filter
        "event_blackout_hours": 2,            # Event öncesi yasak saat
        "high_impact_only": True,             # Sadece HIGH impact eventleri

        # Streak Detection
        "max_consecutive_wins": 7,            # Max arka arkaya kazanç
        "max_consecutive_losses": 4,          # Max arka arkaya kayıp

        # Recent Win Rate
        "min_win_rate": 0.35,                 # Minimum win rate (%35)
        "win_rate_lookback": 20,              # Son kaç trade'e bakılacak
        "min_trades_for_win_rate": 10,        # Minimum trade sayısı

        # Circuit Breaker
        "enable_circuit_breaker": True,
        "circuit_breaker_cooldown_min": 60    # Soğuma süresi (dakika)
    }

    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: Özel konfigürasyon (None = default)
        """
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self._circuit_breaker_until: Optional[datetime] = None
        self._last_check_time: Optional[datetime] = None

    # =========================================================================
    # ANA EVALUATE FONKSİYONU
    # =========================================================================

    def evaluate(
        self,
        symbol: str,
        direction: str,
        # Exchange verisi
        exchange_client: Optional[Any] = None,
        api_response_time_ms: Optional[float] = None,
        orderbook_depth: Optional[int] = None,
        # Portföy verisi
        open_positions: Optional[List[Dict]] = None,
        # Günlük P&L
        daily_pnl_pct: Optional[float] = None,
        session_pnl_pct: Optional[float] = None,
        unrealized_pnl_pct: Optional[float] = None,
        # Margin
        margin_used_pct: Optional[float] = None,
        available_balance: Optional[float] = None,
        # Trade geçmişi
        recent_trades: Optional[List[Dict]] = None,
        position_returns: Optional[Dict[str, List[float]]] = None,
        # Ekstra
        force_check: bool = False
    ) -> SafetyResult:
        """
        Tüm güvenlik kontrollerini çalıştır.

        Args:
            symbol: Trade sembolü (örn: "BTC/USDT")
            direction: "long" veya "short"
            exchange_client: OKX/Binance client (API health için)
            api_response_time_ms: Son API yanıt süresi
            orderbook_depth: Order book derinliği
            open_positions: Açık pozisyonlar listesi
            daily_pnl_pct: Bugünkü toplam P&L (ör: -0.03 = %3 kayıp)
            session_pnl_pct: Bu session'daki P&L
            margin_used_pct: Kullanılan margin oranı (0-1)
            available_balance: Kullanılabilir bakiye
            recent_trades: Son trade'ler
            force_check: True ise cache'i atla

        Returns:
            SafetyResult: Güvenlik değerlendirmesi
        """
        checks: List[SafetyCheck] = []
        critical_failures: List[str] = []
        warnings: List[str] = []

        # =====================================================================
        # 0. CIRCUIT BREAKER CHECK
        # =====================================================================
        if self._circuit_breaker_until:
            if datetime.now(timezone.utc) < self._circuit_breaker_until:
                remaining = (self._circuit_breaker_until - datetime.now(timezone.utc)).seconds // 60
                checks.append(SafetyCheck(
                    name="circuit_breaker",
                    passed=False,
                    severity="critical",
                    message=f"Circuit breaker aktif. {remaining} dakika kaldı.",
                    can_override=False
                ))
                critical_failures.append(f"Circuit breaker: {remaining}dk kaldı")

                return SafetyResult(
                    approved=False,
                    checks=checks,
                    critical_failures=critical_failures,
                    warnings=warnings,
                    summary="⛔ CIRCUIT BREAKER AKTİF"
                )
            else:
                self._circuit_breaker_until = None  # Reset

        # =====================================================================
        # 1. EXCHANGE HEALTH CHECK
        # =====================================================================
        exchange_check = self._check_exchange_health(
            api_response_time_ms=api_response_time_ms,
            orderbook_depth=orderbook_depth,
            exchange_client=exchange_client
        )
        checks.append(exchange_check)

        if not exchange_check.passed:
            if exchange_check.severity == "critical":
                critical_failures.append(exchange_check.message)
            else:
                warnings.append(exchange_check.message)

        # =====================================================================
        # 2. PORTFOLIO CORRELATION CHECK
        # =====================================================================
        correlation_check = self._check_portfolio_correlation(
            symbol=symbol,
            direction=direction,
            open_positions=open_positions
        )
        checks.append(correlation_check)

        if not correlation_check.passed:
            if correlation_check.severity == "critical":
                critical_failures.append(correlation_check.message)
            else:
                warnings.append(correlation_check.message)

        # =====================================================================
        # 3. DAILY LOSS LIMIT CHECK
        # =====================================================================
        effective_daily_pnl = daily_pnl_pct
        if unrealized_pnl_pct is not None:
            effective_daily_pnl = float(daily_pnl_pct or 0.0) + float(unrealized_pnl_pct or 0.0)

        loss_check = self._check_daily_loss_limit(
            daily_pnl_pct=effective_daily_pnl,
            session_pnl_pct=session_pnl_pct
        )
        checks.append(loss_check)

        if not loss_check.passed:
            critical_failures.append(loss_check.message)
            # Circuit breaker'ı aktifle
            if self.config["enable_circuit_breaker"]:
                self._activate_circuit_breaker()

        # =====================================================================
        # 4. MARGIN HEALTH CHECK
        # =====================================================================
        margin_check = self._check_margin_health(
            margin_used_pct=margin_used_pct,
            available_balance=available_balance
        )
        checks.append(margin_check)

        if not margin_check.passed:
            if margin_check.severity == "critical":
                critical_failures.append(margin_check.message)
            else:
                warnings.append(margin_check.message)

        # =====================================================================
        # 5. MACRO EVENT CHECK
        # =====================================================================
        event_check = self._check_macro_events()
        checks.append(event_check)

        if not event_check.passed:
            if event_check.severity == "critical":
                critical_failures.append(event_check.message)
            else:
                warnings.append(event_check.message)

        # =====================================================================
        # 6. STREAK DETECTION CHECK
        # =====================================================================
        streak_check = self._check_trade_streak(recent_trades=recent_trades)
        checks.append(streak_check)

        if not streak_check.passed:
            if streak_check.severity == "critical":
                critical_failures.append(streak_check.message)
            else:
                warnings.append(streak_check.message)

        # =====================================================================
        # 7. RECENT WIN RATE CHECK
        # =====================================================================
        win_rate_check = self._check_recent_win_rate(recent_trades=recent_trades)
        checks.append(win_rate_check)

        if not win_rate_check.passed:
            if win_rate_check.severity == "critical":
                critical_failures.append(win_rate_check.message)
                # Çok düşük win rate = circuit breaker
                if self.config["enable_circuit_breaker"]:
                    self._activate_circuit_breaker()
            else:
                warnings.append(win_rate_check.message)

        # =====================================================================
        # FINAL DECISION
        # =====================================================================
        approved = len(critical_failures) == 0

        # Summary oluştur
        if approved:
            if warnings:
                summary = f"✅ ONAYLANDI ({len(warnings)} uyarı)"
            else:
                summary = "✅ ONAYLANDI - Tüm güvenlik kontrolleri geçti"
        else:
            summary = f"⛔ REDDEDİLDİ - {len(critical_failures)} kritik hata"

        # Log
        self._log_result(symbol, direction, approved, checks, critical_failures, warnings)

        return SafetyResult(
            approved=approved,
            checks=checks,
            critical_failures=critical_failures,
            warnings=warnings,
            summary=summary
        )

    # =========================================================================
    # CHECK METHODS
    # =========================================================================

    def _check_exchange_health(
        self,
        api_response_time_ms: Optional[float],
        orderbook_depth: Optional[int],
        exchange_client: Optional[Any]
    ) -> SafetyCheck:
        """
        Exchange sağlık kontrolü.

        Kontrol edilen:
        - API yanıt süresi
        - Order book derinliği
        - Connection durumu
        """
        issues = []
        severity = "info"

        # API yanıt süresi
        max_ms = self.config["max_api_response_ms"]
        if api_response_time_ms is not None:
            if api_response_time_ms > max_ms:
                issues.append(f"API yavaş: {api_response_time_ms:.0f}ms > {max_ms}ms")
                severity = "critical"
            elif api_response_time_ms > max_ms * 0.7:
                issues.append(f"API yanıt süresi yüksek: {api_response_time_ms:.0f}ms")
                severity = "warning"

        # Order book derinliği
        min_depth = self.config["min_orderbook_depth"]
        if orderbook_depth is not None:
            if orderbook_depth < min_depth:
                issues.append(f"Düşük likidite: depth={orderbook_depth} < {min_depth}")
                severity = "critical" if orderbook_depth < min_depth // 2 else "warning"

        # Sonuç
        passed = severity != "critical"
        message = " | ".join(issues) if issues else "Exchange sağlıklı"

        return SafetyCheck(
            name="exchange_health",
            passed=passed,
            severity=severity,
            message=message,
            details={
                "api_response_ms": api_response_time_ms,
                "orderbook_depth": orderbook_depth
            },
            can_override=False
        )

    def _check_portfolio_correlation(
        self,
        symbol: str,
        direction: str,
        open_positions: Optional[List[Dict]]
    ) -> SafetyCheck:
        """
        Portföy korelasyon kontrolü.

        Aynı korelasyon grubunda çok fazla pozisyon açılmasını engeller.
        Örnek: Zaten 2 BTC-korele coin açıksa, 3. BTC-korele coin engellenir.
        """
        if not open_positions:
            return SafetyCheck(
                name="portfolio_correlation",
                passed=True,
                severity="info",
                message="Açık pozisyon yok",
                can_override=True
            )

        # Symbol'den base coin'i çıkar (örn: "BTC/USDT" -> "BTC")
        base_coin = symbol.split("/")[0].upper() if "/" in symbol else symbol.replace("USDT", "").upper()

        # Bu coin hangi gruplara ait?
        correlation_groups = get_correlation_groups()
        coin_groups = []
        for group_name, coins in correlation_groups.items():
            if base_coin in coins:
                coin_groups.append(group_name)

        if not coin_groups:
            return SafetyCheck(
                name="portfolio_correlation",
                passed=True,
                severity="info",
                message=f"{base_coin} korelasyon grubunda değil",
                can_override=True
            )

        # Her grup için açık pozisyon say
        max_allowed = self.config["max_correlated_positions"]
        issues = []
        severity = "info"

        for group_name in coin_groups:
            group_coins = correlation_groups.get(group_name, [])
            count_in_group = 0

            for pos in open_positions:
                pos_symbol = pos.get("symbol", "")
                pos_base = pos_symbol.split("/")[0].upper() if "/" in pos_symbol else pos_symbol.replace("USDT", "").upper()

                if pos_base in group_coins:
                    count_in_group += 1

            if count_in_group >= max_allowed:
                issues.append(f"{group_name}: {count_in_group}/{max_allowed} pozisyon dolu")
                severity = "critical"
            elif count_in_group == max_allowed - 1:
                issues.append(f"{group_name}: {count_in_group}/{max_allowed} (son slot)")
                severity = "warning" if severity != "critical" else severity

        # Toplam pozisyon kontrolü
        max_total = self.config["max_total_positions"]
        total_positions = len(open_positions)

        if total_positions >= max_total:
            issues.append(f"Max pozisyon: {total_positions}/{max_total}")
            severity = "critical"

        passed = severity != "critical"
        message = " | ".join(issues) if issues else f"{base_coin} eklenmesinde sorun yok"

        return SafetyCheck(
            name="portfolio_correlation",
            passed=passed,
            severity=severity,
            message=message,
            details={
                "base_coin": base_coin,
                "groups": coin_groups,
                "total_positions": total_positions
            },
            can_override=True
        )

    def _check_daily_loss_limit(
        self,
        daily_pnl_pct: Optional[float],
        session_pnl_pct: Optional[float]
    ) -> SafetyCheck:
        """
        Günlük kayıp limiti kontrolü.

        Bugün veya bu session'da çok kaybettiysek trade engellenir.
        """
        issues = []
        severity = "info"

        # Günlük limit
        daily_limit = self.config["daily_loss_limit_pct"]
        if daily_pnl_pct is not None:
            if daily_pnl_pct <= -daily_limit:
                issues.append(f"Günlük kayıp limiti aşıldı: {daily_pnl_pct:.1%} (limit: -{daily_limit:.0%})")
                severity = "critical"
            elif daily_pnl_pct <= -daily_limit * 0.7:
                issues.append(f"Günlük kayıp yüksek: {daily_pnl_pct:.1%}")
                severity = "warning" if severity != "critical" else severity

        # Session limit
        session_limit = self.config["session_loss_limit_pct"]
        if session_pnl_pct is not None:
            if session_pnl_pct <= -session_limit:
                issues.append(f"Session kayıp limiti: {session_pnl_pct:.1%}")
                severity = "critical"

        passed = severity != "critical"
        message = " | ".join(issues) if issues else "Kayıp limitleri içinde"

        return SafetyCheck(
            name="daily_loss_limit",
            passed=passed,
            severity=severity,
            message=message,
            details={
                "daily_pnl_pct": daily_pnl_pct,
                "session_pnl_pct": session_pnl_pct,
                "daily_limit": daily_limit,
                "session_limit": session_limit
            },
            can_override=False  # Bu override edilemez!
        )

    def _check_margin_health(
        self,
        margin_used_pct: Optional[float],
        available_balance: Optional[float]
    ) -> SafetyCheck:
        """
        Margin sağlık kontrolü.

        Margin kullanımı çok yüksekse likidasyon riski var.
        """
        if margin_used_pct is None:
            return SafetyCheck(
                name="margin_health",
                passed=True,
                severity="info",
                message="Margin verisi yok",
                can_override=True
            )

        warning_level = self.config["margin_warning_pct"]
        critical_level = self.config["margin_critical_pct"]

        if margin_used_pct >= critical_level:
            return SafetyCheck(
                name="margin_health",
                passed=False,
                severity="critical",
                message=f"⚠️ KRİTİK MARGIN: {margin_used_pct:.0%} kullanımda (limit: {critical_level:.0%})",
                details={"margin_used_pct": margin_used_pct, "available_balance": available_balance},
                can_override=False
            )
        elif margin_used_pct >= warning_level:
            return SafetyCheck(
                name="margin_health",
                passed=True,  # Uyarı ama geçebilir
                severity="warning",
                message=f"Yüksek margin: {margin_used_pct:.0%}",
                details={"margin_used_pct": margin_used_pct, "available_balance": available_balance},
                can_override=True
            )
        else:
            return SafetyCheck(
                name="margin_health",
                passed=True,
                severity="info",
                message=f"Margin sağlıklı: {margin_used_pct:.0%}",
                details={"margin_used_pct": margin_used_pct, "available_balance": available_balance},
                can_override=True
            )

    def _check_macro_events(self) -> SafetyCheck:
        """
        Makro ekonomik event kontrolü.

        FOMC, CPI, NFP gibi yüksek volatilite yaratacak eventlerden
        önce trade açmayı engeller.
        """
        events = get_upcoming_events()

        if not events:
            return SafetyCheck(
                name="macro_events",
                passed=True,
                severity="info",
                message="Yaklaşan kritik event yok",
                can_override=True
            )

        now = datetime.now(timezone.utc)
        blackout_hours = self.config["event_blackout_hours"]
        high_impact_only = self.config["high_impact_only"]

        upcoming_critical = []

        for event in events:
            event_time = event["datetime"]
            impact = event.get("impact", "HIGH")

            # Sadece HIGH impact mi kontrol ediyoruz?
            if high_impact_only and impact != "HIGH":
                continue

            # Blackout periyodunda mı?
            time_until = (event_time - now).total_seconds() / 3600  # Saat olarak

            if 0 < time_until <= blackout_hours:
                upcoming_critical.append({
                    "name": event["name"],
                    "hours_until": round(time_until, 1),
                    "impact": impact
                })

        if upcoming_critical:
            event_names = [f"{e['name']} ({e['hours_until']}h)" for e in upcoming_critical]
            return SafetyCheck(
                name="macro_events",
                passed=False,
                severity="critical",
                message=f"🚨 YAKIN EVENT: {', '.join(event_names)}",
                details={"events": upcoming_critical},
                can_override=False
            )

        return SafetyCheck(
            name="macro_events",
            passed=True,
            severity="info",
            message="Event blackout yok",
            can_override=True
        )

    def _check_trade_streak(
        self,
        recent_trades: Optional[List[Dict]]
    ) -> SafetyCheck:
        """
        Trade serisi kontrolü.

        - Çok fazla arka arkaya kazanç = overconfidence riski
        - Çok fazla arka arkaya kayıp = tilt riski
        """
        if not recent_trades or len(recent_trades) < 3:
            return SafetyCheck(
                name="trade_streak",
                passed=True,
                severity="info",
                message="Yeterli trade geçmişi yok",
                can_override=True
            )

        # Son N trade'i analiz et
        last_trades = recent_trades[-15:] if len(recent_trades) > 15 else recent_trades

        # Streak hesapla (sondan geriye)
        current_streak = 0
        streak_type = None  # "win" veya "loss"

        for trade in reversed(last_trades):
            pnl = trade.get("pnl_pct", trade.get("pnl", 0))
            if pnl is None:
                continue

            is_win = float(pnl) > 0

            if streak_type is None:
                streak_type = "win" if is_win else "loss"
                current_streak = 1
            elif (streak_type == "win" and is_win) or (streak_type == "loss" and not is_win):
                current_streak += 1
            else:
                break

        max_wins = self.config["max_consecutive_wins"]
        max_losses = self.config["max_consecutive_losses"]

        # Çok fazla kazanç
        if streak_type == "win" and current_streak >= max_wins:
            return SafetyCheck(
                name="trade_streak",
                passed=False,
                severity="warning",  # Kritik değil ama uyarı
                message=f"🎰 {current_streak} arka arkaya kazanç - overconfidence riski!",
                details={"streak_type": "win", "streak_count": current_streak},
                can_override=True  # Override edilebilir
            )

        # Çok fazla kayıp
        if streak_type == "loss" and current_streak >= max_losses:
            return SafetyCheck(
                name="trade_streak",
                passed=False,
                severity="critical",
                message=f"🔴 {current_streak} arka arkaya kayıp - COOLDOWN GEREKLİ!",
                details={"streak_type": "loss", "streak_count": current_streak},
                can_override=False
            )

        return SafetyCheck(
            name="trade_streak",
            passed=True,
            severity="info",
            message=f"Streak: {current_streak} {streak_type or 'yok'}",
            details={"streak_type": streak_type, "streak_count": current_streak},
            can_override=True
        )

    def _check_recent_win_rate(
        self,
        recent_trades: Optional[List[Dict]]
    ) -> SafetyCheck:
        """
        Son N trade'in genel win rate kontrolü.

        Streak'ten farklı olarak, arka arkaya olmasa bile
        genel performansı ölçer.

        Örnek: K-K-W-K-K-W-K-K-K-W-K = %27 win rate
        Streak: max 3 (iyi görünür)
        Win Rate: %27 (kötü - sistem sorunlu)
        """
        min_trades = self.config["min_trades_for_win_rate"]
        lookback = self.config["win_rate_lookback"]
        min_win_rate = self.config["min_win_rate"]

        if not recent_trades or len(recent_trades) < min_trades:
            return SafetyCheck(
                name="recent_win_rate",
                passed=True,
                severity="info",
                message=f"Yeterli trade yok ({len(recent_trades) if recent_trades else 0}/{min_trades})",
                details={"trade_count": len(recent_trades) if recent_trades else 0},
                can_override=True
            )

        # Son N trade'i al
        last_trades = recent_trades[-lookback:] if len(recent_trades) > lookback else recent_trades

        # Win rate hesapla
        wins = 0
        losses = 0
        total_pnl = 0.0

        for trade in last_trades:
            pnl = trade.get("pnl_pct", trade.get("pnl", 0))
            if pnl is None:
                continue

            pnl = float(pnl)
            total_pnl += pnl

            if pnl > 0:
                wins += 1
            else:
                losses += 1

        total = wins + losses
        if total == 0:
            return SafetyCheck(
                name="recent_win_rate",
                passed=True,
                severity="info",
                message="PnL verisi yok",
                can_override=True
            )

        win_rate = wins / total
        avg_pnl = total_pnl / total

        # Değerlendirme
        if win_rate < min_win_rate:
            # Kritik düşük win rate
            return SafetyCheck(
                name="recent_win_rate",
                passed=False,
                severity="critical",
                message=f"📉 Düşük win rate: {win_rate:.0%} (min: {min_win_rate:.0%}) | Son {total} trade'de {wins}W/{losses}L",
                details={
                    "win_rate": round(win_rate, 3),
                    "wins": wins,
                    "losses": losses,
                    "avg_pnl": round(avg_pnl, 4),
                    "total_trades": total
                },
                can_override=False  # Override edilemez - sistem sorunlu
            )
        elif win_rate < 0.45:
            # Uyarı seviyesi
            return SafetyCheck(
                name="recent_win_rate",
                passed=True,  # Geçer ama uyarı
                severity="warning",
                message=f"⚠️ Win rate düşük: {win_rate:.0%} | {wins}W/{losses}L",
                details={
                    "win_rate": round(win_rate, 3),
                    "wins": wins,
                    "losses": losses,
                    "avg_pnl": round(avg_pnl, 4)
                },
                can_override=True
            )
        else:
            return SafetyCheck(
                name="recent_win_rate",
                passed=True,
                severity="info",
                message=f"Win rate: {win_rate:.0%} ({wins}W/{losses}L)",
                details={
                    "win_rate": round(win_rate, 3),
                    "wins": wins,
                    "losses": losses,
                    "avg_pnl": round(avg_pnl, 4)
                },
                can_override=True
            )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _activate_circuit_breaker(self):
        """Circuit breaker'ı aktifle."""
        cooldown_min = self.config["circuit_breaker_cooldown_min"]
        self._circuit_breaker_until = datetime.now(timezone.utc) + timedelta(minutes=cooldown_min)
        log.warning(f"🛑 CIRCUIT BREAKER AKTİF! {cooldown_min} dakika trade yasak.")

    def reset_circuit_breaker(self):
        """Circuit breaker'ı manuel sıfırla."""
        self._circuit_breaker_until = None
        log.info("Circuit breaker sıfırlandı.")

    def _log_result(
        self,
        symbol: str,
        direction: str,
        approved: bool,
        checks: List[SafetyCheck],
        critical_failures: List[str],
        warnings: List[str]
    ):
        """Sonucu logla."""
        try:
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "direction": direction,
                "approved": approved,
                "checks": [
                    {"name": c.name, "passed": c.passed, "severity": c.severity, "message": c.message}
                    for c in checks
                ],
                "critical_failures": critical_failures,
                "warnings": warnings
            }

            log_path = Path(__file__).resolve().parent / "metrics" / "safety_guard_log.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

            # Console log
            status = "✅" if approved else "⛔"
            log.info(f"[SAFETY] {symbol} {direction.upper()} {status}")

            for fail in critical_failures:
                log.warning(f"[SAFETY] ❌ {fail}")

        except BEST_EFFORT_EXCEPTIONS as e:
            log.debug(f"Safety log yazma hatası: {e}")



# --- former _trade_quality_gate_impl_parts/part_03.py ---
# Auto-split from _trade_quality_gate_impl.py, original lines 1132-1345.


# =============================================================================
# ENTEGRASYON FONKSİYONLARI
# =============================================================================

# Global instance
_guard_instance: Optional[TradeSafetyGuard] = None


def get_safety_guard() -> TradeSafetyGuard:
    """Global safety guard instance döndür."""
    global _guard_instance
    if _guard_instance is None:
        _guard_instance = TradeSafetyGuard()
    return _guard_instance


def check_trade_safety(
    symbol: str,
    direction: str,
    open_positions: Optional[List[Dict]] = None,
    daily_pnl_pct: Optional[float] = None,
    margin_used_pct: Optional[float] = None,
    recent_trades: Optional[List[Dict]] = None,
    api_response_time_ms: Optional[float] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Basit wrapper fonksiyon.

    Returns:
        {
            "approved": bool,
            "critical_failures": list,
            "warnings": list,
            "summary": str,
            "checks": list
        }
    """
    guard = get_safety_guard()
    result = guard.evaluate(
        symbol=symbol,
        direction=direction,
        open_positions=open_positions,
        daily_pnl_pct=daily_pnl_pct,
        margin_used_pct=margin_used_pct,
        recent_trades=recent_trades,
        api_response_time_ms=api_response_time_ms,
        **kwargs
    )

    return {
        "approved": result.approved,
        "critical_failures": result.critical_failures,
        "warnings": result.warnings,
        "summary": result.summary,
        "checks": [
            {
                "name": c.name,
                "passed": c.passed,
                "severity": c.severity,
                "message": c.message,
                "can_override": c.can_override
            }
            for c in result.checks
        ]
    }


# =============================================================================
# LEGACY UYUMLULUK - Eski kod için
# =============================================================================

# Eski TradeQualityGate için alias
TradeQualityGate = TradeSafetyGuard


def check_trade_quality(
    decision: Dict[str, Any],
    **kwargs
) -> Dict[str, Any]:
    """
    LEGACY: Eski quality gate API'si için uyumluluk wrapper'ı.

    Yeni kod check_trade_safety() kullanmalı.
    """
    symbol = decision.get("symbol", "UNKNOWN")
    direction = decision.get("direction", decision.get("base", "long"))
    direction = str(direction).lower()
    recent_trades = kwargs.pop("recent_trades", None)
    # Legacy callers pass analysis-only metadata through this wrapper. The
    # safety guard does not consume these fields, so keep them out of evaluate().
    kwargs.pop("multi_data", None)
    kwargs.pop("regime_info", None)
    kwargs.pop("ai_scores", None)

    # Yeni sistemi çağır
    result = check_trade_safety(
        symbol=symbol,
        direction=direction,
        recent_trades=recent_trades,
        **kwargs
    )

    # Eski format'a çevir
    return {
        "approved": result["approved"],
        "score": 100 if result["approved"] else 30,  # Legacy uyumluluk
        "grade": "A" if result["approved"] else "F",
        "reasons": result["critical_failures"],
        "warnings": result["warnings"],
        "reasoning": result["summary"],
        "checks": result["checks"]
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TRADE SAFETY GUARD - YENİ SİSTEM TEST")
    print("=" * 70)

    guard = TradeSafetyGuard()

    # Test 1: Normal durum
    print("\n[TEST 1] Normal durum - Her şey iyi:")
    result1 = guard.evaluate(
        symbol="BTC/USDT",
        direction="long",
        open_positions=[{"symbol": "ETH/USDT", "pnl": 100}],
        daily_pnl_pct=0.02,  # %2 kar
        margin_used_pct=0.30,  # %30 margin
        recent_trades=[
            {"pnl_pct": 1.5},
            {"pnl_pct": -0.5},
            {"pnl_pct": 2.0}
        ],
        api_response_time_ms=150
    )
    print(f"  Approved: {result1.approved}")
    print(f"  Summary: {result1.summary}")

    # Test 2: Günlük kayıp limiti aşıldı
    print("\n[TEST 2] Günlük kayıp limiti aşıldı:")
    result2 = guard.evaluate(
        symbol="ETH/USDT",
        direction="long",
        daily_pnl_pct=-0.06  # %6 kayıp
    )
    print(f"  Approved: {result2.approved}")
    print(f"  Summary: {result2.summary}")
    print(f"  Critical: {result2.critical_failures}")

    # Test 3: Çok fazla korele pozisyon
    print("\n[TEST 3] Korele pozisyon limiti:")
    result3 = guard.evaluate(
        symbol="SOL/USDT",  # BTC-korele
        direction="long",
        open_positions=[
            {"symbol": "BTC/USDT"},
            {"symbol": "ETH/USDT"}  # Zaten 2 BTC-korele var
        ]
    )
    print(f"  Approved: {result3.approved}")
    print(f"  Summary: {result3.summary}")

    # Test 4: 5 arka arkaya kayıp
    print("\n[TEST 4] Kayıp serisi (5 arka arkaya):")
    result4 = guard.evaluate(
        symbol="DOGE/USDT",
        direction="long",
        recent_trades=[
            {"pnl_pct": -1.0},
            {"pnl_pct": -0.5},
            {"pnl_pct": -2.0},
            {"pnl_pct": -1.5},
            {"pnl_pct": -0.8}
        ]
    )
    print(f"  Approved: {result4.approved}")
    print(f"  Summary: {result4.summary}")
    print(f"  Critical: {result4.critical_failures}")

    # Test 5: Yüksek margin kullanımı
    print("\n[TEST 5] Kritik margin seviyesi:")
    result5 = guard.evaluate(
        symbol="LINK/USDT",
        direction="short",
        margin_used_pct=0.85  # %85 - kritik
    )
    print(f"  Approved: {result5.approved}")
    print(f"  Summary: {result5.summary}")

    # Test 6: 8 arka arkaya kazanç
    print("\n[TEST 6] Aşırı kazanç serisi (8 trade):")
    result6 = guard.evaluate(
        symbol="AVAX/USDT",
        direction="long",
        recent_trades=[
            {"pnl_pct": 1.0},
            {"pnl_pct": 2.0},
            {"pnl_pct": 0.5},
            {"pnl_pct": 1.5},
            {"pnl_pct": 3.0},
            {"pnl_pct": 1.2},
            {"pnl_pct": 0.8},
            {"pnl_pct": 2.5}
        ]
    )
    print(f"  Approved: {result6.approved}")
    print(f"  Summary: {result6.summary}")
    print(f"  Warnings: {result6.warnings}")

    print("\n" + "=" * 70)
    print("TEST TAMAMLANDI")
    print("=" * 70)

