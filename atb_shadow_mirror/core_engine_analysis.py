from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ai.llm_cost_optimizer import (
    apply_llm_cost_guard_to_candidates,
    assign_llm_candidate_optimizer_routes,
    build_llm_batch_optimizer_summary,
    build_llm_candidate_shadow_evaluation,
    build_llm_cost_shadow_summary,
    build_llm_runtime_cost_report,
    load_llm_batch_optimizer_history,
    load_llm_candidate_optimizer_history,
    load_llm_cost_optimizer_state,
    notify_llm_cost_optimizer_status,
    save_llm_cost_optimizer_state,
    select_llm_batch_size,
    select_llm_candidate_optimizer_active_profile,
    should_run_full_universe_audit,
    update_llm_cost_optimizer_state_from_shadow,
    update_routine_no_order_gate_state_from_cycle,
    update_weak_skip_gate_state_from_cycle,
)
from atomic_io import enqueue_append_jsonl, safe_append_jsonl, safe_write_text
from core.exceptions import BEST_EFFORT_EXCEPTIONS

from .analysis_batch import EngineAnalysisBatchService
from .analysis_execution import EngineAnalysisExecutionService
from .analysis_pipeline import EngineAnalysisPipelineService

if TYPE_CHECKING:
    from core.engine.bot import BotEngine


log = logging.getLogger(__name__)

_REGIME_FILE = Path("data") / "market_regime.json"
_REGIME_REFRESH_INTERVAL = 900  # 15 dakika
_REGIME_REFRESH_TIMEOUT_SEC = 60.0
_REGIME_REFRESH_LATENCIES: list[float] = []
_REGIME_REFRESH_LATENCY_MAX_SAMPLES = 512


def record_regime_refresh_latency(seconds: float) -> None:
    try:
        value = max(0.0, float(seconds))
    except (TypeError, ValueError):
        return
    _REGIME_REFRESH_LATENCIES.append(value)
    if len(_REGIME_REFRESH_LATENCIES) > _REGIME_REFRESH_LATENCY_MAX_SAMPLES:
        del _REGIME_REFRESH_LATENCIES[: len(_REGIME_REFRESH_LATENCIES) - _REGIME_REFRESH_LATENCY_MAX_SAMPLES]


def get_regime_refresh_latency_stats() -> dict[str, float | int]:
    samples = sorted(_REGIME_REFRESH_LATENCIES)
    if not samples:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0}

    def percentile(pct: float) -> float:
        idx = min(len(samples) - 1, max(0, int(round((len(samples) - 1) * pct))))
        return float(samples[idx])

    return {
        "count": len(samples),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
    }


def evaluate_regime_refresh_latency_health(config: Dict[str, Any] | None = None) -> dict[str, float | int | str]:
    stats = get_regime_refresh_latency_stats()
    threshold = _performance_float(config, "regime_refresh_p99_warn_sec", 5.0, 0.1)
    p99 = float(stats.get("p99", 0.0) or 0.0)
    return {
        **stats,
        "status": "warn" if p99 > threshold else "ok",
        "threshold_seconds": threshold,
    }


def _performance_float(config: Dict[str, Any] | None, key: str, default: float, minimum: float) -> float:
    if not isinstance(config, dict):
        return default
    perf_cfg = config.get("performance")
    if not isinstance(perf_cfg, dict):
        return default
    try:
        return max(minimum, float(perf_cfg.get(key, default)))
    except (TypeError, ValueError):
        return default


def _performance_bool(config: Dict[str, Any] | None, key: str, default: bool) -> bool:
    if not isinstance(config, dict):
        return default
    perf_cfg = config.get("performance")
    if not isinstance(perf_cfg, dict):
        return default
    value = perf_cfg.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return default


def _normalize_optimizer_symbol(symbol: Any) -> str:
    raw = str(symbol or "").strip().upper()
    if ":" in raw:
        raw = raw.split(":", 1)[0]
    return raw


def _optimizer_candidate_bool(config_path: str | Path, key: str, default: bool) -> bool:
    try:
        payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return default
    if not isinstance(payload, dict):
        return default
    llm_cfg = payload.get("llm_cost_optimizer") if isinstance(payload.get("llm_cost_optimizer"), dict) else {}
    optimizer = llm_cfg.get("candidate_optimizer") if isinstance(llm_cfg.get("candidate_optimizer"), dict) else {}
    value = optimizer.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", "disabled"}
    return bool(value)


def _optimizer_candidate_int(config_path: str | Path, key: str, default: int, minimum: int = 0) -> int:
    try:
        payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return max(minimum, int(default))
    if not isinstance(payload, dict):
        return max(minimum, int(default))
    llm_cfg = payload.get("llm_cost_optimizer") if isinstance(payload.get("llm_cost_optimizer"), dict) else {}
    optimizer = llm_cfg.get("candidate_optimizer") if isinstance(llm_cfg.get("candidate_optimizer"), dict) else {}
    try:
        return max(minimum, int(optimizer.get(key, default)))
    except (TypeError, ValueError):
        return max(minimum, int(default))


def _optimizer_force_recent_missed_enabled(config_path: str | Path = "config.json") -> bool:
    return _optimizer_candidate_bool(config_path, "force_recent_missed_enters", True)


def _optimizer_force_recent_final_enabled(config_path: str | Path = "config.json") -> bool:
    return _optimizer_candidate_bool(config_path, "force_recent_final_enters", True)


def _recent_shadow_missed_enter_symbols(history: List[Dict[str, Any]] | None) -> set[str]:
    symbols: set[str] = set()
    for entry in history or []:
        if not isinstance(entry, dict):
            continue
        missed = entry.get("missed_enter_symbols")
        if not isinstance(missed, list):
            continue
        for symbol in missed:
            normalized = _normalize_optimizer_symbol(symbol)
            if normalized:
                symbols.add(normalized)
    return symbols


def _recent_shadow_final_enter_symbols(
    history: List[Dict[str, Any]] | None,
    *,
    config_path: str | Path = "config.json",
) -> set[str]:
    lookback = _optimizer_candidate_int(config_path, "recent_final_enter_lookback_cycles", 3, 1)
    cap = _optimizer_candidate_int(config_path, "max_recent_final_enter_symbols", 20, 0)
    if cap <= 0:
        return set()
    symbols: list[str] = []
    for entry in reversed((history or [])[-lookback:]):
        if not isinstance(entry, dict):
            continue
        final_enters = entry.get("final_enter_symbols")
        if not isinstance(final_enters, list):
            continue
        for symbol in final_enters:
            normalized = _normalize_optimizer_symbol(symbol)
            if normalized and normalized not in symbols:
                symbols.append(normalized)
                if len(symbols) >= cap:
                    return set(symbols)
    return set(symbols)


def _candidate_block_reason(item: Dict[str, Any]) -> str:
    raw_reason = item.get("skip_reason") or item.get("error")
    if not raw_reason and str(item.get("candidate_route") or "") == "blocked":
        raw_reason = item.get("candidate_reason")
    if raw_reason:
        text = str(raw_reason).strip()
    else:
        quality = item.get("data_quality") if isinstance(item.get("data_quality"), dict) else {}
        text = str(quality.get("reason_code") or quality.get("reason") or "").strip()
    if not text:
        return "unknown"
    lowered = text.lower()
    if "empty ohlcv" in lowered or lowered == "empty_ohlcv":
        return "empty_ohlcv"
    if "price divergence" in lowered or lowered == "price_divergence":
        return "price_divergence"
    if "ohlcv bar age" in lowered or lowered in {"stale_at_fetch", "stale_due_to_batch_delay"}:
        return lowered if lowered == "stale_due_to_batch_delay" else "stale_at_fetch"
    if "low_volume" in lowered or "low volume" in lowered:
        return "low_volume"
    if lowered.startswith("hard_gate:"):
        return lowered.split(":", 1)[1].replace(" ", "_")[:80] or "hard_gate"
    return lowered.split(":", 1)[0].replace(" ", "_")[:80]


def _load_hard_safety_gate_config() -> dict[str, Any]:
    try:
        from core.config_loader import load_config

        cfg = load_config()
        return cfg if isinstance(cfg, dict) else {}
    except BEST_EFFORT_EXCEPTIONS as exc:
        log.debug("[ENGINE] hard safety gate config unavailable: %s", exc)
        return {}


def assign_llm_candidate_routes(batch_items: List[Dict[str, Any]]) -> None:
    try:
        from decision.hard_safety_gate import evaluate_hard_safety_gate

        gate_config = _load_hard_safety_gate_config()
    except BEST_EFFORT_EXCEPTIONS as exc:
        evaluate_hard_safety_gate = None
        gate_config = {}
        log.debug("[ENGINE] hard safety gate unavailable before LLM routing: %s", exc)
    for item in batch_items:
        blocked = bool(item.get("skip_reason") or item.get("error"))
        if blocked:
            item["candidate_route"] = "blocked"
            item["candidate_reason"] = _candidate_block_reason(item)
            continue
        if evaluate_hard_safety_gate is not None:
            gate = evaluate_hard_safety_gate(item, gate_config)
            item["hard_safety_gate"] = gate
            if not bool(gate.get("allowed", True)):
                item["candidate_route"] = "blocked"
                item["candidate_reason"] = f"hard_gate:{gate.get('reason') or 'blocked'}"
                continue
        repair = item.get("data_repair") if isinstance(item.get("data_repair"), dict) else {}
        quality = item.get("data_quality") if isinstance(item.get("data_quality"), dict) else {}
        if repair.get("status") == "repaired":
            item["candidate_route"] = "llm"
            item["candidate_reason"] = "repaired_major_data"
        elif quality.get("status") == "usable":
            item["candidate_route"] = "llm"
            item["candidate_reason"] = "usable_data_quality"
        else:
            item["candidate_route"] = "llm"
            item["candidate_reason"] = "clean_data"


def build_llm_candidate_summary(
    batch_items: List[Dict[str, Any]],
    *,
    llm_sent: int,
    batch_size: int = 20,
) -> Dict[str, Any]:
    blocked_by_reason: Dict[str, int] = {}
    major_repaired: List[str] = []
    major_blocked: List[str] = []

    for item in batch_items:
        symbol = str(item.get("symbol") or "")
        route = str(item.get("candidate_route") or "")
        quality = item.get("data_quality") if isinstance(item.get("data_quality"), dict) else {}
        repair = item.get("data_repair") if isinstance(item.get("data_repair"), dict) else {}
        is_major = bool(quality.get("major_symbol"))
        if repair.get("status") == "repaired" and symbol:
            major_repaired.append(symbol)
        if route == "blocked":
            reason = _candidate_block_reason(item)
            blocked_by_reason[reason] = blocked_by_reason.get(reason, 0) + 1
            if is_major and symbol:
                major_blocked.append(symbol)

    try:
        safe_batch_size = max(1, int(batch_size))
    except (TypeError, ValueError):
        safe_batch_size = 20
    llm_batches = (int(llm_sent) + safe_batch_size - 1) // safe_batch_size if llm_sent else 0

    return {
        "schema": "llm-candidate-quality-summary-v1",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "universe_total": len(batch_items),
        "technical_analyzed": len(batch_items),
        "llm_sent": int(llm_sent),
        "blocked_by_reason": dict(sorted(blocked_by_reason.items())),
        "major_repaired": sorted(set(major_repaired)),
        "major_blocked": sorted(set(major_blocked)),
        "llm_batches": llm_batches,
    }


def _llm_runtime_cost_report_path(config_path: str | Path) -> Path:
    cfg_path = Path(config_path)
    if cfg_path.parent != Path("."):
        return cfg_path.parent / "metrics" / "llm_runtime_cost_report.json"
    return Path("metrics") / "llm_runtime_cost_report.json"


def apply_llm_candidate_optimizer_selection(
    *,
    batch_items: List[Dict[str, Any]],
    history: List[Dict[str, Any]] | None = None,
    state: Dict[str, Any] | None = None,
    config_path: str | Path = "config.json",
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if history is None:
        history = load_llm_candidate_optimizer_history()
    if state is None:
        state = load_llm_cost_optimizer_state()
    force_symbols = (
        _recent_shadow_missed_enter_symbols(history) if _optimizer_force_recent_missed_enabled(config_path) else set()
    )
    force_final_symbols = (
        _recent_shadow_final_enter_symbols(history, config_path=config_path)
        if _optimizer_force_recent_final_enabled(config_path)
        else set()
    )
    optimizer_summary = assign_llm_candidate_optimizer_routes(
        batch_items=batch_items,
        config_path=config_path,
        force_symbols=force_symbols,
        force_final_enter_symbols=force_final_symbols,
    )
    active_profile = select_llm_candidate_optimizer_active_profile(
        history=history,
        state=state,
        config_path=config_path,
    )
    full_universe_audit = should_run_full_universe_audit(
        state=state,
        active_profile=active_profile,
        config_path=config_path,
    )
    active = bool(active_profile) and not full_universe_audit
    for item in batch_items:
        if item.get("skip_reason") or item.get("error"):
            continue
        item.setdefault("candidate_route_before_optimizer", item.get("candidate_route"))
        profile_route = None
        if active_profile and isinstance(item.get("candidate_profile_routes"), dict):
            profile_route = item.get("candidate_profile_routes", {}).get(active_profile)
        if active and item.get("candidate_route") == "llm" and profile_route != "llm":
            shadow_reason = str(item.get("candidate_reason_shadow") or f"profile_{active_profile}_technical_only")
            item["candidate_route"] = "technical_only"
            item["candidate_reason"] = f"candidate_optimizer_active;{shadow_reason}"

    cost_guard_summary = apply_llm_cost_guard_to_candidates(
        batch_items,
        cost_report=_read_metrics_json(_llm_runtime_cost_report_path(config_path)),
        config_path=config_path,
    )
    selected: List[Dict[str, Any]] = [
        item for item in batch_items if isinstance(item, dict) and item.get("candidate_route") == "llm"
    ]

    optimizer_summary["active"] = active
    optimizer_summary["active_profile"] = active_profile
    optimizer_summary["full_universe_audit"] = full_universe_audit
    optimizer_summary["cost_guard"] = cost_guard_summary
    optimizer_summary["history_cycles"] = len(history or [])
    optimizer_summary["actual_llm_sent_after_optimizer"] = len(selected)
    return selected, optimizer_summary


def persist_llm_candidate_summary(summary: Dict[str, Any]) -> None:
    try:
        metrics_dir = Path("metrics")
        logs_dir = Path("logs")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        safe_write_text(
            metrics_dir / "llm_candidate_summary.json",
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        audit_file = logs_dir / "llm_candidate_audit.jsonl"
        if not enqueue_append_jsonl(audit_file, summary):
            safe_append_jsonl(audit_file, summary)
    except BEST_EFFORT_EXCEPTIONS as exc:
        log.debug("[ENGINE] LLM candidate summary write failed: %s", exc)


def persist_model_influence_from_batch_items(batch_items: List[Dict[str, Any]]) -> None:
    try:
        from decision.model_influence import append_model_influence_record, build_model_influence_record

        for item in batch_items:
            if not isinstance(item, dict):
                continue
            route = str(item.get("candidate_route") or item.get("route") or "unknown")
            reason = str(item.get("candidate_reason") or item.get("skip_reason") or item.get("error") or "") or None
            confidence = item.get("candidate_score") or item.get("master_confidence") or item.get("confidence") or 0.0
            components = {
                "technical": item.get("technical_score") or item.get("tech_score"),
                "ai": item.get("ai_score"),
                "rl": item.get("rl_score"),
                "sentiment": item.get("sent_score") or item.get("sentiment_score"),
                "regime": item.get("regime") or item.get("market_regime"),
                "llm_candidate_route": route,
            }
            append_model_influence_record(
                build_model_influence_record(
                    symbol=str(item.get("symbol") or item.get("sym") or ""),
                    final_action=f"candidate_route:{route}",
                    confidence_before=confidence,
                    confidence_after=confidence,
                    size_before=0.0,
                    size_after=0.0,
                    leverage_before=1,
                    leverage_after=1,
                    wallet_pct_before=0.0,
                    wallet_pct_after=0.0,
                    components=components,
                    llm_advisory_only=True,
                    block_reason=reason,
                    source="llm_candidate_router",
                    extra={
                        "candidate_route": route,
                        "candidate_reason": reason,
                        "llm_sent": route == "llm",
                        "direct_order_authority": False,
                        "direct_sizing_authority": False,
                        "direct_sl_tp_authority": False,
                    },
                )
            )
    except BEST_EFFORT_EXCEPTIONS as exc:
        log.debug("[ENGINE] model influence write failed: %s", exc)


def persist_llm_candidate_optimizer_shadow(summary: Dict[str, Any]) -> None:
    try:
        metrics_dir = Path("metrics")
        logs_dir = Path("logs")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        safe_write_text(
            metrics_dir / "llm_candidate_optimizer_shadow.json",
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if summary.get("evaluated") is True and summary.get("active_filtering") is not True:
            audit_file = logs_dir / "llm_candidate_optimizer_shadow.jsonl"
            if not enqueue_append_jsonl(audit_file, summary):
                safe_append_jsonl(audit_file, summary)
    except BEST_EFFORT_EXCEPTIONS as exc:
        log.debug("[ENGINE] LLM candidate optimizer shadow write failed: %s", exc)


def persist_llm_cost_shadow_summary(summary: Dict[str, Any]) -> None:
    try:
        metrics_dir = Path("metrics")
        logs_dir = Path("logs")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        safe_write_text(
            metrics_dir / "llm_cost_shadow_summary.json",
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        audit_file = logs_dir / "llm_cost_shadow_audit.jsonl"
        if not enqueue_append_jsonl(audit_file, summary):
            safe_append_jsonl(audit_file, summary)
    except BEST_EFFORT_EXCEPTIONS as exc:
        log.debug("[ENGINE] LLM cost shadow summary write failed: %s", exc)


def persist_llm_batch_optimizer_summary(summary: Dict[str, Any]) -> None:
    try:
        metrics_dir = Path("metrics")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        safe_write_text(
            metrics_dir / "llm_batch_optimizer_summary.json",
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except BEST_EFFORT_EXCEPTIONS as exc:
        log.debug("[ENGINE] LLM batch optimizer summary write failed: %s", exc)


def persist_llm_runtime_cost_report(summary: Dict[str, Any]) -> None:
    try:
        metrics_dir = Path("metrics")
        logs_dir = Path("logs")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        safe_write_text(
            metrics_dir / "llm_runtime_cost_report.json",
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        audit_file = logs_dir / "llm_runtime_cost_report.jsonl"
        if not enqueue_append_jsonl(audit_file, summary):
            safe_append_jsonl(audit_file, summary)
    except BEST_EFFORT_EXCEPTIONS as exc:
        log.debug("[ENGINE] LLM runtime cost report write failed: %s", exc)


def _read_metrics_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def notify_llm_cost_gate_status(
    *,
    candidate_summary: Dict[str, Any],
    cost_report: Dict[str, Any],
    cost_shadow: Dict[str, Any],
    batch_optimizer_summary: Dict[str, Any],
    optimizer_state: Dict[str, Any],
    optimizer_history: List[Dict[str, Any]],
) -> None:
    try:
        prompt_summary = _read_metrics_json(Path("metrics") / "llm_prompt_ab_summary.json")
        sent = notify_llm_cost_optimizer_status(
            candidate_summary=candidate_summary,
            cost_report=cost_report,
            cost_shadow_summary=cost_shadow,
            batch_optimizer_summary=batch_optimizer_summary,
            prompt_ab_summary=prompt_summary,
            state=optimizer_state,
            history=optimizer_history,
        )
        log.info("[ENGINE] LLM cost Telegram status notification queued=%s", sent)
    except BEST_EFFORT_EXCEPTIONS as exc:
        log.debug("[ENGINE] LLM cost Telegram status notification skipped: %s", exc)


class EngineAnalysisService:
    def __init__(self, engine: "BotEngine") -> None:
        self.engine = engine
        self.batch_service = EngineAnalysisBatchService(engine)
        self.pipeline_service = EngineAnalysisPipelineService(engine)
        self.execution_service = EngineAnalysisExecutionService(engine)

    def _touch_loop_watchdog(self, phase: str, **details: Any) -> None:
        touch = getattr(self.engine, "touch_loop_watchdog", None)
        if callable(touch):
            touch(phase, **details)

    async def ticker_for_volume_filter(self, symbol: str, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return await self.batch_service.ticker_for_volume_filter(symbol, item)

    def extract_recent_ohlcv_rows(self, item: Dict[str, Any]) -> List[List[Any]]:
        return self.batch_service.extract_recent_ohlcv_rows(item)

    async def decide_batch_typed(
        self,
        pipeline: Any,
        items: List[Any],
    ) -> Dict[str, Any]:
        return await self.pipeline_service.decide_batch_typed(pipeline, items)

    def uses_external_pipeline(self) -> bool:
        return hasattr(self.engine, "_decision_pipeline") and hasattr(self.engine, "_execution_router")

    def _llm_candidate_base_batch_size(self) -> int:
        config = self.engine.config if isinstance(self.engine.config, dict) else {}
        router = config.get("llm_candidate_router") if isinstance(config.get("llm_candidate_router"), dict) else {}
        try:
            return max(1, min(20, int(router.get("batch_size", 20))))
        except (TypeError, ValueError):
            return 20

    def _llm_candidate_batch_size(self) -> int:
        base_size = self._llm_candidate_base_batch_size()
        return select_llm_batch_size(
            default_batch_size=base_size,
            history=load_llm_batch_optimizer_history(),
        )

    async def _maybe_refresh_regime(self) -> None:
        """Rejim verisini gerektiğinde yeniler (15+ dakika eskiyse)."""
        should_refresh = True
        if _REGIME_FILE.exists():
            try:
                data = json.loads(_REGIME_FILE.read_text(encoding="utf-8"))
                last_update = data.get("last_update")
                if last_update:
                    last_dt = datetime.strptime(last_update, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                    age_sec = (datetime.now(timezone.utc) - last_dt).total_seconds()
                    should_refresh = age_sec > _REGIME_REFRESH_INTERVAL
            except BEST_EFFORT_EXCEPTIONS:
                should_refresh = True

        if not should_refresh:
            return

        try:
            from regime_detector import detect_market_regime
            from runtime.cpu_worker import run_cpu_bound

            timeout_sec = _performance_float(
                self.engine.config,
                "regime_refresh_timeout_sec",
                _REGIME_REFRESH_TIMEOUT_SEC,
                10.0,
            )
            use_process = _performance_bool(self.engine.config, "regime_refresh_use_process", True)
            started = time.perf_counter()
            result = await run_cpu_bound(
                detect_market_regime,
                "BTC/USDT",
                500,
                timeout_sec=timeout_sec,
                use_process=use_process,
            )
            elapsed = time.perf_counter() - started
            record_regime_refresh_latency(elapsed)
            health = evaluate_regime_refresh_latency_health(self.engine.config)
            if health.get("status") == "warn":
                log.warning(
                    "[ENGINE] Regime refresh latency high: p99=%.3fs threshold=%.3fs; cached regime remains available",
                    float(health.get("p99", 0.0) or 0.0),
                    float(health.get("threshold_seconds", 0.0) or 0.0),
                )
            regime = result.get("REGIME", "UNKNOWN")
            log.info(
                "[ENGINE] Rejim güncellendi: %s (adx=%.1f, ema_diff=%.4f)",
                regime,
                result.get("adx_mean", 0),
                result.get("ema_diff_mean", 0),
            )
        except asyncio.TimeoutError:
            log.warning(
                "[ENGINE] Regime refresh timed out after %.0fs; cached regime will be used",
                timeout_sec,
            )
        except BEST_EFFORT_EXCEPTIONS as exc:
            log.warning("[ENGINE] Rejim güncellemesi başarısız: %s", exc)

    async def run_cycle(self) -> None:
        if self.uses_external_pipeline():
            self._touch_loop_watchdog("external_pipeline_start")
            await self.pipeline_service.run_external_pipeline_cycle()
            self._touch_loop_watchdog("external_pipeline_done")
            return

        self._touch_loop_watchdog("regime_refresh_start")
        await self._maybe_refresh_regime()
        self._touch_loop_watchdog("regime_refresh_done")

        self._touch_loop_watchdog("analysis_batch_start")
        batch_items = await self.batch_service.build_batch_items()
        self._touch_loop_watchdog("analysis_batch_done", symbols=len(batch_items))
        if not batch_items:
            return

        # V5 is research-only here. The feature flag is default-off and the
        # shadow payload has no path into decisions or order execution.
        try:
            from decision.v5_breadth_shadow import run_v5_breadth_shadow_cycle

            v5_shadow = run_v5_breadth_shadow_cycle(
                batch_items,
                config=self.engine.config,
            )
            if bool(v5_shadow.get("feature_flag_enabled")):
                self._touch_loop_watchdog(
                    "v5_breadth_shadow_done",
                    ready=bool(v5_shadow.get("ready")),
                    candidates=len(v5_shadow.get("candidates") or []),
                )
                log.info(
                    "[V5_SHADOW] ready=%s reason=%s candidates=%s recorded=%s execution_authority=%s",
                    bool(v5_shadow.get("ready")),
                    v5_shadow.get("reason"),
                    len(v5_shadow.get("candidates") or []),
                    bool(v5_shadow.get("evidence_recorded")),
                    bool(v5_shadow.get("execution_authority")),
                )
        except BEST_EFFORT_EXCEPTIONS as exc:
            log.warning("[V5_SHADOW] research-only evaluation failed: %s", exc)

        assign_llm_candidate_routes(batch_items)
        optimizer_state = load_llm_cost_optimizer_state()
        optimizer_history = load_llm_candidate_optimizer_history()
        valid_items, optimizer_summary = apply_llm_candidate_optimizer_selection(
            batch_items=batch_items,
            history=optimizer_history,
            state=optimizer_state,
        )
        self._touch_loop_watchdog(
            "candidate_selection_done",
            universe=len(batch_items),
            llm_sent=len(valid_items),
        )
        summary = build_llm_candidate_summary(
            batch_items,
            llm_sent=len(valid_items),
            batch_size=self._llm_candidate_batch_size(),
        )
        summary["candidate_optimizer"] = optimizer_summary
        persist_llm_candidate_summary(summary)
        persist_model_influence_from_batch_items(batch_items)
        cost_shadow = build_llm_cost_shadow_summary(batch_items=batch_items)
        persist_llm_cost_shadow_summary(cost_shadow)
        batch_optimizer_summary = build_llm_batch_optimizer_summary(
            llm_sent=len(valid_items),
            current_batch_size=self._llm_candidate_base_batch_size(),
            history=load_llm_batch_optimizer_history(),
        )
        persist_llm_batch_optimizer_summary(batch_optimizer_summary)
        log.info(
            "[ENGINE] LLM candidate summary: universe=%s llm_sent=%s blocked=%s major_repaired=%s major_blocked=%s optimizer_active=%s shadow_llm_sent=%s",
            summary.get("universe_total"),
            summary.get("llm_sent"),
            summary.get("blocked_by_reason"),
            summary.get("major_repaired"),
            summary.get("major_blocked"),
            optimizer_summary.get("active"),
            optimizer_summary.get("shadow_llm_sent"),
        )
        log.info(
            "[ENGINE] LLM cost shadow: provider_policy=%s current_provider_calls=%s optimized_provider_calls=%s shadow=%s",
            cost_shadow.get("provider_policy"),
            cost_shadow.get("current_provider_calls_estimate"),
            cost_shadow.get("optimized_provider_calls_estimate"),
            cost_shadow.get("shadow_mode"),
        )
        log.info(
            "[ENGINE] LLM batch optimizer: current_batch=%s selected_batch=%s current_batches=%s selected_batches=%s",
            batch_optimizer_summary.get("current_batch_size"),
            batch_optimizer_summary.get("selected_batch_size"),
            batch_optimizer_summary.get("current_batches"),
            batch_optimizer_summary.get("selected_batches"),
        )
        log.info(
            "[ENGINE] batch_items detaylari: %s",
            [
                {
                    "sym": item.get("symbol"),
                    "route": item.get("candidate_route"),
                    "reason": item.get("candidate_reason"),
                    "skip": item.get("skip_reason"),
                    "err": item.get("error"),
                }
                for item in batch_items
            ],
        )
        if not valid_items:
            return

        try:
            self._touch_loop_watchdog("enrich_valid_items_start", symbols=len(valid_items))
            await self.batch_service.enrich_valid_items(valid_items)
            self._touch_loop_watchdog("enrich_valid_items_done", symbols=len(valid_items))
            self._touch_loop_watchdog("microstructure_start", symbols=len(valid_items))
            await self.execution_service.feed_microstructure(valid_items)
            self._touch_loop_watchdog("microstructure_done", symbols=len(valid_items))
            llm_cycle_started_epoch = time.time()
            self._touch_loop_watchdog("decision_batch_start", symbols=len(valid_items))
            decisions = await self.pipeline_service.build_decisions(valid_items)
            llm_cycle_duration_sec = time.time() - llm_cycle_started_epoch
            self._touch_loop_watchdog(
                "decision_batch_done",
                symbols=len(valid_items),
                decisions=len(decisions or {}),
                duration_sec=round(float(llm_cycle_duration_sec), 3),
            )
            if not decisions:
                shadow_eval = build_llm_candidate_shadow_evaluation(
                    batch_items=batch_items,
                    decisions={},
                )
                shadow_eval["evaluated"] = False
                shadow_eval["reason"] = "no_decisions_returned"
                shadow_eval["active_filtering"] = bool(optimizer_summary.get("active"))
                persist_llm_candidate_optimizer_shadow(shadow_eval)
                next_state = update_llm_cost_optimizer_state_from_shadow(
                    state=optimizer_state,
                    shadow_eval=shadow_eval,
                    active_profile=optimizer_summary.get("active_profile"),
                    audit_cycle=bool(optimizer_summary.get("full_universe_audit")),
                )
                save_llm_cost_optimizer_state(next_state)
                cost_report = build_llm_runtime_cost_report(
                    cycle_started_epoch=llm_cycle_started_epoch,
                    cycle_duration_sec=llm_cycle_duration_sec,
                    llm_sent=len(valid_items),
                    cost_shadow_summary=cost_shadow,
                )
                next_state = update_weak_skip_gate_state_from_cycle(
                    state=next_state,
                    shadow_eval=shadow_eval,
                    cost_report=cost_report,
                )
                next_state = update_routine_no_order_gate_state_from_cycle(
                    state=next_state,
                    shadow_eval=shadow_eval,
                    cost_report=cost_report,
                )
                save_llm_cost_optimizer_state(next_state)
                persist_llm_runtime_cost_report(cost_report)
                notify_llm_cost_gate_status(
                    candidate_summary=summary,
                    cost_report=cost_report,
                    cost_shadow=cost_shadow,
                    batch_optimizer_summary=batch_optimizer_summary,
                    optimizer_state=next_state,
                    optimizer_history=optimizer_history,
                )
                self._touch_loop_watchdog("cost_report_done", llm_sent=len(valid_items), decisions=0)
                return
            shadow_eval = build_llm_candidate_shadow_evaluation(
                batch_items=batch_items,
                decisions=decisions,
            )
            shadow_eval["active_filtering"] = bool(optimizer_summary.get("active"))
            if optimizer_summary.get("active"):
                shadow_eval["evaluated"] = False
                shadow_eval["reason"] = "active_filtering_no_full_llm_baseline"
            persist_llm_candidate_optimizer_shadow(shadow_eval)
            next_state = update_llm_cost_optimizer_state_from_shadow(
                state=optimizer_state,
                shadow_eval=shadow_eval,
                active_profile=optimizer_summary.get("active_profile"),
                audit_cycle=bool(optimizer_summary.get("full_universe_audit")),
            )
            save_llm_cost_optimizer_state(next_state)
            cost_report = build_llm_runtime_cost_report(
                cycle_started_epoch=llm_cycle_started_epoch,
                cycle_duration_sec=llm_cycle_duration_sec,
                llm_sent=len(valid_items),
                cost_shadow_summary=cost_shadow,
            )
            next_state = update_weak_skip_gate_state_from_cycle(
                state=next_state,
                shadow_eval=shadow_eval,
                cost_report=cost_report,
            )
            next_state = update_routine_no_order_gate_state_from_cycle(
                state=next_state,
                shadow_eval=shadow_eval,
                cost_report=cost_report,
            )
            save_llm_cost_optimizer_state(next_state)
            persist_llm_runtime_cost_report(cost_report)
            notify_llm_cost_gate_status(
                candidate_summary=summary,
                cost_report=cost_report,
                cost_shadow=cost_shadow,
                batch_optimizer_summary=batch_optimizer_summary,
                optimizer_state=next_state,
                optimizer_history=optimizer_history,
            )
            self._touch_loop_watchdog("cost_report_done", llm_sent=len(valid_items), decisions=len(decisions))
            from .sideways_grid import execute_sideways_grid_if_applicable

            grid_handled, grid_result = await execute_sideways_grid_if_applicable(
                getattr(self.engine, "exchange", None),
                valid_items,
                decisions,
                logger=log,
            )
            if grid_handled:
                self._touch_loop_watchdog(
                    "grid_dca_done",
                    symbols=len(valid_items),
                    decisions=len(decisions),
                    status=grid_result.get("status"),
                    submitted=len(grid_result.get("submitted") or []),
                    skipped=len(grid_result.get("skipped") or []),
                )
                return
            self._touch_loop_watchdog("execution_start", symbols=len(valid_items), decisions=len(decisions))
            await self.execution_service.execute_decisions(valid_items, decisions)
            self._touch_loop_watchdog("execution_done", symbols=len(valid_items), decisions=len(decisions))
        finally:
            for item in batch_items:
                item.pop("mtf_data", None)
                item.pop("mtf_features", None)
                item.pop("orderbook", None)
                item.pop("ticker", None)
            batch_items.clear()
            valid_items.clear()
            self._touch_loop_watchdog("analysis_cleanup_done")
