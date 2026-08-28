"""
prometheus_exporter.py
----------------------

Prometheus exporter for AutoTraderBot.  Exposes runtime, trading,
model, and exchange metrics over HTTP in the Prometheus exposition format.

[FAZA 9.4] Enhanced with Grafana-ready metrics:
  a) Win rate real-time
  b) Drawdown level & alarm
  c) Model confidence distribution
  d) Exchange latency monitoring
  e) PnL curve data

By default binds to localhost:9100; override via PROMETHEUS_PORT/PROMETHEUS_BIND.

    python prometheus_exporter.py
"""

from __future__ import annotations


from core.exceptions import BEST_EFFORT_EXCEPTIONS
import json
import logging
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger(__name__)

METRICS_DIR = Path("metrics")


def _read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON file, returning empty dict on error."""
    sidecar_path = path.with_name(path.name + ".tmp")
    candidates = [path]
    try:
        if sidecar_path.exists() and (not path.exists() or sidecar_path.stat().st_mtime >= path.stat().st_mtime):
            candidates = [sidecar_path, path]
    except OSError:
        candidates = [sidecar_path, path] if sidecar_path.exists() else [path]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except BEST_EFFORT_EXCEPTIONS:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _fmt(name: str, value: Any, labels: str = "") -> str:
    """Format a single Prometheus metric line."""
    if labels:
        return f"{name}{{{labels}}} {value}\n"
    return f"{name} {value}\n"


def collect_metrics() -> List[str]:
    """
    Collect metrics from JSON files and return them as lines in
    Prometheus exposition format.  Missing values default to 0.
    """
    lines: List[str] = []

    # =========================================================================
    # 1. RUNTIME STATUS
    # =========================================================================
    status = _read_json(METRICS_DIR / "runtime_status.json")

    open_trades = status.get("open_trades") or 0
    daily_pnl = status.get("daily_pnl") or 0
    kill_switch = status.get("kill_switch") or "OFF"
    kill_val = 1 if str(kill_switch).upper() == "ON" else 0

    lines.append("# HELP autotrader_open_trades Number of open positions\n")
    lines.append("# TYPE autotrader_open_trades gauge\n")
    lines.append(_fmt("autotrader_open_trades", open_trades))

    lines.append("# HELP autotrader_daily_pnl Daily realized PnL in USD\n")
    lines.append("# TYPE autotrader_daily_pnl gauge\n")
    lines.append(_fmt("autotrader_daily_pnl", daily_pnl))

    lines.append("# HELP autotrader_kill_switch Kill switch status (1=ON, 0=OFF)\n")
    lines.append("# TYPE autotrader_kill_switch gauge\n")
    lines.append(_fmt("autotrader_kill_switch", kill_val))

    # =========================================================================
    # 2. [FAZA 9.4a] WIN RATE REAL-TIME
    # =========================================================================
    win_rate = status.get("win_rate") or 0.0
    total_trades = status.get("total_trades") or 0
    winning_trades = status.get("winning_trades") or 0
    losing_trades = status.get("losing_trades") or 0

    lines.append("# HELP autotrader_win_rate Current win rate (0-1)\n")
    lines.append("# TYPE autotrader_win_rate gauge\n")
    lines.append(_fmt("autotrader_win_rate", win_rate))

    lines.append("# HELP autotrader_total_trades Total number of closed trades\n")
    lines.append("# TYPE autotrader_total_trades counter\n")
    lines.append(_fmt("autotrader_total_trades", total_trades))

    lines.append("# HELP autotrader_winning_trades Total winning trades\n")
    lines.append("# TYPE autotrader_winning_trades counter\n")
    lines.append(_fmt("autotrader_winning_trades", winning_trades))

    lines.append("# HELP autotrader_losing_trades Total losing trades\n")
    lines.append("# TYPE autotrader_losing_trades counter\n")
    lines.append(_fmt("autotrader_losing_trades", losing_trades))

    # Online learning win rate (more recent)
    ol_state = _read_json(METRICS_DIR / "online_learning_state.json")
    if ol_state:
        ol_win_rate = ol_state.get("recent_win_rate", 0.0)
        ol_total = ol_state.get("total_trades", 0)
        ol_drift_alarms = ol_state.get("drift_alarms_count", 0)
        lines.append("# HELP autotrader_online_win_rate Recent win rate from online learning\n")
        lines.append("# TYPE autotrader_online_win_rate gauge\n")
        lines.append(_fmt("autotrader_online_win_rate", ol_win_rate))
        lines.append(_fmt("autotrader_online_total_trades", ol_total))
        lines.append(_fmt("autotrader_online_drift_alarms", ol_drift_alarms))

    master_score_status = _read_json(METRICS_DIR / "master_score_calibration_status.json")
    if master_score_status:
        chosen_source = master_score_status.get("chosen_source") or master_score_status.get("best_source") or {}
        if not isinstance(chosen_source, dict):
            chosen_source = {}
        calibration = master_score_status.get("calibration") or {}
        if not isinstance(calibration, dict):
            calibration = {}
        status_value = str(master_score_status.get("status") or "").lower()
        reason_value = str(master_score_status.get("reason") or "").lower()
        ready = 1 if status_value == "trained" and reason_value in {"ready", "success", "trained"} else 0
        lines.append("# HELP autotrader_master_score_learning_ready Master score calibration readiness (1=ready)\n")
        lines.append("# TYPE autotrader_master_score_learning_ready gauge\n")
        lines.append(_fmt("autotrader_master_score_learning_ready", ready))
        lines.append("# HELP autotrader_master_score_samples Usable rows in selected master score learning source\n")
        lines.append("# TYPE autotrader_master_score_samples gauge\n")
        lines.append(_fmt("autotrader_master_score_samples", int(chosen_source.get("usable_rows", 0) or 0)))
        lines.append(_fmt("autotrader_master_score_positive_samples", int(chosen_source.get("positive_rows", 0) or 0)))
        lines.append(_fmt("autotrader_master_score_negative_samples", int(chosen_source.get("negative_rows", 0) or 0)))
        try:
            lines.append("# HELP autotrader_master_score_ece Expected calibration error from latest calibration\n")
            lines.append("# TYPE autotrader_master_score_ece gauge\n")
            lines.append(_fmt("autotrader_master_score_ece", float(calibration.get("ece", 0.0) or 0.0)))
        except (TypeError, ValueError):
            lines.append(_fmt("autotrader_master_score_ece", 0.0))
        try:
            lines.append(_fmt("autotrader_master_score_penalty_multiplier", float(calibration.get("penalty_multiplier", 1.0) or 1.0)))
        except (TypeError, ValueError):
            lines.append(_fmt("autotrader_master_score_penalty_multiplier", 1.0))

    # =========================================================================
    # 3. [FAZA 9.4b] DRAWDOWN LEVEL & ALARM
    # =========================================================================
    max_dd = status.get("max_drawdown") or 0.0
    current_dd = status.get("current_drawdown") or 0.0
    profit_factor = status.get("profit_factor") or 0.0

    lines.append("# HELP autotrader_max_drawdown Maximum drawdown percentage\n")
    lines.append("# TYPE autotrader_max_drawdown gauge\n")
    lines.append(_fmt("autotrader_max_drawdown", max_dd))

    lines.append("# HELP autotrader_current_drawdown Current drawdown percentage\n")
    lines.append("# TYPE autotrader_current_drawdown gauge\n")
    lines.append(_fmt("autotrader_current_drawdown", current_dd))

    lines.append("# HELP autotrader_profit_factor Profit factor (gross profit / gross loss)\n")
    lines.append("# TYPE autotrader_profit_factor gauge\n")
    lines.append(_fmt("autotrader_profit_factor", profit_factor))

    # Drawdown alarm levels
    dd_alarm = 0
    if float(max_dd) > 10:
        dd_alarm = 1  # Warning
    if float(max_dd) > 20:
        dd_alarm = 2  # Critical
    lines.append("# HELP autotrader_drawdown_alarm Drawdown alarm (0=ok, 1=warning, 2=critical)\n")
    lines.append("# TYPE autotrader_drawdown_alarm gauge\n")
    lines.append(_fmt("autotrader_drawdown_alarm", dd_alarm))

    # =========================================================================
    # 4. [FAZA 9.4c] MODEL CONFIDENCE DISTRIBUTION
    # =========================================================================
    last_decisions = _read_json(METRICS_DIR / "last_decisions.json")
    decisions = last_decisions.get("decisions") or []

    # Aggregate confidence stats from last batch
    confs = []
    for dec in decisions:
        if isinstance(dec, dict):
            mc = dec.get("master_confidence") or dec.get("master")
            if mc is not None:
                try:
                    confs.append(float(mc))
                except (TypeError, ValueError):
                    pass

    if confs:
        avg_conf = sum(confs) / len(confs)
        min_conf = min(confs)
        max_conf = max(confs)
        # Confidence buckets
        low_conf = sum(1 for c in confs if c < 0.4)
        mid_conf = sum(1 for c in confs if 0.4 <= c < 0.7)
        high_conf = sum(1 for c in confs if c >= 0.7)
    else:
        avg_conf = min_conf = max_conf = 0.0
        low_conf = mid_conf = high_conf = 0

    lines.append("# HELP autotrader_confidence_avg Average model confidence in last batch\n")
    lines.append("# TYPE autotrader_confidence_avg gauge\n")
    lines.append(_fmt("autotrader_confidence_avg", round(avg_conf, 4)))
    lines.append(_fmt("autotrader_confidence_min", round(min_conf, 4)))
    lines.append(_fmt("autotrader_confidence_max", round(max_conf, 4)))

    lines.append("# HELP autotrader_confidence_bucket Number of signals by confidence tier\n")
    lines.append("# TYPE autotrader_confidence_bucket gauge\n")
    lines.append(_fmt("autotrader_confidence_bucket", low_conf, 'tier="low"'))
    lines.append(_fmt("autotrader_confidence_bucket", mid_conf, 'tier="mid"'))
    lines.append(_fmt("autotrader_confidence_bucket", high_conf, 'tier="high"'))

    # Per-model reliability
    reliability = _read_json(METRICS_DIR / "reliability_live.json")
    rel_scores = reliability.get("scores") or {}
    for model_name, score in rel_scores.items():
        lines.append(_fmt("autotrader_model_reliability", score, f'model="{model_name}"'))

    # =========================================================================
    # 5. [FAZA 9.4d] EXCHANGE LATENCY MONITORING
    # =========================================================================
    mm = _read_json(METRICS_DIR / "metrics.json")

    api_calls = mm.get("api_calls") or 0
    api_errors = mm.get("api_errors") or 0
    api_retries = mm.get("api_retries") or 0
    total_latency = mm.get("total_latency") or 0
    avg_latency = mm.get("avg_latency") or 0.0

    lines.append("# HELP autotrader_api_calls_total Total API calls\n")
    lines.append("# TYPE autotrader_api_calls_total counter\n")
    lines.append(_fmt("autotrader_api_calls_total", api_calls))

    lines.append("# HELP autotrader_api_errors_total Total API errors\n")
    lines.append("# TYPE autotrader_api_errors_total counter\n")
    lines.append(_fmt("autotrader_api_errors_total", api_errors))

    lines.append("# HELP autotrader_api_retries_total Total API retries\n")
    lines.append("# TYPE autotrader_api_retries_total counter\n")
    lines.append(_fmt("autotrader_api_retries_total", api_retries))

    lines.append("# HELP autotrader_total_latency_seconds Cumulative API latency\n")
    lines.append("# TYPE autotrader_total_latency_seconds counter\n")
    lines.append(_fmt("autotrader_total_latency_seconds", total_latency))

    lines.append("# HELP autotrader_avg_latency_seconds Average API latency\n")
    lines.append("# TYPE autotrader_avg_latency_seconds gauge\n")
    lines.append(_fmt("autotrader_avg_latency_seconds", avg_latency))

    # Exchange-specific latency (if tracked)
    exchange_latency = mm.get("exchange_latency") or {}
    for exch_name, lat_val in exchange_latency.items():
        lines.append(_fmt("autotrader_exchange_latency_seconds", lat_val, f'exchange="{exch_name}"'))

    # Exchange connection status
    conn_status = mm.get("connection_status") or {}
    for exch_name, connected in conn_status.items():
        lines.append(_fmt("autotrader_exchange_connected", 1 if connected else 0, f'exchange="{exch_name}"'))

    # Slippage
    avg_slippage = status.get("avg_slippage") or 0.0
    lines.append("# HELP autotrader_avg_slippage Average trade slippage\n")
    lines.append("# TYPE autotrader_avg_slippage gauge\n")
    lines.append(_fmt("autotrader_avg_slippage", avg_slippage))

    # =========================================================================
    # 6. [FAZA 9.4e] PNL CURVE DATA
    # =========================================================================
    lines.append("# HELP autotrader_cumulative_pnl Cumulative PnL in USD\n")
    lines.append("# TYPE autotrader_cumulative_pnl gauge\n")
    cumulative_pnl = status.get("cumulative_pnl") or status.get("total_pnl") or 0.0
    lines.append(_fmt("autotrader_cumulative_pnl", cumulative_pnl))

    lines.append("# HELP autotrader_balance_usdt Current USDT balance\n")
    lines.append("# TYPE autotrader_balance_usdt gauge\n")
    balance = status.get("balance") or status.get("total_balance") or 0.0
    lines.append(_fmt("autotrader_balance_usdt", balance))

    lines.append("# HELP autotrader_equity Current equity\n")
    lines.append("# TYPE autotrader_equity gauge\n")
    equity = status.get("equity") or balance
    lines.append(_fmt("autotrader_equity", equity))

    # Today's PnL by hour (if available)
    hourly_pnl = status.get("hourly_pnl") or {}
    for hour, pnl_val in hourly_pnl.items():
        lines.append(_fmt("autotrader_hourly_pnl", pnl_val, f'hour="{hour}"'))

    # =========================================================================
    # 7. ADDITIONAL OPERATIONAL METRICS
    # =========================================================================

    # Reconnection events
    reconnect_count = mm.get("reconnect_count") or 0
    lines.append("# HELP autotrader_reconnect_count Exchange reconnection attempts\n")
    lines.append("# TYPE autotrader_reconnect_count counter\n")
    lines.append(_fmt("autotrader_reconnect_count", reconnect_count))

    # Trade blocks
    trade_blocked = 1 if status.get("trade_blocked") else 0
    lines.append("# HELP autotrader_trade_blocked Whether trading is blocked (1=blocked)\n")
    lines.append("# TYPE autotrader_trade_blocked gauge\n")
    lines.append(_fmt("autotrader_trade_blocked", trade_blocked))

    # V2 policy/expert operational health
    health = _read_json(METRICS_DIR / "health_status.json")
    v2_health = health.get("v2_operational") or {}
    if isinstance(v2_health, dict):
        v2_metrics = v2_health.get("metrics") or {}
        v2_alerts = v2_health.get("alerts") or []
        if not isinstance(v2_metrics, dict):
            v2_metrics = {}
        alert_count = len(v2_alerts) if isinstance(v2_alerts, list) else 0
        lines.append("# HELP autotrader_v2_operational_alerts Number of v2 operational alerts\n")
        lines.append("# TYPE autotrader_v2_operational_alerts gauge\n")
        lines.append(_fmt("autotrader_v2_operational_alerts", alert_count))
        lines.append(_fmt("autotrader_v2_regime_changes_last_hour", v2_metrics.get("regime_changes_last_hour", 0)))
        lines.append(_fmt("autotrader_v2_hours_since_expert_signal", v2_metrics.get("hours_since_expert_signal", 0)))
        lines.append(_fmt("autotrader_v2_edge_all_blocked_or_frozen", 1 if v2_metrics.get("edge_all_blocked_or_frozen") else 0))
        lines.append(_fmt("autotrader_v2_stale_core_data_count", v2_metrics.get("stale_core_data_count", 0)))
        lines.append(_fmt("autotrader_v2_missing_pipeline_version_count", v2_metrics.get("decision_audit_missing_pipeline_version", 0)))
        lines.append(_fmt("autotrader_v2_no_trade_ratio_24h", v2_metrics.get("no_trade_ratio_24h", 0.0)))

    # Recalibration triggers
    recal = _read_json(METRICS_DIR / "recalibration_trigger.json")
    if recal:
        recal_count = recal.get("recalibration_count") or 0
        lines.append(_fmt("autotrader_recalibration_count", recal_count))

    return lines


class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")
            return
        # [FIX-13.2+13.2B] IP whitelist koruması (X-Forwarded-For destekli)
        allowed_ips = os.getenv("PROMETHEUS_ALLOWED_IPS", "").strip()
        if allowed_ips:
            forwarded = self.headers.get("X-Forwarded-For", "")
            client_ip = forwarded.split(",")[0].strip() if forwarded else self.client_address[0]
            allowed_list = [ip.strip() for ip in allowed_ips.split(",")]
            if client_ip not in allowed_list:
                self.send_response(403)
                self.end_headers()
                self.wfile.write(b"Forbidden: IP not whitelisted")
                return
        # [FIX-13] Opsiyonel Bearer token koruması
        expected_token = os.getenv("PROMETHEUS_TOKEN", "")
        if expected_token:
            auth = self.headers.get("Authorization", "")
            if auth != f"Bearer {expected_token}":
                self.send_response(403)
                self.end_headers()
                self.wfile.write(b"Forbidden")
                return
        body = "".join(collect_metrics()).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default HTTP request logging."""
        pass


def run_server(*, restart_delay_sec: float = 5.0, max_restart_attempts: int | None = None) -> None:
    port = int(os.getenv("PROMETHEUS_PORT", "9100"))
    bind_addr = os.getenv("PROMETHEUS_BIND", "127.0.0.1")
    restart_attempts = 0

    while True:
        server: HTTPServer | None = None
        try:
            server = HTTPServer((bind_addr, port), MetricsHandler)
            log.info("Serving metrics on %s:%d", bind_addr, port)
            server.serve_forever()
            return
        except KeyboardInterrupt:
            log.info("Shutting down")
            return
        except OSError as exc:
            restart_attempts += 1
            log.warning(
                "Prometheus exporter socket failure on %s:%d: %s",
                bind_addr,
                port,
                exc,
            )
            if max_restart_attempts is not None and restart_attempts >= max_restart_attempts:
                log.error("Prometheus exporter giving up after %d restart attempts", restart_attempts)
                return
            time.sleep(max(0.0, float(restart_delay_sec)))
        finally:
            if server is not None:
                try:
                    server.server_close()
                except OSError:
                    pass


if __name__ == "__main__":
    run_server()
