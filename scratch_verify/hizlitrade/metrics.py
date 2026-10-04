from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, start_http_server

FEED_MESSAGES = Counter("hizlitrade_feed_messages_total", "Feed messages", ["venue"])
FEED_ERRORS = Counter("hizlitrade_feed_errors_total", "Feed errors", ["venue"])
FEED_LAST_MESSAGE_UNIX_SECONDS = Gauge(
    "hizlitrade_feed_last_message_unixtime",
    "Local receive timestamp of the latest raw market/oracle event",
    ["venue"],
)
BUS_DROPS = Gauge("hizlitrade_bus_drops", "Dropped events", ["subscriber"])
QUOTE_AGE_MS = Histogram(
    "hizlitrade_quote_age_ms",
    "Age of source timestamp at receipt",
    ["venue"],
    buckets=(1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000),
)
OPPORTUNITIES = Counter("hizlitrade_opportunities_total", "Qualified opportunities", ["strategy", "venue"])
PAPER_FILLS = Counter("hizlitrade_paper_fills_total", "Paper fills", ["venue", "side"])
PAPER_EQUITY = Gauge("hizlitrade_paper_equity_usd", "Paper account equity")
CLOCK_DRIFT_MS = Gauge("hizlitrade_clock_drift_ms", "Wall clock drift versus monotonic clock")
RUNTIME_HEALTHY = Gauge("hizlitrade_runtime_healthy", "1 when runtime health guards permit execution")
ORACLE_BASIS_BPS = Gauge(
    "hizlitrade_oracle_basis_bps",
    "Spot consensus minus oracle price in basis points",
    ["symbol", "oracle", "window_seconds"],
)
ORACLE_OBSERVATION_AGE_MS = Histogram(
    "hizlitrade_oracle_observation_age_ms",
    "Age of oracle observation timestamp at local measurement",
    ["symbol", "oracle", "window_seconds"],
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000),
)


def start_metrics(port: int) -> None:
    start_http_server(port)
