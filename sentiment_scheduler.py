"""
sentiment_scheduler.py
---------------------------------
This script provides a simple scheduler for updating social sentiment
scores at regular intervals using the APScheduler library.  It uses
the ``update_social_sentiment`` function from
``social_sentiment_updater`` to fetch Twitter, Reddit and news
sentiment data and write it to ``metrics/social_sentiment.json``.

Configuration is driven by environment variables:

  SENTIMENT_UPDATE_INTERVAL (int): update frequency in minutes (default 30)
  SENTIMENT_QUERY (str): query term for Twitter and news sentiment (default "crypto")
  SENTIMENT_SUBREDDIT (str): subreddit for Reddit sentiment (default "cryptocurrency")

To run the scheduler, execute this script directly.  It will start a
blocking scheduler that runs until interrupted.  Ensure that
``apscheduler`` is installed in your Python environment.
"""

import os
import logging
from datetime import datetime, timezone
from apscheduler.schedulers.blocking import BlockingScheduler

from analysis.social_sentiment_updater import update_social_sentiment

log = logging.getLogger(__name__)


def _update_job() -> None:
    """Wrapper to call update_social_sentiment with environment overrides."""
    tweet_query = os.getenv("SENTIMENT_QUERY", "crypto")
    reddit_sub = os.getenv("SENTIMENT_SUBREDDIT", "cryptocurrency")
    news_kw = tweet_query
    data = update_social_sentiment(tweet_query, reddit_sub, news_kw)
    log.info(
        "[SentimentScheduler] %sZ updated sentiment: %s",
        datetime.now(timezone.utc).isoformat(),
        data,
    )


def _should_bypass_sentiment() -> bool:
    """Return True if sentiment analysis is disabled in config.json."""
    from pathlib import Path
    import json
    config_path = Path(__file__).resolve().parent / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            weights = cfg.get("decision_weights", {})
            sent_weight = weights.get("sent", weights.get("sentiment", 1.0))
            if sent_weight == 0.0 or sent_weight == 0:
                return True
        except Exception as e:
            log.warning("[SentimentScheduler] Failed to read config.json: %s", e)
    return False


def run_scheduler() -> None:
    """Create and start the blocking scheduler."""
    if _should_bypass_sentiment():
        log.info("[SentimentScheduler] Sentiment weight is 0.0. Sentiment scheduler is disabled via config.json (bypassing idle worker).")
        return
    # Reduce the default update interval from 30 to 15 minutes to
    # better capture rapid shifts in market sentiment.  The interval
    # remains configurable via the SENTIMENT_UPDATE_INTERVAL environment
    # variable (value in minutes).  If unset, a 15‑minute default is used.
    interval_min = int(os.getenv("SENTIMENT_UPDATE_INTERVAL", "15"))
    scheduler = BlockingScheduler()
    scheduler.add_job(
        _update_job,
        'interval',
        minutes=interval_min,
        next_run_time=datetime.now(timezone.utc),
        max_instances=1,
        coalesce=True,
        misfire_grace_time=60,
    )
    log.info(
        "[SentimentScheduler] starting: interval=%s min, query=%s, subreddit=%s",
        interval_min,
        os.getenv("SENTIMENT_QUERY", "crypto"),
        os.getenv("SENTIMENT_SUBREDDIT", "cryptocurrency"),
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("[SentimentScheduler] stopped.")


if __name__ == "__main__":
    run_scheduler()
