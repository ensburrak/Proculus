from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .metrics import FEED_LAST_MESSAGE_UNIX_SECONDS
from .models import OraclePrice, Quote


@dataclass(slots=True)
class SubscriberStats:
    published: int = 0
    dropped: int = 0


class BroadcastBus:
    """Small in-process fan-out bus with bounded per-subscriber queues.

    Slow consumers cannot block market-data ingestion. Their oldest queued event
    is dropped and the drop counter is incremented instead. Subscribers may also
    declare both event classes and a lightweight predicate so high-rate,
    unrelated market data is never enqueued for them in the first place.
    """

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[Any]] = {}
        self._stats: dict[str, SubscriberStats] = {}
        self._event_types: dict[str, tuple[type[Any], ...] | None] = {}
        self._event_filters: dict[str, Callable[[object], bool] | None] = {}

    def subscribe(
        self,
        name: str,
        *,
        maxsize: int = 10_000,
        event_types: tuple[type[Any], ...] | None = None,
        event_filter: Callable[[object], bool] | None = None,
    ) -> asyncio.Queue[Any]:
        if name in self._queues:
            raise ValueError(f"subscriber already exists: {name}")
        if maxsize <= 0:
            raise ValueError("subscriber maxsize must be positive")
        if event_types is not None and not event_types:
            raise ValueError("event_types must be non-empty when provided")
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=maxsize)
        self._queues[name] = queue
        self._stats[name] = SubscriberStats()
        self._event_types[name] = event_types
        self._event_filters[name] = event_filter
        return queue

    def unsubscribe(self, name: str) -> None:
        self._queues.pop(name, None)
        self._stats.pop(name, None)
        self._event_types.pop(name, None)
        self._event_filters.pop(name, None)

    def publish(self, event: Any) -> None:
        # Observe feed liveness once at the fan-out boundary so every raw quote
        # and oracle print is measured consistently regardless of feed adapter.
        if isinstance(event, (Quote, OraclePrice)) and event.recv_ts_ns > 0:
            FEED_LAST_MESSAGE_UNIX_SECONDS.labels(event.venue.value).set(
                event.recv_ts_ns / 1_000_000_000
            )
        for name, queue in self._queues.items():
            accepted_types = self._event_types[name]
            if accepted_types is not None and not isinstance(event, accepted_types):
                continue
            event_filter = self._event_filters[name]
            if event_filter is not None and not event_filter(event):
                continue
            stats = self._stats[name]
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                else:
                    stats.dropped += 1
            queue.put_nowait(event)
            stats.published += 1

    def stats(self) -> dict[str, SubscriberStats]:
        return {name: SubscriberStats(s.published, s.dropped) for name, s in self._stats.items()}
