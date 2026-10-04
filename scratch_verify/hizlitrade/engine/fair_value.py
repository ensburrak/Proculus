from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from decimal import Decimal

from ..models import BinaryMarketSpec, ConsensusPrice


@dataclass(slots=True)
class _Point:
    price: float
    ts_ns: int
    source_ts_ns: int


class RollingVolatility:
    """Estimate log-return volatility on fixed-time closes.

    Consensus feeds can emit many asynchronous updates inside a single second.
    Treating every event as an independent return makes volatility depend on feed
    event rate rather than elapsed market time. This sampler collapses updates
    into fixed buckets, keeps the latest close per bucket, rejects out-of-order
    observations, winsorizes extreme single-bucket returns, and exposes explicit
    warm-up readiness so the signal engine can fail closed before the estimate is
    statistically usable.
    """

    def __init__(
        self,
        window: int = 600,
        floor: float = 1e-7,
        *,
        window_seconds: int | None = None,
        bucket_ms: int = 1_000,
        min_samples: int = 8,
        return_clip_bps: float = 100.0,
    ) -> None:
        resolved_window = window if window_seconds is None else window_seconds
        if resolved_window <= 0:
            raise ValueError("window_seconds must be positive")
        if bucket_ms <= 0:
            raise ValueError("bucket_ms must be positive")
        if min_samples < 2:
            raise ValueError("min_samples must be at least two")
        if floor <= 0:
            raise ValueError("floor must be positive")
        if return_clip_bps <= 0:
            raise ValueError("return_clip_bps must be positive")

        self.window = resolved_window
        self.window_seconds = resolved_window
        self.bucket_ms = bucket_ms
        self.min_samples = min_samples
        self.floor = floor
        self.return_clip_bps = return_clip_bps
        self._bucket_ns = bucket_ms * 1_000_000
        self._window_ns = resolved_window * 1_000_000_000
        self._clip_log_return = return_clip_bps / 10_000.0
        max_points = max(3, math.ceil(self._window_ns / self._bucket_ns) + 2)
        self._points: dict[str, deque[_Point]] = defaultdict(
            lambda: deque(maxlen=max_points)
        )

    def update(self, event: ConsensusPrice) -> None:
        price = float(event.price)
        ts_ns = int(event.recv_ts_ns)
        if price <= 0 or ts_ns <= 0 or not math.isfinite(price):
            return

        points = self._points[event.symbol]
        if points and ts_ns <= points[-1].source_ts_ns:
            # Receive-time ordering is authoritative for research/replay. An
            # older observation must never rewrite a later bucket close.
            return

        bucket_ts_ns = (ts_ns // self._bucket_ns) * self._bucket_ns
        point = _Point(price=price, ts_ns=bucket_ts_ns, source_ts_ns=ts_ns)
        if points and points[-1].ts_ns == bucket_ts_ns:
            points[-1] = point
        else:
            points.append(point)

        cutoff = bucket_ts_ns - self._window_ns
        while points and points[0].ts_ns < cutoff:
            points.popleft()

    def sample_count(self, symbol: str) -> int:
        return len(self._points.get(symbol, ()))

    def _normalized_returns(self, symbol: str) -> list[float]:
        points = self._points.get(symbol)
        if not points or len(points) < 2:
            return []
        normalized: list[float] = []
        previous = points[0]
        for point in list(points)[1:]:
            dt = (point.ts_ns - previous.ts_ns) / 1e9
            if dt > 0 and point.price > 0 and previous.price > 0:
                raw = math.log(point.price / previous.price)
                clipped = min(
                    max(raw, -self._clip_log_return),
                    self._clip_log_return,
                )
                normalized.append(clipped / math.sqrt(dt))
            previous = point
        return normalized

    def is_ready(self, symbol: str) -> bool:
        return len(self._normalized_returns(symbol)) >= self.min_samples

    def sigma_per_sqrt_second(self, symbol: str) -> float:
        normalized = self._normalized_returns(symbol)
        if len(normalized) < self.min_samples:
            return self.floor
        mean = sum(normalized) / len(normalized)
        variance = sum((x - mean) ** 2 for x in normalized) / (len(normalized) - 1)
        return max(math.sqrt(max(variance, 0.0)), self.floor)


class BinaryFairValueModel:
    """Transparent baseline model for Up/Down-style binary crypto markets."""

    def __init__(self, volatility: RollingVolatility) -> None:
        self.volatility = volatility

    def probability_yes(
        self,
        spec: BinaryMarketSpec,
        spot: ConsensusPrice,
        now_ms: int,
        *,
        strike: Decimal | None = None,
        reference_price: Decimal | None = None,
    ) -> Decimal:
        effective_strike = spec.strike if strike is None else strike
        effective_reference = spot.price if reference_price is None else reference_price
        if effective_strike <= 0 or effective_reference <= 0:
            return Decimal("0.5")
        remaining_s = max((spec.expires_at_ms - now_ms) / 1000.0, 0.001)
        sigma = self.volatility.sigma_per_sqrt_second(spec.symbol)
        denom = max(sigma * math.sqrt(remaining_s), 1e-9)
        log_moneyness = math.log(float(effective_reference / effective_strike))
        z = (log_moneyness - 0.5 * sigma * sigma * remaining_s) / denom
        cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        cdf = min(max(cdf, 0.001), 0.999)
        return Decimal(str(cdf))
