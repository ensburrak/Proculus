from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Callable, Collection
from decimal import Decimal, InvalidOperation

from ..bus import BroadcastBus
from ..contracts import resolution_fingerprint, validate_resolution_equivalence_groups
from ..metrics import OPPORTUNITIES
from ..models import (
    ArbitrageOpportunity,
    ArbLeg,
    BinaryMarketSpec,
    ConsensusPrice,
    DecisionTrace,
    EdgeOpportunity,
    FeedReset,
    OraclePrice,
    Quote,
    QuoteKind,
    Side,
    Venue,
)
from .fair_value import BinaryFairValueModel, RollingVolatility

_SPOT_VENUES = {Venue.BINANCE, Venue.OKX, Venue.COINBASE, Venue.HYPERLIQUID}
_DECISION_LOG_INTERVAL_NS = 10_000_000_000


def fee_curve_multiplier(
    price: Decimal,
    venue: Venue | None = None,
    *,
    exponent: Decimal = Decimal(1),
) -> Decimal:
    """Return the collateral-equivalent BUY fee curve multiplier per share."""
    if venue is Venue.LIMITLESS:
        return min(price, Decimal(1) - price)
    if not exponent.is_finite() or exponent < 0:
        raise ValueError("fee exponent must be finite and non-negative")
    base = max(price * (Decimal(1) - price), Decimal(0))
    if base <= 0:
        return Decimal(0)
    try:
        return base**exponent
    except InvalidOperation as exc:
        raise ValueError("invalid fee exponent") from exc


def fee_per_share(
    rate: Decimal,
    price: Decimal,
    venue: Venue | None = None,
    *,
    exponent: Decimal = Decimal(1),
) -> Decimal:
    """Return point-in-time BUY-side fee in collateral units per binary share.

    Polymarket CLOB fee schedules publish both a rate and exponent. Limitless
    keeps its separate collateral-equivalent curve and intentionally ignores the
    Polymarket exponent.
    """
    if rate <= 0:
        return Decimal(0)
    return rate * max(
        fee_curve_multiplier(price, venue, exponent=exponent),
        Decimal(0),
    )


def quote_fee_parameters(
    quote: Quote,
    *,
    fallback_rate: Decimal,
    fallback_exponent: Decimal = Decimal(1),
) -> tuple[Decimal, Decimal]:
    """Resolve fee schedule from executable quote metadata with safe fallbacks."""
    rate = fallback_rate
    raw_rate = quote.metadata.get("fee_rate")
    if raw_rate is not None:
        try:
            parsed_rate = Decimal(str(raw_rate))
            if parsed_rate.is_finite() and parsed_rate >= 0:
                rate = parsed_rate
        except (InvalidOperation, TypeError, ValueError):
            pass

    exponent = fallback_exponent
    if quote.venue is not Venue.LIMITLESS:
        raw_exponent = quote.metadata.get("fee_exponent")
        if raw_exponent is not None:
            try:
                parsed_exponent = Decimal(str(raw_exponent))
                if parsed_exponent.is_finite() and parsed_exponent >= 0:
                    exponent = parsed_exponent
            except (InvalidOperation, TypeError, ValueError):
                pass
    return max(rate, Decimal(0)), max(exponent, Decimal(0))


def quote_is_fresh(quote: Quote, *, now_ns: int, max_age_ns: int) -> bool:
    """Treat missing/zero receive timestamps and stale quotes as non-executable."""
    return quote.recv_ts_ns > 0 and 0 <= now_ns - quote.recv_ts_ns <= max_age_ns


def top_of_book_can_fill(quote: Quote, *, notional_usd: Decimal) -> bool:
    """Conservative BBO-only fallback when full visible depth is unavailable."""
    if quote.ask <= 0 or quote.ask_size <= 0 or notional_usd <= 0:
        return False
    required_shares = notional_usd / quote.ask
    return quote.ask_size >= required_shares


def _visible_asks(quote: Quote) -> tuple[tuple[Decimal, Decimal], ...] | None:
    """Return sorted ask depth; None means venue did not provide L2 metadata."""
    raw = quote.metadata.get("asks_l2")
    if raw is None:
        return None
    if not isinstance(raw, list):
        return ()
    levels: list[tuple[Decimal, Decimal]] = []
    try:
        for item in raw:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                return ()
            price = Decimal(str(item[0]))
            size = Decimal(str(item[1]))
            if price <= 0 or size < 0:
                return ()
            if size > 0:
                levels.append((price, size))
    except (InvalidOperation, TypeError, ValueError):
        return ()
    return tuple(sorted(levels))


def buy_vwap_for_shares(quote: Quote, shares: Decimal) -> Decimal | None:
    """Calculate executable visible-book VWAP or fail closed when depth is insufficient."""
    if shares <= 0 or quote.ask <= 0:
        return None
    levels = _visible_asks(quote)
    if levels is None:
        if quote.ask_size < shares:
            return None
        return quote.ask
    if not levels:
        return None

    remaining = shares
    cost = Decimal(0)
    for price, size in levels:
        take = min(size, remaining)
        cost += take * price
        remaining -= take
        if remaining <= 0:
            return cost / shares
    return None


def estimate_buy_vwap(quote: Quote, *, notional_usd: Decimal) -> Decimal | None:
    """Estimate impact using a conservative share target based on the best ask."""
    if notional_usd <= 0 or quote.ask <= 0:
        return None
    target_shares = notional_usd / quote.ask
    return buy_vwap_for_shares(quote, target_shares)


class EdgeEngine:
    def __init__(
        self,
        bus: BroadcastBus,
        specs: tuple[BinaryMarketSpec, ...],
        *,
        min_net_edge: Decimal,
        slippage: Decimal,
        safety_margin: Decimal,
        order_notional: Decimal,
        max_age_ms: int,
        oracle_reference_max_age_ms: int = 5_000,
        volatility_bucket_ms: int = 1_000,
        volatility_window_seconds: int = 600,
        volatility_min_samples: int = 8,
        volatility_return_clip_bps: Decimal = Decimal("100"),
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.bus = bus
        self.specs = tuple(spec for spec in specs if spec.enabled)
        self.by_symbol: dict[str, list[BinaryMarketSpec]] = defaultdict(list)
        self.by_instrument: dict[str, BinaryMarketSpec] = {}
        for spec in self.specs:
            self.by_symbol[spec.symbol].append(spec)
            self.by_instrument[spec.yes_instrument] = spec
            if spec.no_instrument:
                self.by_instrument[spec.no_instrument] = spec
        self.min_net_edge = min_net_edge
        self.slippage = slippage
        self.safety_margin = safety_margin
        self.order_notional = order_notional
        self.max_age_ns = max_age_ms * 1_000_000
        if oracle_reference_max_age_ms <= 0:
            raise ValueError("oracle_reference_max_age_ms must be positive")
        self.oracle_reference_max_age_ns = oracle_reference_max_age_ms * 1_000_000
        self.clock_ns = clock_ns
        self.latest_consensus: dict[str, ConsensusPrice] = {}
        self.latest_oracles: dict[tuple[str, Venue, int | None], OraclePrice] = {}
        self.latest_quotes: dict[tuple[str, str], Quote] = {}
        self.opening_strikes: dict[str, Decimal] = {}
        self.volatility = RollingVolatility(
            window_seconds=volatility_window_seconds,
            bucket_ms=volatility_bucket_ms,
            min_samples=volatility_min_samples,
            return_clip_bps=float(volatility_return_clip_bps),
        )
        self.model = BinaryFairValueModel(self.volatility)
        self.log = logging.getLogger("hizlitrade.engine.edge")
        self._decision_state: dict[tuple[str, str, str], tuple[str, int]] = {}

    def _emit_decision(
        self,
        spec: BinaryMarketSpec,
        *,
        stage: str,
        reason_code: str,
        reason_tr: str,
        now_ns: int,
        instrument: str = "",
        status: str = "rejected",
        metadata: dict[str, object] | None = None,
        force: bool = False,
    ) -> None:
        key = (spec.market_id, instrument or "*", stage)
        previous = self._decision_state.get(key)
        if previous is not None and not force:
            elapsed = now_ns - previous[1]
            if previous[0] == reason_code and 0 <= elapsed < _DECISION_LOG_INTERVAL_NS:
                return
        self._decision_state[key] = (reason_code, now_ns)
        trace = DecisionTrace(
            stage=stage,
            status=status,
            reason_code=reason_code,
            reason_tr=reason_tr,
            venue=spec.venue,
            market_id=spec.market_id,
            instrument=instrument,
            strategy="fair_value_latency",
            created_ts_ns=now_ns,
            metadata={
                "symbol": spec.symbol,
                "min_net_edge": str(self.min_net_edge),
                **(metadata or {}),
            },
        )
        self.bus.publish(trace)
        self.log.info(
            "paper karar: stage=%s status=%s market=%s instrument=%s neden=%s (%s)",
            stage,
            status,
            spec.market_id,
            instrument or "-",
            reason_tr,
            reason_code,
        )

    def prune_decision_state(self, valid_market_ids: Collection[str]) -> None:
        """Drop throttling state for markets that are no longer configured."""
        valid = frozenset(valid_market_ids)
        for decision_key in tuple(self._decision_state):
            if decision_key[0] not in valid:
                self._decision_state.pop(decision_key, None)

    async def run(self, queue: asyncio.Queue[object]) -> None:
        while True:
            self.process_event(await queue.get())

    def process_event(self, event: object) -> None:
        """Process one event synchronously so live runtime and replay share identical logic."""
        if isinstance(event, FeedReset):
            self.invalidate_venue(event.venue)
            return

        affected: list[BinaryMarketSpec] = []
        if isinstance(event, ConsensusPrice):
            self.latest_consensus[event.symbol] = event
            self.volatility.update(event)
            affected.extend(self.by_symbol.get(event.symbol, []))
        elif isinstance(event, OraclePrice):
            self.latest_oracles[(event.symbol, event.venue, event.window_seconds)] = event
            affected.extend(self._capture_opening_strikes(event))
            affected.extend(self.by_symbol.get(event.symbol, []))
        elif isinstance(event, Quote) and event.kind is QuoteKind.PREDICTION:
            spec = self.by_instrument.get(event.instrument)
            if spec is not None:
                self.latest_quotes[(spec.market_id, event.instrument)] = event
                affected.append(spec)
        else:
            return

        seen: set[str] = set()
        for spec in affected:
            if spec.market_id in seen:
                continue
            seen.add(spec.market_id)
            self._evaluate(spec)

    def invalidate_venue(self, venue: Venue) -> None:
        if venue in _SPOT_VENUES:
            self.latest_consensus.clear()
        stale_oracles = [key for key in self.latest_oracles if key[1] is venue]
        for key in stale_oracles:
            self.latest_oracles.pop(key, None)
        stale = [key for key, quote in self.latest_quotes.items() if quote.venue is venue]
        for key in stale:
            self.latest_quotes.pop(key, None)

    def _capture_opening_strikes(self, event: OraclePrice) -> list[BinaryMarketSpec]:
        affected: list[BinaryMarketSpec] = []
        for spec in self.by_symbol.get(event.symbol, []):
            if spec.strike_mode != "opening_oracle":
                continue
            if spec.market_id in self.opening_strikes:
                continue
            if spec.starts_at_ms is None:
                continue
            if spec.oracle.lower() != event.venue.value:
                continue
            if (
                spec.oracle_window_seconds is not None
                and event.window_seconds != spec.oracle_window_seconds
            ):
                continue
            start = spec.starts_at_ms
            end = start + spec.strike_capture_tolerance_ms
            if not start <= event.publish_ts_ms <= end:
                continue
            self.opening_strikes[spec.market_id] = event.price
            affected.append(spec)
        return affected

    def _effective_strike(self, spec: BinaryMarketSpec) -> Decimal | None:
        if spec.strike_mode == "fixed":
            return spec.strike if spec.strike > 0 else None
        if spec.strike_mode == "opening_oracle":
            return self.opening_strikes.get(spec.market_id)
        return None

    def _required_oracle_reference(
        self,
        spec: BinaryMarketSpec,
    ) -> OraclePrice | None:
        """Return the matching settlement-oracle state for oracle-window contracts.

        The liquid spot consensus remains the volatility source. For recurring
        Polymarket opening-oracle contracts, however, fair value must be anchored
        to the same published oracle family/window that defines settlement rather
        than silently substituting an unrelated spot venue price.
        """
        if (
            spec.venue is not Venue.POLYMARKET
            or spec.strike_mode != "opening_oracle"
            or spec.oracle_window_seconds is None
        ):
            return None
        try:
            oracle_venue = Venue(spec.oracle.lower())
        except ValueError:
            return None
        return self.latest_oracles.get(
            (spec.symbol, oracle_venue, spec.oracle_window_seconds)
        )

    @staticmethod
    def _requires_oracle_reference(spec: BinaryMarketSpec) -> bool:
        return (
            spec.venue is Venue.POLYMARKET
            and spec.strike_mode == "opening_oracle"
            and spec.oracle_window_seconds is not None
        )

    def _evaluate(self, spec: BinaryMarketSpec) -> None:
        now_ns = self.clock_ns()
        spot = self.latest_consensus.get(spec.symbol)
        if spot is None:
            self._emit_decision(
                spec,
                stage="signal_input",
                reason_code="spot_consensus_missing",
                reason_tr="İşlem açılmadı: güncel spot konsensüsü henüz yok.",
                now_ns=now_ns,
            )
            return
        if spot.recv_ts_ns <= 0 or now_ns - spot.recv_ts_ns > self.max_age_ns:
            self._emit_decision(
                spec,
                stage="signal_input",
                reason_code="spot_consensus_stale",
                reason_tr="İşlem açılmadı: spot konsensüsü bayat veya zaman damgası geçersiz.",
                now_ns=now_ns,
                metadata={"spot_recv_ts_ns": spot.recv_ts_ns},
            )
            return
        if not self.volatility.is_ready(spec.symbol):
            self._emit_decision(
                spec,
                stage="signal_input",
                reason_code="volatility_warmup_incomplete",
                reason_tr=(
                    "İşlem açılmadı: sabit-zaman volatilite warm-up örneği henüz "
                    "yeterli değil."
                ),
                now_ns=now_ns,
                metadata={
                    "volatility_samples": self.volatility.sample_count(spec.symbol),
                    "required_return_samples": self.volatility.min_samples,
                    "volatility_bucket_ms": self.volatility.bucket_ms,
                    "volatility_window_seconds": self.volatility.window_seconds,
                },
            )
            return
        now_ms = now_ns // 1_000_000
        if spec.expires_at_ms <= now_ms:
            self._emit_decision(
                spec,
                stage="market_state",
                reason_code="market_expired",
                reason_tr="İşlem açılmadı: piyasanın vadesi dolmuş.",
                now_ns=now_ns,
                metadata={"expires_at_ms": spec.expires_at_ms},
            )
            return
        strike = self._effective_strike(spec)
        if strike is None:
            self._emit_decision(
                spec,
                stage="signal_input",
                reason_code="strike_unavailable",
                reason_tr="İşlem açılmadı: doğrulanmış açılış/strike fiyatı henüz yok.",
                now_ns=now_ns,
            )
            return

        reference_price = spot.price
        reference_source = "spot_consensus"
        oracle_reference_age_ms: int | None = None
        if self._requires_oracle_reference(spec):
            oracle_reference = self._required_oracle_reference(spec)
            if oracle_reference is None:
                self._emit_decision(
                    spec,
                    stage="signal_input",
                    reason_code="oracle_reference_missing",
                    reason_tr=(
                        "İşlem açılmadı: settlement ile aynı kaynak/pencereden güncel "
                        "oracle TWAP referansı henüz yok."
                    ),
                    now_ns=now_ns,
                    metadata={
                        "oracle": spec.oracle,
                        "oracle_window_seconds": spec.oracle_window_seconds,
                    },
                )
                return
            receive_age_ns = now_ns - oracle_reference.recv_ts_ns
            source_age_ms = now_ms - oracle_reference.publish_ts_ms
            if (
                oracle_reference.recv_ts_ns <= 0
                or receive_age_ns < 0
                or receive_age_ns > self.oracle_reference_max_age_ns
                or source_age_ms < 0
                or source_age_ms * 1_000_000 > self.oracle_reference_max_age_ns
                or oracle_reference.price <= 0
            ):
                self._emit_decision(
                    spec,
                    stage="signal_input",
                    reason_code="oracle_reference_stale",
                    reason_tr=(
                        "İşlem açılmadı: settlement oracle TWAP referansı execution "
                        "için bayat veya geçersiz."
                    ),
                    now_ns=now_ns,
                    metadata={
                        "oracle": spec.oracle,
                        "oracle_window_seconds": spec.oracle_window_seconds,
                        "oracle_recv_ts_ns": oracle_reference.recv_ts_ns,
                        "oracle_publish_ts_ms": oracle_reference.publish_ts_ms,
                        "oracle_receive_age_ms": receive_age_ns // 1_000_000,
                        "oracle_source_age_ms": source_age_ms,
                    },
                )
                return
            reference_price = oracle_reference.price
            reference_source = (
                f"{oracle_reference.venue.value}_twap_"
                f"{oracle_reference.window_seconds or 0}s"
            )
            oracle_reference_age_ms = max(
                receive_age_ns // 1_000_000,
                source_age_ms,
            )

        p_yes = self.model.probability_yes(
            spec,
            spot,
            now_ms,
            strike=strike,
            reference_price=reference_price,
        )
        self._evaluate_leg(
            spec,
            spec.yes_instrument,
            p_yes,
            "YES",
            strike,
            now_ns,
            reference_price=reference_price,
            reference_source=reference_source,
            oracle_reference_age_ms=oracle_reference_age_ms,
        )
        if spec.no_instrument:
            self._evaluate_leg(
                spec,
                spec.no_instrument,
                Decimal(1) - p_yes,
                "NO",
                strike,
                now_ns,
                reference_price=reference_price,
                reference_source=reference_source,
                oracle_reference_age_ms=oracle_reference_age_ms,
            )

    def _evaluate_leg(
        self,
        spec: BinaryMarketSpec,
        instrument: str,
        fair: Decimal,
        outcome: str,
        strike: Decimal,
        now_ns: int,
        *,
        reference_price: Decimal,
        reference_source: str,
        oracle_reference_age_ms: int | None,
    ) -> None:
        fingerprint = resolution_fingerprint(spec)
        if fingerprint is None:
            self._emit_decision(
                spec,
                stage="contract_validation",
                reason_code="resolution_semantics_unverified",
                reason_tr="İşlem açılmadı: çözüm/settlement kuralları yeterince doğrulanmamış.",
                now_ns=now_ns,
                instrument=instrument,
            )
            return
        quote = self.latest_quotes.get((spec.market_id, instrument))
        if quote is None:
            self._emit_decision(
                spec,
                stage="execution_input",
                reason_code="prediction_quote_missing",
                reason_tr="İşlem açılmadı: prediction-market emir defteri fiyatı henüz yok.",
                now_ns=now_ns,
                instrument=instrument,
            )
            return
        if quote.metadata.get("indicative"):
            self._emit_decision(
                spec,
                stage="execution_input",
                reason_code="prediction_quote_indicative",
                reason_tr="İşlem açılmadı: mevcut fiyat yalnızca indicative, yürütülebilir değil.",
                now_ns=now_ns,
                instrument=instrument,
            )
            return
        if not quote_is_fresh(quote, now_ns=now_ns, max_age_ns=self.max_age_ns):
            self._emit_decision(
                spec,
                stage="execution_input",
                reason_code="prediction_quote_stale",
                reason_tr="İşlem açılmadı: prediction-market fiyatı bayat.",
                now_ns=now_ns,
                instrument=instrument,
                metadata={"quote_recv_ts_ns": quote.recv_ts_ns},
            )
            return
        executable = estimate_buy_vwap(quote, notional_usd=self.order_notional)
        if executable is None:
            self._emit_decision(
                spec,
                stage="execution_input",
                reason_code="visible_depth_insufficient",
                reason_tr="İşlem açılmadı: istenen paper emir için görünür defter derinliği yetersiz.",
                now_ns=now_ns,
                instrument=instrument,
                metadata={"best_ask": str(quote.ask), "visible_ask_size": str(quote.ask_size)},
            )
            return
        book_impact = executable - quote.ask
        fee_rate, fee_exponent = quote_fee_parameters(
            quote,
            fallback_rate=spec.taker_fee_rate,
        )
        fee = fee_per_share(
            fee_rate,
            executable,
            spec.venue,
            exponent=fee_exponent,
        )
        gross = fair - executable
        net = gross - fee - self.slippage - self.safety_margin
        if net < self.min_net_edge:
            self._emit_decision(
                spec,
                stage="edge_gate",
                reason_code="edge_below_minimum",
                reason_tr="İşlem açılmadı: ücret, slippage ve güvenlik payı sonrası net edge minimum eşiğin altında.",
                now_ns=now_ns,
                instrument=instrument,
                metadata={
                    "outcome": outcome,
                    "fair_probability": str(fair),
                    "executable_price": str(executable),
                    "gross_edge": str(gross),
                    "estimated_fee_per_share": str(fee),
                    "fee_rate": str(fee_rate),
                    "fee_exponent": str(fee_exponent),
                    "slippage": str(self.slippage),
                    "safety_margin": str(self.safety_margin),
                    "net_edge": str(net),
                    "fair_value_reference_price": str(reference_price),
                    "fair_value_reference_source": reference_source,
                    "oracle_reference_age_ms": oracle_reference_age_ms,
                },
            )
            return
        opportunity = EdgeOpportunity(
            strategy="fair_value_latency",
            market_id=spec.market_id,
            canonical_id=spec.canonical_id,
            venue=spec.venue,
            instrument=instrument,
            side=Side.BUY,
            fair_probability=fair,
            executable_price=executable,
            gross_edge=gross,
            estimated_fee=fee,
            estimated_slippage=self.slippage,
            safety_margin=self.safety_margin,
            net_edge=net,
            notional_usd=self.order_notional,
            created_ts_ns=now_ns,
            metadata={
                "outcome": outcome,
                "symbol": spec.symbol,
                "oracle": spec.oracle,
                "oracle_window_seconds": spec.oracle_window_seconds,
                "spot": str(self.latest_consensus[spec.symbol].price),
                "strike": str(strike),
                "fair_value_reference_price": str(reference_price),
                "fair_value_reference_source": reference_source,
                "oracle_reference_age_ms": oracle_reference_age_ms,
                "strike_mode": spec.strike_mode,
                "starts_at_ms": spec.starts_at_ms,
                "expires_at_ms": spec.expires_at_ms,
                "market_duration_ms": (
                    spec.expires_at_ms - spec.starts_at_ms
                    if spec.starts_at_ms is not None
                    else None
                ),
                "settlement_rule": spec.settlement_rule,
                "settlement_capture_tolerance_ms": spec.settlement_capture_tolerance_ms,
                "resolution_fingerprint": fingerprint,
                "best_ask": str(quote.ask),
                "visible_ask_size": str(quote.ask_size),
                "book_impact_per_share": str(book_impact),
                "fee_rate": str(fee_rate),
                "fee_exponent": str(fee_exponent),
                "depth_aware": quote.metadata.get("asks_l2") is not None,
                "volatility_sigma_per_sqrt_second": self.volatility.sigma_per_sqrt_second(
                    spec.symbol
                ),
                "volatility_samples": self.volatility.sample_count(spec.symbol),
                "volatility_bucket_ms": self.volatility.bucket_ms,
                "volatility_window_seconds": self.volatility.window_seconds,
            },
        )
        OPPORTUNITIES.labels(opportunity.strategy, opportunity.venue.value).inc()
        self.bus.publish(opportunity)
        self._emit_decision(
            spec,
            stage="edge_gate",
            reason_code="edge_candidate_approved",
            reason_tr="Sinyal eşiği geçti; fırsat risk ve paper execution kontrolüne gönderildi.",
            now_ns=now_ns,
            instrument=instrument,
            status="candidate",
            metadata={
                "outcome": outcome,
                "fair_probability": str(fair),
                "executable_price": str(executable),
                "net_edge": str(net),
                "fair_value_reference_price": str(reference_price),
                "fair_value_reference_source": reference_source,
                "oracle_reference_age_ms": oracle_reference_age_ms,
                "fee_rate": str(fee_rate),
                "fee_exponent": str(fee_exponent),
                "order_notional_usd": str(self.order_notional),
            },
        )


class CrossVenueArbitrageEngine:
    """Find executable YES + NO pairs costing less than their $1 settlement value."""

    def __init__(
        self,
        bus: BroadcastBus,
        specs: tuple[BinaryMarketSpec, ...],
        *,
        slippage: Decimal,
        safety_margin: Decimal,
        min_net_edge: Decimal,
        order_notional: Decimal,
        max_age_ms: int,
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.bus = bus
        self.specs = tuple(s for s in specs if s.enabled)
        validate_resolution_equivalence_groups(self.specs)
        self.instrument_map: dict[str, tuple[BinaryMarketSpec, str]] = {}
        self.by_canonical: dict[str, list[BinaryMarketSpec]] = defaultdict(list)
        for spec in self.specs:
            self.by_canonical[spec.canonical_id].append(spec)
            self.instrument_map[spec.yes_instrument] = (spec, "YES")
            if spec.no_instrument:
                self.instrument_map[spec.no_instrument] = (spec, "NO")
        self.quotes: dict[str, Quote] = {}
        self.slippage = slippage
        self.safety_margin = safety_margin
        self.min_net_edge = min_net_edge
        self.order_notional = order_notional
        self.max_age_ns = max_age_ms * 1_000_000
        self.clock_ns = clock_ns

    async def run(self, queue: asyncio.Queue[object]) -> None:
        while True:
            self.process_event(await queue.get())

    def process_event(self, event: object) -> None:
        """Process one prediction quote synchronously for live/replay parity."""
        if isinstance(event, FeedReset):
            stale = [
                instrument
                for instrument, quote in self.quotes.items()
                if quote.venue is event.venue
            ]
            for instrument in stale:
                self.quotes.pop(instrument, None)
            return
        if not isinstance(event, Quote) or event.kind is not QuoteKind.PREDICTION:
            return
        mapped = self.instrument_map.get(event.instrument)
        if mapped is None or event.metadata.get("indicative"):
            return
        self.quotes[event.instrument] = event
        self._evaluate(mapped[0].canonical_id)

    def _evaluate(self, canonical_id: str) -> None:
        now_ns = self.clock_ns()
        yes_candidates: list[tuple[BinaryMarketSpec, Quote]] = []
        no_candidates: list[tuple[BinaryMarketSpec, Quote]] = []
        for spec in self.by_canonical.get(canonical_id, []):
            yes = self.quotes.get(spec.yes_instrument)
            if yes is not None and quote_is_fresh(
                yes, now_ns=now_ns, max_age_ns=self.max_age_ns
            ):
                yes_candidates.append((spec, yes))
            if spec.no_instrument:
                no = self.quotes.get(spec.no_instrument)
                if no is not None and quote_is_fresh(
                    no, now_ns=now_ns, max_age_ns=self.max_age_ns
                ):
                    no_candidates.append((spec, no))
        if not yes_candidates or not no_candidates:
            return
        yes_spec, yes_q = min(yes_candidates, key=lambda x: x[1].ask)
        no_spec, no_q = min(no_candidates, key=lambda x: x[1].ask)
        yes_fingerprint = resolution_fingerprint(yes_spec)
        no_fingerprint = resolution_fingerprint(no_spec)
        if (
            yes_fingerprint is None
            or no_fingerprint is None
            or yes_fingerprint != no_fingerprint
        ):
            return
        yes_rate, yes_exponent = quote_fee_parameters(
            yes_q,
            fallback_rate=yes_spec.taker_fee_rate,
        )
        no_rate, no_exponent = quote_fee_parameters(
            no_q,
            fallback_rate=no_spec.taker_fee_rate,
        )
        yes_best_fee = fee_per_share(
            yes_rate,
            yes_q.ask,
            yes_spec.venue,
            exponent=yes_exponent,
        )
        no_best_fee = fee_per_share(
            no_rate,
            no_q.ask,
            no_spec.venue,
            exponent=no_exponent,
        )
        best_total = (
            yes_q.ask + no_q.ask + yes_best_fee + no_best_fee + self.slippage * Decimal(2)
        )
        if best_total <= 0:
            return
        conservative_pair_count = self.order_notional / best_total
        yes_price = buy_vwap_for_shares(yes_q, conservative_pair_count)
        no_price = buy_vwap_for_shares(no_q, conservative_pair_count)
        if yes_price is None or no_price is None:
            return
        yes_fee = fee_per_share(
            yes_rate,
            yes_price,
            yes_spec.venue,
            exponent=yes_exponent,
        )
        no_fee = fee_per_share(
            no_rate,
            no_price,
            no_spec.venue,
            exponent=no_exponent,
        )
        total = yes_price + no_price + yes_fee + no_fee + self.slippage * Decimal(2)
        if total <= 0:
            return
        pair_count = self.order_notional / total
        edge = Decimal(1) - total - self.safety_margin
        if edge < self.min_net_edge:
            return
        arb = ArbitrageOpportunity(
            strategy="cross_venue_binary_arb",
            canonical_id=canonical_id,
            legs=(
                ArbLeg(
                    yes_spec.venue,
                    yes_spec.market_id,
                    yes_spec.yes_instrument,
                    Side.BUY,
                    yes_price,
                    yes_fee,
                ),
                ArbLeg(
                    no_spec.venue,
                    no_spec.market_id,
                    no_spec.no_instrument or "",
                    Side.BUY,
                    no_price,
                    no_fee,
                ),
            ),
            total_cost_per_pair=total,
            locked_value_per_pair=Decimal(1),
            net_edge_per_pair=edge,
            notional_usd=self.order_notional,
            created_ts_ns=now_ns,
            metadata={
                "requires_resolution_equivalence": True,
                "resolution_fingerprint": yes_fingerprint,
                "pair_count": str(pair_count),
                "conservative_depth_check_pairs": str(conservative_pair_count),
                "yes_best_ask": str(yes_q.ask),
                "no_best_ask": str(no_q.ask),
                "yes_vwap": str(yes_price),
                "no_vwap": str(no_price),
            },
        )
        OPPORTUNITIES.labels(arb.strategy, "multi").inc()
        self.bus.publish(arb)
