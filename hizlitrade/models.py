from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Any


class Venue(StrEnum):
    BINANCE = "binance"
    OKX = "okx"
    COINBASE = "coinbase"
    HYPERLIQUID = "hyperliquid"
    POLYMARKET = "polymarket"
    LIMITLESS = "limitless"
    PYTH = "pyth"
    CHAINLINK = "chainlink"


class QuoteKind(StrEnum):
    SPOT = "spot"
    PREDICTION = "prediction"
    ORACLE = "oracle"


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True, slots=True)
class FeedReset:
    venue: Venue
    reason: str
    created_ts_ns: int


@dataclass(frozen=True, slots=True)
class Quote:
    venue: Venue
    instrument: str
    kind: QuoteKind
    bid: Decimal
    ask: Decimal
    bid_size: Decimal = Decimal("0")
    ask_size: Decimal = Decimal("0")
    source_ts_ms: int | None = None
    recv_ts_ns: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / Decimal(2)

    @property
    def spread(self) -> Decimal:
        return self.ask - self.bid

    def validate(self) -> None:
        if self.bid < 0 or self.ask < 0:
            raise ValueError("quote prices cannot be negative")
        if self.ask < self.bid:
            raise ValueError(f"crossed quote: bid={self.bid} ask={self.ask}")


@dataclass(frozen=True, slots=True)
class OraclePrice:
    venue: Venue
    feed_id: str
    symbol: str
    price: Decimal
    confidence: Decimal | None
    publish_ts_ms: int
    recv_ts_ns: int
    window_seconds: int | None = None
    publisher_ts_ms: int | None = None


@dataclass(frozen=True, slots=True)
class MarketResolution:
    """Authoritative venue-published binary market outcome."""

    venue: Venue
    market_id: str
    market_slug: str
    winning_outcome: str
    resolved_ts_ms: int | None
    recv_ts_ns: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OracleBasisObservation:
    symbol: str
    oracle_venue: Venue
    oracle_feed_id: str
    oracle_window_seconds: int | None
    spot_price: Decimal
    oracle_price: Decimal
    basis_bps: Decimal
    observation_age_ms: Decimal
    spot_recv_ts_ns: int
    oracle_recv_ts_ns: int
    created_ts_ns: int


@dataclass(frozen=True, slots=True)
class ConsensusPrice:
    symbol: str
    price: Decimal
    dispersion_bps: Decimal
    venues: tuple[Venue, ...]
    recv_ts_ns: int


@dataclass(frozen=True, slots=True)
class BinaryMarketSpec:
    market_id: str
    canonical_id: str
    venue: Venue
    symbol: str
    yes_instrument: str
    no_instrument: str | None
    strike: Decimal
    expires_at_ms: int
    oracle: str
    taker_fee_rate: Decimal
    enabled: bool = True
    notes: str = ""
    starts_at_ms: int | None = None
    strike_mode: str = "fixed"
    oracle_window_seconds: int | None = None
    strike_capture_tolerance_ms: int = 5_000
    stream_key: str | None = None
    settlement_rule: str = "unknown"
    settlement_capture_tolerance_ms: int = 10_000


@dataclass(frozen=True, slots=True)
class LeadLagFeature:
    market_id: str
    canonical_id: str
    venue: Venue
    instrument: str
    outcome: str
    symbol: str
    spot_price: Decimal
    oracle_price: Decimal
    oracle_basis_bps: Decimal
    spot_momentum_250ms_bps: Decimal | None
    spot_momentum_1s_bps: Decimal | None
    spot_momentum_3s_bps: Decimal | None
    prediction_bid: Decimal
    prediction_ask: Decimal
    prediction_spread: Decimal
    prediction_microprice: Decimal | None
    book_imbalance: Decimal | None
    time_to_expiry_ms: int
    spot_recv_ts_ns: int
    oracle_recv_ts_ns: int
    quote_recv_ts_ns: int
    created_ts_ns: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EdgeOpportunity:
    strategy: str
    market_id: str
    canonical_id: str
    venue: Venue
    instrument: str
    side: Side
    fair_probability: Decimal
    executable_price: Decimal
    gross_edge: Decimal
    estimated_fee: Decimal
    estimated_slippage: Decimal
    safety_margin: Decimal
    net_edge: Decimal
    notional_usd: Decimal
    created_ts_ns: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DecisionTrace:
    """Explainable paper-trading decision emitted for operator observability."""

    stage: str
    status: str
    reason_code: str
    reason_tr: str
    venue: Venue | None
    market_id: str
    instrument: str
    strategy: str
    created_ts_ns: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OrderIntent:
    venue: Venue
    market_id: str
    instrument: str
    side: Side
    price: Decimal
    notional_usd: Decimal
    max_slippage: Decimal
    strategy: str
    created_ts_ns: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Fill:
    venue: Venue
    market_id: str
    instrument: str
    side: Side
    price: Decimal
    shares: Decimal
    notional_usd: Decimal
    fee_usd: Decimal
    paper: bool
    created_ts_ns: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResearchFill:
    """Counterfactual execution evidence that never mutates paper/live capital."""

    venue: Venue
    market_id: str
    instrument: str
    outcome: str
    side: Side
    price: Decimal
    shares: Decimal
    notional_usd: Decimal
    fee_usd: Decimal
    cost_basis_usd: Decimal
    strategy: str
    created_ts_ns: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResearchSettlement:
    """Outcome of a counterfactual research fill, isolated from paper PnL."""

    venue: Venue
    market_id: str
    instrument: str
    outcome: str
    winning_outcome: str
    shares: Decimal
    payout_usd: Decimal
    cost_basis_usd: Decimal
    realized_pnl_usd: Decimal
    oracle_price: Decimal | None
    strike: Decimal
    settled_ts_ns: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PaperSettlement:
    venue: Venue
    market_id: str
    instrument: str
    outcome: str
    winning_outcome: str
    shares: Decimal
    payout_usd: Decimal
    cost_basis_usd: Decimal
    realized_pnl_usd: Decimal
    oracle_price: Decimal | None
    strike: Decimal
    settled_ts_ns: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ArbLeg:
    venue: Venue
    market_id: str
    instrument: str
    side: Side
    price: Decimal
    estimated_fee: Decimal


@dataclass(frozen=True, slots=True)
class ArbitrageOpportunity:
    strategy: str
    canonical_id: str
    legs: tuple[ArbLeg, ...]
    total_cost_per_pair: Decimal
    locked_value_per_pair: Decimal
    net_edge_per_pair: Decimal
    notional_usd: Decimal
    created_ts_ns: int
    metadata: dict[str, Any] = field(default_factory=dict)


def event_to_dict(event: object) -> dict[str, Any]:
    if not is_dataclass(event) or isinstance(event, type):
        raise TypeError(f"event must be a dataclass instance, got {type(event).__name__}")
    data: dict[str, Any] = asdict(event)
    data["event_type"] = type(event).__name__
    safe = _json_safe(data)
    if not isinstance(safe, dict):
        raise TypeError("serialized event must be a mapping")
    return safe


def _json_safe(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value