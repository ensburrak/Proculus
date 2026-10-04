from __future__ import annotations

import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .contracts import validate_resolution_equivalence_groups
from .models import BinaryMarketSpec, Venue


def _canonical_spot_symbol(raw: str) -> str:
    value = raw.upper().replace("-", "").replace("/", "")
    for quote in ("USDT", "USDC", "USD"):
        if value.endswith(quote):
            return value[: -len(quote)] + "USD"
    # Bare Hyperliquid-style base symbols (BTC, XRP, DOGE, ...) are normalized
    # into the same BASEUSD namespace used by the consensus engine.
    if value and value.isalnum():
        return f"{value}USD"
    return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HIZLI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    live_trading: bool = False
    live_ack: str = ""
    live_max_notional_usd: Decimal = Decimal("5")
    live_promotion_report_path: Path = Path("data/promotion_report.json")
    live_promotion_max_age_hours: int = 24

    paper_balance_usd: Decimal = Decimal("250")
    order_notional_usd: Decimal = Decimal("2.50")
    max_open_exposure_pct: Decimal = Decimal("0.20")
    max_market_exposure_pct: Decimal = Decimal("0.05")
    max_symbol_exposure_pct: Decimal = Decimal("0.10")
    max_concurrent_positions: int = 8
    allow_market_pyramiding: bool = False
    max_daily_loss_pct: Decimal = Decimal("0.05")
    max_drawdown_pct: Decimal = Decimal("0.10")
    min_time_to_expiry_ms: int = 3_000
    signal_cooldown_seconds: float = 5.0
    paper_execution_latency_ms: int = 100
    # Empty/"auto" derives the research universe from symbols covered by at
    # least min_spot_venues configured reference feeds. A comma-separated value
    # can explicitly narrow the universe. It never expands beyond reference
    # coverage, so an unpriced coin cannot silently become executable.
    research_symbols: str = "auto"
    # Paper can collect every venue while only allowing fills on selected venues.
    # The Polymarket activation profile pins this to "polymarket".
    paper_execution_venues: str = "polymarket,limitless"

    min_net_edge: Decimal = Decimal("0.025")
    safety_margin: Decimal = Decimal("0.005")
    assumed_slippage: Decimal = Decimal("0.003")
    max_quote_age_ms: int = 1500
    # Settlement-oracle reference freshness is intentionally distinct from
    # prediction-book freshness. Chainlink RTDS TWAPs are allowed a slightly
    # wider but still fail-closed receive/source-age budget.
    oracle_reference_max_age_ms: int = 5_000

    # Fair-value volatility is estimated from fixed receive-time buckets rather
    # than raw feed event counts. Keep these in Settings so live, replay and
    # promotion evidence share the exact same model epoch.
    volatility_bucket_ms: int = 1_000
    volatility_window_seconds: int = 600
    volatility_min_samples: int = 8
    volatility_return_clip_bps: Decimal = Decimal("100")

    # Execution-quality gate: point-in-time microstructure can only veto or
    # downsize an already-qualified fair-value opportunity. It never creates
    # directional authority or increases order size.
    quality_gate_enabled: bool = True
    quality_feature_max_age_ms: int = 1_500
    quality_max_prediction_spread: Decimal = Decimal("0.05")
    quality_max_consensus_dispersion_bps: Decimal = Decimal("20")
    quality_shock_momentum_250ms_bps: Decimal = Decimal("25")
    quality_min_abs_basis_bps: Decimal = Decimal("2")
    quality_min_directional_momentum_1s_bps: Decimal = Decimal("1")
    quality_min_confirmations: int = 3
    quality_min_size_multiplier: Decimal = Decimal("0.25")
    min_spot_venues: int = 2
    clock_jump_tolerance_ms: int = 250

    # Historical Polymarket rollover switch. It remains separate so a host with
    # blocked Polymarket connectivity can still run Limitless research cleanly.
    auto_market_rollover: bool = True
    market_rollover_interval_seconds: int = 30
    market_rollover_horizon_minutes: int = 45
    market_rollover_max_pages: int = 8
    # Authoritative CLOB resolution reconciliation is independent from Chainlink
    # availability. It reads the frozen spec registry, resolves condition IDs and
    # emits only explicit CLOB winner flags; prices are never used to infer winners.
    polymarket_resolution_poll_seconds: int = 15
    polymarket_resolution_retention_seconds: int = 24 * 60 * 60
    polymarket_resolution_max_pages: int = 64

    auto_limitless_rollover: bool = False
    limitless_rollover_interval_seconds: int = 30
    limitless_rollover_max_pages: int = 4
    limitless_resolution_retention_seconds: int = 600

    # Reference-feed defaults cover the current recurring Polymarket crypto set.
    # A symbol only enters research when it is configured on at least
    # ``min_spot_venues`` feeds; runtime consensus still requires fresh quotes.
    binance_symbols: str = "btcusdt,ethusdt,solusdt,xrpusdt,dogeusdt,bnbusdt,zecusdt"
    okx_instruments: str = (
        "BTC-USDT,ETH-USDT,SOL-USDT,XRP-USDT,DOGE-USDT,BNB-USDT,ZEC-USDT,HYPE-USDT"
    )
    coinbase_products: str = "BTC-USD,ETH-USD,SOL-USD,XRP-USD,DOGE-USD,ZEC-USD,HYPE-USD"
    hyperliquid_coins: str = "BTC,ETH,SOL,XRP,DOGE,BNB,ZEC,HYPE"
    chainlink_twap_symbols: str = (
        "btc/usd,eth/usd,sol/usd,xrp/usd,doge/usd,bnb/usd,zec/usd,hype/usd"
    )
    chainlink_twap_windows: str = "30,60"
    polymarket_token_ids: str = ""
    limitless_market_slugs: str = ""
    pyth_feed_ids: str = (
        "0xe62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43,"
        "0xff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace"
    )
    market_specs_path: Path = Path("config/markets.json")

    pyth_api_key: str = Field(default="", validation_alias="PYTH_API_KEY")
    limitless_api_key: str = Field(default="", validation_alias="LIMITLESS_API_KEY")
    polymarket_private_key: str = Field(default="", validation_alias="POLYMARKET_PRIVATE_KEY")
    polymarket_deposit_wallet: str = Field(default="", validation_alias="POLYMARKET_DEPOSIT_WALLET")
    polymarket_signature_type: int = 3

    data_dir: Path = Path("data")
    metrics_port: int = 9108
    log_level: str = "INFO"

    # Optional operational notifications. These transports are best-effort and
    # never participate in trading decisions or execution safety.
    alert_webhook_urls: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    smtp_to: str = ""
    alert_min_interval_seconds: int = 60
    alert_timeout_seconds: float = 5.0

    @field_validator("paper_balance_usd", "order_notional_usd", "live_max_notional_usd")
    @classmethod
    def positive_money(cls, value: Decimal) -> Decimal:
        if value <= 0:
            raise ValueError("monetary settings must be positive")
        return value

    @field_validator(
        "max_open_exposure_pct",
        "max_market_exposure_pct",
        "max_symbol_exposure_pct",
        "max_daily_loss_pct",
        "max_drawdown_pct",
    )
    @classmethod
    def probability_fraction(cls, value: Decimal) -> Decimal:
        if not Decimal(0) < value <= Decimal(1):
            raise ValueError("risk percentages must be in (0, 1]")
        return value

    @field_validator(
        "max_concurrent_positions",
        "min_time_to_expiry_ms",
        "max_quote_age_ms",
        "oracle_reference_max_age_ms",
        "volatility_bucket_ms",
        "volatility_window_seconds",
        "volatility_min_samples",
        "min_spot_venues",
        "clock_jump_tolerance_ms",
        "quality_feature_max_age_ms",
        "quality_min_confirmations",
        "market_rollover_interval_seconds",
        "market_rollover_horizon_minutes",
        "market_rollover_max_pages",
        "polymarket_resolution_poll_seconds",
        "polymarket_resolution_retention_seconds",
        "polymarket_resolution_max_pages",
        "limitless_rollover_interval_seconds",
        "limitless_rollover_max_pages",
        "limitless_resolution_retention_seconds",
        "live_promotion_max_age_hours",
        "metrics_port",
        "smtp_port",
    )
    @classmethod
    def positive_integer(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("integer settings must be positive")
        return value

    @field_validator("volatility_min_samples")
    @classmethod
    def volatility_minimum_samples(cls, value: int) -> int:
        if value < 2:
            raise ValueError("volatility_min_samples must be at least two")
        return value

    @field_validator("polymarket_signature_type")
    @classmethod
    def valid_signature_type(cls, value: int) -> int:
        if value not in {0, 1, 2, 3}:
            raise ValueError("Polymarket signature type must be one of 0, 1, 2 or 3")
        return value

    @field_validator(
        "quality_max_prediction_spread",
        "quality_min_size_multiplier",
    )
    @classmethod
    def quality_probability_fraction(cls, value: Decimal) -> Decimal:
        if not Decimal(0) < value <= Decimal(1):
            raise ValueError("quality probability fractions must be in (0, 1]")
        return value

    @field_validator(
        "quality_max_consensus_dispersion_bps",
        "quality_shock_momentum_250ms_bps",
        "volatility_return_clip_bps",
    )
    @classmethod
    def positive_quality_decimal(cls, value: Decimal) -> Decimal:
        if value <= 0:
            raise ValueError("quality thresholds must be positive")
        return value

    @field_validator(
        "quality_min_abs_basis_bps",
        "quality_min_directional_momentum_1s_bps",
    )
    @classmethod
    def non_negative_quality_decimal(cls, value: Decimal) -> Decimal:
        if value < 0:
            raise ValueError("quality minimums must be non-negative")
        return value

    @field_validator("quality_min_confirmations")
    @classmethod
    def quality_confirmations_range(cls, value: int) -> int:
        if not 1 <= value <= 5:
            raise ValueError("quality_min_confirmations must be between 1 and 5")
        return value

    @field_validator("alert_min_interval_seconds")
    @classmethod
    def non_negative_alert_interval(cls, value: int) -> int:
        if value < 0:
            raise ValueError("alert_min_interval_seconds must be non-negative")
        return value

    @field_validator("alert_timeout_seconds")
    @classmethod
    def positive_alert_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("alert_timeout_seconds must be positive")
        return value

    @field_validator("paper_execution_latency_ms")
    @classmethod
    def non_negative_latency(cls, value: int) -> int:
        if value < 0:
            raise ValueError("paper execution latency must be non-negative")
        return value

    @field_validator("signal_cooldown_seconds")
    @classmethod
    def non_negative_cooldown(cls, value: float) -> float:
        if value < 0:
            raise ValueError("signal cooldown must be non-negative")
        return value

    @staticmethod
    def split_csv(value: str) -> tuple[str, ...]:
        return tuple(item.strip() for item in value.split(",") if item.strip())

    @property
    def binance_symbols_list(self) -> tuple[str, ...]:
        return self.split_csv(self.binance_symbols.lower())

    @property
    def okx_instruments_list(self) -> tuple[str, ...]:
        return self.split_csv(self.okx_instruments.upper())

    @property
    def coinbase_products_list(self) -> tuple[str, ...]:
        return self.split_csv(self.coinbase_products.upper())

    @property
    def hyperliquid_coins_list(self) -> tuple[str, ...]:
        return self.split_csv(self.hyperliquid_coins.upper())

    @property
    def chainlink_twap_symbols_list(self) -> tuple[str, ...]:
        return self.split_csv(self.chainlink_twap_symbols.lower())

    @property
    def chainlink_twap_windows_list(self) -> tuple[int, ...]:
        windows = tuple(int(item) for item in self.split_csv(self.chainlink_twap_windows))
        invalid = set(windows) - {30, 60}
        if invalid:
            raise ValueError(f"Chainlink TWAP windows must be 30 or 60: {sorted(invalid)}")
        return windows

    @property
    def polymarket_token_ids_list(self) -> tuple[str, ...]:
        return self.split_csv(self.polymarket_token_ids)

    @property
    def limitless_market_slugs_list(self) -> tuple[str, ...]:
        return self.split_csv(self.limitless_market_slugs)

    @property
    def pyth_feed_ids_list(self) -> tuple[str, ...]:
        return self.split_csv(self.pyth_feed_ids)

    @property
    def alert_webhook_urls_list(self) -> tuple[str, ...]:
        return self.split_csv(self.alert_webhook_urls)

    @property
    def smtp_to_list(self) -> tuple[str, ...]:
        return self.split_csv(self.smtp_to)

    @property
    def paper_execution_venues_list(self) -> tuple[Venue, ...]:
        values = self.split_csv(self.paper_execution_venues.lower())
        if not values:
            raise ValueError("at least one paper execution venue is required")
        venues: list[Venue] = []
        for value in values:
            try:
                venue = Venue(value)
            except ValueError as exc:
                raise ValueError(f"unsupported paper execution venue: {value}") from exc
            if venue not in {Venue.POLYMARKET, Venue.LIMITLESS}:
                raise ValueError(f"paper execution venue is not a prediction venue: {value}")
            if venue not in venues:
                venues.append(venue)
        return tuple(venues)

    @property
    def research_symbols_list(self) -> tuple[str, ...]:
        configured = self.research_symbols.strip()
        coverage: Counter[str] = Counter()
        feed_symbols = (
            tuple(_canonical_spot_symbol(item) for item in self.binance_symbols_list),
            tuple(_canonical_spot_symbol(item) for item in self.okx_instruments_list),
            tuple(_canonical_spot_symbol(item) for item in self.coinbase_products_list),
            tuple(_canonical_spot_symbol(item) for item in self.hyperliquid_coins_list),
        )
        for venue_symbols in feed_symbols:
            coverage.update(set(venue_symbols))
        eligible = {symbol for symbol, count in coverage.items() if count >= self.min_spot_venues}
        if configured and configured.lower() != "auto":
            requested = {_canonical_spot_symbol(item) for item in self.split_csv(configured)}
            eligible &= requested
        return tuple(sorted(eligible))

    def assert_live_guard(self) -> None:
        if not self.live_trading:
            raise RuntimeError("live trading is disabled (HIZLI_LIVE_TRADING=false)")
        if self.live_ack != "YES_I_ACCEPT_REAL_LOSS":
            raise RuntimeError("live trading acknowledgement is missing")
        if self.order_notional_usd > self.live_max_notional_usd:
            raise RuntimeError("order notional exceeds HIZLI_LIVE_MAX_NOTIONAL_USD")


def load_market_specs(path: Path) -> tuple[BinaryMarketSpec, ...]:
    if not path.exists():
        return ()
    raw = json.loads(path.read_text(encoding="utf-8"))
    specs: list[BinaryMarketSpec] = []
    for item in raw:
        strike_mode = str(item.get("strike_mode", "fixed"))
        if strike_mode not in {"fixed", "opening_oracle"}:
            raise ValueError(f"unsupported strike_mode={strike_mode!r}")
        settlement_rule = str(item.get("settlement_rule", "unknown"))
        if settlement_rule not in {"unknown", "gte_strike", "gt_strike"}:
            raise ValueError(f"unsupported settlement_rule={settlement_rule!r}")
        settlement_tolerance_ms = int(item.get("settlement_capture_tolerance_ms", 10_000))
        if settlement_tolerance_ms <= 0:
            raise ValueError("settlement_capture_tolerance_ms must be positive")
        specs.append(
            BinaryMarketSpec(
                market_id=str(item["market_id"]),
                canonical_id=str(item["canonical_id"]),
                venue=Venue(str(item["venue"])),
                symbol=str(item["symbol"]).upper(),
                yes_instrument=str(item["yes_instrument"]),
                no_instrument=(str(item["no_instrument"]) if item.get("no_instrument") else None),
                strike=Decimal(str(item.get("strike", "0"))),
                expires_at_ms=int(item["expires_at_ms"]),
                oracle=str(item.get("oracle", "unknown")),
                taker_fee_rate=Decimal(str(item.get("taker_fee_rate", "0"))),
                enabled=bool(item.get("enabled", True)),
                notes=str(item.get("notes", "")),
                starts_at_ms=(int(item["starts_at_ms"]) if item.get("starts_at_ms") else None),
                strike_mode=strike_mode,
                oracle_window_seconds=(
                    int(item["oracle_window_seconds"])
                    if item.get("oracle_window_seconds") is not None
                    else None
                ),
                strike_capture_tolerance_ms=int(
                    item.get("strike_capture_tolerance_ms", 5_000)
                ),
                stream_key=(str(item["stream_key"]) if item.get("stream_key") else None),
                settlement_rule=settlement_rule,
                settlement_capture_tolerance_ms=settlement_tolerance_ms,
            )
        )
    result = tuple(specs)
    validate_resolution_equivalence_groups(result)
    return result