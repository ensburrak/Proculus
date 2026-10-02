from __future__ import annotations

import asyncio
from types import SimpleNamespace

import controller_async
import main_bot_async
import risk_manager
from core.decision_pipeline import DecisionPipeline
from runtime.execution_bridge import execute_decision
from runtime.runtime_symbol_universe import resolve_runtime_symbols


def _bull_item(*, edge_validated: bool = True, regime: str = "bull") -> dict:
    return {
        "symbol": "BTC/USDT",
        "runtime_mode": "paper",
        "regime": regime,
        "edge_validated": edge_validated,
        "ta_pack": {
            "base_decision": "long",
            "price": 109.35,
            "rsi": 56.0,
            "adx": 28.0,
            "atr_ratio": 0.012,
            "ema": {"fast": 109.2, "slow": 108.8},
        },
        "mtf_features": {
            "4h": {
                "ema_fast": 108.0,
                "ema_slow": 105.0,
                "macd_hist": 0.30,
                "adx": 26.0,
                "recent_closes": [106.0, 107.0, 108.0, 109.0, 110.0],
                "recent_lows": [105.5, 106.5, 107.5, 108.5, 109.2],
                "recent_highs": [106.5, 107.5, 108.5, 109.5, 110.5],
            },
            "1h": {
                "ema_fast": 109.0,
                "ema_slow": 107.0,
                "macd_hist": 0.10,
                "adx": 24.0,
                "recent_closes": [107.2, 108.0, 108.5, 109.0, 109.5],
                "recent_lows": [106.8, 107.5, 108.0, 108.4, 108.9],
                "recent_highs": [107.5, 108.4, 108.9, 109.4, 109.8],
            },
            "15m": {
                "ema_fast": 109.2,
                "ema_slow": 108.8,
                "macd_hist": 0.04,
                "recent_closes": [109.0, 109.1, 109.0, 109.15, 109.35],
                "recent_lows": [108.9, 109.0, 108.85, 109.0, 109.1],
                "recent_highs": [109.1, 109.2, 109.15, 109.3, 109.45],
            },
        },
    }


def test_controller_async_runs_real_v2_decision_path() -> None:
    result = asyncio.run(controller_async.decide_batch([_bull_item()]))
    decision = result["BTC/USDT"]

    assert decision["action"] == "enter"
    assert decision["direction"] == "long"
    assert decision["pipeline"] == "v2"
    assert decision["setup_id"].startswith("bull_trend.pullback.long")
    assert decision["ai_authority"]["directional"] is False
    assert 0.0 < decision["risk_scale"] <= 1.0


def test_paper_learning_probe_collects_edge_at_capped_size() -> None:
    result = asyncio.run(DecisionPipeline().decide_batch([_bull_item(edge_validated=False)]))
    decision = result["BTC/USDT"]

    assert decision["action"] == "enter"
    assert 0.0 < decision["risk_scale"] <= 0.25
    assert decision["lev"] == 1
    assert decision["edge_contract"]["mode"] == "learning_probe"
    assert decision["learning_probe"]["active"] is True


def test_live_strict_edge_evidence_still_fails_closed() -> None:
    item = _bull_item(edge_validated=False)
    item["runtime_mode"] = "live"
    result = asyncio.run(DecisionPipeline().decide_batch([item]))
    decision = result["BTC/USDT"]

    assert decision["action"] == "hold"
    assert decision["risk_scale"] == 0.0
    assert "edge evidence" in decision["reason"]


def test_shock_regime_is_hard_no_trade() -> None:
    item = _bull_item()
    item["regime"] = "shock"
    result = asyncio.run(controller_async.decide_batch([item]))
    decision = result["BTC/USDT"]

    assert decision["action"] == "hold"
    assert decision["lev"] == 0
    assert decision["risk_scale"] == 0.0


def test_paper_execution_bridge_never_sends_real_order() -> None:
    class DummyExchange:
        async def create_order(self, *args, **kwargs):
            raise AssertionError("paper mode must never call create_order")

    item = _bull_item()
    decision = asyncio.run(controller_async.decide_batch([item]))["BTC/USDT"]
    result = asyncio.run(execute_decision(DummyExchange(), item, decision, {}))

    assert result["status"] == "simulated"
    assert result["order_sent"] is False


def test_legacy_facades_import_against_new_packages() -> None:
    assert callable(main_bot_async.main)
    assert callable(risk_manager.compute_stop_loss)
    assert controller_async.OFFICIAL_DECISION_PIPELINE == "proculus_pipeline_v2"



def test_runtime_symbol_universe_preserves_explicit_symbols() -> None:
    class NoDiscoveryExchange:
        async def load_markets(self):
            raise AssertionError("explicit symbols must bypass discovery")

    cfg = {
        "trade_parameters": {
            "symbols": ["AAVE/USDT:USDT"],
            "symbol_source": "okx_swap_all",
        }
    }
    resolved = asyncio.run(resolve_runtime_symbols(NoDiscoveryExchange(), cfg))
    assert resolved == ["AAVE/USDT:USDT"]


def test_runtime_symbol_universe_applies_okx_swap_filters() -> None:
    old_ms = 1_700_000_000_000

    class DummyExchange:
        async def load_markets(self):
            return {
                "BTC/USDT:USDT": {
                    "symbol": "BTC/USDT:USDT",
                    "id": "BTC-USDT-SWAP",
                    "base": "BTC",
                    "quote": "USDT",
                    "settle": "USDT",
                    "swap": True,
                    "linear": True,
                    "active": True,
                    "info": {"listTime": str(old_ms)},
                },
                "THIN/USDT:USDT": {
                    "symbol": "THIN/USDT:USDT",
                    "id": "THIN-USDT-SWAP",
                    "base": "THIN",
                    "quote": "USDT",
                    "settle": "USDT",
                    "swap": True,
                    "linear": True,
                    "active": True,
                    "info": {"listTime": str(old_ms)},
                },
                "ZEC/USDT:USDT": {
                    "symbol": "ZEC/USDT:USDT",
                    "id": "ZEC-USDT-SWAP",
                    "base": "ZEC",
                    "quote": "USDT",
                    "settle": "USDT",
                    "swap": True,
                    "linear": True,
                    "active": True,
                    "info": {"listTime": str(old_ms)},
                },
            }

        async def fetch_tickers(self):
            return {
                "BTC/USDT:USDT": {"bid": 100.0, "ask": 100.02, "last": 100.01, "quoteVolume": 100_000_000.0},
                "THIN/USDT:USDT": {"bid": 10.0, "ask": 10.001, "last": 10.0, "quoteVolume": 1_000_000.0},
                "ZEC/USDT:USDT": {"bid": 20.0, "ask": 20.001, "last": 20.0, "quoteVolume": 100_000_000.0},
            }

        async def fetch_funding_rates(self, symbols):
            assert symbols == ["BTC/USDT:USDT"]
            return {"BTC/USDT:USDT": {"fundingRate": 0.0001}}

    cfg = {
        "trade_parameters": {
            "symbols": [],
            "symbol_source": "okx_swap_all",
            "symbol_quote": "USDT",
            "symbol_exclude_bases": ["ZEC"],
            "symbol_exclude_prefixes": ["TEST"],
            "always_on_symbols": ["BTC/USDT"],
        },
        "symbol_filters": {
            "min_24h_volume": 50_000_000,
            "max_spread_pct": 0.05,
            "min_listing_days": 90,
            "max_funding_rate_abs": 0.001,
        },
        "performance": {"max_symbols_per_loop": 0},
    }
    resolved = asyncio.run(resolve_runtime_symbols(DummyExchange(), cfg))
    assert resolved == ["BTC/USDT:USDT"]
