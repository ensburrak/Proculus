from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import controller_async
import main_bot_async
import risk_manager
from core.decision_pipeline import DecisionPipeline
from decision.official_pipeline import process_symbol_decision
from runtime.execution_bridge import execute_decision
from runtime.runtime_symbol_universe import resolve_runtime_symbols

ROOT = Path(__file__).resolve().parents[1]


def _bull_item(*, edge_validated: bool = True, regime: str = "bull") -> dict:
    return {
        "symbol": "BTC/USDT",
        "runtime_mode": "paper",
        "regime": regime,
        "edge_validated": edge_validated,
        "ta_pack": {
            "base_decision": "long",
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
                "recent_closes": [106.0, 107.0, 108.0, 109.0, 110.0],
                "recent_lows": [105.5, 106.5, 107.5, 108.5, 109.2],
                "recent_highs": [106.5, 107.5, 108.5, 109.5, 110.5],
            },
            "1h": {
                "ema_fast": 109.0,
                "ema_slow": 107.0,
                "macd_hist": 0.10,
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
    assert decision["stop_atr_mult"] == 1.5
    assert decision["tp_r_target"] == 2.5
    assert decision["max_hold_hours"] == 48.0


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




def test_learning_probe_blocks_untracked_setup() -> None:
    item = _bull_item(edge_validated=False)
    decision = process_symbol_decision(
        item=item,
        config_overrides={
            "pipeline_v2": {
                "strict_edge_evidence": True,
                "allow_edge_cold_start": False,
                "learning_probe_mode": {
                    "enabled": True,
                    "eligible_runtime_modes": ["paper"],
                    "allow_sample_collection_without_edge": True,
                    "max_leverage": 1,
                    "max_size_scale": 0.25,
                    "regime_min_confidence_override": {"bull": 0.55},
                    "tracked_setups": ["some.other.setup.v2"],
                    "target_trades_per_setup": 100,
                },
            }
        },
    )

    assert decision["action"] == "hold"
    assert decision["risk_scale"] == 0.0
    assert decision["reason"] == "learning probe setup not tracked"
    assert decision["learning_probe"]["tracked"] is False

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


def _release_config(*, setup_id: str, allow_live: bool = True, valid_hash: bool = True) -> dict:
    allowed = [setup_id]
    allowed_hash = hashlib.sha256(
        json.dumps(allowed, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if not valid_hash:
        allowed_hash = "0" * 64
    return {
        "pipeline_v2": {"allow_live_exchange_side_effects": True},
        "ai_authority": {"allow_live_exchange_side_effects": True},
        "strategy_release": {
            "requested_release": "v2",
            "required_strategy_oos_evidence_hash": "evidence-sha256",
            "required_strategy_policy_hash": "policy-sha256",
            "allowed_setup_ids": allowed,
            "allowed_setup_ids_hash": allowed_hash,
            "allowed_runtime_modes": ["live"],
            "allow_live": allow_live,
            "allow_testnet": False,
        },
    }


def _live_execution_fixture() -> tuple[dict, dict]:
    setup_id = "bull_trend.pullback.long.15m.v2"
    item = {
        "symbol": "BTC/USDT",
        "runtime_mode": "live",
        "order_size": 0.01,
    }
    decision = {
        "action": "enter",
        "direction": "long",
        "setup_id": setup_id,
        "risk_scale": 0.5,
        "edge_contract": {"validated": True},
    }
    return item, decision


def test_live_execution_release_disabled_fails_closed() -> None:
    class DummyExchange:
        async def create_order(self, *args, **kwargs):
            raise AssertionError("release-disabled live mode must never submit")

    item, decision = _live_execution_fixture()
    cfg = _release_config(setup_id=decision["setup_id"], allow_live=False)
    result = asyncio.run(execute_decision(DummyExchange(), item, decision, cfg))

    assert result["status"] == "live_blocked_by_strategy_release"
    assert result["order_sent"] is False
    assert result["strategy_release"]["reason"] == "live_release_disabled"


def test_live_execution_release_hash_mismatch_fails_closed() -> None:
    class DummyExchange:
        async def create_order(self, *args, **kwargs):
            raise AssertionError("hash-mismatched release must never submit")

    item, decision = _live_execution_fixture()
    cfg = _release_config(setup_id=decision["setup_id"], valid_hash=False)
    result = asyncio.run(execute_decision(DummyExchange(), item, decision, cfg))

    assert result["status"] == "live_blocked_by_strategy_release"
    assert result["order_sent"] is False
    assert result["strategy_release"]["reason"] == "allowed_setup_ids_hash_mismatch"


def test_live_execution_requires_complete_release_contract_before_submit() -> None:
    class DummyExchange:
        def __init__(self) -> None:
            self.calls = []

        async def create_order(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return {"id": "order-1"}

    item, decision = _live_execution_fixture()
    cfg = _release_config(setup_id=decision["setup_id"])
    exchange = DummyExchange()
    result = asyncio.run(execute_decision(exchange, item, decision, cfg))

    assert result["status"] == "submitted"
    assert result["order_sent"] is True
    assert result["strategy_release"]["reason"] == "released"
    assert len(exchange.calls) == 1

def test_strategy_release_is_empty_and_live_fail_closed() -> None:
    from runtime.strategy_release_gate import evaluate_strategy_release

    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    release = config["strategy_release"]
    assert release["allowed_setup_ids"] == []
    assert release["allow_live"] is False
    assert release["allow_testnet"] is False
    assert release["required_strategy_oos_evidence_hash"] == ""
    assert release["required_strategy_policy_hash"] == ""

    decision = {
        "action": "enter",
        "setup_id": "bull_trend.pullback.long.15m.v2",
        "edge_contract": {"validated": True},
    }
    audit = evaluate_strategy_release(
        item={"runtime_mode": "live"},
        decision=decision,
        config=config,
    )
    assert audit["allowed"] is False
    assert audit["reason"] == "runtime_mode_not_released"

