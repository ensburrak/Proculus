from __future__ import annotations

from decision.regime_policy import get_regime_policy
from decision.stochrsi_opportunity import evaluate_stochrsi_opportunity
from decision.strategy_router import route_to_expert
from runtime.stochrsi_parallel import build_stochrsi_parallel_decision


def _trend_item() -> dict:
    return {
        "symbol":"BTC/USDT",
        "regime":"bull",
        "ta_pack":{
            "price":109.35,"rsi":54.0,"adx":28.0,"atr_ratio":0.012,"vol_z":0.8,
            "ema":{"fast":109.2,"slow":108.8},
        },
        "mtf_features":{
            "4h":{"close":110.0,"ema_fast":108.0,"ema_slow":105.0,"ema200":100.0,"macd_hist":0.3,"adx":26.0,
                  "recent_closes":[106,107,108,109,110],"recent_lows":[105,106,107,108,109],"recent_highs":[107,108,109,110,111]},
            "1h":{"close":109.5,"ema_fast":109.0,"ema_slow":107.0,"ema200":103.0,"macd_hist":0.1,"adx":24.0,
                  "recent_closes":[107.2,108,108.5,109,109.5],"recent_lows":[106.8,107.5,108,108.4,108.9],"recent_highs":[107.5,108.4,108.9,109.4,109.8]},
            "15m":{"close":109.35,"prev_close":109.15,"ema_fast":109.2,"ema_slow":108.8,"ema200":104.0,
                   "recent_closes":[109,109.1,109,109.15,109.35],"recent_lows":[108.9,109,108.85,109,109.1],"recent_highs":[109.1,109.2,109.15,109.3,109.45]},
        },
    }


def test_trend_router_uses_v2_expert() -> None:
    item=_trend_item()
    signal=route_to_expert(regime="bull",item=item,ta=item["ta_pack"])
    assert signal is not None
    assert signal.setup_id=="bull_trend.pullback.long.15m.v2"
    assert signal.direction=="long"
    assert signal.confidence >= get_regime_policy("bull")["min_confidence"]


def test_range_requires_dual_extreme_and_reentry() -> None:
    item={
        "regime":"range",
        "ta_pack":{"price":100.0,"rsi":29.0,"stoch_k":55.0,"adx":14.0,"atr_ratio":0.01,"ema":{"fast":100.0,"slow":100.1}},
        "mtf_features":{"15m":{"recent_closes":[100.4,100.2,99.9,99.7,99.8,100.0]}},
    }
    assert route_to_expert(regime="range",item=item,ta=item["ta_pack"]) is None
    item["ta_pack"]["stoch_k"]=12.0
    signal=route_to_expert(regime="range",item=item,ta=item["ta_pack"])
    assert signal is not None
    assert signal.setup_id=="range_revert.low_band_rejection.long.15m.v2"


def test_stochrsi_is_independent_and_true_cross_only() -> None:
    item={
        "regime":"bull",
        "ta_pack":{
            "rsi":52.0,"adx":25.0,"atr_ratio":0.01,"vol_z":0.5,
            "ema":{"fast":101.0,"slow":100.0},
            "stoch_rsi_90_prev_k":8.0,"stoch_rsi_90_prev_d":12.0,
            "stoch_rsi_90_k":18.0,"stoch_rsi_90_d":16.0,
        },
    }
    result=evaluate_stochrsi_opportunity(item=item,ta=item["ta_pack"])
    assert result.action=="enter"
    assert result.direction=="long"
    assert result.setup_id=="stochrsi_opportunity.bull.long.15m.v2"

    item["ta_pack"]["stoch_rsi_90_prev_k"]=20.0
    item["ta_pack"]["stoch_rsi_90_prev_d"]=10.0
    blocked=evaluate_stochrsi_opportunity(item=item,ta=item["ta_pack"])
    assert blocked.action=="hold"


def test_stochrsi_runtime_authority_is_mode_scoped() -> None:
    item={
        "symbol":"BTC/USDT",
        "regime":"bull",
        "runtime_mode":"paper",
        "ta_pack":{
            "rsi":52.0,"adx":25.0,"atr_ratio":0.01,"vol_z":0.5,
            "ema":{"fast":101.0,"slow":100.0},
            "stoch_rsi_90_prev_k":8.0,"stoch_rsi_90_prev_d":12.0,
            "stoch_rsi_90_k":18.0,"stoch_rsi_90_d":16.0,
        },
    }
    cfg={"stochrsi_parallel":{"enabled":True,"paper_orders_enabled":True,"demo_orders_enabled":False,"live_orders_enabled":False}}
    paper=build_stochrsi_parallel_decision(item,cfg)
    assert paper["action"]=="enter"
    assert paper["order_authorized"] is True
    assert paper["merge_into_primary_pipeline"] is False

    item["runtime_mode"]="live"
    live=build_stochrsi_parallel_decision(item,cfg)
    assert live["action"]=="enter"
    assert live["order_authorized"] is False
    assert live["execution_blocker"]=="stochrsi_mode_not_authorized"


def test_probe_confidence_override_does_not_relax_default_policy() -> None:
    item={
        "regime":"range",
        "ta_pack":{"price":100.0,"rsi":30.0,"stoch_k":20.0,"adx":14.0,"atr_ratio":0.01,"ema":{"fast":100.0,"slow":100.1}},
        "mtf_features":{"15m":{"recent_closes":[100.4,100.2,99.9,99.7,99.8,100.0]}},
    }
    assert route_to_expert(regime="range",item=item,ta=item["ta_pack"]) is None
    probe=route_to_expert(regime="range",item=item,ta=item["ta_pack"],min_confidence_override=0.55)
    assert probe is not None
    assert probe.confidence < get_regime_policy("range")["min_confidence"]
    assert probe.setup_id=="range_revert.low_band_rejection.long.15m.v2"
