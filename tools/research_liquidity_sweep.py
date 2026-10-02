#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

import decision.official_pipeline as v2_pipeline
from tools.fresh_two_year_v2_backtest import (
    Candidate,
    acceptance_flags,
    compact_stats,
    f,
    prepare_frame,
    simulate,
    symbol_from_path,
)


def _mtf_not_hostile(row: pd.Series, side: str, *, max_adx: float | None = None) -> bool:
    if max_adx is not None and f(row.get("adx")) > max_adx:
        return False
    h1_fast=f(row.get("ema_fast_1h")); h1_slow=f(row.get("ema_slow_1h"))
    h4_fast=f(row.get("ema_fast_4h")); h4_slow=f(row.get("ema_slow_4h"))
    h4_macd=f(row.get("macd_4h"))
    if side=="long":
        if h1_fast < h1_slow and h4_fast < h4_slow and h4_macd < 0:
            return False
    else:
        if h1_fast > h1_slow and h4_fast > h4_slow and h4_macd > 0:
            return False
    return True


def generate_candidates(symbol: str, frame: pd.DataFrame, profile: dict[str,Any]) -> list[Candidate]:
    lookback=int(profile["lookback"])
    sweep=float(profile["sweep_bps"])/10_000.0
    reclaim=float(profile["reclaim_bps"])/10_000.0
    wick_min=float(profile["wick_ratio_min"])
    vol_min=float(profile.get("vol_z_min",-99.0))
    adx_max=profile.get("adx_max")
    rsi_long_max=float(profile.get("rsi_long_max",45.0))
    rsi_short_min=float(profile.get("rsi_short_min",55.0))
    stop_atr=float(profile.get("stop_atr_mult",1.0))
    tp_r=float(profile.get("tp_r_target",1.75))
    max_hold=int(profile.get("max_hold_bars",64))

    prior_low=frame["low"].shift(1).rolling(lookback,min_periods=lookback).min()
    prior_high=frame["high"].shift(1).rolling(lookback,min_periods=lookback).max()
    out:list[Candidate]=[]

    for idx in range(max(800,lookback+5),len(frame)-1):
        row=frame.iloc[idx]
        lo=f(prior_low.iloc[idx],float("nan")); hi=f(prior_high.iloc[idx],float("nan"))
        if not math.isfinite(lo) or not math.isfinite(hi):
            continue
        o=f(row.get("open")); h=f(row.get("high")); l=f(row.get("low")); c=f(row.get("close"))
        if min(o,h,l,c)<=0 or h<=l:
            continue
        vol_z=f(row.get("vol_z"))
        if vol_z<vol_min:
            continue
        rsi=f(row.get("rsi"))
        rng=h-l
        lower_wick=max(0.0,min(o,c)-l)/rng
        upper_wick=max(0.0,h-max(o,c))/rng

        long_sweep=l < lo*(1.0-sweep)
        long_reclaim=c > lo*(1.0+reclaim)
        long_candle=c>o
        if long_sweep and long_reclaim and long_candle and lower_wick>=wick_min and rsi<=rsi_long_max and _mtf_not_hostile(row,"long",max_adx=adx_max):
            atr=f(row.get("atr"))
            if atr>0:
                out.append(Candidate(
                    symbol=symbol,decision_idx=idx,entry_idx=idx+1,
                    entry_time=pd.Timestamp(frame.iloc[idx+1]["timestamp"]),
                    side="long",
                    setup_id=f"research.liquidity_sweep.{profile['name']}.long.15m",
                    strategy="research_liquidity_sweep",
                    regime="range",
                    confidence=min(0.9,0.68+min(0.08,lower_wick/5.0)+min(0.06,max(0.0,vol_z)/20.0)),
                    risk_scale=0.25,leverage=1.0,atr=atr,decision_price=c,
                    stop_atr_mult=stop_atr,tp_r_target=tp_r,max_hold_bars=max_hold,
                ))

        short_sweep=h > hi*(1.0+sweep)
        short_reclaim=c < hi*(1.0-reclaim)
        short_candle=c<o
        if short_sweep and short_reclaim and short_candle and upper_wick>=wick_min and rsi>=rsi_short_min and _mtf_not_hostile(row,"short",max_adx=adx_max):
            atr=f(row.get("atr"))
            if atr>0:
                out.append(Candidate(
                    symbol=symbol,decision_idx=idx,entry_idx=idx+1,
                    entry_time=pd.Timestamp(frame.iloc[idx+1]["timestamp"]),
                    side="short",
                    setup_id=f"research.liquidity_sweep.{profile['name']}.short.15m",
                    strategy="research_liquidity_sweep",
                    regime="range",
                    confidence=min(0.9,0.68+min(0.08,upper_wick/5.0)+min(0.06,max(0.0,vol_z)/20.0)),
                    risk_scale=0.25,leverage=1.0,atr=atr,decision_price=c,
                    stop_atr_mult=stop_atr,tp_r_target=tp_r,max_hold_bars=max_hold,
                ))
    return out


def profile_grid() -> list[dict[str,Any]]:
    return [
        {"name":"ls20_s05_r00_w35","lookback":20,"sweep_bps":5,"reclaim_bps":0,"wick_ratio_min":0.35,"vol_z_min":0.0,"adx_max":24,"rsi_long_max":42,"rsi_short_min":58,"stop_atr_mult":1.0,"tp_r_target":1.75,"max_hold_bars":64},
        {"name":"ls20_s10_r05_w40","lookback":20,"sweep_bps":10,"reclaim_bps":5,"wick_ratio_min":0.40,"vol_z_min":0.5,"adx_max":22,"rsi_long_max":40,"rsi_short_min":60,"stop_atr_mult":1.0,"tp_r_target":2.0,"max_hold_bars":64},
        {"name":"ls32_s05_r05_w35","lookback":32,"sweep_bps":5,"reclaim_bps":5,"wick_ratio_min":0.35,"vol_z_min":0.0,"adx_max":24,"rsi_long_max":42,"rsi_short_min":58,"stop_atr_mult":1.25,"tp_r_target":2.0,"max_hold_bars":80},
        {"name":"ls32_s10_r10_w45","lookback":32,"sweep_bps":10,"reclaim_bps":10,"wick_ratio_min":0.45,"vol_z_min":0.5,"adx_max":22,"rsi_long_max":40,"rsi_short_min":60,"stop_atr_mult":1.25,"tp_r_target":2.25,"max_hold_bars":96},
        {"name":"ls48_s10_r05_w40","lookback":48,"sweep_bps":10,"reclaim_bps":5,"wick_ratio_min":0.40,"vol_z_min":0.8,"adx_max":25,"rsi_long_max":45,"rsi_short_min":55,"stop_atr_mult":1.25,"tp_r_target":2.25,"max_hold_bars":96},
        {"name":"ls48_s15_r10_w50","lookback":48,"sweep_bps":15,"reclaim_bps":10,"wick_ratio_min":0.50,"vol_z_min":1.0,"adx_max":22,"rsi_long_max":40,"rsi_short_min":60,"stop_atr_mult":1.5,"tp_r_target":2.5,"max_hold_bars":128},
    ]


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",type=Path,required=True)
    ap.add_argument("--days",type=int,default=730)
    ap.add_argument("--holdout-days",type=int,default=180)
    ap.add_argument("--embargo-hours",type=int,default=48)
    ap.add_argument("--initial-balance",type=float,default=10000.0)
    ap.add_argument("--fee-bps",type=float,default=5.0)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()

    cfg=json.loads((ROOT/"config.json").read_text(encoding="utf-8"))
    v2_pipeline._load_config=lambda:cfg

    frames:dict[str,pd.DataFrame]={}
    for path in sorted(args.data_dir.glob("*_15m.parquet")):
        symbol=symbol_from_path(path)
        try:
            frame=prepare_frame(symbol,args.data_dir,args.days)
        except Exception as exc:
            print(f"[SKIP] {symbol}: {type(exc).__name__}: {exc}",flush=True)
            continue
        if len(frame)>=1200:
            frames[symbol]=frame
    if not frames:
        raise SystemExit("no replay frames")

    data_end=max(pd.Timestamp(x["timestamp"].max()) for x in frames.values())
    holdout_start=data_end-pd.Timedelta(days=args.holdout_days)
    development_end=holdout_start-pd.Timedelta(hours=args.embargo_hours)
    validation_start=development_end-pd.Timedelta(days=120)
    train_end=validation_start-pd.Timedelta(hours=args.embargo_hours)

    trials=[]; passers=[]
    for profile in profile_grid():
        all_candidates=[]
        for symbol,frame in frames.items():
            all_candidates.extend(generate_candidates(symbol,frame,profile))
        all_candidates.sort(key=lambda x:(x.entry_time,x.symbol,x.setup_id))
        train=[x for x in all_candidates if x.entry_time<train_end]
        validation=[x for x in all_candidates if validation_start<=x.entry_time<development_end]
        oos=[x for x in all_candidates if x.entry_time>=holdout_start]

        train5=simulate(candidates=train,frames=frames,initial_balance=args.initial_balance,fee_bps=args.fee_bps,slippage_bps=5.0,cfg=cfg)
        train15=simulate(candidates=train,frames=frames,initial_balance=args.initial_balance,fee_bps=args.fee_bps,slippage_bps=15.0,cfg=cfg)
        train_checks={
            "trades_gte_100":int(train5.get("trades",0))>=100,
            "pf5_gte_1_10":float(train5.get("profit_factor",0.0))>=1.10,
            "exp5_gte_0_03":float(train5.get("expectancy_r",-999.0))>=0.03,
            "pf15_gte_1_00":float(train15.get("profit_factor",0.0))>=1.00,
            "exp15_gte_0":float(train15.get("expectancy_r",-999.0))>=0.0,
        }
        trial={
            "profile":profile,
            "candidate_counts":{"all":len(all_candidates),"train":len(train),"validation":len(validation),"oos":len(oos)},
            "train_5bps":compact_stats(train5),
            "train_15bps":compact_stats(train15),
            "train_checks":train_checks,
            "train_pass":all(train_checks.values()),
            "validation":None,
            "validation_pass":False,
            "oos_locked_result":{},
        }
        if trial["train_pass"]:
            val5=simulate(candidates=validation,frames=frames,initial_balance=args.initial_balance,fee_bps=args.fee_bps,slippage_bps=5.0,cfg=cfg)
            val15=simulate(candidates=validation,frames=frames,initial_balance=args.initial_balance,fee_bps=args.fee_bps,slippage_bps=15.0,cfg=cfg)
            val_checks={
                "trades_gte_50":int(val5.get("trades",0))>=50,
                "pf5_gte_1_15":float(val5.get("profit_factor",0.0))>=1.15,
                "exp5_gte_0_05":float(val5.get("expectancy_r",-999.0))>=0.05,
                "dd5_lte_20":float(val5.get("max_drawdown_pct_realized",999.0))<=20.0,
                "pf15_gte_1_05":float(val15.get("profit_factor",0.0))>=1.05,
                "exp15_gte_0":float(val15.get("expectancy_r",-999.0))>=0.0,
            }
            trial["validation"]={"slippage_5bps":compact_stats(val5),"slippage_15bps":compact_stats(val15),"checks":val_checks}
            trial["validation_pass"]=all(val_checks.values())
            if trial["validation_pass"]:
                passers.append((trial,oos))
        trials.append(trial)
        print("LIQUIDITY_SWEEP_TRIAL="+json.dumps(trial,ensure_ascii=False),flush=True)

    selected=None
    if passers:
        passers.sort(key=lambda pair:(
            min(float(pair[0]["validation"]["slippage_5bps"]["profit_factor"]),float(pair[0]["validation"]["slippage_15bps"]["profit_factor"])),
            min(float(pair[0]["validation"]["slippage_5bps"]["expectancy_r"]),float(pair[0]["validation"]["slippage_15bps"]["expectancy_r"])),
        ),reverse=True)
        chosen,oos=passers[0]
        selected=dict(chosen["profile"])
        for bps in (5.0,15.0):
            payload=simulate(candidates=oos,frames=frames,initial_balance=args.initial_balance,fee_bps=args.fee_bps,slippage_bps=bps,cfg=cfg)
            chosen["oos_locked_result"][f"slippage_{int(bps)}bps"]={**compact_stats(payload),"acceptance":acceptance_flags(payload)}

    report={
        "schema":"proculus-liquidity-sweep-research-v1",
        "production_changed":False,
        "selection":"train -> validation -> retrospective holdout",
        "data_end":data_end.isoformat(),
        "train_end_exclusive":train_end.isoformat(),
        "validation_start":validation_start.isoformat(),
        "development_end_exclusive":development_end.isoformat(),
        "holdout_start":holdout_start.isoformat(),
        "profiles":trials,
        "selected":selected,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print("LIQUIDITY_SWEEP_SUMMARY="+json.dumps({
        "selected":selected,
        "profiles":[{
            "name":x["profile"]["name"],
            "counts":x["candidate_counts"],
            "train_5bps":x["train_5bps"],
            "train_15bps":x["train_15bps"],
            "train_pass":x["train_pass"],
            "validation":x["validation"],
            "validation_pass":x["validation_pass"],
            "oos_locked_result":x["oos_locked_result"],
        } for x in trials],
    },ensure_ascii=False),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
