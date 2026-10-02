#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
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


def _aligned(row: pd.Series, side: str, *, adx_min: float, h1_adx_min: float, h4_adx_min: float) -> bool:
    adx=f(row.get("adx"))
    h1_adx=f(row.get("adx_1h"))
    h4_adx=f(row.get("adx_4h"))
    atr=f(row.get("atr_ratio"))
    vol_z=f(row.get("vol_z"))
    rsi=f(row.get("rsi"))
    if adx < adx_min or h1_adx < h1_adx_min or h4_adx < h4_adx_min:
        return False
    if atr < 0.001 or atr > 0.04 or abs(vol_z) >= 3.0:
        return False

    close=f(row.get("close"))
    fast=f(row.get("ema_fast"))
    slow=f(row.get("ema_slow"))
    if min(close,fast,slow) <= 0:
        return False

    if side=="long":
        if not (fast > slow and close >= slow and 45.0 <= rsi <= 72.0):
            return False
    else:
        if not (fast < slow and close <= slow and 28.0 <= rsi <= 55.0):
            return False

    for tf, require_macd in (("1h",False),("4h",True)):
        tf_close=f(row.get(f"close_{tf}"))
        tf_fast=f(row.get(f"ema_fast_{tf}"))
        tf_slow=f(row.get(f"ema_slow_{tf}"))
        raw_e200=row.get(f"ema200_{tf}")
        e200=float(raw_e200) if raw_e200 is not None and pd.notna(raw_e200) and math.isfinite(float(raw_e200)) else None
        raw_macd=row.get(f"macd_{tf}")
        macd=float(raw_macd) if raw_macd is not None and pd.notna(raw_macd) and math.isfinite(float(raw_macd)) else None
        if side=="long":
            if not (tf_fast > tf_slow and tf_close >= tf_slow):
                return False
            if e200 is not None and tf_close <= e200:
                return False
            if require_macd and macd is not None and macd < 0:
                return False
        else:
            if not (tf_fast < tf_slow and tf_close <= tf_slow):
                return False
            if e200 is not None and tf_close >= e200:
                return False
            if require_macd and macd is not None and macd > 0:
                return False
    return True


def generate_profile_candidates(
    *,
    symbol: str,
    frame: pd.DataFrame,
    profile: dict[str,Any],
) -> list[Candidate]:
    lookback=int(profile["lookback"])
    breakout=float(profile["breakout_bps"])/10_000.0
    retest_tol=float(profile["retest_tolerance_bps"])/10_000.0
    reclaim=float(profile["reclaim_bps"])/10_000.0
    window=int(profile["retest_window_bars"])
    vol_min=float(profile.get("vol_z_min",-99.0))
    adx_min=float(profile["adx_min"])
    h1_adx_min=float(profile["h1_adx_min"])
    h4_adx_min=float(profile["h4_adx_min"])
    stop_atr=float(profile.get("stop_atr_mult",1.25))
    target_r=float(profile.get("tp_r_target",2.5))
    max_hold=int(profile.get("max_hold_bars",128))

    prior_high=frame["high"].shift(1).rolling(lookback,min_periods=lookback).max()
    prior_low=frame["low"].shift(1).rolling(lookback,min_periods=lookback).min()
    out:list[Candidate]=[]
    active_long:tuple[int,float] | None=None
    active_short:tuple[int,float] | None=None

    for idx in range(max(800,lookback+5),len(frame)-1):
        row=frame.iloc[idx]
        close=f(row.get("close"))
        vol_z=f(row.get("vol_z"))
        hi=f(prior_high.iloc[idx],float("nan"))
        lo=f(prior_low.iloc[idx],float("nan"))
        if not math.isfinite(hi) or not math.isfinite(lo):
            continue

        if _aligned(row,"long",adx_min=adx_min,h1_adx_min=h1_adx_min,h4_adx_min=h4_adx_min) and vol_z>=vol_min:
            if close > hi*(1.0+breakout):
                active_long=(idx,hi)
        if _aligned(row,"short",adx_min=adx_min,h1_adx_min=h1_adx_min,h4_adx_min=h4_adx_min) and vol_z>=vol_min:
            if close < lo*(1.0-breakout):
                active_short=(idx,lo)

        for side,active in (("long",active_long),("short",active_short)):
            if active is None:
                continue
            break_idx,level=active
            age=idx-break_idx
            if age<=0:
                continue
            if age>window:
                if side=="long":
                    active_long=None
                else:
                    active_short=None
                continue

            if not _aligned(row,side,adx_min=adx_min,h1_adx_min=h1_adx_min,h4_adx_min=h4_adx_min):
                continue

            low=f(row.get("low")); high=f(row.get("high")); open_=f(row.get("open"))
            if side=="long":
                touched=low <= level*(1.0+retest_tol)
                reclaimed=close >= level*(1.0+reclaim)
                candle_ok=close >= open_
            else:
                touched=high >= level*(1.0-retest_tol)
                reclaimed=close <= level*(1.0-reclaim)
                candle_ok=close <= open_
            if not (touched and reclaimed and candle_ok):
                continue

            atr=f(row.get("atr"))
            if atr<=0:
                continue
            confidence=min(0.92,0.70 + min(0.10,max(0.0,(f(row.get("adx"))-adx_min)/100.0)) + min(0.08,max(0.0,vol_z)/20.0))
            out.append(Candidate(
                symbol=symbol,
                decision_idx=idx,
                entry_idx=idx+1,
                entry_time=pd.Timestamp(frame.iloc[idx+1]["timestamp"]),
                side=side,
                setup_id=f"research.breakout_retest.{profile['name']}.{side}.15m",
                strategy="research_breakout_retest",
                regime="bull" if side=="long" else "bear",
                confidence=confidence,
                risk_scale=0.25,
                leverage=1.0,
                atr=atr,
                decision_price=close,
                stop_atr_mult=stop_atr,
                tp_r_target=target_r,
                max_hold_bars=max_hold,
            ))
            if side=="long":
                active_long=None
            else:
                active_short=None
    return out


def profile_grid() -> list[dict[str,Any]]:
    return [
        {"name":"br20_a25_v05","lookback":20,"breakout_bps":5,"retest_tolerance_bps":25,"reclaim_bps":5,"retest_window_bars":6,"adx_min":25,"h1_adx_min":20,"h4_adx_min":20,"vol_z_min":0.5,"stop_atr_mult":1.25,"tp_r_target":2.5,"max_hold_bars":128},
        {"name":"br20_a28_v08","lookback":20,"breakout_bps":10,"retest_tolerance_bps":20,"reclaim_bps":8,"retest_window_bars":6,"adx_min":28,"h1_adx_min":22,"h4_adx_min":22,"vol_z_min":0.8,"stop_atr_mult":1.25,"tp_r_target":2.5,"max_hold_bars":128},
        {"name":"br32_a25_v05","lookback":32,"breakout_bps":8,"retest_tolerance_bps":25,"reclaim_bps":5,"retest_window_bars":8,"adx_min":25,"h1_adx_min":20,"h4_adx_min":20,"vol_z_min":0.5,"stop_atr_mult":1.25,"tp_r_target":2.75,"max_hold_bars":160},
        {"name":"br32_a28_v08","lookback":32,"breakout_bps":10,"retest_tolerance_bps":20,"reclaim_bps":8,"retest_window_bars":8,"adx_min":28,"h1_adx_min":22,"h4_adx_min":22,"vol_z_min":0.8,"stop_atr_mult":1.5,"tp_r_target":3.0,"max_hold_bars":160},
        {"name":"br48_a28_v10","lookback":48,"breakout_bps":10,"retest_tolerance_bps":30,"reclaim_bps":10,"retest_window_bars":10,"adx_min":28,"h1_adx_min":22,"h4_adx_min":22,"vol_z_min":1.0,"stop_atr_mult":1.5,"tp_r_target":3.0,"max_hold_bars":192},
        {"name":"br48_a30_v12","lookback":48,"breakout_bps":12,"retest_tolerance_bps":20,"reclaim_bps":10,"retest_window_bars":8,"adx_min":30,"h1_adx_min":24,"h4_adx_min":24,"vol_z_min":1.2,"stop_atr_mult":1.5,"tp_r_target":3.25,"max_hold_bars":192},
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

    trials=[]
    passers=[]
    for profile in profile_grid():
        all_candidates=[]
        for symbol,frame in frames.items():
            all_candidates.extend(generate_profile_candidates(symbol=symbol,frame=frame,profile=profile))
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
        print("BREAKOUT_RETEST_TRIAL="+json.dumps(trial,ensure_ascii=False),flush=True)

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
        "schema":"proculus-breakout-retest-research-v1",
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
    print("BREAKOUT_RETEST_SUMMARY="+json.dumps({
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
