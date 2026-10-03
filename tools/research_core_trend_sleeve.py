#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MAJORS = ("BTC","ETH","SOL","BNB","XRP","DOGE","AAVE","LINK","LTC","BCH","ADA")

@dataclass
class SymbolResult:
    symbol: str
    return_pct: float
    max_drawdown_pct: float
    trades: int
    profit_factor: float
    avg_trade_pct: float
    curve: pd.Series
    trade_pnls: list[float]


def _symbol_from_path(path: Path) -> str:
    stem=path.stem
    if stem.endswith("_4h"):
        stem=stem[:-3]
    parts=stem.split("_")
    return parts[0] if parts else stem


def _load(path: Path, days: int) -> pd.DataFrame:
    df=pd.read_parquet(path)
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True)
    df=df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    for c in ("open","high","low","close","volume"):
        df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.dropna(subset=["open","high","low","close","volume"]).reset_index(drop=True)
    if not df.empty:
        cutoff=df["timestamp"].max()-pd.Timedelta(days=days+45)
        df=df[df["timestamp"]>=cutoff].reset_index(drop=True)
    return df


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span,adjust=False,min_periods=span).mean()


def _symbol_backtest(
    df: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    target_vol: float,
    one_way_cost: float,
    funding_per_4h: float,
    ema_fast: int = 50,
    ema_slow: int = 200,
) -> SymbolResult | None:
    if len(df)<220:
        return None

    ef=_ema(df["close"],ema_fast)
    es=_ema(df["close"],ema_slow)
    raw=np.where(ef>es,1.0,np.where(ef<es,-1.0,0.0))
    signal=pd.Series(raw,index=df.index).shift(1).fillna(0.0)

    bar_ret=df["close"].pct_change()
    ann_vol=bar_ret.rolling(180,min_periods=90).std(ddof=0)*math.sqrt(6*365)
    scale=(target_vol/ann_vol).clip(lower=0.0,upper=1.0).shift(1).fillna(0.0)
    position=signal*scale

    mask=(df["timestamp"]>=start)&(df["timestamp"]<end)
    idx=np.where(mask.to_numpy())[0]
    idx=idx[idx<len(df)-1]
    if not len(idx):
        return None

    equity=1.0
    peak=1.0
    max_dd=0.0
    previous_position=0.0
    previous_sign=0
    current_trade: dict[str,float] | None=None
    trades: list[float]=[]
    curve_times=[]
    curve_values=[]

    for i in idx:
        pos=float(position.iloc[i])
        sign=int(np.sign(pos))
        open_now=float(df["open"].iloc[i])
        open_next=float(df["open"].iloc[i+1])
        turnover=abs(pos-previous_position)
        cost=turnover*one_way_cost
        funding=abs(pos)*funding_per_4h
        gross=pos*(open_next/open_now-1.0)
        net=gross-cost-funding

        equity*=max(0.0,1.0+net)
        peak=max(peak,equity)
        max_dd=min(max_dd,equity/peak-1.0)
        curve_times.append(pd.Timestamp(df["timestamp"].iloc[i]))
        curve_values.append(equity)

        if sign!=previous_sign:
            if previous_sign!=0 and current_trade is not None:
                trades.append(float(current_trade["net"]))
                current_trade=None
            if sign!=0:
                current_trade={"net":0.0}
        if current_trade is not None:
            current_trade["net"]+=net

        previous_position=pos
        previous_sign=sign

    if previous_sign!=0 and current_trade is not None:
        trades.append(float(current_trade["net"]))

    wins=[x for x in trades if x>0]
    losses=[x for x in trades if x<0]
    pf=sum(wins)/abs(sum(losses)) if losses else (999.0 if wins else 0.0)
    curve=pd.Series(curve_values,index=pd.DatetimeIndex(curve_times),dtype=float)
    return SymbolResult(
        symbol="",
        return_pct=(equity-1.0)*100.0,
        max_drawdown_pct=abs(max_dd)*100.0,
        trades=len(trades),
        profit_factor=float(pf),
        avg_trade_pct=float(np.mean(trades)*100.0) if trades else 0.0,
        curve=curve,
        trade_pnls=trades,
    )


def _portfolio(
    frames: dict[str,pd.DataFrame],
    symbols: list[str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    target_vol: float,
    one_way_cost: float,
    funding_per_4h: float,
    ema_fast: int = 50,
    ema_slow: int = 200,
) -> dict[str,Any]:
    results={}
    for symbol in symbols:
        df=frames.get(symbol)
        if df is None:
            continue
        result=_symbol_backtest(
            df,start=start,end=end,target_vol=target_vol,
            one_way_cost=one_way_cost,funding_per_4h=funding_per_4h,
            ema_fast=ema_fast,ema_slow=ema_slow,
        )
        if result is not None:
            result.symbol=symbol
            results[symbol]=result
    if not results:
        raise RuntimeError("no eligible symbol results")

    equity=pd.concat(
        [result.curve.rename(symbol) for symbol,result in results.items()],
        axis=1,
    ).sort_index().ffill().fillna(1.0).mean(axis=1)
    drawdown=(equity/equity.cummax()-1.0).min()

    trade_pnls=[]
    for result in results.values():
        trade_pnls.extend(result.trade_pnls)
    wins=[x for x in trade_pnls if x>0]
    losses=[x for x in trade_pnls if x<0]
    pf=sum(wins)/abs(sum(losses)) if losses else (999.0 if wins else 0.0)

    return {
        "return_pct":round(float((equity.iloc[-1]-1.0)*100.0),5),
        "max_drawdown_pct":round(float(abs(drawdown)*100.0),5),
        "trades":sum(r.trades for r in results.values()),
        "profit_factor":round(float(pf),5),
        "avg_trade_pct":round(float(np.mean(trade_pnls)*100.0) if trade_pnls else 0.0,5),
        "symbols":len(results),
        "by_symbol":{
            symbol:{
                "return_pct":round(r.return_pct,5),
                "max_drawdown_pct":round(r.max_drawdown_pct,5),
                "trades":r.trades,
                "profit_factor":round(r.profit_factor,5),
                "avg_trade_pct":round(r.avg_trade_pct,5),
            }
            for symbol,r in results.items()
        },
        "_trade_pnls":trade_pnls,
    }


def _public(payload: dict[str,Any]) -> dict[str,Any]:
    return {k:v for k,v in payload.items() if not k.startswith("_")}


def _bootstrap_ci(trades: list[float], seed: int=42) -> dict[str,float]:
    if len(trades)<2:
        return {"mean_trade_pct":0.0,"ci95_low_pct":0.0,"ci95_high_pct":0.0}
    rng=np.random.default_rng(seed)
    values=np.asarray(trades,dtype=float)
    means=np.empty(10000,dtype=float)
    for i in range(len(means)):
        means[i]=rng.choice(values,size=len(values),replace=True).mean()
    low,high=np.quantile(means,[0.025,0.975])
    return {
        "mean_trade_pct":round(float(values.mean()*100.0),5),
        "ci95_low_pct":round(float(low*100.0),5),
        "ci95_high_pct":round(float(high*100.0),5),
    }


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",type=Path,required=True)
    ap.add_argument("--days",type=int,default=730)
    ap.add_argument("--holdout-days",type=int,default=180)
    ap.add_argument("--embargo-hours",type=int,default=48)
    ap.add_argument("--target-vol",type=float,default=0.20)
    ap.add_argument("--fee-bps",type=float,default=5.0)
    ap.add_argument("--funding-bp-per-8h",type=float,default=1.0)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()

    frames={}
    for path in sorted(args.data_dir.glob("*_4h.parquet")):
        symbol=_symbol_from_path(path)
        if symbol not in MAJORS:
            continue
        frame=_load(path,args.days)
        if len(frame)>=220:
            frames[symbol]=frame

    available=[s for s in MAJORS if s in frames]
    if len(available)<8:
        raise SystemExit(f"insufficient major universe coverage: {available}")

    data_start=max(frames[s]["timestamp"].min() for s in available)
    data_end=min(frames[s]["timestamp"].max() for s in available)
    holdout_start=data_end-pd.Timedelta(days=args.holdout_days)
    development_end=holdout_start-pd.Timedelta(hours=args.embargo_hours)
    validation_start=development_end-pd.Timedelta(days=120)
    train_end=validation_start-pd.Timedelta(hours=args.embargo_hours)

    funding_per_4h=(args.funding_bp_per_8h/10000.0)/2.0
    costs={
        "slippage_5bps":(args.fee_bps+5.0)/10000.0,
        "slippage_15bps":(args.fee_bps+15.0)/10000.0,
    }
    periods={
        "train":(data_start,train_end),
        "validation":(validation_start,development_end),
        "retrospective_holdout":(holdout_start,data_end),
    }

    period_results={}
    for period,(start,end) in periods.items():
        period_results[period]={}
        for label,cost in costs.items():
            payload=_portfolio(
                frames,available,start=start,end=end,target_vol=args.target_vol,
                one_way_cost=cost,funding_per_4h=funding_per_4h,
            )
            period_results[period][label]={
                **_public(payload),
                "bootstrap":_bootstrap_ci(payload["_trade_pnls"]),
            }

    rolling=[]
    cursor=data_start
    while cursor+pd.Timedelta(days=120)<=data_end:
        end=cursor+pd.Timedelta(days=120)
        payload=_portfolio(
            frames,available,start=cursor,end=end,target_vol=args.target_vol,
            one_way_cost=costs["slippage_15bps"],funding_per_4h=funding_per_4h,
        )
        rolling.append({
            "start":cursor.isoformat(),"end":end.isoformat(),**_public(payload)
        })
        cursor=end

    neighborhood=[]
    for fast,slow in ((40,180),(40,200),(50,180),(50,200),(50,220),(60,200),(60,220)):
        train=_portfolio(
            frames,available,start=data_start,end=train_end,target_vol=args.target_vol,
            one_way_cost=costs["slippage_15bps"],funding_per_4h=funding_per_4h,
            ema_fast=fast,ema_slow=slow,
        )
        validation=_portfolio(
            frames,available,start=validation_start,end=development_end,target_vol=args.target_vol,
            one_way_cost=costs["slippage_15bps"],funding_per_4h=funding_per_4h,
            ema_fast=fast,ema_slow=slow,
        )
        robust=(
            float(train["return_pct"])>0
            and float(validation["return_pct"])>0
            and float(train["profit_factor"])>=1.15
            and float(validation["profit_factor"])>=1.15
            and float(train["max_drawdown_pct"])<=15.0
            and float(validation["max_drawdown_pct"])<=15.0
        )
        neighborhood.append({
            "ema_fast":fast,
            "ema_slow":slow,
            "train":_public(train),
            "validation":_public(validation),
            "robust":robust,
        })

    leave_one_out=[]
    for omitted in available:
        subset=[s for s in available if s!=omitted]
        train=_portfolio(
            frames,subset,start=data_start,end=train_end,target_vol=args.target_vol,
            one_way_cost=costs["slippage_15bps"],funding_per_4h=funding_per_4h,
        )
        validation=_portfolio(
            frames,subset,start=validation_start,end=development_end,target_vol=args.target_vol,
            one_way_cost=costs["slippage_15bps"],funding_per_4h=funding_per_4h,
        )
        leave_one_out.append({
            "omitted":omitted,
            "train":_public(train),
            "validation":_public(validation),
        })

    t=period_results["train"]["slippage_15bps"]
    v=period_results["validation"]["slippage_15bps"]
    rolling_positive=sum(1 for x in rolling if float(x["return_pct"])>0.0)
    neighborhood_pass=sum(1 for x in neighborhood if x["robust"])
    loo_pass=sum(
        1 for x in leave_one_out
        if float(x["train"]["profit_factor"])>=1.15
        and float(x["validation"]["profit_factor"])>=1.15
        and float(x["train"]["return_pct"])>0
        and float(x["validation"]["return_pct"])>0
    )
    research_checks={
        "train_pf_gte_1_20":float(t["profit_factor"])>=1.20,
        "train_return_positive":float(t["return_pct"])>0,
        "train_dd_lte_15pct":float(t["max_drawdown_pct"])<=15.0,
        "validation_pf_gte_1_20":float(v["profit_factor"])>=1.20,
        "validation_return_positive":float(v["return_pct"])>0,
        "validation_dd_lte_15pct":float(v["max_drawdown_pct"])<=15.0,
        "validation_trades_gte_40":int(v["trades"])>=40,
        "rolling_positive_share_gte_60pct":rolling_positive/max(1,len(rolling))>=0.60,
        "ema_neighborhood_robust_gte_6_of_7":neighborhood_pass>=6,
        "leave_one_out_all_robust":loo_pass==len(leave_one_out),
    }
    shadow_candidate=all(research_checks.values())
    live_release=False
    release_blockers=[]
    if not shadow_candidate:
        release_blockers.append("research_acceptance_failed")
    # Holdout has already been inspected during this research cycle, and
    # bootstrap confidence intervals cross zero. Require forward evidence.
    release_blockers.extend([
        "retrospective_holdout_already_inspected",
        "forward_shadow_duration_missing",
        "bootstrap_ci_lower_bound_not_required_positive_yet",
        "live_tca_missing",
    ])

    report={
        "schema":"proculus-core-trend-sleeve-research-v1",
        "production_changed":False,
        "strategy":{
            "name":"core_trend_4h_ema50_200_voltarget",
            "timeframe":"4h",
            "universe_requested":list(MAJORS),
            "universe_available":available,
            "direction":"long_short",
            "ema_fast":50,
            "ema_slow":200,
            "target_annualized_vol":args.target_vol,
            "per_symbol_exposure_cap":1.0,
            "signal_execution":"closed 4h candle -> next 4h open",
            "funding_assumption":"1 bp per 8h while exposed",
            "fee_bps_per_side":args.fee_bps,
        },
        "periods":{
            "data_start":data_start.isoformat(),
            "train_end_exclusive":train_end.isoformat(),
            "validation_start":validation_start.isoformat(),
            "development_end_exclusive":development_end.isoformat(),
            "retrospective_holdout_start":holdout_start.isoformat(),
            "data_end":data_end.isoformat(),
        },
        "results":period_results,
        "rolling_120d_high_cost":rolling,
        "ema_neighborhood_high_cost":neighborhood,
        "leave_one_out_high_cost":leave_one_out,
        "research_checks":research_checks,
        "shadow_candidate":shadow_candidate,
        "live_release":live_release,
        "release_blockers":release_blockers,
        "limitations":[
            "current-live universe has survivorship bias",
            "historical funding is approximated conservatively rather than replayed exactly",
            "holdout is retrospective because this research cycle already inspected it",
            "trade-level bootstrap confidence intervals are diagnostic and currently not a live-release criterion",
        ],
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print("CORE_TREND_SLEEVE_SUMMARY="+json.dumps({
        "shadow_candidate":shadow_candidate,
        "live_release":live_release,
        "research_checks":research_checks,
        "train_high_cost":{k:v for k,v in t.items() if k!="by_symbol"},
        "validation_high_cost":{k:v for k,v in v.items() if k!="by_symbol"},
        "holdout_high_cost":{k:v for k,v in period_results["retrospective_holdout"]["slippage_15bps"].items() if k!="by_symbol"},
        "rolling_positive":rolling_positive,
        "rolling_total":len(rolling),
        "ema_neighborhood_pass":neighborhood_pass,
        "ema_neighborhood_total":len(neighborhood),
        "leave_one_out_pass":loo_pass,
        "leave_one_out_total":len(leave_one_out),
        "release_blockers":release_blockers,
    },ensure_ascii=False),flush=True)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
