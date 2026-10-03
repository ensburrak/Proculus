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


@dataclass(frozen=True)
class Signal:
    symbol:str
    decision_time:pd.Timestamp
    entry_time:pd.Timestamp
    side:str
    entry_idx:int
    atr:float
    strength:float


@dataclass
class Trade:
    symbol:str
    side:str
    entry_time:pd.Timestamp
    exit_time:pd.Timestamp
    net_pnl:float
    gross_pnl:float
    risk_usd:float
    fees:float
    funding:float


def f(v:Any,default:float=0.0)->float:
    try:
        x=float(v)
    except (TypeError,ValueError):
        return default
    return x if math.isfinite(x) else default


def ema(s:pd.Series,span:int)->pd.Series:
    return s.ewm(span=span,adjust=False,min_periods=span).mean()


def dmi_adx(df:pd.DataFrame,period:int=14)->pd.DataFrame:
    high,low,close=df["high"],df["low"],df["close"]
    pc=close.shift(1)
    tr=pd.concat([(high-low).abs(),(high-pc).abs(),(low-pc).abs()],axis=1).max(axis=1)
    up=high.diff(); down=-low.diff()
    plus_dm=pd.Series(np.where((up>down)&(up>0),up,0.0),index=df.index)
    minus_dm=pd.Series(np.where((down>up)&(down>0),down,0.0),index=df.index)
    atr=tr.ewm(alpha=1.0/period,adjust=False,min_periods=period).mean()
    plus=100.0*plus_dm.ewm(alpha=1.0/period,adjust=False,min_periods=period).mean()/atr.replace(0.0,np.nan)
    minus=100.0*minus_dm.ewm(alpha=1.0/period,adjust=False,min_periods=period).mean()/atr.replace(0.0,np.nan)
    dx=100.0*(plus-minus).abs()/(plus+minus).replace(0.0,np.nan)
    adx=dx.ewm(alpha=1.0/period,adjust=False,min_periods=period).mean()
    return pd.DataFrame({"atr":atr,"adx":adx})


def symbol_from_path(path:Path)->str:
    stem=path.stem
    if stem.endswith("_4h"):
        stem=stem[:-3]
    parts=stem.split("_")
    if len(parts)>=3 and parts[-1]=="USDT":
        return f"{parts[0]}/USDT:USDT"
    return stem.replace("_","/")


def load_frame(path:Path,days:int)->pd.DataFrame:
    df=pd.read_parquet(path)
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True)
    df=df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    for c in ("open","high","low","close","volume"):
        df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.dropna(subset=["open","high","low","close","volume"]).reset_index(drop=True)
    if not df.empty:
        cutoff=df["timestamp"].max()-pd.Timedelta(days=days+80)
        df=df[df["timestamp"]>=cutoff].reset_index(drop=True)
    df["ema50"]=ema(df["close"],50)
    df["ema200"]=ema(df["close"],200)
    dm=dmi_adx(df,14)
    df["atr"]=dm["atr"]; df["adx"]=dm["adx"]
    df["atr_ratio"]=df["atr"]/df["close"].replace(0.0,np.nan)
    return df


def profiles()->list[dict[str,Any]]:
    return [
        {"name":"d20x10_a18","entry_lb":20,"exit_lb":10,"adx_min":18,"stop_atr":2.0,"ema_filter":True,"sides":"both","max_hold":90},
        {"name":"d20x10_a22","entry_lb":20,"exit_lb":10,"adx_min":22,"stop_atr":2.0,"ema_filter":True,"sides":"both","max_hold":90},
        {"name":"d55x20_a18","entry_lb":55,"exit_lb":20,"adx_min":18,"stop_atr":2.0,"ema_filter":True,"sides":"both","max_hold":120},
        {"name":"d55x20_a22","entry_lb":55,"exit_lb":20,"adx_min":22,"stop_atr":2.0,"ema_filter":True,"sides":"both","max_hold":120},
        {"name":"d55x20_a22_short","entry_lb":55,"exit_lb":20,"adx_min":22,"stop_atr":2.0,"ema_filter":True,"sides":"short","max_hold":120},
        {"name":"d55x20_a25_short","entry_lb":55,"exit_lb":20,"adx_min":25,"stop_atr":2.25,"ema_filter":True,"sides":"short","max_hold":120},
        {"name":"d40x15_a22","entry_lb":40,"exit_lb":15,"adx_min":22,"stop_atr":2.0,"ema_filter":True,"sides":"both","max_hold":105},
        {"name":"d40x15_a25","entry_lb":40,"exit_lb":15,"adx_min":25,"stop_atr":2.25,"ema_filter":True,"sides":"both","max_hold":105},
    ]


def prepare_profile(df:pd.DataFrame,p:dict[str,Any])->pd.DataFrame:
    out=df.copy()
    elb=int(p["entry_lb"]); xlb=int(p["exit_lb"])
    out["entry_high"]=out["high"].shift(1).rolling(elb,min_periods=elb).max()
    out["entry_low"]=out["low"].shift(1).rolling(elb,min_periods=elb).min()
    out["exit_high"]=out["high"].shift(1).rolling(xlb,min_periods=xlb).max()
    out["exit_low"]=out["low"].shift(1).rolling(xlb,min_periods=xlb).min()
    return out


def build_signals(frames:dict[str,pd.DataFrame],p:dict[str,Any])->tuple[list[Signal],dict[str,pd.DataFrame]]:
    prepared={}
    rows=[]
    sides=str(p["sides"])
    for symbol,raw in frames.items():
        df=prepare_profile(raw,p)
        prepared[symbol]=df
        if len(df)<260:
            continue
        work=df.iloc[220:-1].copy()
        close=work["close"].astype(float)
        atr=work["atr"].astype(float)
        atrp=work["atr_ratio"].astype(float)
        adx=work["adx"].astype(float)
        e50=work["ema50"].astype(float); e200=work["ema200"].astype(float)
        highch=work["entry_high"].astype(float); lowch=work["entry_low"].astype(float)
        base=atr.gt(0)&atrp.between(0.002,0.08)&adx.ge(float(p["adx_min"]))
        if bool(p.get("ema_filter",True)):
            long_ok=base&e50.gt(e200)&close.gt(e50)&close.gt(highch)
            short_ok=base&e50.lt(e200)&close.lt(e50)&close.lt(lowch)
        else:
            long_ok=base&close.gt(highch)
            short_ok=base&close.lt(lowch)
        strength=(close-highch)/atr.replace(0.0,np.nan)
        short_strength=(lowch-close)/atr.replace(0.0,np.nan)
        tmp=pd.DataFrame({
            "timestamp":work["timestamp"],"symbol":symbol,
            "entry_idx":work.index.to_numpy()+1,"atr":atr,
            "long_ok":long_ok,"short_ok":short_ok,
            "long_strength":strength,"short_strength":short_strength,
        }).replace([np.inf,-np.inf],np.nan)
        rows.append(tmp)
    if not rows:
        return [],prepared
    panel=pd.concat(rows,ignore_index=True)
    chosen=[]
    for _,g in panel.groupby("timestamp",sort=True):
        options=[]
        if sides in {"both","long"}:
            x=g[g["long_ok"]].dropna(subset=["long_strength"])
            if not x.empty:
                r=x.nlargest(1,"long_strength").copy(); r["side"]="long"; r["strength"]=r["long_strength"]; options.append(r)
        if sides in {"both","short"}:
            x=g[g["short_ok"]].dropna(subset=["short_strength"])
            if not x.empty:
                r=x.nlargest(1,"short_strength").copy(); r["side"]="short"; r["strength"]=r["short_strength"]; options.append(r)
        if options:
            merged=pd.concat(options,ignore_index=True).sort_values("strength",ascending=False).head(2)
            chosen.append(merged)
    if not chosen:
        return [],prepared
    sel=pd.concat(chosen,ignore_index=True)
    signals=[]
    for row in sel.itertuples(index=False):
        symbol=str(row.symbol); idx=int(row.entry_idx); df=prepared[symbol]
        if idx>=len(df):
            continue
        signals.append(Signal(
            symbol=symbol,decision_time=pd.Timestamp(row.timestamp),
            entry_time=pd.Timestamp(df.iloc[idx]["timestamp"]),
            side=str(row.side),entry_idx=idx,atr=float(row.atr),strength=float(row.strength),
        ))
    return sorted(signals,key=lambda s:(s.entry_time,-s.strength,s.symbol)),prepared


def adverse_entry(x:float,side:str,slip:float)->float:
    return x*(1+slip) if side=="long" else x*(1-slip)


def adverse_exit(x:float,side:str,slip:float)->float:
    return x*(1-slip) if side=="long" else x*(1+slip)


def path(df:pd.DataFrame,s:Signal,p:dict[str,Any],slip:float)->tuple[float,float,pd.Timestamp,int,str,float]:
    raw=f(df.iloc[s.entry_idx]["open"]); entry=adverse_entry(raw,s.side,slip)
    sd=max(s.atr*float(p["stop_atr"]),entry*0.003)
    stop=entry-sd if s.side=="long" else entry+sd
    end=min(len(df)-1,s.entry_idx+int(p["max_hold"]))
    for i in range(s.entry_idx,end+1):
        row=df.iloc[i]; hi=f(row["high"]); lo=f(row["low"])
        trail=f(row["exit_low"]) if s.side=="long" else f(row["exit_high"])
        if s.side=="long":
            effective=max(stop,trail) if trail>0 else stop
            if lo<=effective:
                return entry,adverse_exit(effective,s.side,slip),pd.Timestamp(row["timestamp"]),i-s.entry_idx+1,"channel_stop",sd
        else:
            effective=min(stop,trail) if trail>0 else stop
            if hi>=effective:
                return entry,adverse_exit(effective,s.side,slip),pd.Timestamp(row["timestamp"]),i-s.entry_idx+1,"channel_stop",sd
    row=df.iloc[end]
    return entry,adverse_exit(f(row["close"]),s.side,slip),pd.Timestamp(row["timestamp"]),end-s.entry_idx+1,"time_stop",sd


def simulate(signals:list[Signal],frames:dict[str,pd.DataFrame],p:dict[str,Any],initial:float,fee_bps:float,slip_bps:float)->dict[str,Any]:
    fee=fee_bps/10000.0; slip=slip_bps/10000.0
    cash=float(initial); open_trades:list[tuple[Trade,float,float]]=[]; closed:list[Trade]=[]; curve=[cash]
    last_entry={}

    def realize(ts:pd.Timestamp)->None:
        nonlocal cash,open_trades
        rem=[]
        for t,gross_settle,exit_cost in open_trades:
            if t.exit_time<=ts:
                cash+=gross_settle-exit_cost-t.funding
                closed.append(t); curve.append(cash)
            else:
                rem.append((t,gross_settle,exit_cost))
        open_trades=rem

    for s in signals:
        realize(s.entry_time)
        if len(open_trades)>=2 or any(x[0].symbol==s.symbol for x in open_trades):
            continue
        prior=last_entry.get(s.symbol)
        if prior is not None and (s.entry_time-prior).total_seconds()<12*3600:
            continue
        df=frames[s.symbol]
        entry,exit_price,exit_time,bars,reason,sd=path(df,s,p,slip)
        if entry<=0:
            continue
        equity=max(cash,0.0); risk_budget=equity*0.0025
        stop_pct=sd/entry
        notional=min(risk_budget/max(stop_pct,1e-9),equity*0.20)
        if notional<=0:
            continue
        qty=notional/entry; sign=1.0 if s.side=="long" else -1.0
        gross=(exit_price-entry)*qty*sign
        entry_fee=notional*fee; exit_fee=abs(exit_price*qty)*fee
        funding=notional*0.0001*(bars//2)
        risk_usd=sd*qty
        net=gross-entry_fee-exit_fee-funding
        cash-=entry_fee
        t=Trade(s.symbol,s.side,s.entry_time,exit_time,net,gross,risk_usd,entry_fee+exit_fee,funding)
        open_trades.append((t,gross,exit_fee)); last_entry[s.symbol]=s.entry_time
    if signals:
        realize(pd.Timestamp.max.tz_localize("UTC"))

    pnl=[x.net_pnl for x in closed]; wins=[x for x in pnl if x>0]; losses=[x for x in pnl if x<0]
    peak=-1e99; dd=0.0
    for x in curve:
        peak=max(peak,x)
        if peak>0: dd=min(dd,(x-peak)/peak)
    rs=[t.net_pnl/t.risk_usd for t in closed if t.risk_usd>0]
    return {
        "return_pct":round(100*(cash/initial-1),4),"max_drawdown_pct":round(abs(dd)*100,4),
        "trades":len(closed),"win_rate_pct":round(100*len(wins)/max(1,len(closed)),4),
        "profit_factor":round(sum(wins)/abs(sum(losses)),5) if losses else (999.0 if wins else 0.0),
        "expectancy_r":round(sum(rs)/max(1,len(rs)),5),
        "fees_paid":round(sum(t.fees for t in closed),4),"funding_paid":round(sum(t.funding for t in closed),4),
    }


def compact(x:dict[str,Any])->dict[str,Any]:
    return {k:x[k] for k in ("return_pct","max_drawdown_pct","trades","win_rate_pct","profit_factor","expectancy_r")}


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",type=Path,required=True)
    ap.add_argument("--days",type=int,default=730)
    ap.add_argument("--holdout-days",type=int,default=180)
    ap.add_argument("--embargo-hours",type=int,default=48)
    ap.add_argument("--initial-balance",type=float,default=10000.0)
    ap.add_argument("--fee-bps",type=float,default=5.0)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()

    frames={}
    for pth in sorted(args.data_dir.glob("*_4h.parquet")):
        symbol=symbol_from_path(pth)
        try:
            df=load_frame(pth,args.days)
        except Exception as exc:
            print(f"[SKIP] {symbol}: {type(exc).__name__}: {exc}",flush=True); continue
        if len(df)>=260: frames[symbol]=df
    if not frames: raise SystemExit("no 4h frames")

    data_end=max(pd.Timestamp(x["timestamp"].max()) for x in frames.values())
    holdout=data_end-pd.Timedelta(days=args.holdout_days)
    dev_end=holdout-pd.Timedelta(hours=args.embargo_hours)
    val_start=dev_end-pd.Timedelta(days=120)
    train_end=val_start-pd.Timedelta(hours=args.embargo_hours)

    trials=[]; passers=[]
    for p in profiles():
        sigs,prepared=build_signals(frames,p)
        train=[x for x in sigs if x.entry_time<train_end]
        val=[x for x in sigs if val_start<=x.entry_time<dev_end]
        oos=[x for x in sigs if x.entry_time>=holdout]
        t5=simulate(train,prepared,p,args.initial_balance,args.fee_bps,5.0)
        t15=simulate(train,prepared,p,args.initial_balance,args.fee_bps,15.0)
        checks={
            "trades_gte_40":t5["trades"]>=40,
            "pf5_gte_1_12":t5["profit_factor"]>=1.12,
            "exp5_gte_0_04":t5["expectancy_r"]>=0.04,
            "pf15_gte_1_05":t15["profit_factor"]>=1.05,
            "exp15_gte_0":t15["expectancy_r"]>=0.0,
            "dd15_lte_15":t15["max_drawdown_pct"]<=15.0,
        }
        row={"profile":p,"counts":{"all":len(sigs),"train":len(train),"validation":len(val),"oos":len(oos)},
             "train_5bps":compact(t5),"train_15bps":compact(t15),"train_checks":checks,
             "train_pass":all(checks.values()),"validation":None,"validation_pass":False,"oos_locked_result":{}}
        if row["train_pass"]:
            v5=simulate(val,prepared,p,args.initial_balance,args.fee_bps,5.0)
            v15=simulate(val,prepared,p,args.initial_balance,args.fee_bps,15.0)
            vc={"trades_gte_20":v5["trades"]>=20,"pf5_gte_1_15":v5["profit_factor"]>=1.15,
                "exp5_gte_0_05":v5["expectancy_r"]>=0.05,"pf15_gte_1_05":v15["profit_factor"]>=1.05,
                "exp15_gte_0":v15["expectancy_r"]>=0.0,"dd15_lte_15":v15["max_drawdown_pct"]<=15.0}
            row["validation"]={"slippage_5bps":compact(v5),"slippage_15bps":compact(v15),"checks":vc}
            row["validation_pass"]=all(vc.values())
            if row["validation_pass"]: passers.append((row,oos,prepared,p))
        trials.append(row)
        print("DONCHIAN_4H_TRIAL="+json.dumps(row,ensure_ascii=False),flush=True)

    selected=None
    if passers:
        passers.sort(key=lambda x:(min(x[0]["validation"]["slippage_5bps"]["profit_factor"],x[0]["validation"]["slippage_15bps"]["profit_factor"]),
                                   min(x[0]["validation"]["slippage_5bps"]["expectancy_r"],x[0]["validation"]["slippage_15bps"]["expectancy_r"])),reverse=True)
        row,oos,prepared,p=passers[0]; selected=dict(p)
        for bps in (5.0,15.0):
            row["oos_locked_result"][f"slippage_{int(bps)}bps"]=compact(simulate(oos,prepared,p,args.initial_balance,args.fee_bps,bps))

    report={"schema":"proculus-4h-donchian-research-v1","production_changed":False,"selection":"train -> validation -> retrospective holdout",
            "data_end":data_end.isoformat(),"train_end_exclusive":train_end.isoformat(),"validation_start":val_start.isoformat(),
            "development_end_exclusive":dev_end.isoformat(),"holdout_start":holdout.isoformat(),"profiles":trials,"selected":selected}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print("DONCHIAN_4H_SUMMARY="+json.dumps({"selected":selected,"profiles":trials},ensure_ascii=False),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
