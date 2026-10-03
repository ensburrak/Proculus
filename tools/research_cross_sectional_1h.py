#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))


@dataclass(frozen=True)
class Signal:
    symbol:str
    decision_time:pd.Timestamp
    entry_time:pd.Timestamp
    side:str
    score:float
    atr:float
    entry_idx:int
    stop_atr_mult:float
    tp_r_target:float
    max_hold_bars:int


@dataclass
class Trade:
    symbol:str
    side:str
    entry_time:pd.Timestamp
    exit_time:pd.Timestamp
    entry_price:float
    exit_price:float
    notional:float
    risk_usd:float
    net_pnl:float
    gross_pnl:float
    fees:float
    funding:float
    exit_reason:str


def f(v:Any,default:float=0.0)->float:
    try:
        x=float(v)
    except (TypeError,ValueError):
        return default
    return x if math.isfinite(x) else default


def ema(s:pd.Series,span:int)->pd.Series:
    return s.ewm(span=span,adjust=False,min_periods=span).mean()


def rsi(close:pd.Series,period:int=14)->pd.Series:
    d=close.diff()
    gain=d.clip(lower=0.0)
    loss=-d.clip(upper=0.0)
    ag=gain.ewm(alpha=1.0/period,adjust=False,min_periods=period).mean()
    al=loss.ewm(alpha=1.0/period,adjust=False,min_periods=period).mean()
    rs=ag/al.replace(0.0,np.nan)
    return (100.0-(100.0/(1.0+rs))).fillna(50.0)


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
    return pd.DataFrame({"atr":atr,"adx":adx,"plus_di":plus,"minus_di":minus})


def symbol_from_path(path:Path)->str:
    stem=path.stem
    if stem.endswith("_1h"):
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
        cutoff=df["timestamp"].max()-pd.Timedelta(days=days+45)
        df=df[df["timestamp"]>=cutoff].reset_index(drop=True)
    return df


def merge_closed_4h(base:pd.DataFrame,h4:pd.DataFrame,col:str)->pd.Series:
    left=pd.DataFrame({"decision_close":base["timestamp"]+pd.Timedelta(hours=1),"_idx":base.index})
    right=h4[["timestamp",col]].copy()
    right["closed_at"]=right["timestamp"]+pd.Timedelta(hours=4)
    merged=pd.merge_asof(
        left.sort_values("decision_close"),
        right.sort_values("closed_at")[["closed_at",col]],
        left_on="decision_close",
        right_on="closed_at",
        direction="backward",
    )
    return merged.sort_values("_idx")[col].reset_index(drop=True)


def prepare_symbol(symbol:str,data_dir:Path,days:int)->pd.DataFrame:
    key=symbol.replace("/","_").replace(":","_")
    h1=load_frame(data_dir/f"{key}_1h.parquet",days)
    if len(h1)<500:
        return h1
    h1["ema20"]=ema(h1["close"],20)
    h1["ema50"]=ema(h1["close"],50)
    h1["ema200"]=ema(h1["close"],200)
    h1["rsi"]=rsi(h1["close"],14)
    dm=dmi_adx(h1,14)
    for c in dm.columns:
        h1[c]=dm[c]
    h1["atr_ratio"]=h1["atr"]/h1["close"].replace(0.0,np.nan)
    h1["ret_12h"]=h1["close"].pct_change(12)
    h1["ret_24h"]=h1["close"].pct_change(24)
    h1["ret_72h"]=h1["close"].pct_change(72)
    h1["vol_24h"]=h1["close"].pct_change().rolling(24,min_periods=24).std(ddof=0)
    vol_mean=h1["volume"].rolling(24,min_periods=24).mean().shift(1)
    vol_std=h1["volume"].rolling(24,min_periods=24).std(ddof=0).shift(1)
    h1["vol_z"]=(h1["volume"]-vol_mean)/vol_std.replace(0.0,np.nan)

    h4=load_frame(data_dir/f"{key}_4h.parquet",days+60)
    if len(h4)>=220:
        h4["ema20_4h"]=ema(h4["close"],20)
        h4["ema50_4h"]=ema(h4["close"],50)
        h4["ema200_4h"]=ema(h4["close"],200)
        h4["close_4h"]=h4["close"]
        h4["rsi_4h"]=rsi(h4["close"],14)
        h4d=dmi_adx(h4,14)
        h4["adx_4h"]=h4d["adx"]
        for col in ("close_4h","ema20_4h","ema50_4h","ema200_4h","rsi_4h","adx_4h"):
            h1[col]=merge_closed_4h(h1,h4,col)
    return h1


def profile_grid()->list[dict[str,Any]]:
    return [
        {"name":"mom24_k1_a22","momentum":"ret_24h","top_k":1,"adx_min":22,"h4_adx_min":18,"rebalance_hours":4,"stop_atr_mult":1.5,"tp_r_target":2.5,"max_hold_bars":24,"sides":"both"},
        {"name":"mom24_k1_a25","momentum":"ret_24h","top_k":1,"adx_min":25,"h4_adx_min":20,"rebalance_hours":4,"stop_atr_mult":1.5,"tp_r_target":3.0,"max_hold_bars":24,"sides":"both"},
        {"name":"mom72_k1_a22","momentum":"ret_72h","top_k":1,"adx_min":22,"h4_adx_min":18,"rebalance_hours":6,"stop_atr_mult":1.75,"tp_r_target":3.0,"max_hold_bars":36,"sides":"both"},
        {"name":"mom72_k1_a25","momentum":"ret_72h","top_k":1,"adx_min":25,"h4_adx_min":20,"rebalance_hours":6,"stop_atr_mult":1.75,"tp_r_target":3.25,"max_hold_bars":36,"sides":"both"},
        {"name":"blend_k1_a25","momentum":"blend","top_k":1,"adx_min":25,"h4_adx_min":20,"rebalance_hours":6,"stop_atr_mult":1.75,"tp_r_target":3.0,"max_hold_bars":36,"sides":"both"},
        {"name":"blend_k1_a25_short","momentum":"blend","top_k":1,"adx_min":25,"h4_adx_min":20,"rebalance_hours":6,"stop_atr_mult":1.75,"tp_r_target":3.0,"max_hold_bars":36,"sides":"short"},
        {"name":"mom72_k2_a25","momentum":"ret_72h","top_k":2,"adx_min":25,"h4_adx_min":20,"rebalance_hours":8,"stop_atr_mult":2.0,"tp_r_target":3.0,"max_hold_bars":48,"sides":"both"},
        {"name":"blend_k2_a28","momentum":"blend","top_k":2,"adx_min":28,"h4_adx_min":22,"rebalance_hours":8,"stop_atr_mult":2.0,"tp_r_target":3.25,"max_hold_bars":48,"sides":"both"},
    ]


def eligible(row:pd.Series,side:str,p:dict[str,Any])->bool:
    close=f(row.get("close")); e20=f(row.get("ema20")); e50=f(row.get("ema50")); e200=f(row.get("ema200"))
    adx=f(row.get("adx")); h4adx=f(row.get("adx_4h")); atrp=f(row.get("atr_ratio"))
    h4c=f(row.get("close_4h")); h4e20=f(row.get("ema20_4h")); h4e50=f(row.get("ema50_4h")); h4e200=f(row.get("ema200_4h"))
    r=f(row.get("rsi")); h4r=f(row.get("rsi_4h")); vz=f(row.get("vol_z"))
    if min(close,e20,e50,e200,h4c,h4e20,h4e50,h4e200)<=0:
        return False
    if adx<float(p["adx_min"]) or h4adx<float(p["h4_adx_min"]):
        return False
    if atrp<0.0015 or atrp>0.05 or abs(vz)>=3.5:
        return False
    if side=="long":
        return bool(e20>e50>e200 and close>e20 and h4e20>h4e50>h4e200 and h4c>h4e20 and 48<=r<=72 and 48<=h4r<=75)
    return bool(e20<e50<e200 and close<e20 and h4e20<h4e50<h4e200 and h4c<h4e20 and 28<=r<=52 and 25<=h4r<=52)


def score_row(row:pd.Series,p:dict[str,Any])->float:
    mode=str(p["momentum"])
    r12=f(row.get("ret_12h")); r24=f(row.get("ret_24h")); r72=f(row.get("ret_72h"))
    vol=max(f(row.get("vol_24h")),1e-6)
    if mode=="ret_24h":
        raw=r24
    elif mode=="ret_72h":
        raw=r72
    else:
        raw=0.55*r24+0.30*r72+0.15*r12
    return raw/vol


def build_signals(frames:dict[str,pd.DataFrame],p:dict[str,Any])->list[Signal]:
    rebalance=int(p["rebalance_hours"])
    top_k=int(p["top_k"])
    sides=str(p["sides"])
    rows:list[pd.DataFrame]=[]

    for symbol,df in frames.items():
        if len(df)<252:
            continue
        work=df.iloc[250:-1].copy()
        if work.empty:
            continue
        work=work[work["timestamp"].dt.hour.mod(rebalance).eq(0)]
        if work.empty:
            continue

        close=work["close"].astype(float)
        e20=work["ema20"].astype(float)
        e50=work["ema50"].astype(float)
        e200=work["ema200"].astype(float)
        adx=work["adx"].astype(float)
        h4adx=work["adx_4h"].astype(float)
        atrp=work["atr_ratio"].astype(float)
        h4c=work["close_4h"].astype(float)
        h4e20=work["ema20_4h"].astype(float)
        h4e50=work["ema50_4h"].astype(float)
        h4e200=work["ema200_4h"].astype(float)
        r=work["rsi"].astype(float)
        h4r=work["rsi_4h"].astype(float)
        vz=work["vol_z"].astype(float)

        base=(
            close.gt(0)&e20.gt(0)&e50.gt(0)&e200.gt(0)&
            h4c.gt(0)&h4e20.gt(0)&h4e50.gt(0)&h4e200.gt(0)&
            adx.ge(float(p["adx_min"]))&
            h4adx.ge(float(p["h4_adx_min"]))&
            atrp.ge(0.0015)&atrp.le(0.05)&
            vz.abs().lt(3.5)
        )
        long_ok=(
            base&
            e20.gt(e50)&e50.gt(e200)&close.gt(e20)&
            h4e20.gt(h4e50)&h4e50.gt(h4e200)&h4c.gt(h4e20)&
            r.between(48,72)&h4r.between(48,75)
        )
        short_ok=(
            base&
            e20.lt(e50)&e50.lt(e200)&close.lt(e20)&
            h4e20.lt(h4e50)&h4e50.lt(h4e200)&h4c.lt(h4e20)&
            r.between(28,52)&h4r.between(25,52)
        )

        vol=work["vol_24h"].astype(float).clip(lower=1e-6)
        mode=str(p["momentum"])
        if mode=="ret_24h":
            raw=work["ret_24h"].astype(float)
        elif mode=="ret_72h":
            raw=work["ret_72h"].astype(float)
        else:
            raw=(
                0.55*work["ret_24h"].astype(float)+
                0.30*work["ret_72h"].astype(float)+
                0.15*work["ret_12h"].astype(float)
            )
        score=raw/vol

        tmp=pd.DataFrame({
            "timestamp":work["timestamp"],
            "symbol":symbol,
            "entry_idx":work.index.to_numpy()+1,
            "score":score,
            "atr":work["atr"].astype(float),
            "long_ok":long_ok,
            "short_ok":short_ok,
        })
        tmp=tmp.replace([np.inf,-np.inf],np.nan).dropna(subset=["score","atr"])
        if not tmp.empty:
            rows.append(tmp)

    if not rows:
        return []

    panel=pd.concat(rows,ignore_index=True)
    selected:list[pd.DataFrame]=[]
    for _,group in panel.groupby("timestamp",sort=True):
        if sides in {"both","long"}:
            longs=group[group["long_ok"]&group["score"].gt(0)]
            if not longs.empty:
                selected.append(longs.nlargest(top_k,"score"))
        if sides in {"both","short"}:
            shorts=group[group["short_ok"]&group["score"].lt(0)]
            if not shorts.empty:
                selected.append(shorts.nsmallest(top_k,"score"))

    if not selected:
        return []

    chosen=pd.concat(selected,ignore_index=True)
    out:list[Signal]=[]
    for row in chosen.itertuples(index=False):
        symbol=str(row.symbol)
        idx=int(row.entry_idx)
        df=frames[symbol]
        if idx>=len(df):
            continue
        side="long" if float(row.score)>0 else "short"
        out.append(Signal(
            symbol=symbol,
            decision_time=pd.Timestamp(row.timestamp),
            entry_time=pd.Timestamp(df.iloc[idx]["timestamp"]),
            side=side,
            score=float(row.score),
            atr=float(row.atr),
            entry_idx=idx,
            stop_atr_mult=float(p["stop_atr_mult"]),
            tp_r_target=float(p["tp_r_target"]),
            max_hold_bars=int(p["max_hold_bars"]),
        ))
    return sorted(out,key=lambda x:(x.entry_time,-abs(x.score),x.symbol))


def adverse_entry(raw:float,side:str,slip:float)->float:
    return raw*(1+slip) if side=="long" else raw*(1-slip)


def adverse_exit(raw:float,side:str,slip:float)->float:
    return raw*(1-slip) if side=="long" else raw*(1+slip)


def trade_path(frame:pd.DataFrame,s:Signal,fee:float,slip:float)->tuple[float,float,pd.Timestamp,int,str,float]:
    raw=f(frame.iloc[s.entry_idx]["open"])
    entry=adverse_entry(raw,s.side,slip)
    sd=max(s.atr*s.stop_atr_mult,entry*0.002)
    stop=entry-sd if s.side=="long" else entry+sd
    target=entry+sd*s.tp_r_target if s.side=="long" else entry-sd*s.tp_r_target
    end=min(len(frame)-1,s.entry_idx+s.max_hold_bars)
    for idx in range(s.entry_idx,end+1):
        row=frame.iloc[idx]; hi=f(row["high"]); lo=f(row["low"])
        stop_hit=lo<=stop if s.side=="long" else hi>=stop
        target_hit=hi>=target if s.side=="long" else lo<=target
        if stop_hit:
            return entry,adverse_exit(stop,s.side,slip),pd.Timestamp(row["timestamp"]),idx-s.entry_idx+1,"stop_loss",sd
        if target_hit:
            return entry,adverse_exit(target,s.side,slip),pd.Timestamp(row["timestamp"]),idx-s.entry_idx+1,"take_profit",sd
    row=frame.iloc[end]
    return entry,adverse_exit(f(row["close"]),s.side,slip),pd.Timestamp(row["timestamp"]),end-s.entry_idx+1,"time_stop",sd


def simulate(signals:list[Signal],frames:dict[str,pd.DataFrame],initial:float,fee_bps:float,slip_bps:float)->dict[str,Any]:
    fee=fee_bps/10000.0; slip=slip_bps/10000.0
    cash=float(initial); open_trades:list[Trade]=[]; closed:list[Trade]=[]; curve=[cash]
    last_symbol_entry:dict[str,pd.Timestamp]={}

    def realize(ts:pd.Timestamp)->None:
        nonlocal cash,open_trades
        remain=[]
        for t in open_trades:
            if t.exit_time<=ts:
                cash+=t.gross_pnl-t.fees-t.funding
                closed.append(t); curve.append(cash)
            else:
                remain.append(t)
        open_trades=remain

    for s in signals:
        realize(s.entry_time)
        if len(open_trades)>=2 or any(t.symbol==s.symbol for t in open_trades):
            continue
        prior=last_symbol_entry.get(s.symbol)
        if prior is not None and (s.entry_time-prior).total_seconds()<4*3600:
            continue
        frame=frames[s.symbol]
        entry,exit_price,exit_time,bars,reason,sd=trade_path(frame,s,fee,slip)
        if entry<=0:
            continue
        equity=max(cash,0.0)
        risk_budget=equity*0.0025
        stop_pct=sd/entry
        notional=min(risk_budget/max(stop_pct,1e-9),equity*0.20)
        if notional<=0:
            continue
        qty=notional/entry; sign=1.0 if s.side=="long" else -1.0
        gross=(exit_price-entry)*qty*sign
        entry_fee=notional*fee; exit_fee=abs(exit_price*qty)*fee
        funding=notional*0.0001*(bars//8)
        net=gross-entry_fee-exit_fee-funding
        risk_usd=sd*qty
        cash-=entry_fee
        open_trades.append(Trade(
            symbol=s.symbol,side=s.side,entry_time=s.entry_time,exit_time=exit_time,
            entry_price=entry,exit_price=exit_price,notional=notional,risk_usd=risk_usd,
            net_pnl=net,gross_pnl=gross,fees=entry_fee+exit_fee,funding=funding,exit_reason=reason
        ))
        last_symbol_entry[s.symbol]=s.entry_time
    if signals:
        realize(pd.Timestamp.max.tz_localize("UTC"))

    pnls=[t.net_pnl for t in closed]; wins=[x for x in pnls if x>0]; losses=[x for x in pnls if x<0]
    peak=-1e99; maxdd=0.0
    for x in curve:
        peak=max(peak,x)
        if peak>0:
            maxdd=min(maxdd,(x-peak)/peak)
    rvals=[t.net_pnl/t.risk_usd for t in closed if t.risk_usd>0]
    return {
        "initial_balance":round(initial,4),"final_balance":round(cash,4),
        "return_pct":round(100*(cash/initial-1),4),"max_drawdown_pct":round(abs(maxdd)*100,4),
        "trades":len(closed),"win_rate_pct":round(100*len(wins)/max(1,len(closed)),4),
        "profit_factor":round(sum(wins)/abs(sum(losses)),5) if losses else (999.0 if wins else 0.0),
        "expectancy_r":round(sum(rvals)/max(1,len(rvals)),5),
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

    frames:dict[str,pd.DataFrame]={}
    for pth in sorted(args.data_dir.glob("*_1h.parquet")):
        symbol=symbol_from_path(pth)
        try:
            df=prepare_symbol(symbol,args.data_dir,args.days)
        except Exception as exc:
            print(f"[SKIP] {symbol}: {type(exc).__name__}: {exc}",flush=True); continue
        if len(df)>=500:
            frames[symbol]=df
    if not frames:
        raise SystemExit("no 1h frames")

    data_end=max(pd.Timestamp(df["timestamp"].max()) for df in frames.values())
    holdout_start=data_end-pd.Timedelta(days=args.holdout_days)
    development_end=holdout_start-pd.Timedelta(hours=args.embargo_hours)
    validation_start=development_end-pd.Timedelta(days=120)
    train_end=validation_start-pd.Timedelta(hours=args.embargo_hours)

    trials=[]; passers=[]
    for p in profile_grid():
        sigs=build_signals(frames,p)
        train=[x for x in sigs if x.entry_time<train_end]
        val=[x for x in sigs if validation_start<=x.entry_time<development_end]
        oos=[x for x in sigs if x.entry_time>=holdout_start]
        t5=simulate(train,frames,args.initial_balance,args.fee_bps,5.0)
        t15=simulate(train,frames,args.initial_balance,args.fee_bps,15.0)
        checks={
            "trades_gte_80":int(t5["trades"])>=80,
            "pf5_gte_1_12":float(t5["profit_factor"])>=1.12,
            "exp5_gte_0_04":float(t5["expectancy_r"])>=0.04,
            "pf15_gte_1_03":float(t15["profit_factor"])>=1.03,
            "exp15_gte_0":float(t15["expectancy_r"])>=0.0,
            "dd15_lte_15":float(t15["max_drawdown_pct"])<=15.0,
        }
        row={"profile":p,"counts":{"all":len(sigs),"train":len(train),"validation":len(val),"oos":len(oos)},
             "train_5bps":compact(t5),"train_15bps":compact(t15),"train_checks":checks,
             "train_pass":all(checks.values()),"validation":None,"validation_pass":False,"oos_locked_result":{}}
        if row["train_pass"]:
            v5=simulate(val,frames,args.initial_balance,args.fee_bps,5.0)
            v15=simulate(val,frames,args.initial_balance,args.fee_bps,15.0)
            vchecks={
                "trades_gte_30":int(v5["trades"])>=30,
                "pf5_gte_1_15":float(v5["profit_factor"])>=1.15,
                "exp5_gte_0_05":float(v5["expectancy_r"])>=0.05,
                "pf15_gte_1_05":float(v15["profit_factor"])>=1.05,
                "exp15_gte_0":float(v15["expectancy_r"])>=0.0,
                "dd15_lte_15":float(v15["max_drawdown_pct"])<=15.0,
            }
            row["validation"]={"slippage_5bps":compact(v5),"slippage_15bps":compact(v15),"checks":vchecks}
            row["validation_pass"]=all(vchecks.values())
            if row["validation_pass"]:
                passers.append((row,oos))
        trials.append(row)
        print("CROSS_SECTIONAL_1H_TRIAL="+json.dumps(row,ensure_ascii=False),flush=True)

    selected=None
    if passers:
        passers.sort(key=lambda pair:(
            min(pair[0]["validation"]["slippage_5bps"]["profit_factor"],pair[0]["validation"]["slippage_15bps"]["profit_factor"]),
            min(pair[0]["validation"]["slippage_5bps"]["expectancy_r"],pair[0]["validation"]["slippage_15bps"]["expectancy_r"]),
        ),reverse=True)
        chosen,oos=passers[0]; selected=dict(chosen["profile"])
        for bps in (5.0,15.0):
            x=simulate(oos,frames,args.initial_balance,args.fee_bps,bps)
            chosen["oos_locked_result"][f"slippage_{int(bps)}bps"]=compact(x)

    report={"schema":"proculus-cross-sectional-1h-research-v1","production_changed":False,
            "selection":"train -> validation -> retrospective holdout","data_end":data_end.isoformat(),
            "train_end_exclusive":train_end.isoformat(),"validation_start":validation_start.isoformat(),
            "development_end_exclusive":development_end.isoformat(),"holdout_start":holdout_start.isoformat(),
            "profiles":trials,"selected":selected}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print("CROSS_SECTIONAL_1H_SUMMARY="+json.dumps({"selected":selected,"profiles":trials},ensure_ascii=False),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
