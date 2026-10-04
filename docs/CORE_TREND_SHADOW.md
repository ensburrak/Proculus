# Core Trend Shadow Runtime

## Purpose

The Core Trend Sleeve is a research-positive 4h trend hypothesis that is intentionally
separate from the frozen 15m V2 setup families.

Research identifier:

`core_trend_4h_ema50_200_voltarget.v1`

The lane is **shadow only**. It has no exchange-order authority and must not be added to
`strategy_release.allowed_setup_ids` from retrospective research.

## Signal contract

For each configured major symbol:

1. Ignore the newest 4h row because it may still be open.
2. Compute EMA(50) and EMA(200) from closed 4h candles only.
3. Long when EMA50 > EMA200; short when EMA50 < EMA200.
4. Estimate annualized realized volatility from closed 4h returns.
5. Scale the target exposure to 20% annualized volatility, capped at 1.0 absolute exposure.
6. Record the current 4h interval open as the forward entry-reference price.
7. Never call `execute_decision` for this lane.

The main V2 and independent StochRSI authorities are unchanged.

## Evidence integrity

Forward observations are stored in:

`reports/core_trend_shadow_events.jsonl`

An observation is uniquely identified by:

- strategy ID;
- symbol;
- closed 4h candle timestamp.

Repeated runtime loops inside the same 4h candle are therefore de-duplicated and cannot
inflate sample count.

Use:

```bash
python tools/core_trend_shadow_report.py
```

to build:

`reports/core_trend_shadow_report.json`

The report resolves each observation against the next recorded interval and applies a
conservative modeled cost. It reports profit factor, average net observation return,
confidence interval, max drawdown, unique days, symbol coverage and compounded return.

## Promotion stages

### Research -> paper candidate

The current policy requires at least:

- 30 distinct forward-shadow days;
- 500 resolved observations;
- 8 symbols;
- profit factor >= 1.20;
- positive average net observation return;
- max drawdown <= 15%;
- positive lower 95% confidence bound.

Passing those criteria creates only a **paper candidate**.

### Live

Live remains fail-closed. A statistical pass is not sufficient.

The report also requires real, non-hypothetical TCA evidence. Backtest or hypothetical TCA
is explicitly rejected. In addition, `core_trend_shadow.forward_promotion.allow_live`
remains `false`, so live release requires a separate reviewed policy change.

## Why historical holdout is not enough

The 180-day retrospective holdout has already been inspected during research. It can
support the decision to start forward shadow collection, but it is no longer unseen data
and must not be reused as a live-capital promotion gate.

The existing research also has trade-level bootstrap lower confidence bounds below zero.
That uncertainty is why the runtime collects new forward evidence rather than promoting
the strategy directly.

## Current research evidence

At the high-cost research scenario, the original study reported positive train,
validation and retrospective-holdout results, plus EMA-neighborhood and
leave-one-symbol-out robustness. Those results make the sleeve worth shadowing; they do
not authorize capital.

The source of truth for future promotion is the new forward evidence generated after this
runtime integration.
