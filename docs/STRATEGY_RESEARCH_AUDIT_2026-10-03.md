# Proculus V2 Strategy Research Audit — 2026-10-03

## Scope

This audit ignores legacy backtest reports and uses the fresh 730-day OKX replay pipeline added for V2.

Baseline methodology:
- 15m decision bars, 1h/4h closed-candle MTF context.
- Next-15m-open entry.
- 5 / 15 / 50 bps adverse slippage scenarios.
- 5 bps per-side fee assumption.
- Conservative funding cost.
- Runtime max-position, cooldown, daily-loss and weekly-loss controls.
- Independent StochRSI90 lane does not vote inside the primary V2 router.
- Model direction authority remains disabled/shadow-only.
- Paper learning-probe may collect samples at capped size; live remains fail-closed without validated edge.

## Fresh two-year baseline

Reference run: GitHub Actions run `37061487274`, head `34164d98b3fc34c6e6ba693c0a227bd00072717d`.

Universe: 36 runtime-eligible OKX swaps.

Period:
- Data start: 2024-10-02.
- Data end: 2026-10-02.
- Development cutoff: 2026-04-03.
- 48h embargo.
- Retrospective 180-day holdout starts 2026-04-05.

### Full-period baseline

| Cost scenario | Return | Max DD | Trades | Win rate | PF | Expectancy |
|---|---:|---:|---:|---:|---:|---:|
| 5 bps slippage | -37.17% | 37.76% | 7,081 | 28.72% | 0.814 | -0.216R |
| 15 bps slippage | -66.14% | 66.18% | 7,390 | 24.45% | 0.618 | -0.488R |
| 50 bps slippage | -87.19% | 87.19% | 5,648 | 12.55% | 0.235 | -1.312R |

### 180-day retrospective holdout

| Cost scenario | Return | Max DD | Trades | Win rate | PF | Expectancy |
|---|---:|---:|---:|---:|---:|---:|
| 5 bps slippage | -10.87% | 10.88% | 2,090 | 28.90% | 0.827 | -0.237R |
| 15 bps slippage | -25.86% | 25.86% | 2,179 | 24.14% | 0.607 | -0.545R |
| 50 bps slippage | -40.89% | 40.89% | 1,542 | 12.26% | 0.219 | -1.413R |

No baseline OOS acceptance scenario passed.

## Setup-level finding

At 5 bps development cost, every setup with a meaningful sample was negative.

Examples:
- `bear_trend.pullback.short.15m.v2`: 2,312 trades, PF 0.834, expectancy -0.197R.
- `bull_trend.pullback.long.15m.v2`: 2,149 trades, PF 0.806, expectancy -0.196R.
- `stochrsi_opportunity.bear.short.15m.v2`: 153 trades, PF 0.793, expectancy -0.202R.
- `stochrsi_opportunity.bull.long.15m.v2`: 162 trades, PF 0.743, expectancy -0.200R.

A two-trade range-long sample looked positive, but it is explicitly rejected as insufficient evidence and must not be promoted.

## Exit-grid research

Reference run: `37066427167`, head `afdd788268fc57a934b11f0c6b1517c8ddf7f9e4`.

Eight trend-only ATR-stop / R-target / max-hold variants were selected on development only.

Result: **0/8 passed** the development robustness gate.

The least-bad 5 bps candidate used a 1.75 ATR stop and 3.0R target:
- return -17.82%
- PF 0.847
- expectancy -0.158R
- max DD 19.13%

Changing exits alone does not create edge.

## Nested trend-entry research

Reference run: `37066842032`, head `9c75ac7f6e1bd0612b1f7c8d6535db82d724e786`.

The research used an inner train/validation split before touching the retrospective 180-day holdout.

Tested families included:
- pullback touch
- EMA cross/reclaim
- stricter ADX / MTF ADX
- tighter RSI bands
- volume-z filters
- EMA gap filters

Result: **no profile passed the train gate**, so validation and holdout were not used for selection.

Examples at 5 bps train cost:
- `touch_adx22`: PF 0.860, expectancy -0.151R.
- `touch_adx28_mtf22`: PF 0.886, expectancy -0.110R.
- `cross_adx22`: PF 0.854, expectancy -0.144R.
- `touch_reclaim20_adx25_mtf22`: PF 0.865, expectancy -0.125R.

This is evidence against the current EMA-pullback family, not merely evidence for a different threshold.

## Current research

Two additional research tracks are running independently:
1. confidence / micro-structure filtering on the pullback family;
2. a structurally different breakout -> retest -> continuation hypothesis.

Neither is authorized for production unless it passes train, validation, cost sensitivity, and subsequent forward evidence.

## Release decision

Current decision: **no strategy promotion to live**.

Reasons:
- baseline PF < 1;
- negative expectancy in development and retrospective holdout;
- exit search did not repair the edge;
- nested entry filtering did not pass train;
- current evidence is insufficient for any live setup approval.

The correct production behavior is therefore:
- keep strict edge evidence fail-closed for live;
- keep model direction authority shadow-only;
- use paper learning-probe only at capped risk;
- promote a setup only after robust evidence, not after a visually attractive backtest slice.
