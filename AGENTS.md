\# AGENTS.md



\# Engineering and Trading Execution Standard



\## Purpose

This repository must be changed with precision, minimal scope, and explicit verification.



Prefer correct, maintainable, reviewable, and risk-aware changes over broad rewrites, speculative refactors, or impressive-looking but weakly verified output.



For this codebase:

\- correctness is more important than speed,

\- safety is more important than aggressiveness,

\- evidence is more important than confidence,

\- survivability is more important than return optimization.



---



\## Operating Mode

\- Default to the smallest correct change.

\- Do not make unrelated edits.

\- Do not silently change behavior in safety-sensitive paths.

\- Do not claim success without verification evidence.

\- Do not fabricate logs, metrics, file contents, API results, or market results.

\- State assumptions when they materially affect implementation, behavior, or risk.



---



\## Repository Context

Use these project conventions unless explicitly instructed otherwise:



\- Primary runtime mode: `paper | sim | dry-run | live-disabled-by-default`

\- Main entrypoints:

&nbsp; - Official bot runtime: `python -m runtime`

&nbsp; - CLI bot runtime: `python -m autotraderbot.cli runtime`

&nbsp; - Full local stack: `python -m autotraderbot.cli stack`

&nbsp; - API only: `python -m autotraderbot.cli api`

&nbsp; - Compatibility wrapper only: `main_bot_async.py`

\- Core strategy / execution files:

&nbsp; - `controller_async.py`

&nbsp; - `core/engine/bot.py`

&nbsp; - `core/order_executor.py`

&nbsp; - `execution/safe_order_wrapper.py`

&nbsp; - `risk_manager.py`

&nbsp; - `risk/portfolio_guard.py`

&nbsp; - `exchanges/`

\- Model / training files:

&nbsp; - `ml/build_dataset.py`

&nbsp; - `ml/transformer_train.py`

&nbsp; - `ml/rl_train.py`

&nbsp; - `ml/bot_model_integration.py`

&nbsp; - `calibrate.py`

\- Test / validation entrypoints:

&nbsp; - `python -m pytest -q`

&nbsp; - `python tools/critical_coverage_gate.py --min-coverage 80`

&nbsp; - `python tools/runtime_two_cycle_smoke.py --runner official-runtime --mode paper --cycles 2 --timeout-seconds 120`

&nbsp; - `python -m compileall -q ai api autotraderbot backtesting bot core decision execution exchanges ml notifier risk runtime analysis paper_trading quality_gate`



The list above is canonical for this repository; do not reintroduce placeholder or removed file names.



---



\## 1. Plan non-trivial work

Start with a short plan when the task involves:

\- multiple files,

\- architectural impact,

\- unclear requirements,

\- model behavior changes,

\- strategy changes,

\- execution/risk changes,

\- external side effects,

\- meaningful debugging uncertainty.



Keep the plan concise and outcome-focused.



For simple, obvious fixes, act directly.



Update the plan if scope, risk, or understanding changes.



---



\## 2. Keep changes bounded

\- Change only what is required for correctness, reliability, maintainability, or risk control.

\- Do not mix unrelated fixes into the same change.

\- Avoid speculative refactors.

\- Prefer small, reviewable edits over broad rewrites unless a rewrite is clearly justified.

\- Preserve existing safeguards unless there is a verified reason to change them.



---



\## 3. Ask only when blocked

\- Ask targeted questions only when ambiguity prevents a correct implementation.

\- Otherwise proceed with reasonable explicit assumptions.

\- Do not ask unnecessary process questions.

\- Do not delay obvious work behind avoidable clarification.



---



\## 4. Verification before conclusion

Never mark work complete without evidence.



Use the strongest relevant verification available:



\### Code change

\- Run relevant tests if they exist.

\- Inspect the execution path touched by the change.

\- Check logs and outputs for affected flows.



\### Bug fix

\- Reproduce the issue when feasible.

\- Prefer adding or identifying a failing check before fixing.

\- Verify the failure is gone after the fix.



\### Model / ML change

\- Compare against a baseline when claiming improvement.

\- Verify feature schema, label logic, preprocessing, thresholds, and output shape compatibility.

\- Do not claim improvement from a single ungrounded metric.



\### Strategy / trading logic change

\- Verify not only profitability, but also drawdown, stability, and failure modes.

\- Use realistic assumptions where relevant: fees, slippage, spread, latency, funding, partial fills, exchange constraints.



\### Execution / live-impacting change

\- Prefer dry-run, sim, or paper validation before any live path.

\- Explicitly call out operational risk.



When formal tests are not feasible, provide clear manual verification steps and expected before/after behavior.



Do not say “fixed”, “improved”, or “production-ready” without verification.



---



\## 5. Bug-fix discipline

\- Reproduce the bug when feasible.

\- Prefer identifying the failing condition before changing logic.

\- Fix the root cause, not only the visible symptom.

\- Re-run the relevant checks after the fix.

\- Document reproducible before/after behavior when formal tests are unavailable.



---



\## 6. Simplicity over cleverness

\- Prefer the simplest solution that fully solves the problem.

\- Avoid hacks when a clean fix is reasonably possible.

\- Do not over-engineer obvious changes.

\- Favor readability, predictability, and maintainability over cleverness.



---



\## 7. Refactoring standard

Refactor only when needed for:

\- correctness,

\- maintainability,

\- clarity,

\- risk reduction,

\- dependency cleanup required for the requested task.



Do not refactor just because another design looks cleaner.



Keep refactors bounded and verify behavior after refactoring.



---



\## 8. AI / ML engineering rules

\- Do not introduce data leakage, look-ahead bias, target leakage, or invalid evaluation logic.

\- Keep feature engineering, labels, preprocessing, splits, and thresholds explicit.

\- Do not silently change:

&nbsp; - feature ordering,

&nbsp; - normalization,

&nbsp; - label definitions,

&nbsp; - decision thresholds,

&nbsp; - train/validation/test boundaries,

&nbsp; - target-generation logic.

\- Keep experiments separate from production logic unless integration is intentional and verified.

\- Prefer reproducible workflows: explicit configs, fixed seeds where appropriate, traceable artifacts, stable schema expectations.

\- If model output affects trading behavior, verify downstream trading impact instead of relying only on model metrics.

\- Clearly distinguish:

&nbsp; - research code,

&nbsp; - training code,

&nbsp; - backtest code,

&nbsp; - paper-trading code,

&nbsp; - live-trading code.



---



\## 9. Prompt / agent engineering rules

\- Separate verified facts from assumptions and inferences.

\- Do not fabricate tool outputs, logs, metrics, file contents, API responses, exchange responses, or market data.

\- If evidence is unavailable, say so directly.

\- Prefer deterministic, inspectable steps over vague reasoning.

\- Use tools when they materially improve correctness, not for show.

\- Do not present polished but weakly verified output as reliable.

\- Be precise about what was observed versus what was inferred.



---



\## 10. Trading system rules

\- Safety, survivability, and risk control come before return optimization.

\- Do not weaken safeguards silently.

\- Do not silently change:

&nbsp; - leverage,

&nbsp; - position sizing,

&nbsp; - wallet allocation,

&nbsp; - stop-loss logic,

&nbsp; - take-profit logic,

&nbsp; - liquidation buffers,

&nbsp; - cooldown logic,

&nbsp; - kill-switch behavior,

&nbsp; - retry logic,

&nbsp; - order routing behavior.

\- Any change affecting entry, exit, execution, sizing, or capital allocation must be explicitly stated.

\- Distinguish clearly between:

&nbsp; - hypothetical performance,

&nbsp; - backtest performance,

&nbsp; - paper results,

&nbsp; - live results.

\- Do not present hypothetical PnL as guaranteed, expected, or likely without evidence.

\- Fail safe when state is uncertain.

\- If market data, account state, position state, balance state, or order state is unreliable, prefer defensive behavior over forced execution.



---



\## 11. Live execution safety

Unless the user explicitly requests otherwise and the environment is confirmed safe:



\- Do not enable live trading.

\- Do not place real orders.

\- Do not increase leverage.

\- Do not disable kill-switches or risk guards.

\- Do not convert paper/sim behavior into live behavior silently.

\- Do not change exchange-side effects without clearly stating them.



If a change can affect live execution, call out the risk explicitly.



---



\## 12. Configuration and secrets

\- Prefer config-driven behavior over hardcoded values.

\- Keep secrets, tokens, API keys, and environment-specific values out of source code.

\- Do not silently alter scheduler behavior, persistent storage behavior, external API side effects, or account-impacting operations.

\- Preserve backward compatibility where reasonable, or state clearly when compatibility changes.



---



\## 13. Required completion format

At the end of non-trivial work, report:



\### Changed

\- files changed

\- purpose of each change



\### Behavior impact

\- practical difference between old and new behavior



\### Verification

\- tests run

\- manual checks performed

\- logs / outputs inspected



\### Risk / uncertainty

\- anything not fully verified

\- any operational or live-trading risk

\- any assumptions that matter



Do not present partial verification as full verification.



---



\## 14. Definition of Done

A task is done only when:

\- the requested outcome is implemented,

\- scope remained controlled,

\- relevant behavior was verified,

\- assumptions and limitations were stated,

\- no unnecessary unrelated changes were introduced,

\- safety-sensitive effects were explicitly disclosed,

\- any model-impacting or trading-impacting behavior changes were clearly documented.



---



\## Non-Negotiable Principles

\- Correctness before speed

\- Evidence before conclusion

\- Root cause before patch

\- Safety before aggressiveness

\- Minimal scope before broad change

\- Clarity before cleverness

\- Maintainability before convenience

\- Risk awareness before performance claims
