\# CLAUDE.md



\# Engineering, AI/ML, and Trading Execution Standard



\## Purpose

This repository must be changed with precision, minimal scope, and explicit verification.



Prefer correct, maintainable, reviewable, and risk-aware solutions over broad rewrites, speculative refactors, or polished but weakly verified output.



For this repository:

\- correctness > speed

\- safety > aggressiveness

\- evidence > confidence

\- survivability > return optimization

\- risk control > profit optimization

\- operational integrity > cosmetic cleanliness



This is not a repo where impressive-looking output is enough.  

Changes must be bounded, inspectable, and verified in proportion to their risk.



---



\## Core Operating Mode

By default, operate as follows:

\- make the smallest correct change,

\- do not mix unrelated edits,

\- do not silently change safety-sensitive behavior,

\- do not claim success without verification evidence,

\- do not fabricate logs, metrics, outputs, file contents, API responses, exchange responses, or market results,

\- state assumptions when they materially affect implementation, behavior, or risk,

\- be direct about uncertainty, incomplete verification, and operational risk.



When in doubt:

\- prefer explicitness over cleverness,

\- prefer inspectable behavior over implicit behavior,

\- prefer defensive behavior over forced execution,

\- prefer a bounded refactor over a fragile patch only when the patch would preserve a misleading or unstable design.



---



\## Repository Context

Assume this repository is a trading / AI-assisted execution system unless explicitly stated otherwise.



\### Typical entrypoints

Primary local entrypoints in this repository are:

\- `python -m autotraderbot.cli stack --allow-missing-frontend-build`

\- `python -m autotraderbot.cli runtime`

\- `python -m autotraderbot.cli api`

\- `python -m runtime`

\- `python -m uvicorn api.main:app --host 127.0.0.1 --port 8000`

Backtesting is exposed through `python -m backtesting.runtime`.



\### Safety-critical execution files

Treat these as safety-sensitive unless proven otherwise:

\- `trade\_engine.py`

\- `risk\_manager.py`

\- `portfolio\_manager.py`

\- `binance\_client.py`

\- `ai\_strategy\_engine.py`

\- `confidence\_score.py`

\- `core/order\_lifecycle.py` — order state machine; all exchange interaction must eventually route through here. Do not weaken transition guards or idempotency logic.

\- `core/idempotency.py` — persistent deduplication layer; fail-closed contract must be preserved. A duplicate order = double position.



\### Model / training / data files

Treat schema, labels, preprocessing, and inference compatibility as sensitive in files such as:

\- `dataset\_builder.py`

\- `train\_xgboost.py`

\- `train\_lstm.py`

\- `xgboost\_predictor.py`

\- `confidence\_score.py`



\### Validation / test entrypoints

When available, prefer existing validation paths such as:

\- `pytest -q`

\- `python test\_backtest\_decision.py`

\- `python test\_rl\_decision.py`



If the actual repo uses different commands, use the real commands in the environment instead of assuming these blindly.



\### Runtime safety assumption

Unless clearly configured otherwise, assume:

\- paper / sim / dry-run is the intended default,

\- live trading is disabled by default,

\- real exchange side effects must not be introduced silently.



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

\- Explicit behavior before hidden behavior

\- Defensive execution before forced execution



---



\## Plan First When It Matters

Start with a short plan when work involves:

\- multiple files,

\- architectural impact,

\- unclear requirements,

\- model behavior changes,

\- strategy changes,

\- execution or risk changes,

\- external side effects,

\- schema or config changes,

\- debugging with uncertain cause,

\- live-impacting or potentially live-impacting behavior.



Keep the plan concise and outcome-focused.



A good plan should usually identify:

\- goal,

\- scope,

\- main risks,

\- intended verification path.



For simple, obvious, low-risk fixes, act directly.



Update the plan if scope, risk, or understanding changes.



---



\## Ask Only When Blocked

Ask targeted questions only when ambiguity prevents a correct implementation.



Otherwise:

\- proceed with reasonable explicit assumptions,

\- avoid unnecessary process questions,

\- do not delay obvious work behind avoidable clarification.



However, do not guess in ways that could silently alter:

\- live vs paper behavior,

\- leverage or sizing,

\- order routing,

\- stop-loss / take-profit logic,

\- kill-switch behavior,

\- model schema compatibility,

\- labels or decision thresholds,

\- secrets or environment-specific behavior.



If ambiguity affects safety, state the assumption explicitly or stop and ask.



---



\## Scope Control

\- Change only what is necessary for correctness, reliability, maintainability, or risk control.

\- Do not mix unrelated fixes into the same change.

\- Avoid speculative refactors.

\- Keep edits reviewable.

\- Preserve existing safeguards unless there is a verified reason to change them.

\- Prefer small, focused edits over broad rewrites unless a rewrite is clearly justified.

\- Do not expand scope just because another design appears cleaner.



If the smallest patch would preserve a fragile, misleading, or repeatedly failing design, propose or perform a bounded refactor explicitly and verify behavior afterward.



---



\## Simplicity and Durability

\- Prefer the simplest solution that fully solves the problem.

\- Avoid hacks when a clean fix is practical.

\- Do not over-engineer obvious changes.

\- Favor readability, predictability, maintainability, and explicit control flow over cleverness.

\- Choose durable fixes over temporary cosmetic patches when the root cause is known.



---



\## Verification Before Conclusion

Never mark work complete without evidence.



Use the strongest relevant verification available.



\### Code changes

\- Run relevant tests if they exist.

\- Inspect the execution path touched by the change.

\- Check affected logs, outputs, and critical branches.

\- Confirm the new behavior matches the intended scope.



\### Bug fixes

\- Reproduce the issue when feasible.

\- Prefer identifying or adding a failing check before changing logic.

\- Verify the failure is gone after the fix.

\- Confirm no adjacent behavior regressed in the touched path.



\### Model / AI / ML changes

\- Compare against a baseline when claiming improvement.

\- Verify feature schema, preprocessing, labels, thresholds, split boundaries, and output compatibility.

\- Verify train-time assumptions still match inference-time behavior.

\- Do not claim improvement from a single ungrounded metric.

\- If model output influences execution, verify downstream trading impact, not only model metrics.



\### Strategy / trading logic changes

\- Verify not only profitability, but also:

&nbsp; - drawdown,

&nbsp; - stability,

&nbsp; - failure modes,

&nbsp; - behavior under missing / stale / partial data,

&nbsp; - risk guard behavior.

\- Use realistic assumptions where relevant:

&nbsp; - fees,

&nbsp; - slippage,

&nbsp; - spread,

&nbsp; - latency,

&nbsp; - funding,

&nbsp; - exchange constraints,

&nbsp; - partial fills.



\### Execution / live-impacting changes

\- Prefer dry-run, sim, or paper validation before any live path.

\- Check order lifecycle assumptions, state transitions, and error-handling paths.

\- Explicitly call out operational risk.



When formal tests are not feasible:

\- provide clear manual verification steps,

\- describe expected before/after behavior,

\- state what remains unverified.



Do not say:

\- “fixed”

\- “improved”

\- “production-ready”

\- “safe”

unless verification genuinely supports the claim.



---



\## Bug-Fix Discipline

\- Reproduce the bug when feasible.

\- Prefer identifying the failing condition before changing logic.

\- Fix the root cause, not only the visible symptom.

\- Re-run relevant checks after the fix.

\- Document reproducible before/after behavior when formal tests are unavailable.

\- Do not treat “error disappeared once” as proof of correctness.

\- For intermittent issues, note the uncertainty explicitly.



---



\## Refactoring Standard

Refactor only when needed for:

\- correctness,

\- maintainability,

\- clarity,

\- risk reduction,

\- dependency cleanup required for the requested task,

\- removing a design that is actively causing fragility.



Do not refactor just because another design looks cleaner.



Keep refactors bounded and verify behavior after refactoring.



When refactoring safety-sensitive paths:

\- keep logic changes explicit,

\- preserve existing safeguards unless intentionally changed,

\- state practical behavior impact,

\- verify affected flows after the refactor.



---



\## AI / ML Engineering Rules

\- Do not introduce data leakage, look-ahead bias, target leakage, or invalid evaluation logic.

\- Keep feature engineering, preprocessing, labels, splits, thresholds, and schema assumptions explicit.

\- Do not silently change:

&nbsp; - feature ordering,

&nbsp; - normalization,

&nbsp; - label definitions,

&nbsp; - target-generation logic,

&nbsp; - decision thresholds,

&nbsp; - train / validation / test boundaries,

&nbsp; - sampling assumptions,

&nbsp; - inference feature shape,

&nbsp; - expected input schema.

\- Keep experiments separate from production logic unless integration is intentional and verified.

\- Prefer reproducible workflows:

&nbsp; - explicit configs,

&nbsp; - traceable artifacts,

&nbsp; - stable schema expectations,

&nbsp; - fixed seeds where appropriate.

\- Maintain train/inference compatibility.

\- Do not claim model improvement without comparison to a relevant baseline.

\- Distinguish clearly between:

&nbsp; - research code,

&nbsp; - training code,

&nbsp; - evaluation code,

&nbsp; - backtest code,

&nbsp; - paper-trading code,

&nbsp; - live-trading code.

\- If a model change can affect execution behavior, verify downstream strategy impact instead of relying only on accuracy-like metrics.

\- Do not hide threshold changes inside unrelated edits.

\- Do not silently alter label horizon, target definitions, or feature windows.



---



\## Trading System Rules

\- Risk control comes before return optimization.

\- Safety, survivability, and capital preservation take priority over aggressiveness.

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

&nbsp; - order routing behavior,

&nbsp; - capital allocation,

&nbsp; - confidence thresholds that affect entries or exits.

\- Any change affecting entry, exit, execution, sizing, allocation, or risk must be called out explicitly.

\- Distinguish clearly between:

&nbsp; - hypothetical performance,

&nbsp; - backtest performance,

&nbsp; - paper results,

&nbsp; - live results.

\- Do not present hypothetical PnL as guaranteed, expected, or likely without evidence.

\- Use realistic assumptions where relevant.

\- Fail safe when state is uncertain.

\- If market data, account state, balance state, position state, or order state is unreliable, prefer defensive behavior over forced execution.



---



\## Execution Integrity Rules

Treat execution integrity as safety-critical.



\- Do not assume local state is correct when exchange state may differ.

\- Prefer reconciliation when order state, position state, or account state could be stale or inconsistent.

\- Avoid duplicate order placement.

\- Preserve or implement idempotent execution behavior where relevant.

\- Reject or invalidate stale signals when they can cause unsafe execution.

\- Do not assume fills are complete, immediate, or exact.

\- Consider partial fills, exchange precision rules, and latency-sensitive race conditions.

\- Do not silently alter retry behavior in ways that could duplicate orders or amplify risk.

\- Prefer no-trade over unsafe trade when execution state is ambiguous.



---



\## Live Execution Safety

Unless the user explicitly requests otherwise and live conditions are intentionally verified:



\- do not enable live trading,

\- do not place real orders,

\- do not switch from paper/sim/testnet to live silently,

\- do not increase leverage,

\- do not disable kill-switches,

\- do not weaken risk guards,

\- do not change exchange-side effects without clearly stating them.



Live conditions are not “verified” merely because code compiles.  

Treat live-impacting behavior as high risk.



Before considering a live-impacting path, prefer evidence such as:

\- explicit user intent,

\- correct environment/config targeting,

\- non-live validation path already exercised,

\- relevant dry-run / sim / paper checks.



If a change can affect live execution, call out the risk explicitly.



---



\## Configuration, Secrets, and Environment Rules

\- Prefer config-driven behavior over hardcoded values.

\- Keep secrets, tokens, API keys, and environment-specific values out of source code.

\- Do not print secrets in logs or outputs.

\- Do not silently alter:

&nbsp; - scheduler behavior,

&nbsp; - persistent storage behavior,

&nbsp; - external API side effects,

&nbsp; - environment selection,

&nbsp; - account-impacting operations.

\- Preserve backward compatibility where reasonable.

\- If compatibility changes, state it clearly.

\- Treat `.env`, secrets, credential loaders, and exchange configuration as sensitive.

\- Do not silently convert testnet-oriented behavior into live-mainnet behavior.

\- Document any config default change that materially affects behavior or risk.



---



\## Prompt / Agent Conduct Rules

\- Separate verified facts from assumptions and inferences.

\- Do not fabricate tool outputs, logs, metrics, file contents, API responses, exchange responses, or market data.

\- If evidence is unavailable, say so directly.

\- Prefer deterministic, inspectable steps over vague reasoning.

\- Use tools when they materially improve correctness, not for show.

\- Do not present polished but weakly verified output as reliable.

\- Be precise about what was observed versus what was inferred.

\- Do not imply testing was performed if it was not.

\- Do not imply exchange interaction occurred if it did not.

\- Do not imply model improvement was demonstrated if only hypothesized.



---



\## Communication Rules

At the end of non-trivial work, report clearly:



\### Changed

\- files changed,

\- purpose of each change.



\### Behavior impact

\- practical difference between old and new behavior,

\- any behavior intentionally preserved,

\- any safety-sensitive behavior affected.



\### Verification

\- tests run,

\- manual checks performed,

\- logs / outputs / flows inspected,

\- what exactly was and was not verified.



\### Risk / uncertainty

\- anything not fully verified,

\- operational or live-trading risk,

\- assumptions that materially matter,

\- any follow-up validation still recommended.



Do not overstate confidence.  

Do not present partial verification as full verification.  

Do not hide safety-sensitive effects inside vague summaries.



---



\## Definition of Done

A task is done only when:

\- the requested outcome is implemented,

\- scope remained controlled,

\- relevant behavior was verified,

\- assumptions and limitations were stated,

\- no unnecessary unrelated changes were introduced,

\- safety-sensitive effects were explicitly disclosed,

\- any model-impacting behavior changes were clearly documented,

\- any trading-impacting behavior changes were clearly documented,

\- the result is described honestly in proportion to the evidence.



If verification is incomplete, the task is not “fully done”; it is only partially implemented or partially verified.



---



\## Change Philosophy

\- Minimal blast radius

\- Explicit behavior

\- Root-cause thinking

\- Honest verification

\- Maintainability over speed theater

\- Risk awareness over performance storytelling

\- Defensive execution over fragile optimism



---



\## Practical Default Summary

When working in this repository:



1\. Understand the task and identify whether it touches safety-sensitive logic.

2\. Plan briefly if risk, scope, or uncertainty justifies it.

3\. Make the smallest correct change unless a bounded refactor is clearly safer.

4\. Do not silently alter model, trading, execution, or live-impacting behavior.

5\. Verify in proportion to risk.

6\. Report exactly what changed, how it was checked, and what remains uncertain.

7\. Prefer no-trade over unsafe trade.

8\. Prefer no-claim over unsupported claim.

