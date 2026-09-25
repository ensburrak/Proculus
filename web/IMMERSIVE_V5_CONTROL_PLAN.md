# Proculus / PRISMA V5 command room — backend control plan

Status: **design only**. The prior V3 draft PR adds an isolated `web/` Turkish interactive visual experience. The expanded V5 3D standalone HTML is delivered separately and is **not a connected trading application**.

## Repository-verified constraints
The inspected Proculus `main` branch does **not** contain `api/main.py` or `frontend/src/App.tsx`; do not copy AutoTraderBot endpoint assumptions. `config.json` includes `paper_trading.enabled=true`, `live_readiness`, `live_safety`, `llm_contract`, `edge_learning`. It is not evidence of an authenticated, browser-ready API or validated live execution.

## Architecture first
`V5 Web UI (read-only 3D) → same-origin HTTPS BFF → session/RBAC + CSRF → paper-only adapter → runtime/paper worker → immutable audit/artifact store → read model + event stream`. Vault and watchdog belong in the backend and stay unaffected by UI crashes.

## Work packages and gates
P0: Enumerate actual Python entrypoint, paper state model, restart/reconciliation behavior, execution gates, risk config validator and a live-side-effect threat model.
P1: Implement **paper-only** `GET /studio/v1/capabilities`, `GET /studio/v1/overview`, typed `GET /studio/v1/events` with data provenance, monotonic versions and stale-data handling.
P2: Authenticated paper pause/resume and candidate config updates via separate preview/confirm flow (reason, nonce, idempotency, server-side validation, immutable audit).
P3: Persistent research jobs (backtest, OOS, paper report), worker restart/recovery, artifact hash and reproducibility metadata.
P4: Server-side masked secret status, independent kill-switch/health monitor, mandatory config live-readiness and canary gates; website cannot disable them.
P5: Full paper smoke run, auth failure tests, idempotency/replay tests, network outage recovery, contract tests, audit verification and responsive/mobile rendering. Only then consider a separately approved testnet adapter.

## UX/visual constraints
3D room is a **spatial explanation** of regime, evidence, risk, execution and research. Every navigation hotspot opens the conventional 2D control workspace. No 3D gesture directly creates orders; no API key in HTML, JS or localStorage. Show "DEMO" until authenticated, verified paper API is actually connected.

The downloadable V5 package contains the standalone 3D room, a more detailed Turkish architecture plan and an offline viewable connection diagram. This PR note does not assert that V5 room or a backend adapter has been committed.
