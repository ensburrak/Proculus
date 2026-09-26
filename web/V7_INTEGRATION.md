# Proculus / PRISMA V7 — localhost paper evidence extension

Status as of 2026-09-25: **unmerged design and evidence prototype**, not a remotely hosted trading dashboard.

## Native repository addition
`web/paper_observer_v7.py` is a Python-standard-library, `127.0.0.1`-bound, GET-only observer. It serves a local user-installed `Proculus_Immersive_v7.html` file and seven explicit `/api/studio/v1` GET resources (`health`, `capabilities`, `snapshot`, `config`, `setup_edge_stats`, `edge_calibration_proposals`, `edge_events_tail`). Paper config fields are allowlisted and recursively redacted. Reports return source and missing/available provenance, and bounded file reads. All HTTP write verbs are rejected. Host and Origin checks apply. This observer does not start a bot, schedule jobs or operate an exchange; keep it bound to localhost.

## Full standalone V7 (provided separately)
The downloadable `Proculus_Immersive_v7.html` file contains **29 navigable pages** with ten additional evidence-first research and 3D visual modules: spatial Digital Twin, Visual Strategy Studio, deterministic Shock Lab, read-only paper Execution Replay, labelled-JSON Model Arena, job **blueprints**, local tab-only alarms, imported Fleet Atlas, visual Data Universe and deterministic Turkish Evidence Assistant. The independent download embeds all CSS/JS and optional WebGL geometry; it does not claim that 3D visualization equals live runtime connectivity.

## Install locally
Copy the independently provided `Proculus_Immersive_v7.html` file into the repo's `web/` folder or pass an explicit site directory:
```powershell
python .\web\paper_observer_v7.py --repo . --site .\web --port 8096
```
Open the localhost URL printed by the observer. Its startup requires the actual Proculus `config.json` in the specified repo. If a report file is missing, the observer returns `available: false`; it must not fabricate trading evidence.

## Remaining work before real control
A real authenticated paper-only BFF, role-scoped sessions, CSRF protection, explicit preview/confirm, idempotent durable paper jobs, immutable server audit, independent watchdog and vault, threat model, contract tests and controlled paper E2E acceptance are **not built** by the V7 UI or this observer. Never put exchange secrets in localStorage or expose the observer on the public internet.

No React/Vite frontend is assumed to exist on the inspected Proculus main branch. The downloadable V7 site is not committed wholesale into this feature PR; only its safe observer and documentation are part of this branch.
