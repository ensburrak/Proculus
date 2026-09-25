# PRISMA V9 localhost operator — release candidate (not production)

2026-09-25 feature branch implementation:
- Added `studio_control/research_jobs.py` durable offline worker with owner-isolated SQLite jobs, valid OHLC CSV, genuine next-open EMA and chronological logistic baseline, single-worker file lease, cancel, idempotency and SHA artifact checking.
- `web/paper_operator_v9.py`: 127.0.0.1-only token-guarded localhost HTTP server; Host/Origin restrictions; explicit reason and OFFLINE confirmation; no trading/config/model deployment routes; older paper evidence API retained.
- `web/operator.html`: functional Turkish native localhost operator panel for CSV upload, jobs, cancellation, artifacts and source truth. Linked from the original visual `web/index.html`.
- `tests/test_studio_offline_v9.py`, `tests/test_studio_http_v9.py`, and GitHub Actions workflow for standalone worker + actual local HTTP tests.
- Full stand-alone Immersive V9 3D HTML is a **separate downloadable user artifact**, not merged into this source tree. Copy it to `web/Proculus_Immersive_v9.html` if desired.

## Safe local start

```powershell
$env:STUDIO_RESEARCH_TOKEN = [Convert]::ToBase64String([Security.Cryptography.RandomNumberGenerator]::GetBytes(32))
python .\web\paper_operator_v9.py --repo . --site .\web --port 8096
```

Open localhost `/operator.html`. Never set this service to bind 0.0.0.0, reverse proxy it publicly or enter exchange keys in the web form. Service does not start Proculus trading runtime.

## What is verified

The GitHub `V9 Studio Offline Acceptance` worker-only workflow previously completed successfully; subsequent HTTP workflow runs must also complete. The separately downloadable, more extensive V9 engine passed isolated tests with actual synthetic HTTP localhost requests. They are two related but not byte-identical variants.

The inspected Proculus `main` branch lacks the referenced `backtesting.runtime` and `ml.rl_train` package directories. `run_full_backtest.py` and `train_rl_gpu.py` being present does **not** make the original engines runnable. This feature cannot claim canonical RL/backtester web orchestration or live trading.

## Production blocking checklist

Per-user session/RBAC, CSRF, scoped API keys, internet-facing BFF, TLS and firewall, vault secret provider, separate kill-switch/watchdog fault injection, encrypted offsite restore, canonical engine dependency/capability reconciliation, minimum paper soak and independent security review remain unverified. The separate localhost token alone is insufficient for remote deployment. Keep the PR draft; no production/live-money approval is implied.
