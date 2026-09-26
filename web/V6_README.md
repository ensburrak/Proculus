# PRISMA V6 — real paper evidence, not live trading control

This feature branch now has `web/paper_observer_v6.py`: a Python standard-library-only HTTP server bound to **127.0.0.1**. It serves the separate Proculus V6 website if the user copies `Proculus_Immersive_v6.html` into this `web/` directory. It exposes **only seven allowlisted GET endpoints** under `/api/studio/v1/`:

```
health
capabilities
snapshot
config
setup_edge_stats
edge_calibration_proposals
edge_events_tail
```

The last four endpoints read only whitelisted paper evidence and a subset of the local config; sensitive keys are recursively redacted. Server metadata includes available/missing sources and SHA256 where applicable. All `POST`, `PUT`, `PATCH`, `DELETE`, and `OPTIONS` methods are denied. Host and Origin restrictions mitigate cross-origin and DNS-rebinding access. No CORS is enabled. This is **not a hardened internet deployment service**, and should not be published outside localhost.

## Install and run

From a local Proculus checkout:

```powershell
copy C:\path\to\download\Proculus_Immersive_v6.html .\web\Proculus_Immersive_v6.html
python .\web\paper_observer_v6.py --repo . --port 8096
```

Then visit the local URL printed by the observer, open **Operasyon Merkezi**, and verify the `health` endpoint before checking paper artifacts. If the files do not exist the JSON returns `available:false` and **does not fabricate results**. The V6 website includes a catalog of 66 distinct UI/source/paper-GET entries and its own offline 3D command room; it is distributed separately rather than committed in its entirety here.

**No browser-safe authenticated trading-control API exists on the inspected Proculus main branch.** This observer has no order placement, pause/resume, model deployment, training jobs or config write authority. For those, follow `web/IMMERSIVE_V5_CONTROL_PLAN.md`: separate paper-only BFF, short-lived session, CSRF, RBAC, idempotency, bounded job workers and immutable audit. Do not expose API keys or use this observer as a bridge to live exchange trading. Running the observer does not start Proculus runtime.

## Validation required

- Check API health and 7 GET shapes on localhost.
- Verify all HTTP write methods reject with 405.
- Confirm Host/Origin rejection, secret masking, missing-file states and JSONL oversized-line behavior.
- Verify responsive UI, offline fallback and the site's clear distinction between paper report, simulated demo and verified runtime.
- Run independent security review before exposing any remote or write-capable endpoint.

The standalone ZIP has a more complete observer version and a local-only integration test. This repository feature branch is unmerged and not runtime-certified.
