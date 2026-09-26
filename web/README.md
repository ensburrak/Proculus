# Proculus / PRISMA Immersive Studio

This repository's inspected `main` tree did not include a self-contained `frontend/` React application. To avoid altering the trading runtime, this feature branch adds a standalone Turkish-language web experience under `web/`:

- `web/index.html` — responsive Proculus site (research, regime, risk, model explainability, local settings).
- `web/studio.js` — pinned Three.js procedural 3D PRISMA mascot and animation, cinematic intro, optional Turkish speech, voice-only page navigation, local microphone amplitude response.

## Run
From the repository root:
```bash
python -m http.server 8090 --directory web
```
Open localhost port 8090. Internet access is required for the Three.js CDN; CSS fallback remains usable offline. Some microphone and speech features require a compatible browser and a secure localhost/HTTPS origin.

## Scope
This integration is **visual and client-side only**. The reviewed Proculus branch has no validated browser-safe, authenticated API contract for trading actions. Consequently these site controls do not send real orders or manipulate bot runtime. Do not paste API keys into JavaScript or expose local bot processes to the public internet. Any future backend adapter needs authenticated same-origin proxying, CSRF protection, RBAC and explicit operator attestation before write actions are considered.

The larger standalone HTML `Proculus_Immersive_v3.html` supplied in the user deliverable contains additional dashboard sections. This lighter repository site is a non-destructive first integration, not an assertion that every runtime module has been wired to a backend.
