# Model Registry

This file documents production model artifact expectations. The machine-readable
source of truth is `models/model_registry.json` when that file exists.

## Current Artifact Status

| Type | Model ID | Artifact | Status |
| --- | --- | --- | --- |
| transformer | unavailable | models/transformer_latest.pt | missing in this workspace |
| rl_ppo | unavailable | models/rl_ppo_latest.zip | missing in this workspace |

## Operational Rule

Only artifacts referenced by `active_models` in `models/model_registry.json` and
present on disk are eligible for runtime loading. Historical checkpoints and
candidates should be archived under `models/archive/` after operator approval,
not loaded implicitly.

Do not document transformer or RL/PPO as active unless the artifact file exists,
validation has passed, and the runtime model-load check succeeds.
