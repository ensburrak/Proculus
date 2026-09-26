"""Read-only repository readiness inventory. NEVER authorizes live trading."""
from __future__ import annotations
import json
from datetime import datetime,timezone
from pathlib import Path

FILES={
    "trading_entrypoint":"main_bot_async.py",
    "config_validator":"config_validator.py",
    "stop_order_watchdog":"stop_order_watchdog.py",
    "health_monitor":"health_monitor.py",
    "sys_watchdog":"sys_watchdog.py",
    "canonical_backtesting":"backtesting/runtime.py",
    "canonical_ml_training":"ml/rl_train.py",
    "authenticated_bff":"web/live_operator_api.py",
    "rbac_iam":"web/iam_provider.py",
    "independent_watchdog_probe":"web/independent_watchdog_probe.py",
    "tls_deployment_manifest":"deploy/live_tls_manifest.json",
}
def inspect(repo:Path)->dict:
    repo=repo.resolve()
    present={key:(repo/path).is_file() for key,path in FILES.items()}
    blockers=[key for key,exists in present.items() if not exists]
    try:
        cfg=json.loads((repo/"config.json").read_text(encoding="utf8"))
        assert isinstance(cfg,dict)
        live=cfg.get("live_safety") or {}
        canary=live.get("canary") or {}
        watchdog=live.get("independent_watchdog") or {}
        declared={
            "paper_trading_configured":isinstance(cfg.get("paper_trading"),dict),
            "live_readiness_configured":isinstance(cfg.get("live_readiness"),dict),
            "canary_configured":canary.get("enabled") is True,
            "independent_watchdog_declared":watchdog.get("required") is True,
            "capital_isolation_declared":(live.get("capital_isolation") or {}).get("required") is True,
        }
    except (OSError,ValueError,TypeError,AssertionError):
        declared={"valid_config":False}
    blockers += ["config:"+key for key,valid in declared.items() if not valid]
    blockers += [
       "actual_cloud_https_iam_mfa_unverified","live_exchange_key_permissions_unverified",
       "actual_independent_watchdog_fault_injection_unverified",
       "real_paper_testnet_soak_and_fill_reconciliation_unverified",
       "independent_security_review_and_rollback_unverified",
    ]
    return {
       "schema":"proculus_v10_readonly_preflight",
       "observed_utc":datetime.now(timezone.utc).isoformat(),
       "repo_module_presence":present,"declared_config_only":declared,
       "blockers":blockers,"production_authorized":False,
       "live_exchange_order_endpoint":False,"paper_only_observer":True,
       "warning":"Source inspection is not live-deployment certification",
    }
