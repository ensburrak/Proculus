from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RiskConfig:
    daily_loss_limit_pct: float = 0.005
    max_portfolio_risk_pct: float = 0.02
    max_leverage: int = 2
    max_wallet_pct: float = 0.20


def load_risk_config() -> RiskConfig:
    try:
        cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cfg = {}
    risk = cfg.get("risk") if isinstance(cfg, dict) and isinstance(cfg.get("risk"), dict) else {}
    trading = cfg.get("trading") if isinstance(cfg, dict) and isinstance(cfg.get("trading"), dict) else {}
    return RiskConfig(
        daily_loss_limit_pct=float(risk.get("daily_loss_limit_pct", 0.005) or 0.005),
        max_portfolio_risk_pct=float(risk.get("max_portfolio_risk_pct", 0.02) or 0.02),
        max_leverage=min(2, int(trading.get("max_leverage", 2) or 2)),
        max_wallet_pct=float(risk.get("max_wallet_pct", 0.20) or 0.20),
    )
