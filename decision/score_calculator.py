from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .score_weights import resolve_decision_weights
from .tech_score import base_decision_from_ema, enhanced_tech_score

CONFIG_FILE = Path(__file__).resolve().parents[1] / "config.json"


def load_decision_weights(*args: Any, **kwargs: Any) -> None:
    return None


def get_decision_weights() -> dict[str, float]:
    return resolve_decision_weights()


def is_expert_score_mode() -> bool:
    return True


def is_rl_in_ai() -> bool:
    return False


def compute_symbol_signal(*, item: dict[str, Any], ai_part: dict[str, Any] | None = None, **kwargs: Any):
    ta = item.get("ta_pack") if isinstance(item.get("ta_pack"), dict) else {}
    ema = ta.get("ema") if isinstance(ta.get("ema"), dict) else {}
    base = str(ta.get("base_decision") or base_decision_from_ema(ema.get("fast"), ema.get("slow")))
    tech = enhanced_tech_score(ta, base)
    state = SimpleNamespace(
        sym=str(item.get("symbol") or ""),
        item=item,
        ta=ta,
        base_dec=base,
        tech_score=float(tech.get("score", 0.5)),
        ai_part=dict(ai_part or {}),
        master_raw=float(tech.get("score", 0.5)),
        master=float(tech.get("score", 0.5)),
    )
    return state, None


def persist_last_decisions(symbol_inputs: list[dict[str, Any]], out: dict[str, dict[str, Any]], logger: Any = None) -> None:
    try:
        target = CONFIG_FILE.parent / "metrics" / "last_decisions.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        return


def log_decision_summary(out: dict[str, dict[str, Any]], symbol_inputs: list[dict[str, Any]], logger: Any = None) -> None:
    if logger is not None:
        logger.info("[V2] decision batch completed symbols=%s", len(out))
