from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .controller_batch_service import BatchDecisionRuntime
from .official_pipeline import process_symbol_decision

OFFICIAL_DECISION_PIPELINE = "proculus_pipeline_v2"
TRADING_SYSTEM_PROMPT = "Deterministic strategy experts own direction; AI is advisory quality-only."


def build_batch_prompt(batch_items: list[dict[str, Any]]) -> str:
    return json.dumps(batch_items, ensure_ascii=False, default=str)


def is_llm_tech_only_mode() -> bool:
    return False


def build_batch_runtime(*, controller_file: Path) -> BatchDecisionRuntime:
    return BatchDecisionRuntime(
        process_symbol_decision=process_symbol_decision,
        config_file=controller_file.parent / "config.json",
    )
