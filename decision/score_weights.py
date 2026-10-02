from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CONFIG_FILE = Path(__file__).resolve().parents[1] / "config.json"


def _load_config() -> dict[str, Any]:
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def resolve_decision_weights() -> dict[str, float]:
    raw = _load_config().get("decision_weights")
    raw = raw if isinstance(raw, dict) else {}
    values = {
        "ai": max(0.0, float(raw.get("ai", 0.0) or 0.0)),
        "tech": max(0.0, float(raw.get("tech", 1.0) or 0.0)),
        "sent": max(0.0, float(raw.get("sent", 0.0) or 0.0)),
    }
    total = sum(values.values())
    if total <= 0.0:
        return {"ai": 0.0, "tech": 1.0, "sent": 0.0}
    return {key: value / total for key, value in values.items()}
