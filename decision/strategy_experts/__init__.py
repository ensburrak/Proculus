from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ExpertSignal:
    setup_id: str
    direction: str
    confidence: float
    strategy: str
    reasoning: list[str]
    max_leverage: int = 1
    stop_atr_mult: float = 1.5
    tp_r_target: float = 2.5
    max_hold_hours: float = 48.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def f(value: Any) -> float | None:
    try:
        parsed=float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed else None


def tf(item: dict[str, Any], name: str) -> dict[str, Any]:
    mtf=item.get("mtf_features")
    if not isinstance(mtf, dict):
        return {}
    row=mtf.get(name)
    return row if isinstance(row, dict) else {}


def series(row: dict[str, Any], key: str) -> list[float]:
    raw=row.get(key)
    if not isinstance(raw,(list,tuple)):
        return []
    out=[]
    for value in raw:
        parsed=f(value)
        if parsed is not None:
            out.append(parsed)
    return out
