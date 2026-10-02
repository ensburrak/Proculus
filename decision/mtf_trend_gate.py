from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MtfTrendGateResult:
    allowed: bool
    reason: str
    audit: dict[str, Any]


def _f(value: Any) -> float | None:
    try:
        parsed = float(value)
        return parsed if parsed == parsed else None
    except (TypeError, ValueError):
        return None


def _series(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return []
    out = []
    for raw in value:
        parsed = _f(raw)
        if parsed is not None:
            out.append(parsed)
    return out


def _snap(item: dict[str, Any], tf: str) -> dict[str, Any]:
    mtf = item.get("mtf_features")
    features = mtf.get(tf, {}) if isinstance(mtf, dict) and isinstance(mtf.get(tf), dict) else {}
    closes = _series(features.get("recent_closes"))
    lows = _series(features.get("recent_lows"))
    highs = _series(features.get("recent_highs"))
    return {
        "close": _f(features.get("close")) or (closes[-1] if closes else None),
        "prev_close": _f(features.get("prev_close")) or (closes[-2] if len(closes) >= 2 else None),
        "ema_fast": _f(features.get("ema_fast") or features.get("ema20")),
        "ema_slow": _f(features.get("ema_slow") or features.get("ema50")),
        "ema200": _f(features.get("ema200") or features.get("ema_200")),
        "macd_hist": _f(features.get("macd_hist") or features.get("macd")),
        "recent_low": min(lows[-5:]) if lows else None,
        "recent_high": max(highs[-5:]) if highs else None,
        "closes": closes,
    }


def _aligned(s: dict[str, Any], side: str, momentum: bool = False) -> bool:
    close, fast, slow = s.get("close"), s.get("ema_fast"), s.get("ema_slow")
    if close is None or fast is None or slow is None:
        return False
    ema200 = s.get("ema200")
    macd = s.get("macd_hist")
    if side == "long":
        ok = close >= slow and fast > slow
        if ema200 is not None:
            ok = ok and close > ema200
        if momentum and macd is not None:
            ok = ok and macd >= 0
        return ok
    if side == "short":
        ok = close <= slow and fast < slow
        if ema200 is not None:
            ok = ok and close < ema200
        if momentum and macd is not None:
            ok = ok and macd <= 0
        return ok
    return False


def _resumed(s: dict[str, Any], side: str) -> bool:
    close, prev, fast, slow = s.get("close"), s.get("prev_close"), s.get("ema_fast"), s.get("ema_slow")
    if close is None or fast is None or slow is None:
        return False
    tol = 0.004
    if side == "long":
        low = s.get("recent_low")
        pullback = (low is not None and low <= fast * (1 + tol)) or close <= fast * (1 + tol)
        resume = close >= fast and (prev is None or close > prev)
        return bool(pullback and resume)
    high = s.get("recent_high")
    pullback = (high is not None and high >= fast * (1 - tol)) or close >= fast * (1 - tol)
    resume = close <= fast and (prev is None or close < prev)
    return bool(pullback and resume)


def evaluate_mtf_trend_gate(*, side: str, item: dict[str, Any]) -> MtfTrendGateResult:
    h4, h1 = _snap(item, "4h"), _snap(item, "1h")
    lower_tf = "5m"
    lower = _snap(item, lower_tf)
    if lower.get("close") is None:
        lower_tf = "15m"
        lower = _snap(item, lower_tf)

    ready = all(s.get("close") is not None and s.get("ema_fast") is not None and s.get("ema_slow") is not None for s in (h4, h1, lower))
    audit = {"h4": h4, "h1": h1, "lower_tf": lower_tf, "lower": lower, "mode": "full_mtf" if ready else "insufficient_mtf"}
    if not ready:
        return MtfTrendGateResult(False, "mtf_missing", audit)

    macro = _aligned(h4, side, momentum=True)
    structure = _aligned(h1, side)
    resumption = _resumed(lower, side)
    audit.update({"macro_ok": macro, "structure_ok": structure, "resumption_ok": resumption})
    return MtfTrendGateResult(bool(macro and structure and resumption), "mtf_pass" if macro and structure and resumption else "mtf_block", audit)
