"""
meta_labeler.py
----------------

Bu stub modul, meta‑labelling veya ikinci gorus modeli icin basit bir
fonksiyon sunar.  Gercek implementasyon, AI/teknik/sentiment skorlarini
kullanarak bir trade'in kazancli olup olmayacagina dair olasilik tahmini
uretebilir.  Mevcut projenizde bu file eksik oldugu icin import
errorlarini engellemek amaciyla basit bir versiyon eklenmistir.

Fonksiyonlar:
    compute_meta_probability(master_conf, ai_score, tech_score, sent_score, default)
        Verilen skorlari kullanarak 0..1 araliginda bir meta olasilik return.

Kullanim:
    from .meta_labeler import compute_meta_probability
"""
from __future__ import annotations

from core.exceptions import BEST_EFFORT_EXCEPTIONS
import logging
import os
from dataclasses import dataclass
from typing import Optional

from atomic_io import safe_read_json
from runtime_paths import PROJECT_ROOT, get_artifact_path


_WEIGHTS_PATH = get_artifact_path("logistic_weights.json")
LOGISTIC_WEIGHTS_SCHEMA = "meta-logistic-weights-v1"
_REQUIRED_WEIGHT_KEYS = ("w0", "w_ai", "w_tech", "w_sent")
_REQUIRED_FEATURE_COLUMNS = ("ai_score", "tech_score", "sent_score")


def _weight_candidates() -> list:
    return [
        _WEIGHTS_PATH,
        PROJECT_ROOT / "state" / "artifacts" / _WEIGHTS_PATH.name,
        PROJECT_ROOT / _WEIGHTS_PATH.name,
    ]


def _load_logistic_weights() -> dict:
    """Best-effort atomic loader for logistic meta-label weights."""
    for path in _weight_candidates():
        if not path.exists():
            continue
        data = safe_read_json(path, default={})
        if isinstance(data, dict):
            return data
    return {}


def _validated_calibrated_weights() -> tuple[dict | None, str]:
    try:
        data = _load_logistic_weights()
    except (OSError, ValueError, TypeError):
        return None, "logistic_weights_unreadable"
    if not isinstance(data, dict) or not data:
        return None, "missing_logistic_weights"
    if data.get("schema") != LOGISTIC_WEIGHTS_SCHEMA:
        return None, "invalid_or_legacy_logistic_weights"
    if not all(isinstance(data.get(k), (int, float)) for k in _REQUIRED_WEIGHT_KEYS):
        return None, "invalid_logistic_weight_values"
    try:
        trained_on = int(data.get("trained_on"))
    except (TypeError, ValueError):
        return None, "invalid_logistic_weight_provenance"
    if trained_on <= 0:
        return None, "invalid_logistic_weight_provenance"
    label_classes = data.get("label_classes")
    if not isinstance(label_classes, list) or {int(v) for v in label_classes if isinstance(v, (int, float))} != {0, 1}:
        return None, "invalid_logistic_weight_label_classes"
    feature_columns = data.get("feature_columns")
    if not isinstance(feature_columns, list) or tuple(str(v) for v in feature_columns) != _REQUIRED_FEATURE_COLUMNS:
        return None, "invalid_logistic_weight_features"
    if not str(data.get("source") or "").strip() or not str(data.get("created_at_utc") or "").strip():
        return None, "invalid_logistic_weight_provenance"
    return data, "ready"


def _has_calibrated_weights() -> bool:
    """Return True only when calibrated logistic weights are available on disk.

    The historic ``IS_STUB_IMPLEMENTATION`` flag was hard-coded to ``True`` even
    after a real logistic regression and calibrated weight file landed. That
    forced ``decision.imports`` onto the identity fallback path and silently
    disabled meta-labelling in production. The flag is now derived from the
    actual artefact: stub semantics only apply when the weights file is missing
    or unreadable, which matches the documented contract ("set
    ENABLE_STUB_META_LABELER=1 to opt into the unweighted average").
    """
    data, _reason = _validated_calibrated_weights()
    return data is not None


def get_meta_labeling_status() -> dict:
    weights, reason = _validated_calibrated_weights()
    if weights is None:
        return {
            "active": False,
            "calibrated": False,
            "reason": reason,
            "schema": LOGISTIC_WEIGHTS_SCHEMA,
        }
    return {
        "active": True,
        "calibrated": True,
        "reason": "ready",
        "schema": LOGISTIC_WEIGHTS_SCHEMA,
        "source": weights.get("source"),
        "trained_on": weights.get("trained_on"),
        "label_classes": weights.get("label_classes"),
        "feature_columns": weights.get("feature_columns"),
    }


IS_STUB_IMPLEMENTATION = not _has_calibrated_weights()


@dataclass
class MetaLabelResult:
    """Result of meta-label computation with separate probability and suppression."""
    probability: float
    suppression_active: bool
    multiplier: float
    reason: str


def compute_meta_probability(
    master_conf: float,
    ai_score: Optional[float],
    tech_score: Optional[float],
    sent_score: Optional[float],
    default: float = 1.0,
) -> float:
    """
    AI, teknik analiz ve sentiment skorlarina gore bir meta olasilik
    accountlar.  Varsayilan implementasyon olarak, proje kokunde bulunan
    ``logistic_weights.json`` filesindaki lojistik regresyon katsayilari
    kullanilarak sigmoid fonksiyonu uzerinden bir olasilik accountlanir.
    Bu file bulunamaz veya okunamazsa, mevcut skorlarin basit ortalamasi
    alinir.  Eger hicbir skor mevcut degilse ``default`` valuei return.

    Lojistik fonksiyon:

        p = 1 / (1 + exp(-(w0 + w_ai*ai_score + w_tech*tech_score + w_sent*sent_score)))

    Burada ``w0`` sabit terim (intercept), ``w_ai`` AI skorunun katsayisi,
    ``w_tech`` teknik skor katsayisi ve ``w_sent`` sentiment skor katsayisidir.

    Args:
        master_conf (float): Birlesik guven skoru (0..1).  Bu parametre
            burada kullanilmaz ancak imza icin tutulmustur.
        ai_score (float|None): AI bileseninin skoru (0..1)
        tech_score (float|None): Teknik bilesenin skoru (0..1)
        sent_score (float|None): Sentiment bilesenin skoru (0..1)
        default (float): Mevcut skor yoksa return default value.

    Returns:
        float: 0..1 araliginda meta etiket olasiligi.
    """
    # Kullanilabilir skorlari topla
    scores: dict[str, float] = {}
    try:
        if ai_score is not None:
            scores["ai"] = float(ai_score)
    except BEST_EFFORT_EXCEPTIONS as _exc:
        logging.getLogger(__name__).warning("compute_meta_probability score cast error: %s", _exc)
    try:
        if tech_score is not None:
            scores["tech"] = float(tech_score)
    except BEST_EFFORT_EXCEPTIONS as _exc:
        logging.getLogger(__name__).warning("compute_meta_probability score cast error: %s", _exc)
    try:
        if sent_score is not None:
            scores["sent"] = float(sent_score)
    except BEST_EFFORT_EXCEPTIONS as _exc:
        logging.getLogger(__name__).warning("compute_meta_probability score cast error: %s", _exc)

    # Eger hic skor yoksa default valuei return
    if not scores:
        try:
            return float(default)
        except BEST_EFFORT_EXCEPTIONS:
            return 1.0

    weights, _reason = _validated_calibrated_weights()

    if weights:
        # Lojistik regresyon katsayilari kullanarak olasilik accountla
        w0 = float(weights.get("w0", 0.0))
        # Katsayilar fileda varsa ilgili skora uygula, yoksa 0 kabul et
        w_ai = float(weights.get("w_ai", 0.0))
        w_tech = float(weights.get("w_tech", 0.0))
        w_sent = float(weights.get("w_sent", 0.0))
        # Eksik skorlar icin sifir katsayilari yok sayilir
        z = w0
        z += w_ai * scores.get("ai", 0.0)
        z += w_tech * scores.get("tech", 0.0)
        z += w_sent * scores.get("sent", 0.0)
        # Sigmoid fonksiyonu (O13 FIX: overflow korumasi)
        try:
            import math
            z = max(-20.0, min(20.0, z))
            p = 1.0 / (1.0 + math.exp(-z))
        except BEST_EFFORT_EXCEPTIONS:
            p = 0.5
        # Aralik disina tasmamasi icin kliple
        if p < 0.0:
            p = 0.0
        elif p > 1.0:
            p = 1.0
        return p

    if str(os.getenv("ENABLE_STUB_META_LABELER", "") or "").strip().lower() not in {"1", "true", "yes", "on"}:
        try:
            return float(default)
        except BEST_EFFORT_EXCEPTIONS:
            return 1.0

    # Explicitly enabled stub fallback: simple score average.
    avg = sum(scores.values()) / len(scores)
    if avg < 0.0:
        avg = 0.0
    elif avg > 1.0:
        avg = 1.0
    return avg


def compute_meta_label(
    master_conf: float,
    ai_score: Optional[float],
    tech_score: Optional[float],
    sent_score: Optional[float],
    default: float = 1.0,
    suppression_threshold: float = 0.40,
    suppression_floor: float = 0.65,
) -> MetaLabelResult:
    """
    Compute meta-label with separated probability and suppression logic.

    Calls ``compute_meta_probability()`` to obtain the raw probability, then
    applies suppression/modulation:

    - If ``prob < suppression_threshold`` the signal is suppressed and the
      multiplier is clamped to ``suppression_floor``.
    - Otherwise the multiplier ramps linearly from ``suppression_floor`` to
      ``1.0`` as probability increases.

    Args:
        master_conf: Combined confidence score (0..1).
        ai_score: AI component score (0..1) or None.
        tech_score: Technical component score (0..1) or None.
        sent_score: Sentiment component score (0..1) or None.
        default: Fallback probability when no scores are available.
        suppression_threshold: Probability below which suppression activates.
        suppression_floor: Minimum multiplier applied during suppression.

    Returns:
        MetaLabelResult with probability, suppression flag, multiplier, and reason.
    """
    prob = compute_meta_probability(master_conf, ai_score, tech_score, sent_score, default=default)
    prob = max(0.0, min(1.0, float(prob)))

    # If default was returned (no scores), pass through without suppression
    if ai_score is None and tech_score is None and sent_score is None:
        return MetaLabelResult(
            probability=prob,
            suppression_active=False,
            multiplier=prob,
            reason="no_scores_available",
        )

    raw_scores: list[float] = []
    for value in (ai_score, tech_score, sent_score):
        if value is None:
            continue
        try:
            raw_scores.append(max(0.0, min(1.0, float(value))))
        except BEST_EFFORT_EXCEPTIONS:
            continue
    evidence_prob = sum(raw_scores) / len(raw_scores) if raw_scores else prob
    suppression_metric = min(prob, evidence_prob)

    if suppression_metric < suppression_threshold:
        return MetaLabelResult(
            probability=prob,
            suppression_active=True,
            multiplier=suppression_floor,
            reason=(
                f"suppression_metric {suppression_metric:.3f} < threshold {suppression_threshold} "
                f"(prob={prob:.3f}, evidence={evidence_prob:.3f})"
            ),
        )

    multiplier = suppression_floor + (1.0 - suppression_floor) * prob
    return MetaLabelResult(
        probability=prob,
        suppression_active=False,
        multiplier=multiplier,
        reason=f"modulated: prob={prob:.3f} multiplier={multiplier:.3f}",
    )
