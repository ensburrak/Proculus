"""
anomaly_detector.py
-------------------

Fiyat anomalisi tespit modulu.  Son bilinen fiyat ile update fiyat
arasindaki farkin ATR'nin belirli bir katini asip asmadigini check
eder.  Esik asilirsa ``True`` return ve circuit_breaker'a anomali
kaydi yapilmalidir.

Kullanim::

    from anomaly_detector import is_anomalous_price

    if is_anomalous_price(last_price=100.0, current_price=115.0, atr_value=3.0, threshold=3.0):
        record_anomaly()
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def is_anomalous_price(
    last_price: float | None,
    current_price: float | None,
    atr_value: float | None,
    threshold: float = 3.0,
) -> bool:
    """Son fiyat ile update fiyat arasinda anormal bir sapma olup olmadigini check eder.

    Fiyat farki ``threshold * ATR`` valueini asarsa anomali olarak valuelendirilir.
    ATR mevcut degilse, percentsel change uzerinden yedek check yapilir
    (default: %10 uzeri anomali).

    Parameters
    ----------
    last_price:
        Sembol icin bilinen son fiyat.  ``None`` ise check yapilamaz,
        ``False`` return.
    current_price:
        update (yeni gelen) fiyat.  ``None`` ise check yapilamaz.
    atr_value:
        Average True Range valuei.  ``None`` veya ``<= 0`` ise percentsel
        fallback kullanilir.
    threshold:
        ATR carpani.  Fiyat farki ``threshold * atr_value`` uzerindeyse
        anomali sayilir.  Varsayilan 3.0.

    Returns
    -------
    bool
        ``True`` → fiyat anomalisi tespit edildi, ``False`` → normal.
    """
    if last_price is None or current_price is None:
        return False

    try:
        last_f = float(last_price)
        curr_f = float(current_price)
    except (TypeError, ValueError):
        return False

    if last_f <= 0 or curr_f <= 0:
        return False

    price_diff = abs(curr_f - last_f)

    # ATR tabanli check
    if atr_value is not None:
        try:
            atr_f = float(atr_value)
        except (TypeError, ValueError):
            atr_f = 0.0

        if atr_f > 0:
            if price_diff > threshold * atr_f:
                pct = (price_diff / last_f) * 100
                log.warning(
                    "Fiyat anomalisi: %.6f -> %.6f (fark=%.6f > %.1f*ATR=%.6f) [%%%.2f]",
                    last_f, curr_f, price_diff, threshold, atr_f, pct,
                )
                return True
            return False

    # Fallback: ATR yoksa percentsel change checku (%10 esik)
    pct_change = (price_diff / last_f) * 100
    PCT_THRESHOLD = 10.0
    if pct_change > PCT_THRESHOLD:
        log.warning(
            "Fiyat anomalisi (ATR yok, yuzdesel): %.6f -> %.6f (%%%.2f > %%%.1f)",
            last_f, curr_f, pct_change, PCT_THRESHOLD,
        )
        return True

    return False
