# -*- coding: utf-8 -*-
"""
report_generator.py

Haftalik veya gunluk performans raporlari olusturmak icin kullanilabilecek
bir yardimci arac. Bu script, ``trade_log.json`` filesini okuyarak
 temel metrikleri (kazanc/zarar, win‑rate, en buyuk dusus, trade
sayisi) accountlar ve bunlari basit bir Markdown rapor olarak yazdirir.

Kullanim:
    (venv) python report_generator.py --output report.md

Opsiyonel parametreler ile belirli bir tarih araligi, symbol filtresi
veya metrikler belirlenebilir. Varsayilan olarak tum veriyi raporlar.
"""
from __future__ import annotations


from core.exceptions import BEST_EFFORT_EXCEPTIONS
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


def generate_report(trade_log_path: Path, output_path: Path, start_date: str | None = None, end_date: str | None = None) -> None:
    """Trade log filesini okuyup performans raporu uret.

    Args:
        trade_log_path: trade_log.json filesinin yolu.
        output_path: Yazilacak rapor (Markdown) filesi.
        start_date: ISO tarih (YYYY‑MM‑DD) baslangic filtresi.
        end_date: ISO tarih (YYYY‑MM‑DD) bitis filtresi.
    """
    if not trade_log_path.exists():
        print(f"[report_generator] trade_log.json not found: {trade_log_path}")
        return
    try:
        with trade_log_path.open('r', encoding='utf-8') as f:
            trades = json.load(f)
    except BEST_EFFORT_EXCEPTIONS as e:
        print(f"[report_generator] trade_log.json could not be read: {e}")
        return
    if not isinstance(trades, list):
        print("[report_generator] trade_log.json beklenen formatta degil.")
        return
    # JSON listeden DataFrame
    df = pd.DataFrame(trades)
    if df.empty:
        print("[report_generator] Raporlanacak veri yok.")
        return
    # Tarih filtresi uygula
    if start_date:
        try:
            start_dt = datetime.fromisoformat(start_date)
            df = df[df['timestamp_open'] >= start_dt.isoformat()]
        except BEST_EFFORT_EXCEPTIONS as exc:
            logger.warning("Invalid start date filter %r: %s", start_date, exc)
    if end_date:
        try:
            end_dt = datetime.fromisoformat(end_date)
            df = df[df['timestamp_open'] <= end_dt.isoformat()]
        except BEST_EFFORT_EXCEPTIONS as exc:
            logger.warning("Invalid end date filter %r: %s", end_date, exc)
    # PnL accountla (varsayim: pnl_abs alani var ve USDT cinsinden)
    total_pnl = df['pnl_abs'].sum() if 'pnl_abs' in df else 0.0
    win_trades = df[df['pnl_abs'] > 0].shape[0]
    lose_trades = df[df['pnl_abs'] <= 0].shape[0]
    total_trades = df.shape[0]
    win_rate = (win_trades / total_trades) * 100 if total_trades > 0 else 0.0
    # En buyuk geri cekilme (max drawdown) sutunu varsa
    max_dd = df['max_drawdown_pct'].max() if 'max_drawdown_pct' in df else None
    # Rapora yaz
    lines = []
    lines.append(f"# Performans Raporu\n")
    lines.append(f"Toplam Islem Sayisi: {total_trades}")
    lines.append(f"Toplam PnL (USDT): {total_pnl:.2f}")
    lines.append(f"Kazanan Islem Sayisi: {win_trades}")
    lines.append(f"Kaybeden Islem Sayisi: {lose_trades}")
    lines.append(f"Win‑Rate: {win_rate:.2f}%")
    if max_dd is not None:
        lines.append(f"En Buyuk Cekilme (Max DD %): {max_dd:.2f}")
    lines.append("\n## Sembol Bazli Ozet\n")
    if 'symbol' in df:
        grouped = df.groupby('symbol')['pnl_abs'].sum().reset_index()
        for _, row in grouped.iterrows():
            lines.append(f"* {row['symbol']}: {row['pnl_abs']:.2f} USDT toplam PnL")
    # Yaz
    try:
        output_path.write_text('\n'.join(lines), encoding='utf-8')
        print(f"[report_generator] Rapor created: {output_path}")
    except BEST_EFFORT_EXCEPTIONS as e:
        print(f"[report_generator] Rapor yazilamadi: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trade performans raporu olustur.')
    parser.add_argument('--output', type=str, default='trade_report.md', help='Yazilacak rapor filesi')
    parser.add_argument('--start', type=str, default=None, help='Baslangic tarihi (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='Bitis tarihi (YYYY-MM-DD)')
    args = parser.parse_args()
    trade_log_path = Path('trade_log.json')
    out_path = Path(args.output)
    generate_report(trade_log_path, out_path, args.start, args.end)
