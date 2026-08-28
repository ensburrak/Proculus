from core.exceptions import BEST_EFFORT_EXCEPTIONS
import json
import os
import subprocess
import sys

from runtime_paths import get_symbols_okx_path

# Proje dizinini path'e ekle
sys.path.insert(0, os.path.abspath("."))


def _build_backtest_command(base_symbols):
    formatted_symbols = [f"{sym}:USDT" for sym in base_symbols]
    return [
        sys.executable,
        "-m",
        "backtesting.runtime",
        "--days",
        "365",
        "--mode",
        "event_driven",
        "--use-ai",
        "--symbols",
        *formatted_symbols,
    ]


def _build_startup_lines(symbol_count):
    return [
        "=" * 70,
        "AUTO TRADER BOT - 1 SENELIK KAPSAMLI AI BACKTEST (TUM SEMBOLLER)",
        "=" * 70,
        f"Test edilecek sembol sayisi: {symbol_count}",
        "Veri Araligi: Son 365 Gun (1 Yil)",
        "Aktif Katmanlar:",
        "  - AI/ML Karar Motoru (Transformer + RL)",
        "  - LLM Sagduyu Motoru (ChatGPT + DeepSeek - Replay)",
        "  - Gercekci Maliyet (Komisyon + Slippage + Funding Fee)",
        "  - Dinamik Risk (ATR Tabanli SL/TP, Kademeli Cikis)",
        "=" * 70,
        "Test basliyor, bu islem AI modullerini de kullandigi icin uzun surebilir...",
        "=" * 70,
    ]


def main():
    # 1. Sembolleri yukle
    try:
        with get_symbols_okx_path().open("r", encoding="utf-8") as f:
            base_symbols = json.load(f)
    except BEST_EFFORT_EXCEPTIONS as e:
        print(f"Hata - Semboller yuklenemedi: {e}")
        return

    cmd = _build_backtest_command(base_symbols)
    formatted_symbols = cmd[cmd.index("--symbols") + 1 :]

    for line in _build_startup_lines(len(formatted_symbols)):
        print(line)

    # 2. Ayni interpreter ile calistir; PATH'teki farkli/kirik python'a dusme.
    completed = subprocess.run(cmd)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
