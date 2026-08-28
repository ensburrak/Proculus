"""
user_control.py
---------------

[FIX-2.1+2.3+2.4] Bu modül yeniden yazıldı:
- Legacy handler silindi — tüm komutlar bot/command_handler'a delege edilir.
- is_paused artık state_manager property'si (merkezi state).
- Long-polling timeout=30 (anında komut tepkisi).
- send_message telegram_notifier üzerinden (rate limiter otomatik).
"""

from __future__ import annotations


from core.exceptions import BEST_EFFORT_EXCEPTIONS
import os
import time
import json
import logging
import threading
import requests
from typing import Optional, Dict, Any, Callable
from pathlib import Path
from runtime_paths import METRICS_DIR, get_trade_log_path

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent
RUNTIME_STATUS_FILE = METRICS_DIR / "runtime_status.json"
TRADE_LOG_FILE = get_trade_log_path()


class UserControl:
    """
    Telegram Bot API üzerinden polling yaparak komutları dinler.
    [FIX-2.1] Tüm komutlar bot/command_handler'a delege edilir.
    [FIX-2.3] is_paused state_manager property'si.
    """
    
    def __init__(self, token: str, chat_id: str, poll_interval: float = 2.0):
        self.token = token
        self.chat_id = chat_id
        if not self.token:
            logger.warning("[UserControl] TELEGRAM_BOT_TOKEN eksik!")
        
        self.poll_interval = poll_interval
        self.offset: Optional[int] = None
        self._is_paused_fallback = False
        self._stop_event = threading.Event()
        
        # Manuel emir kuyruğu
        self.manual_orders_queue = []
        self.force_close_all = False
        self.reload_config_requested = False
        self.market_request = None
        self.blacklist_req = None
        
        # Cache
        self.latest_balance = 0.0
        self.open_trades_snapshot = {}

    # [FIX-2.3] is_paused property — state_manager ile merkezi
    @property
    def is_paused(self) -> bool:
        try:
            from state_manager import is_paused
            return is_paused()
        except ImportError:
            return self._is_paused_fallback

    @is_paused.setter
    def is_paused(self, value: bool):
        try:
            from state_manager import request_pause, clear_pause
            if value:
                request_pause("UserControl")
            else:
                clear_pause()
        except ImportError:
            self._is_paused_fallback = value

    def send_message(self, text: str) -> None:
        """[FIX-4] telegram_notifier üzerinden gönderir — rate limiter otomatik."""
        try:
            from telegram_notifier import send_message as tg_send
            tg_send(text, parse_mode="HTML")
        except ImportError:
            if not self.token or not self.chat_id:
                return
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = {"chat_id": self.chat_id, "text": text, "parse_mode": "Markdown"}
            try:
                requests.post(url, json=data, timeout=5)
            except BEST_EFFORT_EXCEPTIONS as e:
                logger.warning("[UserControl] Mesaj gönderme hatası: %s", e)

    def _read_runtime_status(self) -> Dict[str, Any]:
        """Runtime status dosyasını okur."""
        try:
            if RUNTIME_STATUS_FILE.exists():
                return json.loads(RUNTIME_STATUS_FILE.read_text(encoding="utf-8"))
        except BEST_EFFORT_EXCEPTIONS:
            pass
        return {}

    def _handle_command(self, cmd: str, args: str) -> str:
        """[FIX-2.1+2.1B] TEK YETKİLİ: bot/command_handler._handle_user_command
        Fallback: import başarısızsa minimal komut seti çalışır."""
        try:
            from bot.command_handler import _handle_user_command
            return _handle_user_command(cmd, args)
        except ImportError:
            logger.warning("[UserControl] bot.command_handler import edilemedi — minimal fallback aktif")
            return self._minimal_fallback_handler(cmd, args)

    def _minimal_fallback_handler(self, cmd: str, args: str) -> str:
        """[FIX-2.1B] bot.command_handler yoksa temel komutlar hâlâ çalışır."""
        c = cmd.strip().lower().lstrip("/")
        if c == "help":
            return "📚 Mevcut komutlar: /status, /pause, /resume, /help\n⚠️ Tam komut işleyici yüklenemedi."
        elif c == "status":
            status = self._read_runtime_status()
            return f"🤖 Bot: {'Çalışıyor' if status.get('running') else 'Bilinmiyor'}\n⏸ Paused: {self.is_paused}"
        elif c == "pause":
            self.is_paused = True
            return "⏸ Bot duraklatıldı."
        elif c == "resume":
            self.is_paused = False
            return "▶️ Bot devam ediyor."
        return "❓ Bilinmeyen komut. /help deneyin."

    def process_commands(self) -> None:
        """
        Main loop tarafından periyodik olarak çağrılır.
        Telegram'dan yeni mesaj var mı diye bakar ve varsa yanıtlar.
        """
        if not self.token:
            return

        base_url = f"https://api.telegram.org/bot{self.token}"
        try:
            # [FIX-2.4] Long-polling — 30sn timeout ile anında tepki
            params = {"timeout": 30, "offset": self.offset}
            resp = requests.get(f"{base_url}/getUpdates", params=params, timeout=35)
            data = resp.json()
            
            if not data.get("ok"):
                return
                
            result = data.get("result", [])
            for update in result:
                self.offset = update.get("update_id", 0) + 1
                msg = update.get("message", {})
                chat_id = msg.get("chat", {}).get("id")
                text = msg.get("text", "").strip()
                
                # Güvenlik: sadece yetkili chat_id
                if not self.chat_id or chat_id is None:
                    continue
                try:
                    if int(chat_id) != int(self.chat_id):
                        continue
                except (TypeError, ValueError):
                    continue
                
                if text.startswith("/"):
                    parts = text.split(maxsplit=1)
                    cmd = parts[0]
                    args = parts[1] if len(parts) > 1 else ""
                    
                    response_text = self._handle_command(cmd, args)
                    self.send_message(response_text)
                    
        except BEST_EFFORT_EXCEPTIONS as e:
            logger.debug("Telegram poll error: %s", e)
