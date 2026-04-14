"""
channels/access_control.py

Per-channel user allowlist.

Controls which Telegram user IDs are allowed to send messages that reach
the agent pipeline.  Unapproved users get a polite rejection; they are
never silently dropped (that would make debugging painful).

Configuration is stored in settings.json under the key "channel_allowlist":
{
  "telegram": ["123456789", "987654321"],
  "discord":  []
}

An empty list means ALL users are allowed (open mode — fine for a private
bot only you know the token for, but you should add your ID for safety).

Thread-safe: reads and writes go through a threading.Lock so the Telegram
polling thread and the GUI settings thread cannot race.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.settings import Settings
    from channels.inbound_message import Channel

log = logging.getLogger("MyAIAgentHub.access_control")

_SETTINGS_KEY = "channel_allowlist"


class AccessControl:
    """
    Manages per-channel user ID allowlists.

    Usage
    -----
    ac = AccessControl(settings)
    if not ac.is_allowed(Channel.TELEGRAM, str(update.effective_user.id)):
        return  # ignore message
    """

    def __init__(self, settings: "Settings") -> None:
        self._settings = settings
        self._lock = threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────────

    def is_allowed(self, channel: "Channel", user_id: str) -> bool:
        """
        Return True if user_id is permitted on this channel.
        An empty allowlist means OPEN (everyone allowed).
        """
        allowlist = self._get_list(channel.value)
        if not allowlist:
            return True  # open mode
        allowed = str(user_id) in [str(uid) for uid in allowlist]
        if not allowed:
            log.warning(
                "Access denied: user_id=%s on channel=%s (not in allowlist of %d)",
                user_id, channel.value, len(allowlist),
            )
        return allowed

    def get_allowlist(self, channel_name: str) -> list[str]:
        """Return allowlist for a channel (empty = open)."""
        return self._get_list(channel_name)

    def add_user(self, channel_name: str, user_id: str) -> None:
        """Add a user ID to a channel's allowlist."""
        with self._lock:
            data = self._load()
            lst = data.setdefault(channel_name, [])
            uid = str(user_id)
            if uid not in [str(x) for x in lst]:
                lst.append(uid)
                self._save(data)
                log.info("Access control: added user_id=%s to channel=%s", uid, channel_name)

    def remove_user(self, channel_name: str, user_id: str) -> None:
        """Remove a user ID from a channel's allowlist."""
        with self._lock:
            data = self._load()
            lst = data.get(channel_name, [])
            data[channel_name] = [x for x in lst if str(x) != str(user_id)]
            self._save(data)
            log.info("Access control: removed user_id=%s from channel=%s", user_id, channel_name)

    def set_open(self, channel_name: str) -> None:
        """Set channel to open mode (empty allowlist = all users allowed)."""
        with self._lock:
            data = self._load()
            data[channel_name] = []
            self._save(data)

    def status(self) -> dict:
        """Return a summary dict for the GUI settings panel."""
        data = self._load()
        return {
            ch: {"mode": "open" if not lst else "restricted", "count": len(lst)}
            for ch, lst in data.items()
        }

    # ── Internal ───────────────────────────────────────────────────────────

    def _load(self) -> dict:
        raw = self._settings.get(_SETTINGS_KEY, {})
        if not isinstance(raw, dict):
            return {}
        return raw

    def _save(self, data: dict) -> None:
        self._settings.set(_SETTINGS_KEY, data)

    def _get_list(self, channel_name: str) -> list[str]:
        with self._lock:
            data = self._load()
            return [str(x) for x in data.get(channel_name, [])]
