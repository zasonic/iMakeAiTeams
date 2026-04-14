"""
channels/inbound_message.py

Normalized inbound message dataclass.

Every channel adapter (Telegram, Discord, GUI) converts its native message
format into an InboundMessage before passing it to the shared pipeline.
This means the router, orchestrator, memory, and agent loop never need
to know which channel a message came from.

Channel-specific metadata (chat_id, message_id for replies) is stored in
`raw_meta` so adapters can use it for sending replies without polluting
the core data model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Channel(str, Enum):
    GUI      = "gui"
    TELEGRAM = "telegram"
    DISCORD  = "discord"


class ChatType(str, Enum):
    PRIVATE = "private"   # 1-to-1 DM
    GROUP   = "group"     # group chat / channel


@dataclass
class InboundMessage:
    """
    A channel-agnostic representation of a single inbound user message.

    Fields
    ------
    channel         Which adapter produced this message.
    chat_type       Private DM or group chat.
    user_id         Channel-specific user identifier (string for portability).
    username        Display name for logging and memory attribution.
    text            The raw message text after stripping bot mention prefixes.
    conversation_id Stable session key: "<channel>:<user_id>" for DMs,
                    "<channel>:<chat_id>" for groups (set by adapter).
    timestamp       UTC time the message was received.
    raw_meta        Adapter-private data needed for replies
                    (e.g. Telegram chat_id, message_id).
    agent_mode      True when the message should trigger the full agentic
                    coding loop rather than a single-turn chat response.
                    Adapters set this when the text starts with "/run",
                    "/agent", or "/code".
    """
    channel:         Channel
    chat_type:       ChatType
    user_id:         str
    username:        str
    text:            str
    conversation_id: str
    timestamp:       datetime          = field(default_factory=lambda: datetime.now(timezone.utc))
    raw_meta:        dict[str, Any]    = field(default_factory=dict)
    agent_mode:      bool              = False

    # ── Convenience helpers ────────────────────────────────────────────────

    @property
    def is_agent_trigger(self) -> bool:
        """True if the text starts with an agent-mode slash command."""
        return self.agent_mode or self.text.lstrip().startswith(
            ("/run ", "/agent ", "/code ", "/run\n", "/agent\n", "/code\n")
        )

    @property
    def clean_text(self) -> str:
        """Message text with agent-mode prefix stripped."""
        for prefix in ("/run ", "/agent ", "/code "):
            if self.text.lstrip().startswith(prefix):
                return self.text.lstrip()[len(prefix):]
        return self.text

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (
            f"InboundMessage(channel={self.channel.value!r}, "
            f"user={self.username!r}, "
            f"agent={self.agent_mode}, "
            f"text={preview!r})"
        )
