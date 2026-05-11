"""
services/memory/buffer.py — Per-conversation rolling buffer + summarizer.

The short-term tier of the three-tier memory system. Holds the last
~60 turns per conversation in an in-memory deque, decides when the
buffer needs summarizing (length trigger OR topic-shift detection),
and squashes the buffer into a single ``[Earlier conversation summary]``
message via the local model when the trigger fires.

Hard-trim fallback: if the local model is unavailable but the buffer
crosses 50 entries, we drop the oldest messages down to 30 so the
buffer can't grow unbounded on installs without Ollama/LM Studio.
"""

from __future__ import annotations

import logging
from collections import deque

from models import SessionHistory

log = logging.getLogger("iMakeAiTeams.memory.buffer")

# ── Public constants (re-exported via services/memory/__init__.py) ───────────
# The orchestrator's monolithic-vs-split decisions and a few tests
# anchor on these — keeping them at module scope means callers can
# patch them in tests without reaching into the manager.
_SUMMARIZE_LENGTH_TRIGGER = 30
_TOPIC_SHIFT_WINDOW       = 3

_SUMMARY_PROMPT = (
    "Summarize this conversation segment in 3–5 sentences. "
    "Focus on: decisions made, open questions, and any preferences or commitments "
    "the user expressed. Be specific — preserve names, numbers, and dates."
)


class _BufferStore:
    """Owns the per-conversation deques.

    Instantiated once per ``MemoryManager``. Methods take the
    ``conversation_id`` and the ``local_client`` reference (forwarded
    from the manager, not stored, so test doubles can swap clients per
    call).
    """

    def __init__(self):
        self._buffers: dict[str, deque] = {}

    def get_deque(self, conversation_id: str) -> deque:
        if conversation_id not in self._buffers:
            self._buffers[conversation_id] = deque(maxlen=60)
        return self._buffers[conversation_id]

    def add(self, conversation_id: str, role: str, content: str) -> None:
        self.get_deque(conversation_id).append({"role": role, "content": content})

    def snapshot(self, conversation_id: str) -> list:
        return list(self.get_deque(conversation_id))

    def should_summarize(self, conversation_id: str) -> bool:
        """Length trigger OR topic-shift trigger (Improvement 3)."""
        buf = self.get_deque(conversation_id)
        if len(buf) >= _SUMMARIZE_LENGTH_TRIGGER:
            return True
        if len(buf) >= _TOPIC_SHIFT_WINDOW + 2:
            recent = list(buf)[-_TOPIC_SHIFT_WINDOW:]
            earlier = list(buf)[: len(buf) - _TOPIC_SHIFT_WINDOW]
            recent_words = set(
                w.lower() for m in recent
                for w in m["content"].split() if len(w) > 4
            )
            earlier_words = set(
                w.lower() for m in earlier
                for w in m["content"].split() if len(w) > 4
            )
            overlap = recent_words & earlier_words
            if earlier_words and len(overlap) / max(len(recent_words), 1) < 0.1:
                return True
        return False

    def summarize(
        self,
        conversation_id: str,
        local_client,
        history: SessionHistory,
    ) -> str | None:
        """Squash the buffer into one summary message, or hard-trim if the
        local model is unavailable. Returns the summary text or None."""
        buf = self.get_deque(conversation_id)
        if len(buf) < 4:
            return None
        if not local_client or not local_client.is_available():
            if len(buf) >= 50:
                overflow = len(buf) - 30
                original_len = len(buf)
                for _ in range(overflow):
                    buf.popleft()
                log.info(
                    "Hard-trimmed conversation buffer from %d to %d messages",
                    original_len, len(buf),
                )
                history.add(
                    "hard_trim",
                    f"Hard-trimmed buffer from {original_len} to {len(buf)} messages",
                )
            return None
        messages_text = "\n".join(
            f"{m['role'].upper()}: {m['content'][:300]}" for m in list(buf)[-20:]
        )
        try:
            summary = local_client.chat(
                _SUMMARY_PROMPT,
                f"Conversation to summarize:\n\n{messages_text}",
                max_tokens=300,
            )
            original_count = len(list(buf))
            buf.clear()
            buf.append({"role": "system", "content": f"[Earlier conversation summary: {summary}]"})
            history.add(
                "summarized",
                f"Summarized {original_count} messages into compact form",
            )
            return summary
        except Exception as exc:
            log.debug("Buffer summarization failed: %s", exc)
            return None
