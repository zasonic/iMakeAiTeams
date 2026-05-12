"""
services/memory/_context.py — MemoryContext dataclass + deflection scrubber.

Private module: imports come through ``services.memory`` so the public
surface stays unchanged. ``MemoryContext`` is the typed payload every
recall pipeline produces; ``_scrub_deflections`` is shared by the
session-fact extractor and the explicit-memory save path so a model
narration of its own failures never gets stored as a memory.
"""

from __future__ import annotations

import re as _re
from dataclasses import dataclass, field


# Patterns that match sentences whose PURPOSE is to record an assistant
# failure/limitation. Deliberately narrow — false positives erase real
# content, which is worse than letting a few deflections through.
_DEFLECTION_PATTERNS: tuple[_re.Pattern[str], ...] = tuple(
    _re.compile(p, _re.IGNORECASE) for p in (
        # "the assistant <failure verb>"
        r"the assistant\s+(was unable|could not|did not have|does not have|"
        r"offered to (search|help|look)|suggested checking|"
        r"recommended consulting|clarified that[^.]{0,40}(could not|did not have|cannot|limit)|"
        r"explained (it|that it) (could not|cannot|does not|did not)|"
        r"stated[^.]{0,40}(could not|did not have)|"
        r"indicated[^.]{0,40}(could not|did not have))",
        # Capability-denial framing
        r"the assistant\s+(lacks|cannot|can'?t)\s+(access|the ability|information|specific information|details)",
        # "the AI" variants
        r"the AI\s+(was unable|could not|did not have|does not have|cannot|doesn'?t have)",
        # Self-referential limitations
        r"(I|the model)\s+(don'?t|do not|cannot|can'?t)\s+have\s+(access|real-time|current|specific)",
    )
)

# Sentence splitter: split on . ! ? followed by whitespace or end-of-string.
_SENTENCE_SPLIT = _re.compile(r"(?<=[.!?])\s+")


def _scrub_deflections(text: str) -> str:
    """Remove sentences that narrate assistant failures/limitations.

    Returns the text with deflection sentences removed. If every sentence
    is a deflection, returns empty string (caller should discard the fact).
    """
    if not text:
        return text
    sentences = _SENTENCE_SPLIT.split(text)
    kept = []
    for sentence in sentences:
        is_deflection = any(p.search(sentence) for p in _DEFLECTION_PATTERNS)
        if not is_deflection:
            kept.append(sentence)
    return " ".join(kept).strip()


@dataclass
class MemoryContext:
    """Recalled-memory payload produced by ``MemoryManager.get_context()``.

    The four fields cover the three memory tiers plus the rolling buffer:
      - ``recent_messages``: short-term in-memory deque (per conversation)
      - ``session_facts``:   per-conversation extracted facts (SQLite)
      - ``rag_chunks``:      RAG retrieval results (chromadb-equivalent)
      - ``memories``:        long-term cross-session memories
    """
    recent_messages: list = field(default_factory=list)
    session_facts:   list = field(default_factory=list)
    rag_chunks:      list = field(default_factory=list)
    memories:        list = field(default_factory=list)

    def to_system_suffix(self) -> str:
        # Section headers are deliberately phrased to mirror the trigger
        # conditions of canonical denial templates ("I don't have personal
        # information about you"). Pre-filling those exact slots with real
        # data dampens the denial reflex on small local models.
        parts = []
        if self.session_facts:
            parts.append(
                "## Personal information about the user\n"
                "(These are facts the user told you. Reference them naturally.)\n\n" +
                "\n".join(f"- {f}" for f in self.session_facts)
            )
        if self.rag_chunks:
            parts.append(
                "## Reference documents the user has provided\n"
                "(The user uploaded these. Use them to answer their question.)\n\n" +
                "\n---\n".join(self.rag_chunks)
            )
        if self.memories:
            parts.append(
                "## Information the user has shared in prior conversations\n"
                "(You have access to this — it was stored from previous sessions.)\n\n" +
                "\n".join(f"- {m}" for m in self.memories)
            )
        return "\n\n".join(parts) if parts else ""
