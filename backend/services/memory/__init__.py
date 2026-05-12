"""
services/memory/__init__.py — Public surface of the memory package.

Before the Layer-B2 split, ``services/memory.py`` was a 968-line file
with eight loosely-related concerns. This package keeps the public
contract identical — every prior ``from services.memory import …``
still works — while the bodies live in domain-bounded sub-modules:

  - ``_context.py``       MemoryContext dataclass + deflection scrubber
  - ``buffer.py``         per-conversation deque, summarizer, hard-trim
  - ``session_facts.py``  fact-extraction pipeline + global counters
  - ``rag.py``            retrieval + per-turn MemoryContext assembly
  - ``write_gate.py``     MemoryWriteGate + trust scan + pending-review
                          + pending-write CRUD + save_explicit_memory

The MemoryManager class below is the thin façade: holds the sub-module
instances, owns the per-conversation ``SessionHistory`` state, and
delegates every public method to the matching sub-module helper.

Mutable module-level globals from the legacy module (the
``_extract_attempts``/``_extract_failures`` counters that
``services.health_monitor`` reads) are re-exposed via PEP 562
``__getattr__`` so callers see a live view rather than a stale snapshot
that a ``from import`` would freeze at import time.
"""

from __future__ import annotations

import logging

from models import SessionHistory

from ._context import MemoryContext, _scrub_deflections
from .buffer import (
    _BufferStore,
    _SUMMARIZE_LENGTH_TRIGGER,
    _TOPIC_SHIFT_WINDOW,
)
from .rag import _RagAssembler, SIMILARITY_THRESHOLD
from .session_facts import _FactExtractor
from .write_gate import (
    MemoryWriteGate,
    _trust_scan,
    _write_to_pending_review,
    approve_pending,
    approve_pending_write,
    deny_pending_write,
    get_pending_count,
    get_pending_review,
    list_pending_writes,
    reject_pending,
    save_explicit_memory as _save_explicit_memory,
)

log = logging.getLogger("iMakeAiTeams.memory")

# Re-exported names: keep the public symbol list explicit so a future
# refactor can see what the façade is committing to.
__all__ = [
    # Dataclasses + constants
    "MemoryContext",
    "MemoryWriteGate",
    "MemoryManager",
    "SIMILARITY_THRESHOLD",
    # Pending review CRUD
    "get_pending_review",
    "approve_pending",
    "reject_pending",
    "get_pending_count",
    # Pending write CRUD
    "list_pending_writes",
    "approve_pending_write",
    "deny_pending_write",
]


def __getattr__(name: str):
    """PEP 562 hook for mutable module-level state.

    ``services.health_monitor`` does:

        from services.memory import _extract_attempts, _extract_failures

    Those counters live on ``session_facts`` and are mutated on every
    extraction. A normal ``from … import`` would bind the value at
    import time and never update. This forwards each read to
    ``session_facts`` at call site so the values stay live.

    Anything not in the forward set falls through to the default
    AttributeError so we don't silently mask typos.
    """
    if name in ("_extract_attempts", "_extract_failures"):
        from . import session_facts
        return getattr(session_facts, name)
    raise AttributeError(f"module 'services.memory' has no attribute {name!r}")


class MemoryManager:
    """Public façade. Owns per-instance SessionHistory state; delegates
    all other work to the sub-module helpers.

    The constructor signature mirrors the pre-split MemoryManager so
    callers (``core/api/__init__.py``, tests) compile unchanged.
    """

    def __init__(self, rag_index, semantic_search_mod, local_client, settings=None):
        self.rag      = rag_index
        self.semantic = semantic_search_mod
        self.local    = local_client
        self.write_gate = MemoryWriteGate(local_client, settings)
        self._buffer  = _BufferStore()
        # Back-compat: the legacy MemoryManager exposed ``_buffers`` as a dict
        # on the instance. Several tests reach in to seed/inspect deques
        # directly. Re-bind the underlying dict here so existing
        # ``mem._buffers[conv_id]`` reads/writes still work.
        self._buffers = self._buffer._buffers
        self._rag_assembler = _RagAssembler(rag_index, semantic_search_mod)
        self._fact_extractor = _FactExtractor(local_client, self.write_gate)
        self._histories: dict[str, SessionHistory] = {}

    # ── SessionHistory (per-instance state) ──────────────────────────────

    def _get_history(self, conversation_id: str) -> SessionHistory:
        if conversation_id not in self._histories:
            self._histories[conversation_id] = SessionHistory()
        return self._histories[conversation_id]

    def get_session_history(self, conversation_id: str) -> list[dict]:
        history = self._get_history(conversation_id)
        return [
            {
                "event_type": e.event_type,
                "detail":     e.detail,
                "timestamp":  e.timestamp,
            }
            for e in history.recent(50)
        ]

    # ── Buffer (delegates to _BufferStore) ───────────────────────────────

    def add_to_buffer(self, conversation_id: str, role: str, content: str) -> None:
        self._buffer.add(conversation_id, role, content)

    def get_buffer(self, conversation_id: str) -> list:
        return self._buffer.snapshot(conversation_id)

    def should_summarize(self, conversation_id: str) -> bool:
        return self._buffer.should_summarize(conversation_id)

    def summarize_buffer(self, conversation_id: str) -> str | None:
        return self._buffer.summarize(
            conversation_id, self.local, self._get_history(conversation_id),
        )

    # ── Retrieval (delegates to _RagAssembler) ───────────────────────────

    def get_context(
        self,
        conversation_id: str,
        user_message:    str,
        agent_id:        str | None = None,
    ) -> MemoryContext:
        return self._rag_assembler.get_context(
            conversation_id=conversation_id,
            user_message=user_message,
            buffer_snapshot=self._buffer.snapshot(conversation_id),
            history=self._get_history(conversation_id),
            agent_id=agent_id,
        )

    # ── Fact extraction (delegates to _FactExtractor) ────────────────────

    def _resolve_pending_facts(self, conversation_id: str, user_message: str) -> None:
        # Kept as an underscore-prefixed method for back-compat with any
        # test that patches this on the manager directly.
        self._fact_extractor.resolve_pending(conversation_id, user_message)

    def extract_facts(
        self, conversation_id: str, user_msg: str, assistant_msg: str,
    ) -> None:
        self._fact_extractor.extract(
            conversation_id, user_msg, assistant_msg,
            history=self._get_history(conversation_id),
        )

    # ── Explicit-memory save (delegates to write_gate.save_explicit_memory) ─

    def save_explicit_memory(self, content: str, category: str = "fact") -> str:
        return _save_explicit_memory(content, category)
