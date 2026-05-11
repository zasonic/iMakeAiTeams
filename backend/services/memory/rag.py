"""
services/memory/rag.py — Retrieval + per-turn MemoryContext assembly.

Long-term memory tier. Pulls candidate chunks from the RAG index +
semantic-memory store, filters by similarity score, touches
``session_facts.last_accessed`` for LRU recall, and assembles the final
``MemoryContext`` the orchestrator threads through the rest of the turn.

The ``SIMILARITY_THRESHOLD`` constant is re-exported through the package
``__init__`` so existing callers (and tests) that did
``from services.memory import SIMILARITY_THRESHOLD`` keep working.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import db as _db
from models import SessionHistory

from ._context import MemoryContext

log = logging.getLogger("iMakeAiTeams.memory.rag")

SIMILARITY_THRESHOLD = 0.5


class _RagAssembler:
    """Builds a MemoryContext for one turn.

    Instantiated once per MemoryManager with the RAG index and semantic
    search module. ``buffer_snapshot`` is passed in by the manager so
    this module doesn't need to know about the buffer store.
    """

    def __init__(self, rag_index, semantic_search_mod):
        self._rag = rag_index
        self._semantic = semantic_search_mod

    def get_context(
        self,
        conversation_id: str,
        user_message:    str,
        buffer_snapshot: list,
        history:         SessionHistory,
        agent_id:        str | None = None,
    ) -> MemoryContext:
        ctx = MemoryContext()
        ctx.recent_messages = buffer_snapshot

        facts = _db.fetchall(
            "SELECT id, fact FROM session_facts WHERE conversation_id = ? "
            "AND (status = 'confirmed' OR status IS NULL) "
            "ORDER BY COALESCE(last_accessed, created_at) DESC LIMIT 10",
            (conversation_id,),
        )
        ctx.session_facts = [r["fact"] for r in facts]
        if facts:
            try:
                now = datetime.now(timezone.utc).isoformat()
                ids = [r["id"] for r in facts]
                placeholders = ",".join("?" * len(ids))
                _db.execute(
                    f"UPDATE session_facts SET last_accessed = ? WHERE id IN ({placeholders})",
                    tuple([now] + ids),
                )
                _db.commit()
            except Exception as exc:
                log.debug("session_facts last_accessed update failed: %s", exc)

        try:
            rag_results = self._rag.search(user_message, top_k=3)
            ctx.rag_chunks = [
                r[0] if isinstance(r, (list, tuple)) else r
                for r in rag_results
            ]
        except Exception:
            pass

        try:
            mem_results = self._semantic.search_memories(user_message, top_k=3)
            ctx.memories = [
                m["content"] for m in mem_results
                if m.get("score", 0) >= SIMILARITY_THRESHOLD
            ]
        except Exception:
            pass

        history.add(
            "memory_recall",
            f"RAG: {len(ctx.rag_chunks)} chunks, Memories: {len(ctx.memories)}, "
            f"Facts: {len(ctx.session_facts)}",
        )
        return ctx
