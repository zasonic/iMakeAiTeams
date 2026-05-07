"""
core/api/memory.py — Memory and semantic-search bridge methods.
"""

from __future__ import annotations

from core.service_guard import requires as _requires

from services import memory as _memory_svc
from services import semantic_search

from ._base import BaseAPI


class MemoryAPI(BaseAPI):

    def search_memories_semantic(self, query: str, top_k: int = 5) -> list:
        return semantic_search.search_memories(query, top_k=top_k)

    def search_documents_semantic(self, query: str, top_k: int = 10,
                                  doc_type: str = "") -> list:
        return semantic_search.search_documents(
            query, top_k=top_k, doc_type=doc_type or None
        )

    def semantic_search_available(self) -> bool:
        status = self._status.get("semantic_search", {})
        return bool(status.get("ok")) and semantic_search.is_available()

    @_requires("memory_manager", default={"error": "memory unavailable"})
    def save_memory(self, content: str, category: str = "fact") -> dict:
        mem_id = self._memory.save_explicit_memory(content, category)
        return {"id": mem_id}

    def get_stale_memories(self, days: int = 30) -> list:
        """
        Return memory entries not accessed in the last `days` days.
        Used by the frontend Stale Memories panel so users can review and delete.
        """
        return semantic_search.get_stale_memories(days=days)

    def delete_memory_entry(self, entry_id: str) -> dict:
        """Delete a specific memory entry from both SQLite and ChromaDB."""
        ok = semantic_search.delete_memory_entry(entry_id)
        return {"ok": ok}

    # ── MINJA-style memory injection gate (Phase 5) ──────────────────────────

    def list_pending_writes(self, limit: int = 100) -> list:
        return _memory_svc.list_pending_writes(limit)

    def approve_pending_write(self, pending_id: str) -> dict:
        return _memory_svc.approve_pending_write(pending_id)

    def deny_pending_write(self, pending_id: str) -> dict:
        return _memory_svc.deny_pending_write(pending_id)
