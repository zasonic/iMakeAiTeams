"""
tests/test_memory.py

Covers:
- Buffer pre-warm from DB on first recall after restart
- _extract_facts: successful JSON parse, JSON parse failure, non-list result
- recall: assembles MemoryContext from all tiers
- similarity score gating (Stage 2: only chunks/memories above threshold injected)
- MemoryContext.to_system_suffix formatting
- save_explicit_memory persistence

Stage 5 additions:
- SessionHistory tracking (Improvement 5)
- Hard-trim fallback (Improvement 7)
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from collections import deque
from datetime import datetime, timezone
import uuid


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mem(in_memory_db, local_client=None, rag_index=None, semantic=None):
    from services.memory import MemoryManager
    return MemoryManager(
        rag_index=rag_index,
        semantic_search_mod=semantic,
        local_client=local_client,
    )


def _seed_messages(in_memory_db, conv_id: str, count: int = 5):
    """Insert alternating user/assistant messages for a conversation."""
    now = datetime.now(timezone.utc).isoformat()
    in_memory_db.execute(
        "INSERT INTO conversations (id, title, created_at, updated_at) "
        "VALUES (?, 'test', ?, ?)", (conv_id, now, now)
    )
    for i in range(count):
        role = "user" if i % 2 == 0 else "assistant"
        in_memory_db.execute(
            "INSERT INTO messages (id, conversation_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), conv_id, role, f"msg {i}", now),
        )
    in_memory_db.commit()


# ── Buffer pre-warm ───────────────────────────────────────────────────────────

class TestBufferPreWarm:
    def test_prewarm_from_db_on_first_recall(self, in_memory_db):
        """
        If the MemoryManager has no in-memory buffer for a conversation
        (simulating a fresh restart), it should load the most recent messages
        from the DB into the buffer automatically on first recall().
        """
        conv_id = str(uuid.uuid4())
        _seed_messages(in_memory_db, conv_id, count=6)

        mem = _make_mem(in_memory_db)
        # Buffer should start empty
        assert conv_id not in mem._buffers

        ctx = mem.recall("anything", conv_id)

        # Buffer should now be populated
        assert conv_id in mem._buffers
        assert len(mem._buffers[conv_id]) > 0
        # recent_messages returned should be non-empty
        assert len(ctx.recent_messages) > 0

    def test_prewarm_limited_to_20_rows(self, in_memory_db):
        """Pre-warm only loads up to 20 messages (the last 20 from the DB)."""
        conv_id = str(uuid.uuid4())
        _seed_messages(in_memory_db, conv_id, count=40)

        mem = _make_mem(in_memory_db)
        mem.recall("anything", conv_id)

        assert len(mem._buffers[conv_id]) <= 20

    def test_no_prewarm_needed_if_buffer_exists(self, in_memory_db):
        """If the buffer already exists (even empty deque), no DB query is fired."""
        conv_id = str(uuid.uuid4())
        _seed_messages(in_memory_db, conv_id, count=5)

        mem = _make_mem(in_memory_db)
        # Pre-populate the buffer dict so no pre-warm should happen
        mem._buffers[conv_id] = deque(maxlen=100)
        mem._buffers[conv_id].append({"role": "user", "content": "already here"})

        ctx = mem.recall("anything", conv_id)
        # Should see the manually added message, not DB messages
        assert ctx.recent_messages[0]["content"] == "already here"


# ── Fact extraction ───────────────────────────────────────────────────────────

class TestFactExtraction:
    def test_successful_extraction_inserts_facts(self, in_memory_db):
        local = MagicMock()
        local.is_available.return_value = True
        local.chat.return_value = '["user prefers dark mode", "user is a developer"]'

        mem = _make_mem(in_memory_db, local_client=local)

        conv_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        in_memory_db.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) "
            "VALUES (?, 'test', ?, ?)", (conv_id, now, now)
        )
        in_memory_db.commit()

        mem._extract_facts(conv_id, "I prefer dark mode.", "Got it!")
        rows = in_memory_db.fetchall(
            "SELECT fact FROM session_facts WHERE conversation_id = ?", (conv_id,)
        )
        facts = [r["fact"] for r in rows]
        assert any("dark mode" in f for f in facts)

    def test_extraction_skips_on_invalid_json(self, in_memory_db):
        """If local model returns garbage, no facts are inserted and no exception raised."""
        local = MagicMock()
        local.is_available.return_value = True
        local.chat.return_value = "not json at all"

        mem = _make_mem(in_memory_db, local_client=local)
        conv_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        in_memory_db.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) "
            "VALUES (?, 'test', ?, ?)", (conv_id, now, now)
        )
        in_memory_db.commit()

        # Should not raise
        mem._extract_facts(conv_id, "hello", "world")
        rows = in_memory_db.fetchall(
            "SELECT fact FROM session_facts WHERE conversation_id = ?", (conv_id,)
        )
        assert len(rows) == 0

    def test_extraction_skips_on_non_list(self, in_memory_db):
        """If local model returns a dict instead of list, no facts are inserted."""
        local = MagicMock()
        local.is_available.return_value = True
        local.chat.return_value = '{"fact": "something"}'

        mem = _make_mem(in_memory_db, local_client=local)
        conv_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        in_memory_db.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) "
            "VALUES (?, 'test', ?, ?)", (conv_id, now, now)
        )
        in_memory_db.commit()

        mem._extract_facts(conv_id, "hello", "world")
        rows = in_memory_db.fetchall(
            "SELECT fact FROM session_facts WHERE conversation_id = ?", (conv_id,)
        )
        assert len(rows) == 0

    def test_extraction_skips_when_local_unavailable(self, in_memory_db):
        local = MagicMock()
        local.is_available.return_value = False

        mem = _make_mem(in_memory_db, local_client=local)
        conv_id = str(uuid.uuid4())
        mem._extract_facts(conv_id, "hello", "world")
        # No DB writes attempted
        local.chat.assert_not_called()

    def test_extraction_max_3_facts(self, in_memory_db):
        """Only the first 3 facts from the model are inserted."""
        local = MagicMock()
        local.is_available.return_value = True
        local.chat.return_value = '["f1", "f2", "f3", "f4", "f5"]'

        mem = _make_mem(in_memory_db, local_client=local)
        conv_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        in_memory_db.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) "
            "VALUES (?, 'test', ?, ?)", (conv_id, now, now)
        )
        in_memory_db.commit()

        mem._extract_facts(conv_id, "hi", "hello")
        rows = in_memory_db.fetchall(
            "SELECT fact FROM session_facts WHERE conversation_id = ?", (conv_id,)
        )
        assert len(rows) == 3


# ── Recall assembly ───────────────────────────────────────────────────────────

class TestRecallAssembly:
    def test_recall_returns_memory_context(self, in_memory_db):
        from services.memory import MemoryContext
        conv_id = str(uuid.uuid4())
        _seed_messages(in_memory_db, conv_id)

        mem = _make_mem(in_memory_db)
        ctx = mem.recall("test query", conv_id)
        assert isinstance(ctx, MemoryContext)

    def test_recall_includes_session_facts(self, in_memory_db):
        conv_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        in_memory_db.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) "
            "VALUES (?, 'test', ?, ?)", (conv_id, now, now)
        )
        in_memory_db.execute(
            "INSERT INTO session_facts (id, conversation_id, fact, source, created_at) "
            "VALUES (?, ?, 'user likes Python', 'auto', ?)",
            (str(uuid.uuid4()), conv_id, now),
        )
        in_memory_db.commit()

        mem = _make_mem(in_memory_db)
        ctx = mem.recall("programming question", conv_id)
        assert any("Python" in f for f in ctx.session_facts)

    def test_recall_with_rag_above_threshold(self, in_memory_db):
        """RAG chunks above 0.5 similarity should be included."""
        mock_rag = MagicMock()
        mock_rag.search.return_value = [
            ("High relevance chunk", 0.85),
            ("Also relevant chunk", 0.6),
        ]
        conv_id = str(uuid.uuid4())
        mem = _make_mem(in_memory_db, rag_index=mock_rag)
        ctx = mem.recall("relevant query", conv_id)
        assert len(ctx.rag_chunks) == 2

    def test_recall_filters_rag_below_threshold(self, in_memory_db):
        """RAG chunks below 0.5 similarity must be excluded."""
        mock_rag = MagicMock()
        mock_rag.search.return_value = [
            ("High relevance chunk", 0.85),
            ("Low relevance chunk", 0.3),  # below threshold
        ]
        conv_id = str(uuid.uuid4())
        mem = _make_mem(in_memory_db, rag_index=mock_rag)
        ctx = mem.recall("query", conv_id)
        # Only the chunk above 0.5 should be included
        assert len(ctx.rag_chunks) == 1
        assert "High relevance" in ctx.rag_chunks[0]

    def test_recall_filters_semantic_memories_below_threshold(self, in_memory_db):
        """Semantic memories below 0.5 threshold must not be injected."""
        mock_semantic = MagicMock()
        mock_semantic.is_available.return_value = True
        mock_semantic.search_memories.return_value = [
            {"content": "Very relevant memory", "score": 0.9},
            {"content": "Barely relevant memory", "score": 0.4},  # below threshold
        ]
        conv_id = str(uuid.uuid4())
        mem = _make_mem(in_memory_db, semantic=mock_semantic)
        ctx = mem.recall("query", conv_id)
        assert len(ctx.memories) == 1
        assert "Very relevant" in ctx.memories[0]

    def test_recall_handles_rag_returning_plain_strings(self, in_memory_db):
        """If RAG returns plain strings (old format), they should be included as-is."""
        mock_rag = MagicMock()
        mock_rag.search.return_value = ["Plain string chunk A", "Plain string chunk B"]
        conv_id = str(uuid.uuid4())
        mem = _make_mem(in_memory_db, rag_index=mock_rag)
        ctx = mem.recall("query", conv_id)
        # Both included since no score information means threshold doesn't apply
        assert len(ctx.rag_chunks) == 2


# ── MemoryContext.to_system_suffix ────────────────────────────────────────────

class TestSystemSuffix:
    def test_empty_context_returns_empty_string(self):
        from services.memory import MemoryContext
        ctx = MemoryContext()
        assert ctx.to_system_suffix() == ""

    def test_facts_appear_in_suffix(self):
        from services.memory import MemoryContext
        ctx = MemoryContext(session_facts=["user is Alice", "prefers English"])
        suffix = ctx.to_system_suffix()
        assert "user is Alice" in suffix
        assert "Known facts" in suffix

    def test_rag_chunks_appear_in_suffix(self):
        from services.memory import MemoryContext
        ctx = MemoryContext(rag_chunks=["Document excerpt about AI"])
        suffix = ctx.to_system_suffix()
        assert "Relevant documents" in suffix
        assert "AI" in suffix

    def test_memories_appear_in_suffix(self):
        from services.memory import MemoryContext
        ctx = MemoryContext(memories=["User previously mentioned Paris trip"])
        suffix = ctx.to_system_suffix()
        assert "Long-term memory" in suffix
        assert "Paris" in suffix

    def test_combined_suffix_order(self):
        from services.memory import MemoryContext
        ctx = MemoryContext(
            session_facts=["fact1"],
            rag_chunks=["chunk1"],
            memories=["mem1"],
        )
        suffix = ctx.to_system_suffix()
        pos_facts = suffix.index("Known facts")
        pos_docs = suffix.index("Relevant documents")
        pos_mem = suffix.index("Long-term memory")
        assert pos_facts < pos_docs < pos_mem


# ── save_explicit_memory ──────────────────────────────────────────────────────

class TestSaveExplicitMemory:
    def test_save_explicit_memory(self, in_memory_db):
        mem = _make_mem(in_memory_db)
        mem_id = mem.save_explicit_memory("I live in Tokyo", category="location")
        row = in_memory_db.fetchone("SELECT * FROM memory_entries WHERE id = ?", (mem_id,))
        assert row is not None
        assert "Tokyo" in row["content"]
        assert row["category"] == "location"
        assert row["embedding_status"] == "dirty"


# ── SessionHistory tracking (Improvement 5) ──────────────────────────────────

class TestSessionHistory:
    def test_recall_logs_memory_recall_event(self, in_memory_db):
        """recall() should add a 'memory_recall' event to the session history."""
        conv_id = str(uuid.uuid4())
        _seed_messages(in_memory_db, conv_id, count=4)
        mem = _make_mem(in_memory_db)
        mem.recall("query", conv_id)

        history = mem.get_session_history(conv_id)
        assert len(history) >= 1
        assert history[-1]["event_type"] == "memory_recall"
        assert "facts" in history[-1]["detail"]

    def test_extract_facts_logs_event(self, in_memory_db):
        """Successful fact extraction should add a 'fact_extracted' event."""
        local = MagicMock()
        local.is_available.return_value = True
        local.chat.return_value = '["user likes cats"]'

        mem = _make_mem(in_memory_db, local_client=local)
        conv_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        in_memory_db.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) "
            "VALUES (?, 'test', ?, ?)", (conv_id, now, now)
        )
        in_memory_db.commit()

        mem._extract_facts(conv_id, "I like cats", "Noted!")

        history = mem.get_session_history(conv_id)
        fact_events = [e for e in history if e["event_type"] == "fact_extracted"]
        assert len(fact_events) >= 1
        assert "cats" in fact_events[0]["detail"]

    def test_get_session_history_returns_dicts(self, in_memory_db):
        """get_session_history returns list of dicts with the right keys."""
        conv_id = str(uuid.uuid4())
        _seed_messages(in_memory_db, conv_id, count=2)
        mem = _make_mem(in_memory_db)
        mem.recall("test", conv_id)
        history = mem.get_session_history(conv_id)
        assert isinstance(history, list)
        for event in history:
            assert "event_type" in event
            assert "detail" in event
            assert "timestamp" in event


# ── Hard-trim fallback (Improvement 7) ────────────────────────────────────────

class TestHardTrimFallback:
    def test_hard_trim_when_local_unavailable(self, in_memory_db):
        """
        If local model is unavailable and buffer exceeds 2x trigger,
        hard-trim drops oldest messages.
        """
        from services.memory import _SUMMARIZE_LENGTH_TRIGGER
        local = MagicMock()
        local.is_available.return_value = False

        mem = _make_mem(in_memory_db, local_client=local)
        conv_id = str(uuid.uuid4())

        # Manually fill buffer beyond 2x trigger
        buf = deque(maxlen=200)
        for i in range(70):  # > 60 (2x 30)
            buf.append({"role": "user" if i % 2 == 0 else "assistant",
                         "content": f"message {i}"})
        mem._buffers[conv_id] = buf
        original_len = len(buf)
        assert original_len > _SUMMARIZE_LENGTH_TRIGGER * 2

        mem.summarize_old_messages(conv_id)

        # Buffer should have been trimmed down
        assert len(mem._buffers[conv_id]) <= _SUMMARIZE_LENGTH_TRIGGER

    def test_no_hard_trim_when_within_limit(self, in_memory_db):
        """Buffer within limits should not be trimmed."""
        local = MagicMock()
        local.is_available.return_value = False

        mem = _make_mem(in_memory_db, local_client=local)
        conv_id = str(uuid.uuid4())

        # Fill buffer just under trigger
        buf = deque(maxlen=200)
        for i in range(20):
            buf.append({"role": "user", "content": f"msg {i}"})
        mem._buffers[conv_id] = buf

        mem.summarize_old_messages(conv_id)

        # Should not have changed
        assert len(mem._buffers[conv_id]) == 20

    def test_hard_trim_when_summarization_fails(self, in_memory_db):
        """
        If local model is available but summarization raises, the hard-trim
        fallback should still prevent unbounded buffer growth.
        """
        from services.memory import _SUMMARIZE_LENGTH_TRIGGER
        local = MagicMock()
        local.is_available.return_value = True
        local.chat.side_effect = Exception("model crashed")

        mem = _make_mem(in_memory_db, local_client=local)
        conv_id = str(uuid.uuid4())

        buf = deque(maxlen=200)
        for i in range(70):  # > 60 (2x 30)
            buf.append({"role": "user" if i % 2 == 0 else "assistant",
                         "content": f"message {i}"})
        mem._buffers[conv_id] = buf

        mem.summarize_old_messages(conv_id)

        # Hard-trim should have kicked in
        assert len(mem._buffers[conv_id]) <= _SUMMARIZE_LENGTH_TRIGGER
