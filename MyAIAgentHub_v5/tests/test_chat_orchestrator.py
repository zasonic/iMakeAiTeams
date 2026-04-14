"""
tests/test_chat_orchestrator.py

Covers:
- Routing decisions: claude path, local path, agent override
- History cap (MAX_HISTORY_MESSAGES)
- Token tracking for both streaming and non-streaming paths
- Message persistence
- Conversation CRUD helpers

Stage 5 updates:
- ChatResult dataclass returned from send() instead of raw dict
- RouteDecision imported from models.py instead of Route from router.py
- Budget enforcement tests (Improvement 2)
- ExecutionTarget resolution test (Improvement 6)
"""

import pytest
from unittest.mock import MagicMock, patch, call
import json


def _make_orchestrator(in_memory_db, claude_client, local_client, settings, routing="claude"):
    """Build a ChatOrchestrator with a mocked router."""
    from services.chat_orchestrator import ChatOrchestrator
    from models import RouteDecision
    from services.memory import MemoryManager

    router = MagicMock()
    router.classify.return_value = RouteDecision(model=routing, complexity="complex",
                                                  reasoning="test")

    # Memory manager that does nothing for indexing
    mem = MemoryManager(rag_index=None, semantic_search_mod=None, local_client=local_client)
    return ChatOrchestrator(claude_client, local_client, router, mem, settings)


# ── Routing decisions ─────────────────────────────────────────────────────────

class TestRoutingDecisions:
    def test_routes_to_claude_by_default(self, in_memory_db, claude_client,
                                          local_client_unavailable, settings):
        orch = _make_orchestrator(in_memory_db, claude_client, local_client_unavailable,
                                   settings, routing="claude")
        conv_id = orch.create_conversation()

        claude_client.chat_multi_turn = MagicMock(return_value={
            "text": "Hello!", "input_tokens": 5, "output_tokens": 3
        })
        result = orch.send(conv_id, "Hi there")

        assert result.model == "claude-sonnet-4-20250514"
        assert result.text == "Hello!"
        claude_client.chat_multi_turn.assert_called_once()

    def test_routes_to_local_when_requested(self, in_memory_db, claude_client,
                                             local_client_available, settings):
        orch = _make_orchestrator(in_memory_db, claude_client, local_client_available,
                                   settings, routing="local")
        conv_id = orch.create_conversation()

        result = orch.send(conv_id, "summarize this")
        assert result.text == "local response"
        # Claude should NOT have been called
        claude_client.chat_multi_turn.assert_not_called()

    def test_agent_model_pref_claude_overrides_router(self, in_memory_db, claude_client,
                                                       local_client_available, settings):
        """An agent with model_preference='claude' must go to Claude regardless of router."""
        from services.chat_orchestrator import ChatOrchestrator
        from services.memory import MemoryManager
        from models import RouteDecision

        # Seed an agent that prefers Claude
        in_memory_db.execute(
            "INSERT INTO agents (id, name, description, system_prompt, model_preference, "
            "max_tokens, is_builtin, created_at, updated_at) VALUES "
            "('ag1', 'TestAgent', 'desc', 'You help.', 'claude', 4096, 0, '2024-01-01', '2024-01-01')"
        )
        in_memory_db.commit()

        router = MagicMock()
        router.classify.return_value = RouteDecision(model="local", complexity="simple", reasoning="")
        mem = MemoryManager(None, None, local_client_available)
        orch = ChatOrchestrator(claude_client, local_client_available, router, mem, settings)
        conv_id = orch.create_conversation(agent_id="ag1")

        claude_client.chat_multi_turn = MagicMock(return_value={
            "text": "Claude says hi", "input_tokens": 10, "output_tokens": 5
        })
        result = orch.send(conv_id, "hello", agent_id="ag1")
        assert result.text == "Claude says hi"
        # Router classify should NOT have been called (agent pref takes priority)
        router.classify.assert_not_called()

    def test_agent_model_pref_local_overrides_router(self, in_memory_db, claude_client,
                                                      local_client_available, settings):
        """An agent with model_preference='local' must go to local."""
        from services.chat_orchestrator import ChatOrchestrator
        from services.memory import MemoryManager
        from models import RouteDecision

        in_memory_db.execute(
            "INSERT INTO agents (id, name, description, system_prompt, model_preference, "
            "max_tokens, is_builtin, created_at, updated_at) VALUES "
            "('ag2', 'LocalAgent', 'desc', 'You help.', 'local', 4096, 0, '2024-01-01', '2024-01-01')"
        )
        in_memory_db.commit()

        router = MagicMock()
        router.classify.return_value = RouteDecision(model="claude", complexity="complex", reasoning="")
        mem = MemoryManager(None, None, local_client_available)
        orch = ChatOrchestrator(claude_client, local_client_available, router, mem, settings)
        conv_id = orch.create_conversation(agent_id="ag2")

        result = orch.send(conv_id, "hello", agent_id="ag2")
        assert result.text == "local response"
        claude_client.chat_multi_turn.assert_not_called()


# ── History cap ───────────────────────────────────────────────────────────────

class TestHistoryCap:
    def test_history_capped_at_max(self, in_memory_db, claude_client,
                                    local_client_unavailable, settings):
        """
        Even if the DB has 60 messages, only MAX_HISTORY_MESSAGES are sent to the model.
        """
        from services.chat_orchestrator import MAX_HISTORY_MESSAGES
        from datetime import datetime, timezone

        orch = _make_orchestrator(in_memory_db, claude_client, local_client_unavailable,
                                   settings, routing="claude")
        conv_id = orch.create_conversation()

        # Insert 60 messages directly into the DB (alternating user/assistant)
        now = datetime.now(timezone.utc).isoformat()
        import uuid as _uuid
        for i in range(60):
            role = "user" if i % 2 == 0 else "assistant"
            in_memory_db.execute(
                "INSERT INTO messages (id, conversation_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (str(_uuid.uuid4()), conv_id, role, f"message {i}", now),
            )
        in_memory_db.commit()

        captured_messages = []

        def capture(system, msgs, **kwargs):
            captured_messages.extend(msgs)
            return {"text": "ok", "input_tokens": 1, "output_tokens": 1}

        claude_client.chat_multi_turn = capture

        orch.send(conv_id, "new message")

        # The new user message itself is appended, so we compare ≤ MAX + 1
        # (the +1 is the message we just sent, which was already inserted before fetch)
        assert len(captured_messages) <= MAX_HISTORY_MESSAGES + 1, (
            f"Expected ≤ {MAX_HISTORY_MESSAGES + 1} messages, got {len(captured_messages)}"
        )


# ── Token tracking ────────────────────────────────────────────────────────────

class TestTokenTracking:
    def test_non_streaming_tokens_recorded(self, in_memory_db, claude_client,
                                            local_client_unavailable, settings):
        orch = _make_orchestrator(in_memory_db, claude_client, local_client_unavailable,
                                   settings, routing="claude")
        conv_id = orch.create_conversation()

        claude_client.chat_multi_turn = MagicMock(return_value={
            "text": "answer", "input_tokens": 42, "output_tokens": 17
        })
        result = orch.send(conv_id, "question")

        assert result.tokens_in == 42
        assert result.tokens_out == 17
        assert result.cost_usd > 0.0

        # Verify token_usage row was written
        row = in_memory_db.fetchone(
            "SELECT tokens_in, tokens_out FROM token_usage WHERE conversation_id = ?",
            (conv_id,)
        )
        assert row is not None
        assert row["tokens_in"] == 42
        assert row["tokens_out"] == 17

    def test_streaming_tokens_recorded(self, in_memory_db, claude_client,
                                        local_client_unavailable, settings):
        orch = _make_orchestrator(in_memory_db, claude_client, local_client_unavailable,
                                   settings, routing="claude")
        conv_id = orch.create_conversation()

        # Simulate stream_multi_turn returning (text, usage)
        mock_usage = MagicMock()
        mock_usage.input_tokens = 55
        mock_usage.output_tokens = 22
        claude_client.stream_multi_turn = MagicMock(return_value=("streamed text", mock_usage))

        on_token = MagicMock()
        result = orch.send(conv_id, "stream this", on_token=on_token)

        assert result.tokens_in == 55
        assert result.tokens_out == 22
        assert result.text == "streamed text"

    def test_local_route_zero_cost(self, in_memory_db, claude_client,
                                    local_client_available, settings):
        orch = _make_orchestrator(in_memory_db, claude_client, local_client_available,
                                   settings, routing="local")
        conv_id = orch.create_conversation()
        result = orch.send(conv_id, "local question")
        assert result.cost_usd == 0.0


# ── Budget enforcement (Improvement 2) ────────────────────────────────────────

class TestBudgetEnforcement:
    def test_budget_exceeded_blocks_call(self, in_memory_db, claude_client,
                                          local_client_unavailable, settings):
        """When cumulative cost exceeds budget, send() returns a budget-exceeded ChatResult."""
        settings.set("max_conversation_budget_usd", 1.0)
        orch = _make_orchestrator(in_memory_db, claude_client, local_client_unavailable,
                                   settings, routing="claude")
        conv_id = orch.create_conversation()

        # Insert a fake token_usage row with $1.50 cost (exceeds $1.00 budget)
        import uuid as _uuid
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        in_memory_db.execute(
            "INSERT INTO token_usage (id, conversation_id, model, tokens_in, tokens_out, "
            "cost_usd, routed_reason, created_at) VALUES (?, ?, 'claude', 100, 50, 1.50, 'test', ?)",
            (str(_uuid.uuid4()), conv_id, now),
        )
        in_memory_db.commit()

        result = orch.send(conv_id, "hello after budget exceeded")
        assert result.route_reason == "budget_exceeded"
        assert "budget limit" in result.text
        # Claude should NOT have been called
        claude_client.chat_multi_turn.assert_not_called()

    def test_budget_warning_emitted_near_threshold(self, in_memory_db, claude_client,
                                                     local_client_unavailable, settings):
        """When cumulative cost passes warning threshold, budget_warning is set."""
        settings.set("max_conversation_budget_usd", 1.0)
        settings.set("budget_warning_threshold_pct", 80.0)
        orch = _make_orchestrator(in_memory_db, claude_client, local_client_unavailable,
                                   settings, routing="claude")
        conv_id = orch.create_conversation()

        # Insert $0.79 of usage — just under 80%
        import uuid as _uuid
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        in_memory_db.execute(
            "INSERT INTO token_usage (id, conversation_id, model, tokens_in, tokens_out, "
            "cost_usd, routed_reason, created_at) VALUES (?, ?, 'claude', 100, 50, 0.79, 'test', ?)",
            (str(_uuid.uuid4()), conv_id, now),
        )
        in_memory_db.commit()

        claude_client.chat_multi_turn = MagicMock(return_value={
            "text": "response", "input_tokens": 500, "output_tokens": 200
        })
        result = orch.send(conv_id, "more stuff")
        # After this call, spent ~$0.79 + new cost should cross 80% of $1.00
        # The budget_warning should be non-empty
        assert result.budget_warning  # should be truthy

    def test_zero_budget_means_unlimited(self, in_memory_db, claude_client,
                                          local_client_unavailable, settings):
        """Budget of 0 means no limit."""
        settings.set("max_conversation_budget_usd", 0.0)
        orch = _make_orchestrator(in_memory_db, claude_client, local_client_unavailable,
                                   settings, routing="claude")
        conv_id = orch.create_conversation()

        claude_client.chat_multi_turn = MagicMock(return_value={
            "text": "ok", "input_tokens": 1, "output_tokens": 1
        })
        result = orch.send(conv_id, "hello")
        assert result.text == "ok"  # Should not be blocked


# ── Execution target (Improvement 6) ─────────────────────────────────────────

class TestExecutionTarget:
    def test_resolve_target_claude(self, in_memory_db, claude_client,
                                    local_client_unavailable, settings):
        orch = _make_orchestrator(in_memory_db, claude_client, local_client_unavailable, settings)
        target = orch._resolve_target("claude", None)
        assert target.backend == "claude"
        assert target.max_tokens == 4096

    def test_resolve_target_local(self, in_memory_db, claude_client,
                                   local_client_available, settings):
        orch = _make_orchestrator(in_memory_db, claude_client, local_client_available, settings)
        target = orch._resolve_target("local", None)
        assert target.backend == "local"
        assert target.max_tokens == 2048  # min(4096, 2048)

    def test_resolve_target_with_agent(self, in_memory_db, claude_client,
                                        local_client_unavailable, settings):
        orch = _make_orchestrator(in_memory_db, claude_client, local_client_unavailable, settings)
        agent = {"max_tokens": "8192"}
        target = orch._resolve_target("claude", agent)
        assert target.max_tokens == 8192


# ── ChatResult dataclass (Improvement 1) ─────────────────────────────────────

class TestChatResult:
    def test_send_returns_chat_result(self, in_memory_db, claude_client,
                                       local_client_unavailable, settings):
        from models import ChatResult
        orch = _make_orchestrator(in_memory_db, claude_client, local_client_unavailable,
                                   settings, routing="claude")
        conv_id = orch.create_conversation()

        claude_client.chat_multi_turn = MagicMock(return_value={
            "text": "test", "input_tokens": 1, "output_tokens": 1
        })
        result = orch.send(conv_id, "hello")
        assert isinstance(result, ChatResult)

    def test_chat_result_to_dict(self, in_memory_db, claude_client,
                                  local_client_unavailable, settings):
        from models import ChatResult
        orch = _make_orchestrator(in_memory_db, claude_client, local_client_unavailable,
                                   settings, routing="claude")
        conv_id = orch.create_conversation()

        claude_client.chat_multi_turn = MagicMock(return_value={
            "text": "test", "input_tokens": 1, "output_tokens": 1
        })
        result = orch.send(conv_id, "hello")
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["text"] == "test"
        assert "model" in d
        assert "cost_usd" in d


# ── Conversation CRUD ─────────────────────────────────────────────────────────

class TestConversationCRUD:
    def test_create_and_list(self, in_memory_db, claude_client,
                              local_client_unavailable, settings):
        orch = _make_orchestrator(in_memory_db, claude_client, local_client_unavailable, settings)
        c1 = orch.create_conversation(title="First")
        c2 = orch.create_conversation(title="Second")
        convs = orch.list_conversations()
        ids = [c["id"] for c in convs]
        assert c1 in ids and c2 in ids

    def test_update_title(self, in_memory_db, claude_client,
                           local_client_unavailable, settings):
        orch = _make_orchestrator(in_memory_db, claude_client, local_client_unavailable, settings)
        cid = orch.create_conversation(title="Old")
        orch.update_conversation_title(cid, "New Title")
        row = in_memory_db.fetchone("SELECT title FROM conversations WHERE id = ?", (cid,))
        assert row["title"] == "New Title"

    def test_delete_conversation(self, in_memory_db, claude_client,
                                  local_client_unavailable, settings):
        orch = _make_orchestrator(in_memory_db, claude_client, local_client_unavailable, settings)
        cid = orch.create_conversation()
        orch.delete_conversation(cid)
        row = in_memory_db.fetchone("SELECT id FROM conversations WHERE id = ?", (cid,))
        assert row is None

    def test_get_messages_empty(self, in_memory_db, claude_client,
                                 local_client_unavailable, settings):
        orch = _make_orchestrator(in_memory_db, claude_client, local_client_unavailable, settings)
        cid = orch.create_conversation()
        msgs = orch.get_conversation_messages(cid)
        assert msgs == []

    def test_auto_title_from_first_message(self, in_memory_db, claude_client,
                                             local_client_unavailable, settings):
        orch = _make_orchestrator(in_memory_db, claude_client, local_client_unavailable,
                                   settings, routing="claude")
        conv_id = orch.create_conversation()

        claude_client.chat_multi_turn = MagicMock(return_value={
            "text": "Sure!", "input_tokens": 1, "output_tokens": 1
        })
        orch.send(conv_id, "My important question about elephants")
        row = in_memory_db.fetchone("SELECT title FROM conversations WHERE id = ?", (conv_id,))
        assert "elephant" in row["title"].lower()
