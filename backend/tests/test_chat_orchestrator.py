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

        assert result.model == claude_client._model
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

        # Pre-load the conversation just under the 80% threshold. Sonnet is
        # priced at $3/$15 per million tokens, so the new request below adds
        # ~$0.01 in cost which crosses 0.80 of the $1.00 budget.
        import uuid as _uuid
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        in_memory_db.execute(
            "INSERT INTO token_usage (id, conversation_id, model, tokens_in, tokens_out, "
            "cost_usd, routed_reason, created_at) VALUES (?, ?, 'claude', 100, 50, 0.795, 'test', ?)",
            (str(_uuid.uuid4()), conv_id, now),
        )
        in_memory_db.commit()

        claude_client.chat_multi_turn = MagicMock(return_value={
            "text": "response", "input_tokens": 500, "output_tokens": 700
        })
        result = orch.send(conv_id, "more stuff")
        # After this call, spent ~$0.795 + ~$0.012 should cross 80% of $1.00
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


# ── Codebase review fixes ─────────────────────────────────────────────────────

class TestReviewBugFixes:
    """Regression tests for Bugs 1, 4, 5, 6 from CODEBASE_REVIEW_REPORT.md."""

    def test_bug1_rag_trim_path_does_not_raise(self, in_memory_db, claude_client,
                                                local_client_unavailable, settings):
        """Bug 1: RAG-trim branch must not reference removed `_active_mem_suffix`."""
        from services.memory import MemoryContext

        orch = _make_orchestrator(in_memory_db, claude_client, local_client_unavailable,
                                   settings, routing="claude")
        conv_id = orch.create_conversation()

        # Force memory.get_context to return more RAG chunks than `complex`'s
        # cap of 8 so the trim branch fires.
        big_ctx = MemoryContext(
            session_facts=[],
            rag_chunks=[f"chunk {i}" for i in range(20)],
            memories=[],
        )
        orch.memory.get_context = MagicMock(return_value=big_ctx)
        orch.memory.should_summarize = MagicMock(return_value=False)
        orch.memory.add_to_buffer = MagicMock()
        orch.memory.extract_facts = MagicMock()

        claude_client.chat_multi_turn = MagicMock(return_value={
            "text": "answer", "input_tokens": 5, "output_tokens": 3
        })

        # Must not raise AttributeError / NameError / UnboundLocalError.
        result = orch.send(conv_id, "tell me about elephants in detail please")
        assert result.text == "answer"

    def test_bug4_router_log_records_post_escalation_response_empty(
        self, in_memory_db, claude_client, local_client_available, settings,
    ):
        """Bug 4: After empty local triggers escalation, router_log must
        record response_empty=False with the post-escalation response."""
        from services.chat_orchestrator import ChatOrchestrator
        from services.memory import MemoryManager
        from models import RouteDecision

        router = MagicMock()
        router.classify.return_value = RouteDecision(
            model="local", complexity="complex", reasoning="test"
        )
        mem = MemoryManager(None, None, local_client_available)
        orch = ChatOrchestrator(claude_client, local_client_available, router, mem, settings)
        conv_id = orch.create_conversation()

        # Local returns empty — triggers the empty-response escalation gate.
        local_client_available.chat_unified.return_value = {
            "text": "", "input_tokens": 0, "output_tokens": 0,
        }
        local_client_available.stream_unified.return_value = {
            "text": "", "input_tokens": 0, "output_tokens": 0,
        }

        # Claude (escalation target) returns a substantive response.
        long_response = "This is a fully formed Claude answer that escalation produced for the user's question."
        claude_client.chat_multi_turn = MagicMock(return_value={
            "text": long_response, "input_tokens": 5, "output_tokens": 25,
        })

        result = orch.send(conv_id, "Tell me about quantum mechanics please")

        assert result.text == long_response
        assert len(result.text.strip()) >= 20

        row = in_memory_db.fetchone(
            "SELECT response_empty FROM router_log WHERE conversation_id = ?",
            (conv_id,),
        )
        assert row is not None
        assert row["response_empty"] == 0, (
            f"router_log should record post-escalation response_empty=False, got {row['response_empty']}"
        )

    def test_bug5_concurrent_sends_share_budget_state(
        self, in_memory_db, claude_client, local_client_unavailable, settings,
    ):
        """Bug 5: With two overlapping sends, the second's budget warning must
        reflect the first's cost (i.e. SUM is re-read inside the post-write lock)."""
        import threading
        import time

        settings.set("max_conversation_budget_usd", 1.0)
        settings.set("budget_warning_threshold_pct", 50.0)

        orch = _make_orchestrator(
            in_memory_db, claude_client, local_client_unavailable,
            settings, routing="claude",
        )
        conv_id = orch.create_conversation()

        # Each send: 100k input + 20k output Sonnet tokens → ~$0.60 cost.
        # Two overlapping sends together cross the 50% warning threshold.
        barrier = threading.Barrier(2, timeout=10)

        def slow_llm(*_args, **_kwargs):
            barrier.wait()  # both threads enter LLM call simultaneously
            time.sleep(0.05)
            return {"text": "ok", "input_tokens": 100_000, "output_tokens": 20_000}

        claude_client.chat_multi_turn = MagicMock(side_effect=slow_llm)

        results = {}

        def call_send(i):
            results[i] = orch.send(conv_id, f"message {i} please tell me")

        t1 = threading.Thread(target=call_send, args=(0,))
        t2 = threading.Thread(target=call_send, args=(1,))
        t1.start(); t2.start()
        t1.join(); t2.join()

        warnings = [r.budget_warning for r in results.values() if r and r.budget_warning]
        assert len(warnings) >= 1, f"expected at least one budget warning, got {warnings}"

        # Extract the dollar amounts reported in each warning. The send that
        # ran second must report > $1.00 because it sees both costs summed.
        # The pre-fix bug used a stale `spent` so each warning would only
        # show its own ~$0.60 contribution, never the cumulative total.
        import re
        spent_amounts = []
        for w in warnings:
            m = re.search(r"\$([\d.]+)/", w)
            if m:
                spent_amounts.append(float(m.group(1)))

        assert max(spent_amounts) > 1.0, (
            f"second send did not see first's cost; spent_amounts={spent_amounts}"
        )

    def test_bug6_sqlite_error_rolls_back_assistant_and_token_usage(
        self, in_memory_db, claude_client, local_client_unavailable, settings,
    ):
        """Bug 6: A SQLite error during the post-LLM transaction must leave
        the assistant message and token_usage row both absent (atomic)."""
        import sqlite3
        import db as _db_mod

        orch = _make_orchestrator(
            in_memory_db, claude_client, local_client_unavailable,
            settings, routing="claude",
        )
        conv_id = orch.create_conversation()

        claude_client.chat_multi_turn = MagicMock(return_value={
            "text": "answer", "input_tokens": 5, "output_tokens": 3,
        })

        real_conn = _db_mod.get_db()

        # sqlite3.Connection.execute can't be patched directly, so wrap the
        # connection in a delegator that fails on INSERT INTO token_usage.
        class FailingTokenUsageConn:
            def __init__(self, real):
                self._real = real

            def execute(self, sql, *args, **kwargs):
                if "INSERT INTO token_usage" in sql:
                    raise sqlite3.OperationalError("simulated mid-write error")
                return self._real.execute(sql, *args, **kwargs)

            def commit(self):
                return self._real.commit()

            def rollback(self):
                return self._real.rollback()

            def __getattr__(self, name):
                return getattr(self._real, name)

        wrapper = FailingTokenUsageConn(real_conn)
        with patch.object(_db_mod, "_conn", wrapper):
            with pytest.raises(sqlite3.OperationalError):
                orch.send(conv_id, "trigger the rollback please")

        user_count = in_memory_db.fetchone(
            "SELECT COUNT(*) AS c FROM messages "
            "WHERE conversation_id = ? AND role = 'user'",
            (conv_id,),
        )["c"]
        asst_count = in_memory_db.fetchone(
            "SELECT COUNT(*) AS c FROM messages "
            "WHERE conversation_id = ? AND role = 'assistant'",
            (conv_id,),
        )["c"]
        tok_count = in_memory_db.fetchone(
            "SELECT COUNT(*) AS c FROM token_usage WHERE conversation_id = ?",
            (conv_id,),
        )["c"]

        # The user-message INSERT runs in its own earlier transaction so it
        # persists. The assistant-message + token_usage transaction must
        # have rolled back together.
        assert user_count == 1
        assert asst_count == 0 and tok_count == 0, (
            f"expected both rolled back; got asst={asst_count}, tok={tok_count}"
        )


# ── MAST failure-mode tagging (Phase 4) ───────────────────────────────────────

class TestMastFailureTagging:
    def test_failed_turn_writes_non_null_mast_category(
        self, in_memory_db, claude_client, local_client_unavailable, settings,
    ):
        """A turn that returns ``[Error: ...]`` from the worker must persist
        a non-NULL mast_category on the router_log row."""
        orch = _make_orchestrator(
            in_memory_db, claude_client, local_client_unavailable,
            settings, routing="claude",
        )
        conv_id = orch.create_conversation()

        # Force the worker invocation to fail so HubRouter.invoke returns
        # a "[Error: ...]" WorkerResult, which is the failure signal that
        # the orchestrator routes through classify_failure().
        claude_client.chat_multi_turn = MagicMock(
            side_effect=RuntimeError("simulated worker failure"),
        )
        # Stub the classifier so the test does not depend on a live API.
        orch.hub_router.classify_failure = MagicMock(return_value="3.2")

        orch.send(conv_id, "please run a quick failure for me")

        row = in_memory_db.fetchone(
            "SELECT mast_category, had_error FROM router_log "
            "WHERE conversation_id = ?",
            (conv_id,),
        )
        assert row is not None
        assert row["had_error"] == 1
        assert row["mast_category"] == "3.2"
        orch.hub_router.classify_failure.assert_called_once()
