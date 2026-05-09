"""tests/camel/test_orchestrator_integration.py — CaMeL routing in chat_orchestrator.

The four scenarios the spec calls for:

  1. Flag off                     → existing path runs (regression).
  2. Flag on + retrieved chunks   → CaMeL runs; camel_log row written.
  3. Flag on, no retrieved chunks → existing path runs (CaMeL only fires
                                    when RAG context is in play).
  4. Flag on AND reader_actor_split_enabled=True → CaMeL wins, camel_log
                                    written, no reader_actor branch entered.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest


def _make_orchestrator(
    in_memory_db, claude_client, local_client, settings,
    mem_chunks=None, session_facts=None,
):
    from services.chat_orchestrator import ChatOrchestrator
    from services.memory import MemoryContext, MemoryManager
    from models import RouteDecision

    mem_chunks = list(mem_chunks or [])
    session_facts = list(session_facts or [])

    memory = MemoryManager(
        rag_index=None, semantic_search_mod=None, local_client=local_client,
    )

    def _stub_get_context(conversation_id, user_message, agent_id=None):
        return MemoryContext(
            recent_messages=[],
            session_facts=list(session_facts),
            rag_chunks=list(mem_chunks),
            memories=[],
        )
    memory.get_context = _stub_get_context

    router = MagicMock()
    router.classify.return_value = RouteDecision(
        model="claude", complexity="simple", reasoning="test",
    )

    return ChatOrchestrator(
        claude_client, local_client, router, memory, settings,
    )


def _scripted_claude(claude_client, scripts):
    iter_scripts = iter(scripts)
    captured_calls = []

    def _call(system, messages, max_tokens=4096):
        captured_calls.append({"system": system, "messages": list(messages)})
        try:
            text = next(iter_scripts)
        except StopIteration:
            text = ""
        return {"text": text, "input_tokens": 5, "output_tokens": 5}

    claude_client.chat_unified = _call
    claude_client.stream_unified = _call
    return captured_calls


class TestFlagOff:
    def test_camel_off_runs_monolithic_path_no_camel_log_row(
        self, in_memory_db, claude_client, local_client_unavailable, settings,
    ):
        settings.set("camel_enabled", False)
        orch = _make_orchestrator(
            in_memory_db, claude_client, local_client_unavailable, settings,
            mem_chunks=["doc1"],  # CaMeL would fire if flag were on.
        )
        conv_id = orch.create_conversation()
        _scripted_claude(claude_client, ["normal answer"])

        result = orch.send(conv_id, "Hi there, how are you?")

        assert result.text == "normal answer"
        rows = in_memory_db.fetchall(
            "SELECT * FROM camel_log WHERE conversation_id = ?", (conv_id,),
        )
        assert rows == []
        # router_log shows the legacy monolithic row.
        rows = in_memory_db.fetchall(
            "SELECT agent_role FROM router_log WHERE conversation_id = ?",
            (conv_id,),
        )
        assert len(rows) == 1
        assert rows[0]["agent_role"] == "monolithic"


class TestFlagOnWithChunks:
    def test_camel_on_with_chunks_writes_camel_log(
        self, in_memory_db, claude_client, local_client_unavailable, settings,
    ):
        settings.set("camel_enabled", True)
        orch = _make_orchestrator(
            in_memory_db, claude_client, local_client_unavailable, settings,
            mem_chunks=["my notes about projects"],
        )
        conv_id = orch.create_conversation()

        # The privileged client emits a small fixed plan; there's no
        # quarantined call for this test (the plan is trivially "output a
        # constant"), so we just need one scripted Claude reply for the
        # privileged invocation.
        plan = "output = 'CaMeL produced this answer.'\n"
        _scripted_claude(claude_client, [plan])

        result = orch.send(conv_id, "What's in my notes?")

        assert result.text == "CaMeL produced this answer."

        rows = in_memory_db.fetchall(
            "SELECT plan_source, executed_steps, capability_violations, "
            "blocked_calls, output_text FROM camel_log "
            "WHERE conversation_id = ?",
            (conv_id,),
        )
        assert len(rows) == 1
        row = rows[0]
        assert "output = 'CaMeL produced this answer.'" in row["plan_source"]
        assert row["executed_steps"] >= 1
        assert row["capability_violations"] == 0
        assert json.loads(row["blocked_calls"]) == []
        assert row["output_text"] == "CaMeL produced this answer."

        # router_log tagged camel for analytics queries.
        rl = in_memory_db.fetchall(
            "SELECT agent_role FROM router_log WHERE conversation_id = ?",
            (conv_id,),
        )
        assert len(rl) == 1
        assert rl[0]["agent_role"] == "camel"


class TestFlagOnNoChunks:
    def test_camel_on_no_chunks_uses_monolithic_path(
        self, in_memory_db, claude_client, local_client_unavailable, settings,
    ):
        settings.set("camel_enabled", True)
        orch = _make_orchestrator(
            in_memory_db, claude_client, local_client_unavailable, settings,
            mem_chunks=[],  # no RAG → CaMeL must skip.
        )
        conv_id = orch.create_conversation()
        _scripted_claude(claude_client, ["plain answer"])

        result = orch.send(conv_id, "Hello")

        assert result.text == "plain answer"
        rows = in_memory_db.fetchall(
            "SELECT * FROM camel_log WHERE conversation_id = ?", (conv_id,),
        )
        assert rows == []


class TestCamelBeatsReaderActor:
    def test_both_flags_on_camel_wins_no_reader_actor_rows(
        self, in_memory_db, claude_client, local_client_unavailable, settings,
    ):
        settings.set("camel_enabled", True)
        settings.set("reader_actor_split_enabled", True)
        orch = _make_orchestrator(
            in_memory_db, claude_client, local_client_unavailable, settings,
            mem_chunks=["chunk-A"],
        )
        conv_id = orch.create_conversation()

        plan = "output = 'CaMeL again.'\n"
        _scripted_claude(claude_client, [plan])

        result = orch.send(conv_id, "Anything?")

        assert result.text == "CaMeL again."

        # camel_log written, exactly one row.
        cl = in_memory_db.fetchall(
            "SELECT * FROM camel_log WHERE conversation_id = ?", (conv_id,),
        )
        assert len(cl) == 1

        # No reader/actor router_log rows. The camel turn writes a single
        # row tagged ``camel``.
        rl = in_memory_db.fetchall(
            "SELECT agent_role FROM router_log WHERE conversation_id = ?",
            (conv_id,),
        )
        roles = [r["agent_role"] for r in rl]
        assert "reader" not in roles
        assert "actor" not in roles
        assert "camel" in roles
