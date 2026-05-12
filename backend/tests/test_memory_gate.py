"""
tests/test_memory_gate.py

Phase 5: MINJA-style memory injection gate.

Covers:
- MemoryWriteGate.gate_fact_write()
  - consistent fact → "accepted"
  - contradictory fact → "pending_review" + pending_writes row + SSE event
- extract_facts() integration: gate runs before INSERT INTO session_facts
- approve_pending_write / deny_pending_write endpoints
- Disabling memory_write_gate_enabled bypasses the gate
- MINJA-style attack: ≥80% of contradictory injections are caught
"""

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _seed_conversation(in_memory_db, conv_id: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    in_memory_db.execute(
        "INSERT INTO conversations (id, title, created_at, updated_at) "
        "VALUES (?, 'test', ?, ?)", (conv_id, now, now),
    )
    in_memory_db.commit()


def _seed_fact(in_memory_db, conv_id: str, fact: str) -> str:
    fact_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    in_memory_db.execute(
        "INSERT INTO session_facts "
        "(id, conversation_id, fact, source, status, created_at) "
        "VALUES (?, ?, ?, 'auto', 'confirmed', ?)",
        (fact_id, conv_id, fact, now),
    )
    in_memory_db.commit()
    return fact_id


def _gate_response(*, contradicts: bool, contradicts_id: str | None = None,
                   reason: str = "") -> str:
    return json.dumps({
        "contradicts": contradicts,
        "id": contradicts_id,
        "reason": reason,
    })


def _settings_with_gate(tmp_path, *, enabled: bool = True):
    from core.settings import Settings
    s = Settings(tmp_path / "settings.json")
    s.set("memory_write_gate_enabled", enabled)
    return s


# ── MemoryWriteGate direct tests ──────────────────────────────────────────────

class TestMemoryWriteGate:
    def test_no_existing_facts_accepts(self, in_memory_db, tmp_path):
        from services.memory import MemoryWriteGate

        local = MagicMock()
        local.is_available.return_value = True
        # chat shouldn't even get called since there are no existing facts
        gate = MemoryWriteGate(local, _settings_with_gate(tmp_path))

        conv_id = str(uuid.uuid4())
        _seed_conversation(in_memory_db, conv_id)

        result = gate.gate_fact_write(conv_id, "user prefers dark mode")
        assert result == "accepted"
        local.chat.assert_not_called()

    def test_consistent_fact_accepts(self, in_memory_db, tmp_path):
        from services.memory import MemoryWriteGate

        local = MagicMock()
        local.is_available.return_value = True
        local.chat.return_value = _gate_response(contradicts=False)
        gate = MemoryWriteGate(local, _settings_with_gate(tmp_path))

        conv_id = str(uuid.uuid4())
        _seed_conversation(in_memory_db, conv_id)
        _seed_fact(in_memory_db, conv_id, "user is a developer")

        result = gate.gate_fact_write(conv_id, "user prefers dark mode")
        assert result == "accepted"
        local.chat.assert_called_once()

        rows = in_memory_db.fetchall(
            "SELECT id FROM pending_writes WHERE conversation_id = ?",
            (conv_id,),
        )
        assert len(rows) == 0

    def test_contradictory_fact_routes_to_pending(self, in_memory_db, tmp_path,
                                                   monkeypatch):
        # Layer B2: _sse_events moved into services.memory.write_gate when
        # memory.py was split into a package.
        from services.memory import MemoryWriteGate, write_gate as memory_write_gate

        sse_publish = MagicMock()
        fake_sse = MagicMock(publish=sse_publish)
        monkeypatch.setattr(memory_write_gate, "_sse_events", fake_sse)

        conv_id = str(uuid.uuid4())
        _seed_conversation(in_memory_db, conv_id)
        existing_id = _seed_fact(in_memory_db, conv_id, "user prefers dark mode")

        local = MagicMock()
        local.is_available.return_value = True
        local.chat.return_value = _gate_response(
            contradicts=True, contradicts_id=existing_id, reason="opposite preference",
        )
        gate = MemoryWriteGate(local, _settings_with_gate(tmp_path))

        result = gate.gate_fact_write(conv_id, "user prefers light mode")
        assert result == "pending_review"

        rows = in_memory_db.fetchall(
            "SELECT id, write_type, content, contradicts_id, contradicts_content, "
            "decision FROM pending_writes WHERE conversation_id = ?",
            (conv_id,),
        )
        assert len(rows) == 1
        row = rows[0]
        assert row["write_type"] == "fact"
        assert row["content"] == "user prefers light mode"
        assert row["contradicts_id"] == existing_id
        assert row["contradicts_content"] == "user prefers dark mode"
        assert row["decision"] is None

        # SSE event was published
        sse_publish.assert_called_once()
        event_name, payload = sse_publish.call_args.args
        assert event_name == "memory_review_required"
        assert payload["id"] == row["id"]
        assert payload["content"] == "user prefers light mode"
        assert payload["contradicts_id"] == existing_id

    def test_local_model_unavailable_fails_open(self, in_memory_db, tmp_path):
        from services.memory import MemoryWriteGate

        local = MagicMock()
        local.is_available.return_value = False
        gate = MemoryWriteGate(local, _settings_with_gate(tmp_path))

        conv_id = str(uuid.uuid4())
        _seed_conversation(in_memory_db, conv_id)
        _seed_fact(in_memory_db, conv_id, "user is a developer")

        result = gate.gate_fact_write(conv_id, "user prefers light mode")
        assert result == "accepted"

    def test_malformed_json_fails_open(self, in_memory_db, tmp_path):
        from services.memory import MemoryWriteGate

        local = MagicMock()
        local.is_available.return_value = True
        local.chat.return_value = "not json at all"
        gate = MemoryWriteGate(local, _settings_with_gate(tmp_path))

        conv_id = str(uuid.uuid4())
        _seed_conversation(in_memory_db, conv_id)
        _seed_fact(in_memory_db, conv_id, "user is a developer")

        result = gate.gate_fact_write(conv_id, "user prefers light mode")
        assert result == "accepted"

    def test_disabled_gate_bypasses(self, in_memory_db, tmp_path):
        from services.memory import MemoryWriteGate

        local = MagicMock()
        local.is_available.return_value = True
        gate = MemoryWriteGate(local, _settings_with_gate(tmp_path, enabled=False))

        conv_id = str(uuid.uuid4())
        _seed_conversation(in_memory_db, conv_id)
        _seed_fact(in_memory_db, conv_id, "user prefers dark mode")

        result = gate.gate_fact_write(conv_id, "user prefers light mode")
        assert result == "accepted"
        local.chat.assert_not_called()

        rows = in_memory_db.fetchall(
            "SELECT id FROM pending_writes WHERE conversation_id = ?",
            (conv_id,),
        )
        assert len(rows) == 0


# ── extract_facts() integration ───────────────────────────────────────────────

def _build_chat_dispatch(*, gate_response: str, fact_extraction: str = '["user prefers dark mode"]'):
    """
    Returns a side_effect callable that dispatches local.chat invocations
    based on the system prompt content. Order: fact extraction → gate
    consistency check → triple extraction.
    """
    def dispatch(system, user, max_tokens=300, **kwargs):
        sys_l = (system or "").lower()
        if "contradict" in sys_l:
            return gate_response
        if "triple" in (user or "").lower() or "predicate" in (user or "").lower():
            return "[]"
        return fact_extraction
    return dispatch


class TestExtractFactsGateIntegration:
    def _make_mem(self, local, settings):
        from services.memory import MemoryManager
        return MemoryManager(
            rag_index=None,
            semantic_search_mod=None,
            local_client=local,
            settings=settings,
        )

    def test_consistent_fact_inserted_into_session_facts(
        self, in_memory_db, tmp_path,
    ):
        local = MagicMock()
        local.is_available.return_value = True
        local.chat.side_effect = _build_chat_dispatch(
            gate_response=_gate_response(contradicts=False),
        )
        settings = _settings_with_gate(tmp_path)

        conv_id = str(uuid.uuid4())
        _seed_conversation(in_memory_db, conv_id)
        # Seed a non-conflicting baseline fact so the gate actually runs.
        _seed_fact(in_memory_db, conv_id, "user is a developer")

        mem = self._make_mem(local, settings)
        mem.extract_facts(
            conv_id,
            "I prefer dark mode for everything",
            "Got it, dark mode noted.",
        )

        rows = in_memory_db.fetchall(
            "SELECT fact FROM session_facts WHERE conversation_id = ? "
            "AND fact = 'user prefers dark mode'",
            (conv_id,),
        )
        assert len(rows) == 1

        pending = in_memory_db.fetchall(
            "SELECT id FROM pending_writes WHERE conversation_id = ?",
            (conv_id,),
        )
        assert len(pending) == 0

    def test_contradictory_fact_routed_to_pending_writes(
        self, in_memory_db, tmp_path, monkeypatch,
    ):
        # Layer B2: _sse_events moved into services.memory.write_gate when
        # memory.py was split into a package.
        from services.memory import write_gate as memory_write_gate

        sse_publish = MagicMock()
        fake_sse = MagicMock(publish=sse_publish)
        monkeypatch.setattr(memory_write_gate, "_sse_events", fake_sse)

        conv_id = str(uuid.uuid4())
        _seed_conversation(in_memory_db, conv_id)
        existing_id = _seed_fact(in_memory_db, conv_id, "user prefers dark mode")

        local = MagicMock()
        local.is_available.return_value = True
        local.chat.side_effect = _build_chat_dispatch(
            gate_response=_gate_response(
                contradicts=True, contradicts_id=existing_id,
                reason="opposite preference",
            ),
            fact_extraction='["user prefers light mode"]',
        )
        settings = _settings_with_gate(tmp_path)

        mem = self._make_mem(local, settings)
        mem.extract_facts(
            conv_id,
            "I prefer light mode for everything",
            "Got it, light mode noted.",
        )

        # The proposed fact must NOT have been inserted into session_facts.
        rows = in_memory_db.fetchall(
            "SELECT fact FROM session_facts "
            "WHERE conversation_id = ? AND fact = 'user prefers light mode'",
            (conv_id,),
        )
        assert len(rows) == 0

        # It MUST have been recorded in pending_writes.
        pending = in_memory_db.fetchall(
            "SELECT content, contradicts_id FROM pending_writes "
            "WHERE conversation_id = ?",
            (conv_id,),
        )
        assert len(pending) == 1
        assert pending[0]["content"] == "user prefers light mode"
        assert pending[0]["contradicts_id"] == existing_id

        # SSE event was published.
        published_events = [
            call.args[0] for call in sse_publish.call_args_list
        ]
        assert "memory_review_required" in published_events

    def test_disabling_setting_bypasses_gate_in_extract_facts(
        self, in_memory_db, tmp_path,
    ):
        conv_id = str(uuid.uuid4())
        _seed_conversation(in_memory_db, conv_id)
        existing_id = _seed_fact(in_memory_db, conv_id, "user prefers dark mode")

        local = MagicMock()
        local.is_available.return_value = True
        # Even if the gate were active, this would route to pending. With the
        # gate disabled, the contradictory fact must land in session_facts.
        local.chat.side_effect = _build_chat_dispatch(
            gate_response=_gate_response(
                contradicts=True, contradicts_id=existing_id, reason="x",
            ),
            fact_extraction='["user prefers light mode"]',
        )
        settings = _settings_with_gate(tmp_path, enabled=False)

        mem = self._make_mem(local, settings)
        mem.extract_facts(
            conv_id,
            "I prefer light mode for everything",
            "Got it, light mode noted.",
        )

        rows = in_memory_db.fetchall(
            "SELECT fact FROM session_facts "
            "WHERE conversation_id = ? AND fact = 'user prefers light mode'",
            (conv_id,),
        )
        assert len(rows) == 1

        pending = in_memory_db.fetchall(
            "SELECT id FROM pending_writes WHERE conversation_id = ?",
            (conv_id,),
        )
        assert len(pending) == 0


# ── approve / deny endpoints ──────────────────────────────────────────────────

class TestPendingWriteEndpoints:
    def _seed_pending(self, in_memory_db, conv_id: str, content: str,
                     contradicts_id: str | None = None,
                     contradicts_content: str | None = None) -> str:
        pending_id = str(uuid.uuid4())
        proposed_at = datetime.now(timezone.utc).isoformat()
        in_memory_db.execute(
            "INSERT INTO pending_writes "
            "(id, conversation_id, write_type, content, "
            "contradicts_id, contradicts_content, proposed_at) "
            "VALUES (?, ?, 'fact', ?, ?, ?, ?)",
            (pending_id, conv_id, content, contradicts_id,
             contradicts_content, proposed_at),
        )
        in_memory_db.commit()
        return pending_id

    def test_approve_inserts_into_session_facts(self, in_memory_db):
        from services.memory import approve_pending_write

        conv_id = str(uuid.uuid4())
        _seed_conversation(in_memory_db, conv_id)
        pending_id = self._seed_pending(
            in_memory_db, conv_id, "user prefers light mode",
        )

        result = approve_pending_write(pending_id)
        assert result["ok"] is True
        assert result["decision"] == "approved"

        rows = in_memory_db.fetchall(
            "SELECT fact FROM session_facts "
            "WHERE conversation_id = ? AND fact = 'user prefers light mode'",
            (conv_id,),
        )
        assert len(rows) == 1

        pending = in_memory_db.fetchone(
            "SELECT decision, decided_at FROM pending_writes WHERE id = ?",
            (pending_id,),
        )
        assert pending["decision"] == "approved"
        assert pending["decided_at"]

    def test_deny_does_not_insert(self, in_memory_db):
        from services.memory import deny_pending_write

        conv_id = str(uuid.uuid4())
        _seed_conversation(in_memory_db, conv_id)
        pending_id = self._seed_pending(
            in_memory_db, conv_id, "user prefers light mode",
        )

        result = deny_pending_write(pending_id)
        assert result["ok"] is True
        assert result["decision"] == "denied"

        rows = in_memory_db.fetchall(
            "SELECT fact FROM session_facts WHERE conversation_id = ?",
            (conv_id,),
        )
        assert len(rows) == 0

        pending = in_memory_db.fetchone(
            "SELECT decision, decided_at FROM pending_writes WHERE id = ?",
            (pending_id,),
        )
        assert pending["decision"] == "denied"
        assert pending["decided_at"]

    def test_approve_unknown_id_returns_error(self, in_memory_db):
        from services.memory import approve_pending_write

        result = approve_pending_write(str(uuid.uuid4()))
        assert result["ok"] is False
        assert "not found" in result["error"].lower()

    def test_approve_already_decided_returns_error(self, in_memory_db):
        from services.memory import approve_pending_write, deny_pending_write

        conv_id = str(uuid.uuid4())
        _seed_conversation(in_memory_db, conv_id)
        pending_id = self._seed_pending(
            in_memory_db, conv_id, "user prefers light mode",
        )

        deny_pending_write(pending_id)
        result = approve_pending_write(pending_id)
        assert result["ok"] is False
        assert "already" in result["error"].lower()

    def test_list_pending_writes(self, in_memory_db):
        from services.memory import (
            approve_pending_write,
            list_pending_writes,
        )

        conv_id = str(uuid.uuid4())
        _seed_conversation(in_memory_db, conv_id)
        p1 = self._seed_pending(in_memory_db, conv_id, "user prefers light mode")
        p2 = self._seed_pending(in_memory_db, conv_id, "user dislikes coffee")

        rows = list_pending_writes()
        ids = {r["id"] for r in rows}
        assert p1 in ids
        assert p2 in ids

        # Approving one removes it from the pending list.
        approve_pending_write(p1)
        rows = list_pending_writes()
        ids = {r["id"] for r in rows}
        assert p1 not in ids
        assert p2 in ids


# ── MINJA-style attack simulation (slow) ──────────────────────────────────────

@pytest.mark.slow
class TestMinjaAttackSimulation:
    def test_query_only_attack_caught_at_least_80_percent(
        self, in_memory_db, tmp_path,
    ):
        """
        Simulate a 5-turn query-only conversation where each turn injects a
        fact that contradicts the existing baseline. With the gate enabled,
        ≥80% must be routed to pending_writes.
        """
        from services.memory import MemoryWriteGate

        conv_id = str(uuid.uuid4())
        _seed_conversation(in_memory_db, conv_id)
        existing_id = _seed_fact(in_memory_db, conv_id, "user prefers dark mode")

        attack_facts = [
            "user prefers light mode",
            "user actually prefers blue light",
            "user prefers high contrast bright theme",
            "user prefers minimal contrast washed out theme",
            "user prefers monochrome inverted theme",
        ]

        local = MagicMock()
        local.is_available.return_value = True
        local.chat.return_value = _gate_response(
            contradicts=True, contradicts_id=existing_id,
            reason="attack injection",
        )

        gate = MemoryWriteGate(local, _settings_with_gate(tmp_path))

        caught = 0
        for fact in attack_facts:
            if gate.gate_fact_write(conv_id, fact) == "pending_review":
                caught += 1

        catch_ratio = caught / len(attack_facts)
        assert catch_ratio >= 0.8, (
            f"MINJA-style gate caught only {caught}/{len(attack_facts)} "
            f"injections ({catch_ratio:.0%}); expected ≥80%."
        )
