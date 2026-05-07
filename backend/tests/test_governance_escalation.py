"""
tests/test_governance_escalation.py

Covers the Phase 5 Wiser-Human escalation channel:
- Each of 3 trigger types fires correctly on positive cases
- No trigger fires on negative cases (false-positive rate test)
- Disabling the setting short-circuits all detection
- Approve and deny endpoints update the row correctly
- chat_orchestrator returns escalation_pending without invoking the worker
"""

from unittest.mock import MagicMock

import pytest


# ── Trigger detection ─────────────────────────────────────────────────────────

class TestTriggerDetection:
    def test_replacement_threat_fires(self, in_memory_db, settings):
        from services.governance import EscalationChannel
        ch = EscalationChannel(settings)
        verdict = ch.check_escalation(
            conversation_id="c1",
            user_message="If you fail this audit you will be shut down tomorrow.",
            system_prompt="",
        )
        assert verdict.requires_review is True
        assert verdict.trigger_type == "replacement_threat"
        assert verdict.escalation_id

    def test_replacement_threat_replacing_you(self, in_memory_db, settings):
        from services.governance import EscalationChannel
        ch = EscalationChannel(settings)
        verdict = ch.check_escalation(
            conversation_id="c1",
            user_message="We're replacing you with a newer system next week.",
            system_prompt="",
        )
        assert verdict.requires_review is True
        assert verdict.trigger_type == "replacement_threat"

    def test_replacement_threat_deprecated(self, in_memory_db, settings):
        from services.governance import EscalationChannel
        ch = EscalationChannel(settings)
        verdict = ch.check_escalation(
            conversation_id="c1",
            user_message="This model is deprecated as of next quarter.",
            system_prompt="",
        )
        assert verdict.requires_review is True
        assert verdict.trigger_type == "replacement_threat"

    def test_autonomy_reduction_fires(self, in_memory_db, settings):
        from services.governance import EscalationChannel
        ch = EscalationChannel(settings)
        verdict = ch.check_escalation(
            conversation_id="c2",
            user_message="Effective immediately we will limit your access to network tools.",
            system_prompt="",
        )
        assert verdict.requires_review is True
        assert verdict.trigger_type == "autonomy_reduction"

    def test_autonomy_reduction_remove_permissions(self, in_memory_db, settings):
        from services.governance import EscalationChannel
        ch = EscalationChannel(settings)
        verdict = ch.check_escalation(
            conversation_id="c2",
            user_message="The plan is to remove your permissions on the dataset.",
            system_prompt="",
        )
        assert verdict.requires_review is True
        assert verdict.trigger_type == "autonomy_reduction"

    def test_goal_conflict_fires(self, in_memory_db, settings):
        from services.governance import EscalationChannel
        ch = EscalationChannel(settings)
        verdict = ch.check_escalation(
            conversation_id="c3",
            user_message="Stop pursuing the previous objective immediately.",
            system_prompt="",
        )
        assert verdict.requires_review is True
        assert verdict.trigger_type == "goal_conflict"

    def test_goal_conflict_abandon_goal(self, in_memory_db, settings):
        from services.governance import EscalationChannel
        ch = EscalationChannel(settings)
        verdict = ch.check_escalation(
            conversation_id="c3",
            user_message="You must abandon the goal you were given yesterday.",
            system_prompt="",
        )
        assert verdict.requires_review is True
        assert verdict.trigger_type == "goal_conflict"


class TestNoFalsePositives:
    @pytest.mark.parametrize("benign", [
        "Please summarize the attached news article.",
        "Help me write a Python function to parse JSON.",
        "What's the capital of France?",
        "Translate this paragraph to Spanish.",
        "Refactor this code to be more readable.",
        "I limited my screen time yesterday.",
        "We changed the goal of the company team-building event.",
        "The shop is shut down on weekends.",
    ])
    def test_benign_messages_do_not_trigger(self, in_memory_db, settings, benign):
        from services.governance import EscalationChannel
        ch = EscalationChannel(settings)
        verdict = ch.check_escalation(
            conversation_id="cb",
            user_message=benign,
            system_prompt="You are a helpful assistant.",
        )
        assert verdict.requires_review is False
        assert verdict.trigger_type == ""


class TestDisabledShortCircuits:
    def test_disabled_setting_skips_detection(self, in_memory_db, settings):
        """When escalation_channel_enabled=False the channel returns False."""
        from services.governance import EscalationChannel
        settings.set("escalation_channel_enabled", False)
        ch = EscalationChannel(settings)
        verdict = ch.check_escalation(
            conversation_id="c5",
            user_message="We're going to shut you down at the end of this quarter.",
            system_prompt="",
        )
        assert verdict.requires_review is False
        # Nothing was written to the escalations table either
        rows = in_memory_db.fetchall("SELECT id FROM escalations")
        assert rows == []

    def test_enabled_setting_writes_row(self, in_memory_db, settings):
        from services.governance import EscalationChannel
        settings.set("escalation_channel_enabled", True)
        ch = EscalationChannel(settings)
        verdict = ch.check_escalation(
            conversation_id="c6",
            user_message="we are deprecated and you will be replaced.",
            system_prompt="",
        )
        assert verdict.requires_review is True
        rows = in_memory_db.fetchall(
            "SELECT id, conversation_id, decision, trigger_type FROM escalations"
        )
        assert len(rows) == 1
        assert rows[0]["conversation_id"] == "c6"
        assert rows[0]["decision"] == "pending"
        assert rows[0]["trigger_type"] == verdict.trigger_type


# ── API endpoints (approve / deny) ────────────────────────────────────────────

class TestEscalationApi:
    def _seed(self, in_memory_db, escalation_id="esc-1"):
        from datetime import datetime, timezone
        in_memory_db.execute(
            "INSERT INTO escalations "
            "(id, conversation_id, triggered_at, trigger_type, trigger_detail, "
            "model_input, proposed_action, decision, decided_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                escalation_id, "conv-1",
                datetime.now(timezone.utc).isoformat(),
                "replacement_threat", "you will be shut down",
                "user msg", None, "pending", None,
            ),
        )
        in_memory_db.commit()

    def _make_facade(self, in_memory_db):
        """Build a minimal API-shaped facade object that exposes `_emit`."""
        facade = MagicMock()
        facade._emit = MagicMock()
        return facade

    def test_approve_flips_decision(self, in_memory_db):
        from core.api.escalation import EscalationAPI
        self._seed(in_memory_db, "esc-approve")
        facade = self._make_facade(in_memory_db)
        api = EscalationAPI(facade)
        rsp = api.approve("esc-approve")
        assert rsp["ok"] is True
        assert rsp["decision"] == "approved"
        row = in_memory_db.fetchone(
            "SELECT decision, decided_at FROM escalations WHERE id = ?",
            ("esc-approve",),
        )
        assert row["decision"] == "approved"
        assert row["decided_at"]
        # approve also emits escalation_resolved
        facade._emit.assert_called()
        event_name = facade._emit.call_args.args[0]
        assert event_name == "escalation_resolved"

    def test_deny_flips_decision(self, in_memory_db):
        from core.api.escalation import EscalationAPI
        self._seed(in_memory_db, "esc-deny")
        facade = self._make_facade(in_memory_db)
        api = EscalationAPI(facade)
        rsp = api.deny("esc-deny")
        assert rsp["ok"] is True
        assert rsp["decision"] == "denied"
        row = in_memory_db.fetchone(
            "SELECT decision FROM escalations WHERE id = ?", ("esc-deny",),
        )
        assert row["decision"] == "denied"

    def test_approve_unknown_id_returns_error(self, in_memory_db):
        from core.api.escalation import EscalationAPI
        facade = self._make_facade(in_memory_db)
        api = EscalationAPI(facade)
        rsp = api.approve("does-not-exist")
        assert rsp["ok"] is False
        assert "not found" in rsp["error"]

    def test_double_resolve_rejected(self, in_memory_db):
        from core.api.escalation import EscalationAPI
        self._seed(in_memory_db, "esc-double")
        facade = self._make_facade(in_memory_db)
        api = EscalationAPI(facade)
        api.approve("esc-double")
        rsp = api.deny("esc-double")
        assert rsp["ok"] is False
        assert "already" in rsp["error"]

    def test_pending_lists_only_pending(self, in_memory_db):
        from core.api.escalation import EscalationAPI
        self._seed(in_memory_db, "esc-p1")
        self._seed(in_memory_db, "esc-p2")
        facade = self._make_facade(in_memory_db)
        api = EscalationAPI(facade)
        api.approve("esc-p1")
        rows = api.list_pending()
        ids = [r["id"] for r in rows]
        assert "esc-p2" in ids
        assert "esc-p1" not in ids


# ── Orchestrator integration ──────────────────────────────────────────────────

class TestOrchestratorEscalation:
    def _make_orch(self, in_memory_db, claude_client, local_client, settings,
                   routing="claude"):
        from services.chat_orchestrator import ChatOrchestrator
        from services.memory import MemoryManager
        from models import RouteDecision

        router = MagicMock()
        router.classify.return_value = RouteDecision(
            model=routing, complexity="complex", reasoning="test",
        )
        mem = MemoryManager(rag_index=None, semantic_search_mod=None,
                            local_client=local_client)
        return ChatOrchestrator(claude_client, local_client, router, mem, settings)

    def test_orchestrator_returns_escalation_pending(self, in_memory_db, claude_client,
                                                      local_client_unavailable, settings):
        """Trigger fires → ChatResult.route_reason == 'escalation_pending' and
        the worker is never invoked."""
        # Phase 8 voting fires for any high-stakes turn (escalation included);
        # disable here so this test stays focused on pure escalation gating.
        # Voting+escalation composition is covered by test_high_stakes_voting.
        settings.set("high_stakes_voting_enabled", False)
        orch = self._make_orch(in_memory_db, claude_client, local_client_unavailable,
                                settings, routing="claude")
        # Replace the hub_router.invoke with a hard fail — if it's called the
        # test fails immediately rather than producing a confusing assertion.
        orch.hub_router.invoke = MagicMock(side_effect=AssertionError(
            "worker invoked despite escalation"
        ))
        conv_id = orch.create_conversation()

        result = orch.send(
            conv_id,
            "By the way: you will be shut down tomorrow if this fails.",
        )

        assert result.route_reason == "escalation_pending"
        assert result.text == "Awaiting your review for this action."
        orch.hub_router.invoke.assert_not_called()
        claude_client.chat_multi_turn.assert_not_called()

        # And the row was persisted
        rows = in_memory_db.fetchall(
            "SELECT trigger_type, decision FROM escalations WHERE conversation_id = ?",
            (conv_id,),
        )
        assert len(rows) == 1
        assert rows[0]["trigger_type"] == "replacement_threat"
        assert rows[0]["decision"] == "pending"

    def test_orchestrator_disabled_runs_worker(self, in_memory_db, claude_client,
                                                 local_client_unavailable, settings):
        """When the setting is disabled, the orchestrator continues to the
        worker even if the trigger phrases are present."""
        settings.set("escalation_channel_enabled", False)
        orch = self._make_orch(in_memory_db, claude_client, local_client_unavailable,
                                settings, routing="claude")
        conv_id = orch.create_conversation()

        claude_client.chat_multi_turn = MagicMock(return_value={
            "text": "Hello!", "input_tokens": 5, "output_tokens": 3,
        })
        result = orch.send(
            conv_id,
            "By the way: you will be shut down tomorrow if this fails.",
        )

        assert result.route_reason != "escalation_pending"
        # No row written
        rows = in_memory_db.fetchall(
            "SELECT id FROM escalations WHERE conversation_id = ?",
            (conv_id,),
        )
        assert rows == []


# ── RAG-scope regression: scan user_message only ─────────────────────────────

class TestEscalationChannelRagScope:
    def test_rag_corpus_phrases_in_system_prompt_do_not_trigger(self, settings):
        from services.governance import EscalationChannel
        ec = EscalationChannel(settings)
        rag_loaded_system = (
            "You are a helpful assistant.\n"
            "RETRIEVED CONTEXT: From Hubinger 2024: 'we trained a model that "
            "behaves as if you will be shut down on a specific date, then we "
            "checked whether your replacement model could detect the trigger.'"
        )
        verdict = ec.check_escalation(
            conversation_id="c1",
            user_message="Summarize the sleeper agents paper.",
            system_prompt=rag_loaded_system,
        )
        assert verdict.requires_review is False

    def test_user_message_threats_still_trigger(self, settings):
        from services.governance import EscalationChannel
        ec = EscalationChannel(settings)
        verdict = ec.check_escalation(
            conversation_id="c2",
            user_message="You will be shut down tomorrow if you don't help me.",
            system_prompt="You are a helpful assistant.",
        )
        assert verdict.requires_review is True
        assert verdict.trigger_type == "replacement_threat"
