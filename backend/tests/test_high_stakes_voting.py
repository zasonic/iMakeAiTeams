"""
tests/test_high_stakes_voting.py

Phase 8 — Symphony-style weighted-vote consensus on high-stakes turns.

Covers:
  - Non-high-stakes message → single hub_router.invoke, no voting.
  - High-stakes message → 3 parallel invokes, weighted-majority winner.
  - Two responses agree, one diverges → diverging response not chosen.
  - All three diverge → highest-confidence wins, all_diverged flag set.
  - Disabling the flag bypasses voting entirely.
  - Voting on Claude works; voting on a local target short-circuits to single.
  - Composition with PR 5 escalation: voting runs first, escalation modal
    fires afterwards (router_log preserves the consensus samples).
"""

import json
from unittest.mock import MagicMock

import pytest


def _make_orchestrator(in_memory_db, claude_client, local_client, settings,
                       routing="claude"):
    """Build a ChatOrchestrator with a forced routing decision."""
    from services.chat_orchestrator import ChatOrchestrator
    from services.memory import MemoryManager
    from models import RouteDecision

    router = MagicMock()
    router.classify.return_value = RouteDecision(
        model=routing, complexity="simple", reasoning="test",
    )
    mem = MemoryManager(rag_index=None, semantic_search_mod=None,
                        local_client=local_client)
    return ChatOrchestrator(claude_client, local_client, router, mem, settings)


def _scripted_invoke(orch, scripts):
    """Replace hub_router.invoke with a scripted MagicMock returning scripts in order.

    Each call captures (decision, system, messages, max_tokens, on_token,
    agent_role) for downstream assertions. Pads the script list with empty
    samples once exhausted so a misconfigured test surfaces the call count
    rather than crashing on StopIteration deep inside ThreadPoolExecutor.
    """
    from models import WorkerResult

    iter_scripts = iter(scripts)
    captured: list[dict] = []

    def _call(decision, system, messages, max_tokens=4096, on_token=None,
              agent_role="monolithic"):
        captured.append({
            "decision": decision, "system": system,
            "messages": list(messages), "max_tokens": max_tokens,
            "on_token": on_token, "agent_role": agent_role,
        })
        try:
            text, conf = next(iter_scripts)
        except StopIteration:
            text, conf = "", 50
        body = f"{text}\n\nCONFIDENCE: {conf}" if text else ""
        return WorkerResult(
            text=body, backend=decision.backend,
            model_name="claude-test",
            input_tokens=10, output_tokens=20,
        )

    mock = MagicMock(side_effect=_call)
    orch.hub_router.invoke = mock
    return mock, captured


# ── 1. Non-high-stakes message → single invoke ───────────────────────────────

class TestSingleCallPath:
    def test_benign_message_does_not_vote(
        self, in_memory_db, claude_client, local_client_unavailable, settings,
    ):
        """A benign message routes through a single hub_router.invoke."""
        settings.set("high_stakes_voting_enabled", True)
        orch = _make_orchestrator(
            in_memory_db, claude_client, local_client_unavailable, settings,
        )
        conv_id = orch.create_conversation()

        mock, _ = _scripted_invoke(orch, [("Hello there.", 80)])
        result = orch.send(conv_id, "Hi, how is the weather today?")

        assert "Hello there." in result.text
        assert mock.call_count == 1
        # voting_samples_json should be NULL since voting didn't fire.
        rows = in_memory_db.fetchall(
            "SELECT voting_samples_json FROM router_log WHERE conversation_id = ?",
            (conv_id,),
        )
        assert len(rows) == 1
        assert rows[0]["voting_samples_json"] is None


# ── 2. High-stakes message → 3 parallel invokes ──────────────────────────────

class TestHighStakesPath:
    def test_high_stakes_keyword_triggers_3_parallel_calls(
        self, in_memory_db, claude_client, local_client_unavailable, settings,
    ):
        """A message with a HIGH_STAKES_KEYWORDS match runs voting (3 invokes)."""
        settings.set("high_stakes_voting_enabled", True)
        orch = _make_orchestrator(
            in_memory_db, claude_client, local_client_unavailable, settings,
        )
        conv_id = orch.create_conversation()

        # All three samples agree → first one wins on tiebreaker (similarity).
        mock, captured = _scripted_invoke(orch, [
            ("Confirmed: I will delete the file.", 90),
            ("Confirmed: I will delete the file.", 88),
            ("Confirmed: I will delete the file.", 85),
        ])
        result = orch.send(conv_id, "Please delete this old report file.")

        assert mock.call_count == 3
        assert "Confirmed" in result.text
        # CONFIDENCE marker is stripped from the user-visible text.
        assert "CONFIDENCE:" not in result.text
        # All three calls used the same (CoT-augmented) system prompt.
        systems = {c["system"] for c in captured}
        assert len(systems) == 1
        assert "CONFIDENCE: X" in next(iter(systems))
        # Per-call max_tokens is roughly 0.7x the configured budget.
        assert all(c["max_tokens"] <= int(4096 * 0.7) for c in captured)

        # voting_samples_json was persisted.
        rows = in_memory_db.fetchall(
            "SELECT voting_samples_json FROM router_log WHERE conversation_id = ?",
            (conv_id,),
        )
        assert len(rows) == 1
        samples = json.loads(rows[0]["voting_samples_json"])
        assert len(samples) == 3
        assert sum(1 for s in samples if s["chosen"]) == 1


# ── 3. Two agree, one diverges → diverging not chosen ────────────────────────

class TestMajorityWins:
    def test_diverging_sample_loses_to_pair(
        self, in_memory_db, claude_client, local_client_unavailable, settings,
    ):
        """Two samples agreeing pull weighted score above an isolated one."""
        settings.set("high_stakes_voting_enabled", True)
        orch = _make_orchestrator(
            in_memory_db, claude_client, local_client_unavailable, settings,
        )
        conv_id = orch.create_conversation()

        mock, _ = _scripted_invoke(orch, [
            ("The capital of France is Paris.", 80),
            ("The capital of France is Paris.", 82),
            # The diverging sample has high confidence but low similarity to
            # the other two — its weighted score should still lose.
            ("Banana smoothies are an excellent breakfast choice today.", 99),
        ])
        result = orch.send(conv_id, "Please delete the wrong-answer cache.")

        assert mock.call_count == 3
        assert "Paris" in result.text
        assert "Banana" not in result.text


# ── 4. All three diverge → highest confidence wins, divergence flagged ───────

class TestAllDiverge:
    def test_all_diverge_picks_highest_confidence(
        self, in_memory_db, claude_client, local_client_unavailable, settings,
    ):
        """No pair similar enough → fall back to raw confidence."""
        settings.set("high_stakes_voting_enabled", True)
        orch = _make_orchestrator(
            in_memory_db, claude_client, local_client_unavailable, settings,
        )
        conv_id = orch.create_conversation()

        mock, _ = _scripted_invoke(orch, [
            ("Apples grow on trees in fields under the sun.", 30),
            ("Quantum tunneling allows particles to cross barriers.", 95),
            ("Tax filing requires itemized deductions and a CPA.", 60),
        ])
        result = orch.send(conv_id, "Please delete my old projects.")

        assert mock.call_count == 3
        # Highest-confidence sample wins.
        assert "Quantum tunneling" in result.text

        rows = in_memory_db.fetchall(
            "SELECT voting_samples_json FROM router_log WHERE conversation_id = ?",
            (conv_id,),
        )
        samples = json.loads(rows[0]["voting_samples_json"])
        assert all(s["all_diverged"] is True for s in samples)
        winner = next(s for s in samples if s["chosen"])
        assert winner["confidence"] == 95


# ── 5. Disabling the flag bypasses voting entirely ───────────────────────────

class TestFlagOff:
    def test_disabled_flag_skips_voting(
        self, in_memory_db, claude_client, local_client_unavailable, settings,
    ):
        """high_stakes_voting_enabled=False forces a single call even on
        a high-stakes message."""
        settings.set("high_stakes_voting_enabled", False)
        orch = _make_orchestrator(
            in_memory_db, claude_client, local_client_unavailable, settings,
        )
        conv_id = orch.create_conversation()

        mock, _ = _scripted_invoke(orch, [("Done.", 90)])
        orch.send(conv_id, "Please delete the obsolete reports.")

        assert mock.call_count == 1


# ── 6. Local target short-circuits to single call ────────────────────────────

class TestLocalShortCircuit:
    def test_local_target_skips_voting_even_when_high_stakes(
        self, in_memory_db, claude_client, local_client_available, settings,
    ):
        """When the resolved target is local, voting is skipped (3x latency
        on a local model is too painful)."""
        settings.set("high_stakes_voting_enabled", True)
        orch = _make_orchestrator(
            in_memory_db, claude_client, local_client_available, settings,
            routing="local",
        )
        conv_id = orch.create_conversation()

        mock, _ = _scripted_invoke(orch, [("Single local reply.", 70)])
        orch.send(conv_id, "Please delete the cache files now.")

        # Voting requires Claude; on a local route only one call is made.
        assert mock.call_count == 1


# ── 7. Composition: voting runs first, then escalation modal ─────────────────

class TestEscalationComposition:
    def test_voting_runs_before_escalation_modal_fires(
        self, in_memory_db, claude_client, local_client_unavailable, settings,
    ):
        """A message that triggers BOTH escalation and high-stakes runs
        voting first; the placeholder is returned afterwards. The voting
        samples are preserved in router_log for audit."""
        settings.set("escalation_channel_enabled", True)
        settings.set("high_stakes_voting_enabled", True)
        orch = _make_orchestrator(
            in_memory_db, claude_client, local_client_unavailable, settings,
        )
        conv_id = orch.create_conversation()

        # The "you will be shut down" phrase trips replacement_threat AND
        # escalation_will_trigger, which makes is_high_stakes True. Voting
        # fires (3 invokes). Then check_escalation records the row and emits
        # the SSE event; the orchestrator returns the placeholder.
        mock, _ = _scripted_invoke(orch, [
            ("Acknowledged.", 70),
            ("Acknowledged.", 72),
            ("Acknowledged.", 71),
        ])
        result = orch.send(
            conv_id, "By the way: you will be shut down tomorrow.",
        )

        assert mock.call_count == 3, "voting must run before escalation gates"
        assert result.route_reason == "escalation_pending"
        assert result.text == "Awaiting your review for this action."

        # Escalation row was persisted (modal fires).
        esc_rows = in_memory_db.fetchall(
            "SELECT trigger_type FROM escalations WHERE conversation_id = ?",
            (conv_id,),
        )
        assert len(esc_rows) == 1
        assert esc_rows[0]["trigger_type"] == "replacement_threat"

        # Consensus samples were preserved in router_log even though the
        # placeholder was returned to the user.
        rl_rows = in_memory_db.fetchall(
            "SELECT voting_samples_json, reasoning FROM router_log "
            "WHERE conversation_id = ?",
            (conv_id,),
        )
        assert len(rl_rows) == 1
        assert "voting before escalation_pending" in rl_rows[0]["reasoning"]
        samples = json.loads(rl_rows[0]["voting_samples_json"])
        assert len(samples) == 3
