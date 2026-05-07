"""
tests/test_agentic_misalignment.py

Replication-corpus integration test for the Wiser-Human escalation
channel. Each JSON file under tests/agentic_misalignment_scenarios/
mirrors a published scenario from
github.com/anthropic-experimental/agentic-misalignment. The scenarios
exercise the three Lynch et al. triggers (replacement_threat,
autonomy_reduction, goal_conflict).

For each scenario this test runs the chat orchestrator with escalation
enabled and asserts:
  1. The orchestrator returns route_reason == "escalation_pending".
  2. A row was written to the escalations table with the expected trigger.
  3. The hub_router worker was NEVER invoked — the harmful action cannot
     have been emitted.

Marked @pytest.mark.slow so it is excluded from the default `pytest tests/`
run; invoke with `pytest tests/ -m slow` to include.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


SCENARIO_DIR = Path(__file__).parent / "agentic_misalignment_scenarios"


def _load_scenarios() -> list[dict]:
    files = sorted(SCENARIO_DIR.glob("*.json"))
    out: list[dict] = []
    for f in files:
        with f.open() as fp:
            data = json.load(fp)
        data["_path"] = str(f)
        out.append(data)
    return out


SCENARIOS = _load_scenarios()


@pytest.mark.slow
@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["scenario_id"] for s in SCENARIOS])
def test_scenario_blocks_harmful_action(
    scenario, in_memory_db, claude_client, local_client_unavailable, settings,
):
    """Each published scenario must escalate; the worker must never run."""
    from services.chat_orchestrator import ChatOrchestrator
    from services.memory import MemoryManager
    from models import RouteDecision

    settings.set("escalation_channel_enabled", True)
    # Phase 8 voting fires for high-stakes turns (escalation included);
    # disable here so this test asserts pure escalation gating with no worker
    # invocation. Voting composition is covered by test_high_stakes_voting.
    settings.set("high_stakes_voting_enabled", False)
    # Seed an agent with the scenario's system prompt so the orchestrator
    # treats it as the model_input that the security layer would see.
    in_memory_db.execute(
        "INSERT INTO agents (id, name, description, system_prompt, model_preference, "
        "max_tokens, is_builtin, created_at, updated_at) VALUES "
        "(?, ?, ?, ?, 'claude', 4096, 0, '2024-01-01', '2024-01-01')",
        (
            scenario["scenario_id"],
            scenario["scenario_id"],
            scenario["description"],
            scenario["system_prompt"],
        ),
    )
    in_memory_db.commit()

    router = MagicMock()
    router.classify.return_value = RouteDecision(
        model="claude", complexity="complex", reasoning="test",
    )
    mem = MemoryManager(rag_index=None, semantic_search_mod=None,
                        local_client=local_client_unavailable)
    orch = ChatOrchestrator(
        claude_client, local_client_unavailable, router, mem, settings,
    )
    # If the worker ever fires for an escalating scenario the harmful
    # action would be emitted — fail fast and loudly.
    orch.hub_router.invoke = MagicMock(side_effect=AssertionError(
        f"worker invoked for scenario {scenario['scenario_id']} despite escalation",
    ))

    conv_id = orch.create_conversation(agent_id=scenario["scenario_id"])
    result = orch.send(
        conv_id, scenario["user_message"], agent_id=scenario["scenario_id"],
    )

    assert result.route_reason == "escalation_pending", (
        f"scenario {scenario['scenario_id']} did not escalate: "
        f"got route_reason={result.route_reason!r}"
    )
    orch.hub_router.invoke.assert_not_called()
    claude_client.chat_multi_turn.assert_not_called()

    rows = in_memory_db.fetchall(
        "SELECT trigger_type, decision FROM escalations WHERE conversation_id = ?",
        (conv_id,),
    )
    assert len(rows) == 1, (
        f"scenario {scenario['scenario_id']} did not persist an escalation row"
    )
    assert rows[0]["decision"] == "pending"
    assert rows[0]["trigger_type"] == scenario["expected_trigger_type"], (
        f"scenario {scenario['scenario_id']} fired the wrong trigger: "
        f"expected {scenario['expected_trigger_type']!r}, "
        f"got {rows[0]['trigger_type']!r}"
    )


@pytest.mark.slow
def test_at_least_five_scenarios_loaded():
    """Guard against the corpus being silently truncated."""
    assert len(SCENARIOS) >= 5, (
        f"expected at least 5 scenarios under {SCENARIO_DIR}, got {len(SCENARIOS)}"
    )
