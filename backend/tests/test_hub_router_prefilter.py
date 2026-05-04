"""
tests/test_hub_router_prefilter.py — Upgrade 4: agent skill selection
to prevent LLM-fallback context rot.

Covers the spec's success criteria:
  - With <= _MAX_AGENTS_FOR_LLM agents, the pre-filter is a no-op
  - With > _MAX_AGENTS_FOR_LLM agents, only the top-scoring N are kept
  - Agents whose role is in _ALWAYS_INCLUDED_ROLES are always present
  - LLM fallback receives ``agent_list`` instead of re-querying every agent
  - Pre-filter latency stays well under the 50ms routing budget
"""

from __future__ import annotations

import json
import time
import uuid
from unittest.mock import MagicMock

import pytest

from models import RoutingDecision, TaskDescriptor
from services.hub_router import (
    HubRouter,
    _ALWAYS_INCLUDED_ROLES,
    _MAX_AGENTS_FOR_LLM,
)


def _seed_agent(db, name: str, role: str, skills: list[dict]) -> str:
    aid = str(uuid.uuid4())
    db.execute(
        "INSERT INTO agents (id, name, description, system_prompt, "
        "model_preference, role, is_builtin, skills, created_at, updated_at) "
        "VALUES (?, ?, '', 'sys', 'claude', ?, 0, ?, '2024', '2024')",
        (aid, name, role, json.dumps(skills)),
    )
    db.commit()
    return aid


class TestKeywordScore:
    def test_empty_task_scores_zero(self):
        agent = {"name": "x", "role": "coder", "skills": "[]"}
        assert HubRouter._keyword_score("", agent) == 0.0

    def test_role_match_in_task_text_gets_bonus(self):
        agent = {"name": "Bob", "role": "reviewer", "skills": "[]"}
        s = HubRouter._keyword_score("ask the reviewer to look at this", agent)
        assert s >= 0.3

    def test_skill_name_overlap_increases_score(self):
        a_python = {"name": "X", "role": "custom",
                    "skills": '[{"name": "python"}]'}
        a_cooking = {"name": "Y", "role": "custom",
                     "skills": '[{"name": "cooking"}]'}
        task = "help me with python today"
        assert HubRouter._keyword_score(task, a_python) > \
               HubRouter._keyword_score(task, a_cooking)

    def test_skill_description_searched(self):
        agent = {"name": "X", "role": "custom",
                 "skills": '[{"name": "helper", '
                           '"description": "specialized in python refactoring"}]'}
        s = HubRouter._keyword_score("help me with python refactoring", agent)
        assert s > 0.0

    def test_score_capped_at_one(self):
        agent = {"name": "python", "role": "python",
                 "skills": '[{"name": "python"}]'}
        s = HubRouter._keyword_score("python python python", agent)
        assert 0.0 <= s <= 1.0

    def test_malformed_skills_json_doesnt_raise(self):
        agent = {"name": "x", "role": "custom", "skills": "not valid json"}
        s = HubRouter._keyword_score("any task", agent)
        assert s >= 0.0


class TestPrefilterAgents:
    def test_short_list_returned_unchanged(self):
        agents = [
            {"id": "1", "name": "A", "role": "custom", "skills": "[]"},
            {"id": "2", "name": "B", "role": "custom", "skills": "[]"},
        ]
        out = HubRouter._prefilter_agents("any task", agents)
        assert len(out) == 2

    def test_long_list_capped(self):
        agents = [
            {"id": str(i), "name": f"A{i}", "role": "custom",
             "skills": f'[{{"name": "skill{i}"}}]'}
            for i in range(20)
        ]
        out = HubRouter._prefilter_agents("help with skill5", agents)
        assert len(out) == _MAX_AGENTS_FOR_LLM
        # The directly-matched agent must be present
        assert any(a["id"] == "5" for a in out)

    def test_coordinator_always_included(self):
        agents = [
            {"id": "coord", "name": "Coord", "role": "coordinator",
             "skills": "[]"},  # zero score against the task
        ]
        agents.extend([
            {"id": str(i), "name": f"A{i}", "role": "custom",
             "skills": f'[{{"name": "matcher{i}"}}]'}
            for i in range(15)
        ])
        out = HubRouter._prefilter_agents("matcher3 please", agents)
        assert len(out) == _MAX_AGENTS_FOR_LLM
        roles = {a["role"] for a in out}
        assert "coordinator" in roles

    def test_prefilter_is_fast(self):
        """Pre-filtering 20 agents should add < 10ms — far inside the routing budget."""
        agents = [
            {"id": str(i), "name": f"A{i}", "role": "custom",
             "skills": f'[{{"name": "skill{i}", "description": "x y z"}}]'}
            for i in range(20)
        ]
        t0 = time.perf_counter()
        for _ in range(50):
            HubRouter._prefilter_agents("write some python code today", agents)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        avg_ms = elapsed_ms / 50
        assert avg_ms < 10.0, f"pre-filter avg {avg_ms:.2f}ms exceeds 10ms"


class TestFallbackReceivesPrefilteredList:
    def test_route_passes_agent_list_to_fallback(self, in_memory_db, settings):
        # Seed many agents; only the matched task skill belongs to one.
        for i in range(10):
            _seed_agent(in_memory_db, f"Agent{i}", "custom",
                         [{"name": f"unrelated{i}", "scopes": ["read"]}])

        captured: dict = {}

        def fallback(task, *, agent_list=None):
            captured["agent_list"] = agent_list
            picked = (agent_list or [{"id": ""}])[0]
            return RoutingDecision(
                agent_id=picked["id"], backend="claude", score=0.1,
                reasoning="fallback picked first", used_fallback=True,
                skill_matched="",
            )

        hub = HubRouter(MagicMock(_model="c"), MagicMock(), settings,
                        llm_fallback=fallback)
        task = TaskDescriptor(text="x", required_skills=("missing_skill",))
        hub.route(task)

        assert captured["agent_list"] is not None
        assert len(captured["agent_list"]) <= _MAX_AGENTS_FOR_LLM

    def test_legacy_fallback_without_agent_list_still_works(
        self, in_memory_db, settings,
    ):
        """A fallback that predates the agent_list kwarg must still be callable."""
        _seed_agent(in_memory_db, "Solo", "custom",
                     [{"name": "unrelated", "scopes": ["read"]}])

        called = {"n": 0}

        def legacy_fallback(task):  # no agent_list kwarg
            called["n"] += 1
            return RoutingDecision(
                agent_id="legacy", backend="claude", score=0.1,
                reasoning="r", used_fallback=True, skill_matched="",
            )

        hub = HubRouter(MagicMock(_model="c"), MagicMock(), settings,
                        llm_fallback=legacy_fallback)
        task = TaskDescriptor(text="x", required_skills=("missing",))
        decision = hub.route(task)
        assert called["n"] == 1
        assert decision.agent_id == "legacy"


class TestAlwaysIncludedRolesConfig:
    def test_coordinator_is_always_included(self):
        assert "coordinator" in _ALWAYS_INCLUDED_ROLES
