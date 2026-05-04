"""
tests/test_agent_registry.py

Covers:
- update_agent allowlist enforcement
- create / duplicate / delete agents
- builtin agent protection
- team CRUD
"""

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def seed(in_memory_db):
    """Seed agents once per test."""
    from services.agent_registry import seed_agents
    seed_agents()
    return in_memory_db


# ── Allowlist enforcement ─────────────────────────────────────────────────────

class TestAllowlistEnforcement:
    def test_update_disallowed_field_raises(self, in_memory_db):
        from services.agent_registry import create_agent, update_agent
        agent = create_agent("MyAgent", "desc", "You help.", model_preference="auto")
        with pytest.raises(ValueError, match="Unknown/disallowed"):
            update_agent(agent["id"], is_builtin=1)

    def test_update_disallowed_sql_injection_attempt(self, in_memory_db):
        from services.agent_registry import create_agent, update_agent
        agent = create_agent("HackAgent", "desc", "You help.", model_preference="auto")
        with pytest.raises(ValueError, match="Unknown/disallowed"):
            update_agent(agent["id"], **{"name; DROP TABLE agents; --": "evil"})

    def test_update_allowed_field_succeeds(self, in_memory_db):
        from services.agent_registry import create_agent, update_agent, get_agent
        agent = create_agent("GoodAgent", "desc", "You help.", model_preference="auto")
        update_agent(agent["id"], description="Updated description")
        refreshed = get_agent(agent["id"])
        assert refreshed["description"] == "Updated description"

    def test_update_nonexistent_agent_raises(self, in_memory_db):
        from services.agent_registry import update_agent
        with pytest.raises(ValueError, match="not found"):
            update_agent("00000000-0000-0000-0000-000000000000", description="x")


# ── Create / Delete ───────────────────────────────────────────────────────────

class TestCreateDeleteAgent:
    def test_create_agent(self, in_memory_db):
        from services.agent_registry import create_agent, get_agent
        result = create_agent("TestBot", "A test bot", "You test things.")
        agent = get_agent(result["id"])
        assert agent["name"] == "TestBot"
        assert agent["is_builtin"] == 0

    def test_create_sets_defaults(self, in_memory_db):
        from services.agent_registry import create_agent, get_agent
        result = create_agent("DefaultBot", "desc", "prompt")
        agent = get_agent(result["id"])
        assert agent["model_preference"] == "auto"
        assert agent["temperature"] == pytest.approx(0.7)
        assert agent["max_tokens"] == 4096

    def test_delete_custom_agent(self, in_memory_db):
        from services.agent_registry import create_agent, delete_agent, get_agent
        result = create_agent("Temp", "desc", "prompt")
        delete_agent(result["id"])
        assert get_agent(result["id"]) is None

    def test_delete_nonexistent_is_silent(self, in_memory_db):
        from services.agent_registry import delete_agent
        # Should not raise
        delete_agent("00000000-0000-0000-0000-000000000099")


# ── Builtin protection ────────────────────────────────────────────────────────

class TestBuiltinProtection:
    def _get_builtin_id(self, in_memory_db):
        row = in_memory_db.fetchone("SELECT id FROM agents WHERE is_builtin = 1 LIMIT 1")
        assert row is not None, "No builtin agents seeded"
        return row["id"]

    def test_cannot_edit_builtin(self, in_memory_db):
        from services.agent_registry import update_agent
        bid = self._get_builtin_id(in_memory_db)
        with pytest.raises(ValueError, match="Built-in"):
            update_agent(bid, description="hacked")

    def test_cannot_delete_builtin(self, in_memory_db):
        from services.agent_registry import delete_agent
        bid = self._get_builtin_id(in_memory_db)
        with pytest.raises(ValueError, match="Built-in"):
            delete_agent(bid)

    def test_duplicate_builtin_creates_custom(self, in_memory_db):
        from services.agent_registry import duplicate_agent, get_agent
        bid = self._get_builtin_id(in_memory_db)
        result = duplicate_agent(bid, "My Custom Copy")
        copied = get_agent(result["id"])
        assert copied["is_builtin"] == 0
        assert copied["name"] == "My Custom Copy"

    def test_duplicate_then_edit(self, in_memory_db):
        from services.agent_registry import duplicate_agent, update_agent, get_agent
        bid = self._get_builtin_id(in_memory_db)
        copy = duplicate_agent(bid, "Editable Copy")
        update_agent(copy["id"], description="changed")
        refreshed = get_agent(copy["id"])
        assert refreshed["description"] == "changed"


# ── List / GetByName ──────────────────────────────────────────────────────────

class TestListAgents:
    def test_list_agents_returns_builtins(self, in_memory_db):
        from services.agent_registry import list_agents
        agents = list_agents()
        names = [a["name"] for a in agents]
        assert "General Assistant" in names

    def test_get_agent_by_name(self, in_memory_db):
        from services.agent_registry import get_agent_by_name
        agent = get_agent_by_name("General Assistant")
        assert agent is not None
        assert agent["is_builtin"] == 1

    def test_get_agent_by_name_missing(self, in_memory_db):
        from services.agent_registry import get_agent_by_name
        assert get_agent_by_name("DoesNotExist") is None


# ── Team CRUD ─────────────────────────────────────────────────────────────────

class TestTeamCRUD:
    def _coord_id(self, in_memory_db):
        row = in_memory_db.fetchone("SELECT id FROM agents LIMIT 1")
        return row["id"]

    def test_create_team(self, in_memory_db):
        from services.agent_registry import create_team, list_teams
        coord_id = self._coord_id(in_memory_db)
        create_team("Alpha Team", "Test team", coord_id)
        teams = list_teams()
        names = [t["name"] for t in teams]
        assert "Alpha Team" in names

    def test_add_and_remove_member(self, in_memory_db):
        from services.agent_registry import (
            create_team, add_team_member, remove_team_member, get_team_with_members
        )
        coord_id = self._coord_id(in_memory_db)
        team = create_team("Beta", "desc", coord_id)
        # Add a different agent
        other = in_memory_db.fetchone(
            "SELECT id FROM agents WHERE id != ? LIMIT 1", (coord_id,)
        )
        if other:
            add_team_member(team["id"], other["id"])
            team_full = get_team_with_members(team["id"])
            member_ids = [m["id"] for m in team_full["members"]]
            assert other["id"] in member_ids
            remove_team_member(team["id"], other["id"])
            team_full2 = get_team_with_members(team["id"])
            member_ids2 = [m["id"] for m in team_full2["members"]]
            assert other["id"] not in member_ids2

    def test_delete_team(self, in_memory_db):
        from services.agent_registry import create_team, delete_team, list_teams
        coord_id = self._coord_id(in_memory_db)
        team = create_team("Temp Team", "desc", coord_id)
        delete_team(team["id"])
        teams = list_teams()
        ids = [t["id"] for t in teams]
        assert team["id"] not in ids
