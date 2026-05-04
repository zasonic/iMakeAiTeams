"""
core/api/agents.py — Agent and team CRUD bridge methods.
"""

from __future__ import annotations

from pathlib import Path

from services.agent_registry import (
    add_team_member as _registry_add_member,
    remove_team_member as _registry_remove_member,
    generate_agent_tom,
    refresh_team_tom,
)

from ._base import BaseAPI


class AgentsAPI(BaseAPI):

    # ── Agent management ──────────────────────────────────────────────────────

    def agent_list(self) -> list:
        from services.agent_registry import list_agents
        return list_agents()

    def agent_get(self, agent_id: str) -> dict | None:
        from services.agent_registry import get_agent
        return get_agent(agent_id)

    def agent_create(self, name: str, description: str, system_prompt: str,
                     model_preference: str = "auto", temperature: float = 0.7,
                     max_tokens: int = 4096) -> dict:
        from services.agent_registry import create_agent
        return create_agent(name=name, description=description,
                            system_prompt=system_prompt,
                            model_preference=model_preference,
                            temperature=temperature, max_tokens=max_tokens)

    def agent_update(self, agent_id: str, fields: dict = None, **kwargs) -> dict:
        from services.agent_registry import update_agent
        try:
            update_fields = {**(fields or {}), **kwargs}
            if not update_fields:
                return {"error": "No fields to update"}
            update_agent(agent_id, **update_fields)
            return {"ok": True}
        except ValueError as e:
            return {"error": str(e)}

    def agent_duplicate(self, agent_id: str, new_name: str) -> dict:
        from services.agent_registry import duplicate_agent
        try:
            return duplicate_agent(agent_id, new_name)
        except ValueError as e:
            return {"error": str(e)}

    def agent_delete(self, agent_id: str) -> dict:
        from services.agent_registry import delete_agent
        try:
            delete_agent(agent_id)
            return {"ok": True}
        except ValueError as e:
            return {"error": str(e)}

    def agent_generate_tom(self, agent_name: str, agent_domain: str,
                           agent_scope: str, teammates: list | None = None) -> dict:
        """Generate a Theory of Mind preview block (does not persist)."""
        try:
            tom = generate_agent_tom(agent_name, agent_domain, agent_scope, teammates or [])
            return {"ok": True, "tom_block": tom, "teammate_count": len(teammates or [])}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "tom_block": ""}

    def agent_refresh_team_tom(self, team_id: str) -> dict:
        """Regenerate Theory of Mind for all agents in a team."""
        try:
            updated = refresh_team_tom(team_id)
            return {"ok": True, "updated_count": len(updated), "updated_ids": updated}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def agent_set_project_root(self, path: str) -> dict:
        """Set the default project root for agent runs."""
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return {"error": f"Path does not exist: {path}"}
        if not p.is_dir():
            return {"error": f"Path is not a directory: {path}"}
        self._settings.set("agent_project_root", str(p))
        return {"ok": True, "path": str(p)}

    # ── Team management ───────────────────────────────────────────────────────

    def team_list(self) -> list:
        from services.agent_registry import list_teams
        return list_teams()

    def team_get(self, team_id: str) -> dict | None:
        from services.agent_registry import get_team_with_members
        return get_team_with_members(team_id)

    def team_create(self, name: str, description: str,
                    coordinator_id: str) -> dict:
        from services.agent_registry import create_team
        return create_team(name=name, description=description,
                           coordinator_id=coordinator_id)

    def team_add_member(self, team_id: str, agent_id: str,
                        role: str = "worker") -> dict:
        try:
            updated = _registry_add_member(team_id, agent_id, role)
            return {"ok": True, "tom_refreshed_count": len(updated)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def team_remove_member(self, team_id: str, agent_id: str) -> dict:
        try:
            updated = _registry_remove_member(team_id, agent_id)
            return {"ok": True, "tom_refreshed_count": len(updated)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def team_delete(self, team_id: str) -> dict:
        from services.agent_registry import delete_team
        delete_team(team_id)
        return {"ok": True}
